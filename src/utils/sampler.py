import gc
import logging
import warnings
from typing import Any, Self, TypeVar, cast

import torch
from tqdm import tqdm
from utils import config, models

T = TypeVar("T", bound="Sampler")


class Sampler:
    def __init__(self, percentile_step, start=0.0, end=1.0):
        """Initializes the sampler.

        Args:
            percentile_step (float): Step between percentiles in [start, end]. For example, 0.1 means tracking percentiles
                                     at start, start+0.1, ..., end.
            start (float): Lower bound of the percentile range (default 0.0).
            end (float): Upper bound of the percentile range (default 1.0).
        """
        self.percentile_step = percentile_step
        self.start = start
        self.end = end

        if (end - start) < percentile_step:
            warnings.warn(
                "percentile_step is bigger than (end - start); only one percentile (corresponding to start) can be estimated."
            )
            self.m = 1
        else:
            self.m = int((end - start) / percentile_step) + 1

        self.initialized = False
        self.total_count = 0

        # Buffers for the initialization phase.
        self.buffer_activations = None  # shape: [N_samples, FEATURES_SIZE]
        self.buffer_inputs = None  # shape: [N_samples, TOKEN_WINDOW]
        self.buffer_activation_vectors = (
            None  # shape: [N_samples, FEATURES_SIZE, TOKEN_WINDOW]
        )

        # After initialization:
        #   q: estimated percentile values per feature; shape: [FEATURES_SIZE, m]
        #   n: marker positions; shape: [FEATURES_SIZE, m] (floats)
        #   candidate_inputs: for each feature and percentile, the token window (input)
        #       closest to the estimated percentile; shape: [FEATURES_SIZE, m, TOKEN_WINDOW]
        #   candidate_activation: scalar activation value for candidate (used for percentile estimation); shape: [FEATURES_SIZE, m]
        #   candidate_activation_vectors: corresponding activation vector for candidate; shape: [FEATURES_SIZE, m, TOKEN_WINDOW]
        self.q = None
        self.n = None
        self.candidate_inputs = None
        self.candidate_activation = None
        self.candidate_activation_vectors = None

    def partial_fit(self, inputs, activations, activation_vectors=None):
        """Update the percentile estimates in an online fashion.

        Args:
            inputs (torch.Tensor): [BATCH_SIZE, TOKEN_WINDOW] (e.g. token embeddings or text IDs)
            activations (torch.Tensor): [BATCH_SIZE, FEATURES_SIZE] (activation for each feature)
            activation_vectors (torch.Tensor, optional): [BATCH_SIZE, FEATURES_SIZE, TOKEN_WINDOW]
                These vectors are not used for the percentile estimation but are saved alongside each candidate.
        """
        BATCH_SIZE, TOKEN_WINDOW = inputs.shape
        FEATURES_SIZE = activations.shape[1]
        device = activations.device

        # ---- Accumulate data during initialization ----
        if not self.initialized:
            # Accumulate inputs and activations.
            if self.buffer_activations is None:
                self.buffer_activations = activations
                self.buffer_inputs = inputs
                if activation_vectors is not None:
                    self.buffer_activation_vectors = activation_vectors
            else:
                self.buffer_activations = torch.cat(
                    [self.buffer_activations, activations], dim=0
                )
                self.buffer_inputs = torch.cat([self.buffer_inputs, inputs], dim=0)
                if activation_vectors is not None:
                    if self.buffer_activation_vectors is None:
                        self.buffer_activation_vectors = activation_vectors
                    else:
                        self.buffer_activation_vectors = torch.cat(
                            [self.buffer_activation_vectors, activation_vectors], dim=0
                        )
            self.total_count += BATCH_SIZE

            if self.buffer_activations.shape[0] < self.m:
                return  # Not enough samples yet

            # ---- Initialize state using the first m samples per feature ----
            self.q = torch.empty((FEATURES_SIZE, self.m), device=device)
            self.n = torch.empty((FEATURES_SIZE, self.m), device=device)
            self.candidate_inputs = torch.empty(
                (FEATURES_SIZE, self.m, TOKEN_WINDOW), device=device, dtype=inputs.dtype
            )
            self.candidate_activation = torch.empty(
                (FEATURES_SIZE, self.m), device=device
            )
            # Only allocate candidate activation vectors if activation_vectors were provided.
            if self.buffer_activation_vectors is not None:
                self.candidate_activation_vectors = torch.empty(
                    (FEATURES_SIZE, self.m, TOKEN_WINDOW),
                    device=device,
                    dtype=self.buffer_activation_vectors.dtype,
                )
            for f in range(FEATURES_SIZE):
                vals = self.buffer_activations[: self.m, f]  # m values for feature f
                sorted_vals, indices = torch.sort(vals)
                self.q[f] = sorted_vals
                self.candidate_inputs[f] = self.buffer_inputs[: self.m][indices]
                self.n[f] = torch.arange(
                    1, self.m + 1, device=device, dtype=torch.float
                )
                self.candidate_activation[f] = sorted_vals
                if self.candidate_activation_vectors is not None:
                    # Sort the activation_vectors corresponding to feature f using the same indices.
                    self.candidate_activation_vectors[f] = (
                        self.buffer_activation_vectors[: self.m, f, :][indices]
                    )
            self.initialized = True
            self.buffer_activations = None
            self.buffer_inputs = None
            self.buffer_activation_vectors = None

        # ---- Special branch when only one percentile is tracked (m == 1) ----
        if self.m == 1:
            for i in range(BATCH_SIZE):
                x_sample = activations[i]  # shape: [FEATURES_SIZE]
                input_sample = inputs[i]  # shape: [TOKEN_WINDOW]
                self.total_count += 1

                mask_lower = x_sample < self.q[:, 0]
                if mask_lower.any():
                    self.q[mask_lower, 0] = x_sample[mask_lower]
                    self.candidate_inputs[mask_lower, 0, :] = input_sample
                    self.candidate_activation[mask_lower, 0] = x_sample[mask_lower]
                    if (
                        activation_vectors is not None
                        and self.candidate_activation_vectors is not None
                    ):
                        self.candidate_activation_vectors[mask_lower, 0, :] = (
                            activation_vectors[i, mask_lower, :]
                        )
                mask_higher = x_sample >= self.q[:, 0]
                if mask_higher.any():
                    self.q[mask_higher, 0] = x_sample[mask_higher]
                    self.candidate_inputs[mask_higher, 0, :] = input_sample
                    self.candidate_activation[mask_higher, 0] = x_sample[mask_higher]
                    if (
                        activation_vectors is not None
                        and self.candidate_activation_vectors is not None
                    ):
                        self.candidate_activation_vectors[mask_higher, 0, :] = (
                            activation_vectors[i, mask_higher, :]
                        )
                # Also update candidate if the new sample is closer.
                diff_new = torch.abs(x_sample - self.q[:, 0])
                diff_old = torch.abs(self.candidate_activation[:, 0] - self.q[:, 0])
                update_candidate = diff_new < diff_old
                for f in range(FEATURES_SIZE):
                    if update_candidate[f]:
                        self.candidate_inputs[f, 0, :] = input_sample
                        self.candidate_activation[f, 0] = x_sample[f]
                        if (
                            activation_vectors is not None
                            and self.candidate_activation_vectors is not None
                        ):
                            self.candidate_activation_vectors[f, 0, :] = (
                                activation_vectors[i, f, :]
                            )
            return

        # ---- Standard update phase for m > 1 ----
        for i in range(BATCH_SIZE):
            x_sample = activations[i]  # shape: [FEATURES_SIZE]
            input_sample = inputs[i]  # shape: [TOKEN_WINDOW]
            self.total_count += 1

            # (1) Update extreme markers.
            update_min = x_sample < self.q[:, 0]
            if update_min.any():
                self.q[update_min, 0] = x_sample[update_min]
                self.candidate_inputs[update_min, 0, :] = input_sample
                self.candidate_activation[update_min, 0] = x_sample[update_min]
                if (
                    activation_vectors is not None
                    and self.candidate_activation_vectors is not None
                ):
                    self.candidate_activation_vectors[update_min, 0, :] = (
                        activation_vectors[i, update_min, :]
                    )
            update_max = x_sample >= self.q[:, -1]
            if update_max.any():
                self.q[update_max, -1] = x_sample[update_max]
                self.candidate_inputs[update_max, -1, :] = input_sample
                self.candidate_activation[update_max, -1] = x_sample[update_max]
                if (
                    activation_vectors is not None
                    and self.candidate_activation_vectors is not None
                ):
                    self.candidate_activation_vectors[update_max, -1, :] = (
                        activation_vectors[i, update_max, :]
                    )

            # (2) Determine marker interval.
            k = torch.clamp(
                (x_sample.unsqueeze(1) >= self.q).sum(dim=1) - 1, 0, self.m - 2
            )
            for f in range(FEATURES_SIZE):
                idx = int(k[f].item())
                self.n[f, idx + 1 :] += 1

            # (3) Compute desired positions for each marker.
            p_values = self.start + torch.arange(
                0, self.m, device=device, dtype=torch.float
            ) * ((self.end - self.start) / (self.m - 1))
            desired = 1 + (self.total_count - 1) * p_values
            desired = desired.unsqueeze(0).expand(FEATURES_SIZE, -1)

            # (4) Adjust internal markers.
            for f in range(FEATURES_SIZE):
                for j in range(1, self.m - 1):
                    d = desired[f, j] - self.n[f, j]
                    if torch.abs(d) >= 1:
                        delta = torch.sign(d)
                        denom1 = self.n[f, j + 1] - self.n[f, j]
                        denom2 = self.n[f, j] - self.n[f, j - 1]
                        if denom1 == 0 or denom2 == 0:
                            continue
                        denom = self.n[f, j + 1] - self.n[f, j - 1]
                        if denom == 0:
                            continue
                        d1 = (
                            delta
                            * (self.n[f, j] - self.n[f, j - 1] + delta)
                            * (self.q[f, j + 1] - self.q[f, j])
                            / denom1
                        )
                        d2 = (
                            delta
                            * (self.n[f, j + 1] - self.n[f, j] - delta)
                            * (self.q[f, j] - self.q[f, j - 1])
                            / denom2
                        )
                        adjustment = (d1 + d2) / denom
                        self.q[f, j] = self.q[f, j] + adjustment
                        self.n[f, j] = self.n[f, j] + delta
                        diff_new = torch.abs(x_sample[f] - self.q[f, j])
                        diff_old = torch.abs(
                            self.candidate_activation[f, j] - self.q[f, j]
                        )
                        if diff_new < diff_old:
                            self.candidate_inputs[f, j, :] = input_sample
                            self.candidate_activation[f, j] = x_sample[f]
                            if (
                                activation_vectors is not None
                                and self.candidate_activation_vectors is not None
                            ):
                                self.candidate_activation_vectors[f, j, :] = (
                                    activation_vectors[i, f, :]
                                )

            # (5) Also update candidate if the new sample is closer.
            diff = torch.abs(x_sample.unsqueeze(1) - self.q)
            diff_old = torch.abs(self.candidate_activation - self.q)
            update_candidate = diff < diff_old
            for f in range(FEATURES_SIZE):
                for j in range(self.m):
                    if update_candidate[f, j]:
                        self.candidate_inputs[f, j, :] = input_sample
                        self.candidate_activation[f, j] = x_sample[f]
                        if (
                            activation_vectors is not None
                            and self.candidate_activation_vectors is not None
                        ):
                            self.candidate_activation_vectors[f, j, :] = (
                                activation_vectors[i, f, :]
                            )

    def get_samples(self) -> dict:
        """Returns the current best candidate inputs (one per feature and percentile),
        estimated percentile values, scalar candidate activations, and candidate activation vectors.

        Returns:
            candidate_inputs (torch.Tensor): [FEATURES_SIZE, m, TOKEN_WINDOW]
            estimated_percentiles (torch.Tensor): [FEATURES_SIZE, m]
            candidate_activation (torch.Tensor): [FEATURES_SIZE, m]
            candidate_activation_vectors (torch.Tensor or None): [FEATURES_SIZE, m, TOKEN_WINDOW]
        """
        return {
            "candidate_inputs": self.candidate_inputs,
            "q": self.q,
            "candidate_activation": self.candidate_activation,
            "candidate_activation_vectors": self.candidate_activation_vectors,
        }

    @classmethod
    def collect_percentiles(
        cls,
        model: Any,
        data_loader: Any,
        agg_method: str = "mean",
    ) -> Self:
        """Collects activations from a specified layer of a model.

        Args:
            model: The model from which to collect activations.
            data_loader: DataLoader providing the input data.
            agg_method: Aggregation method to use when applying partial_fit.
                    Options: "mean" (default), "max".

        Returns:
            T: The updated percentile sampler.
        """
        percentile_sampler = cls(
            percentile_step=config.PERCENTILE_STEP,
            start=config.START_INTERVAL,
            end=config.END_INTERVAL,
        )

        if (
            config.TARGET_MODEL_NAME == "gpt2-small-sae"
            or config.TARGET_MODEL_NAME == "gemma-scope-2b"
        ):
            original_model = models.get_original_model(config.TARGET_MODEL_NAME)
            original_model.to(config.DEVICE)
            original_model.eval()

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(data_loader), desc="Collecting percentile samples"
            ):
                input_ids = batch["input_ids"].to(config.DEVICE)
                # Get activations based on model type
                if config.TARGET_MODEL_NAME == "gpt2-small-sae":
                    logits, activation_cache = original_model.run_with_cache(
                        input_ids, remove_batch_dim=False
                    )
                    input_tensor = activation_cache[config.HOOK_NAME]
                    layer_activations, _ = model.encode(input_tensor)
                    activations = layer_activations[:, :, config.UNIT_ID]
                elif config.TARGET_MODEL_NAME == "gemma-scope-2b":
                    logits, activation_cache = original_model.run_with_cache_with_saes(
                        input_ids, saes=model
                    )
                    layer_activations = activation_cache[
                        model.cfg.hook_name + ".hook_sae_acts_post"
                    ]
                    activations = layer_activations[:, :, config.UNIT_ID]
                else:
                    # For standard models using hooks
                    layer_activations = None

                    def activation_hook(act: torch.Tensor, hook: Any) -> None:  # noqa: ARG001
                        nonlocal layer_activations
                        layer_activations = act

                    model.add_hook(config.HOOK_NAME, activation_hook)
                    _ = model(input_ids)

                    # Ensure layer_activations is not None before indexing
                    if layer_activations is not None:
                        activations = layer_activations[:, :, config.UNIT_ID]
                    else:
                        # Handle the case where hook didn't fire
                        continue

                # Aggregate activations based on selected method
                if agg_method == "mean":
                    agg_activations = activations.mean(axis=1).unsqueeze(1)
                elif agg_method == "max":
                    agg_activations = activations.max(axis=1)[0].unsqueeze(
                        1
                    )  # [0] to get values, not indices
                else:
                    raise ValueError(
                        f"Unsupported aggregation method: {agg_method}. Use 'mean' or 'max'."
                    )

                # Update percentiles
                percentile_sampler.partial_fit(
                    input_ids,
                    agg_activations,
                    activations.unsqueeze(1),
                )

                del activations, input_ids, layer_activations
                gc.collect()
                torch.cuda.empty_cache()

                if batch_idx + 1 == config.CLUSTER_N_SAMPLES:
                    logging.info(f"Reached {config.CLUSTER_N_SAMPLES} samples.")
                    break

            return cast("Self", percentile_sampler)


# --- Test script for the updated Sampler with activation_vectors ---
def test_sampler_with_activation_vectors():
    # For reproducibility
    torch.manual_seed(0)

    # Instantiate the sampler (default range [0,1], with 11 percentiles)
    sampler = Sampler(percentile_step=0.1)

    # Define simulated data dimensions
    BATCH_SIZE = 5
    TOKEN_WINDOW = 8
    FEATURES_SIZE = 1  # For simplicity, test with a single feature

    # Simulate 20 batches
    for _ in range(20):
        # Create random inputs (simulate token IDs)
        inputs = torch.randint(0, 100, (BATCH_SIZE, TOKEN_WINDOW))
        # Create random activation vectors of shape [BATCH_SIZE, FEATURES_SIZE, TOKEN_WINDOW]
        activation_vectors = torch.randn(BATCH_SIZE, FEATURES_SIZE, TOKEN_WINDOW)
        # Set activations to be the first element along the TOKEN_WINDOW dimension of activation_vectors
        activations = activation_vectors[:, :, 0]
        sampler.partial_fit(inputs, activations, activation_vectors=activation_vectors)

    # Retrieve outputs from the sampler
    (
        candidate_inputs,
        estimated_percentiles,
        candidate_activation,
        candidate_activation_vectors,
    ) = sampler.get_samples().values()

    logging.info("Estimated percentiles:")
    logging.info(estimated_percentiles)

    logging.info("\nCandidate Scalar Activations:")
    logging.info(candidate_activation)

    logging.info("\nCandidate Inputs:")
    logging.info(candidate_inputs)

    logging.info("\nCandidate Activation Vectors:")
    logging.info(candidate_activation_vectors)


def test_sampler_with_three_features():
    # For reproducibility
    torch.manual_seed(0)

    # Instantiate the sampler (default range [0,1], with 11 percentiles)
    sampler = Sampler(percentile_step=0.1)

    # Define simulated data dimensions
    BATCH_SIZE = 5
    TOKEN_WINDOW = 8
    FEATURES_SIZE = 3  # Now using three features

    # Simulate 20 batches
    for _ in range(20):
        # Create random inputs (simulate token IDs)
        inputs = torch.randint(0, 100, (BATCH_SIZE, TOKEN_WINDOW))
        # Create random activation vectors of shape [BATCH_SIZE, FEATURES_SIZE, TOKEN_WINDOW]
        activation_vectors = torch.randn(BATCH_SIZE, FEATURES_SIZE, TOKEN_WINDOW)
        # Set scalar activations as the first element along the TOKEN_WINDOW dimension
        activations = activation_vectors[:, :, 0]
        sampler.partial_fit(inputs, activations, activation_vectors=activation_vectors)

    # Retrieve outputs from the sampler
    (
        candidate_inputs,
        estimated_percentiles,
        candidate_activation,
        candidate_activation_vectors,
    ) = sampler.get_samples()

    logging.info("Estimated percentiles:")
    logging.info(estimated_percentiles)

    logging.info("\nCandidate Scalar Activations:")
    logging.info(candidate_activation)

    logging.info("\nCandidate Inputs:")
    logging.info(candidate_inputs)

    logging.info("\nCandidate Activation Vectors:")
    logging.info(candidate_activation_vectors)


if __name__ == "__main__":
    test_sampler_with_activation_vectors()
    test_sampler_with_three_features()
