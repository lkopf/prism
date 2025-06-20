from collections.abc import Callable
from typing import Any, cast

import torch
from torch import nn


class SAEPath:
    """Paths for GPT-2 small sparse autoencoder models.

    Note:
        Adapted from:
        https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/paths.py
    """

    id2hook = {
        "mlp.hook_post": "mlp_post_act",
        "hook_attn_out": "resid_delta_attn",
        "hook_resid_mid": "resid_post_attn",
        "hook_mlp_out": "resid_delta_mlp",
        "hook_resid_post": "resid_post_mlp",
    }

    @classmethod
    def get_path(cls, version: str, location: str, layer_index: int) -> str:
        """Get the path for a specific version, location, and layer index.

        Args:
            version (str): The version method to use for generating the path.
            location (str): The location identifier to be translated if necessary.
            layer_index (int): The index of the layer for which the path is needed.

        Returns:
            str: The generated path based on the version method, translated location, and layer index.

        Raises:
            ValueError: If the specified version method is not found in the class.

        """
        version_method = getattr(cls, version, None)

        if version_method is None:
            raise ValueError(
                f"Version '{version}' not found. "
                f"Available versions: {[method for method in dir(cls) if not method.startswith('_')]}"
            )
        # Translate location if it exists in the dictionary
        translated_location = cls.id2hook.get(location, location)

        return version_method(translated_location, layer_index)

    @staticmethod
    def v1(location: str, layer_index: int) -> str:
        """Details:
        - Number of autoencoder latents: 32768
        - Number of training tokens: ~64M
        - Activation function: ReLU
        - L1 regularization strength: 0.01
        - Layer normed inputs: false
        - NeuronRecord files:
            `az://openaipublic/sparse-autoencoder/gpt2-small/{location}/collated_activations/{layer_index}/{latent_index}.json`
        """
        if location not in ["mlp_post_act", "resid_delta_mlp"]:
            raise ValueError(f"Invalid location: {location}")
        if layer_index not in range(12):
            raise ValueError(
                f"layer_index must be between 0 and 11, but got {layer_index}"
            )
        return f"az://openaipublic/sparse-autoencoder/gpt2-small/{location}/autoencoders/{layer_index}.pt"

    @staticmethod
    def v4(location: str, layer_index: int) -> str:
        """Details:
        same as v1
        """
        if location not in ["mlp_post_act", "resid_delta_mlp"]:
            raise ValueError(f"Invalid location: {location}")
        if layer_index not in range(12):
            raise ValueError(
                f"layer_index must be between 0 and 11, but got {layer_index}"
            )
        return f"az://openaipublic/sparse-autoencoder/gpt2-small/{location}_v4/autoencoders/{layer_index}.pt"

    @staticmethod
    def v5_32k(location: str, layer_index: int) -> str:
        """Details:
        - Number of autoencoder latents: 2**15 = 32768
        - Number of training tokens:  TODO
        - Activation function: TopK(32)
        - L1 regularization strength: n/a
        - Layer normed inputs: true
        """
        if location not in [
            "resid_delta_attn",
            "resid_delta_mlp",
            "resid_post_attn",
            "resid_post_mlp",
        ]:
            raise ValueError(f"Invalid location: {location}")
        if layer_index not in range(12):
            raise ValueError(
                f"layer_index must be between 0 and 11, but got {layer_index}"
            )
        # note: it's actually 2**15 and 2**17 ~= 131k
        return f"az://openaipublic/sparse-autoencoder/gpt2-small/{location}_v5_32k/autoencoders/{layer_index}.pt"

    @staticmethod
    def v5_128k(location: str, layer_index: int) -> str:
        """Details:
        - Number of autoencoder latents: 2**17 = 131072
        - Number of training tokens: TODO
        - Activation function: TopK(32)
        - L1 regularization strength: n/a
        - Layer normed inputs: true
        """
        if location not in [
            "resid_delta_attn",
            "resid_delta_mlp",
            "resid_post_attn",
            "resid_post_mlp",
        ]:
            raise ValueError(f"Invalid location: {location}")
        if layer_index not in range(12):
            raise ValueError(
                f"layer_index must be between 0 and 11, but got {layer_index}"
            )
        # note: it's actually 2**15 and 2**17 ~= 131k
        return f"az://openaipublic/sparse-autoencoder/gpt2-small/{location}_v5_128k/autoencoders/{layer_index}.pt"


def ln(
    x: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalizes the input tensor `x` by subtracting the mean and dividing by the standard deviation.

    Args:
        x (torch.Tensor): The input tensor to be normalized.
        eps (float, optional): A small value added to the standard deviation to avoid division by zero.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the normalized tensor,
        the mean, and the standard deviation.
    """
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std


class Autoencoder(nn.Module):
    """Sparse autoencoder

    Implements:
        latents = activation(encoder(x - pre_bias) + latent_bias)
        recons = decoder(latents) + pre_bias

    Note:
        Adapted from:
        https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/model.py
    """

    def __init__(
        self,
        n_latents: int,
        n_inputs: int,
        activation: Callable | None = None,
        tied: bool = False,
        normalize: bool = False,
    ) -> None:
        """:param n_latents: dimension of the autoencoder latent
        :param n_inputs: dimensionality of the original data (e.g residual stream, number of MLP hidden units)
        :param activation: activation function
        :param tied: whether to tie the encoder and decoder weights
        """
        super().__init__()

        self.pre_bias = nn.Parameter(torch.zeros(n_inputs))
        self.encoder: nn.Module = nn.Linear(n_inputs, n_latents, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        self.activation = activation if activation is not None else nn.ReLU()
        if tied:
            self.decoder: nn.Linear | TiedTranspose = TiedTranspose(self.encoder)
        else:
            self.decoder = nn.Linear(n_latents, n_inputs, bias=False)
        self.normalize = normalize

        self.stats_last_nonzero: torch.Tensor
        self.latents_activation_frequency: torch.Tensor
        self.latents_mean_square: torch.Tensor
        self.register_buffer(
            "stats_last_nonzero", torch.zeros(n_latents, dtype=torch.long)
        )
        self.register_buffer(
            "latents_activation_frequency", torch.ones(n_latents, dtype=torch.float)
        )
        self.register_buffer(
            "latents_mean_square", torch.zeros(n_latents, dtype=torch.float)
        )

    def encode_pre_act(
        self, x: torch.Tensor, latent_slice: slice | None = None
    ) -> torch.Tensor:
        """:param x: input data (shape: [batch, n_inputs])
        :param latent_slice: slice of latents to compute
            Example: latent_slice = slice(0, 10) to compute only the first 10 latents.
        :return: autoencoder latents before activation (shape: [batch, n_latents])
        """
        if latent_slice is None:
            latent_slice = slice(None)
        x = x - self.pre_bias
        # Add explicit type cast to help mypy understand weight is a Tensor
        latents_pre_act = nn.functional.linear(
            x,
            self.encoder.weight[latent_slice],  # type: ignore[index]
            self.latent_bias[latent_slice],
        )
        return latents_pre_act

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Preprocess the input tensor by normalizing it if required.

        Args:
            x (torch.Tensor): The input tensor to preprocess.

        Returns:
            tuple[torch.Tensor, dict[str, Any]]: A tuple containing the preprocessed tensor
            and a dictionary with the mean and standard deviation used for normalization
            (if normalization is applied).
        """
        if not self.normalize:
            return x, {}
        x, mu, std = ln(x)
        return x, {"mu": mu, "std": std}

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """:param x: input data (shape: [batch, n_inputs])
        :return: autoencoder latents (shape: [batch, n_latents])
        """
        x, info = self.preprocess(x)
        return self.activation(self.encode_pre_act(x)), info

    def decode(
        self, latents: torch.Tensor, info: dict[str, Any] | None = None
    ) -> torch.Tensor:
        """:param latents: autoencoder latents (shape: [batch, n_latents])
        :return: reconstructed data (shape: [batch, n_inputs])
        """
        ret = self.decoder(latents) + self.pre_bias
        if self.normalize:
            if info is None:
                raise ValueError("info cannot be None when normalization is enabled")
            ret = ret * info["std"] + info["mu"]
        return ret

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """:param x: input data (shape: [batch, n_inputs])
        :return:  autoencoder latents pre activation (shape: [batch, n_latents])
                  autoencoder latents (shape: [batch, n_latents])
                  reconstructed data (shape: [batch, n_inputs])
        """
        x, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)
        latents = self.activation(latents_pre_act)
        recons = self.decode(latents, info)

        # set all indices of self.stats_last_nonzero where (latents != 0) to 0
        self.stats_last_nonzero *= (latents == 0).all(dim=0).long()
        self.stats_last_nonzero += 1

        return latents_pre_act, latents, recons

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> "Autoencoder":
        """Create an Autoencoder instance from a given state dictionary.

        Args:
            state_dict (dict[str, torch.Tensor]): A dictionary containing the state of the autoencoder.
            strict (bool, optional): Whether to strictly enforce that the keys in `state_dict` match the keys
                                     returned by `state_dict` function. Defaults to True.

        Returns:
            Autoencoder: An instance of the Autoencoder class initialized with the provided state dictionary.
        """
        n_latents, d_model = state_dict["encoder.weight"].shape

        # Retrieve activation
        activation_class_name = state_dict.pop("activation", "ReLU")
        activation_class = ACTIVATIONS_CLASSES.get(str(activation_class_name), nn.ReLU)
        normalize = (
            activation_class_name == "TopK"
        )  # NOTE: hacky way to determine if normalization is enabled
        activation_state_dict = cast(
            "dict[str, torch.Tensor]", state_dict.pop("activation_state_dict", {})
        )
        if hasattr(activation_class, "from_state_dict"):
            activation = activation_class.from_state_dict(
                activation_state_dict, strict=strict
            )
        else:
            activation = activation_class()
            if hasattr(activation, "load_state_dict"):
                activation.load_state_dict(activation_state_dict, strict=strict)

        autoencoder = cls(
            n_latents, d_model, activation=activation, normalize=normalize
        )
        # Load remaining state dict
        autoencoder.load_state_dict(state_dict, strict=strict)
        return autoencoder

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        sd = super().state_dict(destination, prefix, keep_vars)
        sd[prefix + "activation"] = self.activation.__class__.__name__
        if hasattr(self.activation, "state_dict"):
            sd[prefix + "activation_state_dict"] = self.activation.state_dict()
        return sd


class TiedTranspose(nn.Module):
    """A custom PyTorch module that ties the weights of a linear layer to its transpose.

    This module is useful for implementing tied autoencoders where the decoder weights
    are the transpose of the encoder weights.

    Args:
        linear (nn.Linear): The linear layer whose weights will be tied to this module.

    Attributes:
        linear (nn.Linear): The linear layer whose weights are tied.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Performs the forward pass using the transposed weights of the linear layer.

        weight() -> torch.Tensor:
            Returns the transposed weights of the linear layer.

        bias() -> torch.Tensor:
            Returns the bias of the linear layer.
    """

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the linear transformation.
        """
        if self.linear.bias is not None:
            raise ValueError("The linear layer's bias must be None.")
        return nn.functional.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        """Returns the transposed weight tensor of the linear layer.

        Returns:
            torch.Tensor: The transposed weight tensor.
        """
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        """Returns the bias tensor of the linear layer.

        Returns:
            torch.Tensor: The bias tensor of the linear layer.
        """
        return self.linear.bias


class TopK(nn.Module):
    """A PyTorch module that selects the top-k values along the last dimension of the input tensor,
    applies a post-activation function to these values, and sets all other values to zero.

    Args:
        k (int): The number of top values to select.
        postact_fn (Callable, optional): The post-activation function to apply to the top-k values.
                                         Defaults to nn.ReLU().

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Applies the top-k selection and post-activation function to the input tensor.

        state_dict(destination=None, prefix="", keep_vars=False):
            Returns the state dictionary of the module, including the value of k and the name of the post-activation function.

        from_state_dict(state_dict: dict[str, torch.Tensor], strict: bool = True) -> "TopK":
            Creates an instance of the TopK module from a state dictionary.
    """

    def __init__(self, k: int, postact_fn: Callable | None = None) -> None:
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn if postact_fn is not None else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the top-k selection and post-activation function.
        """
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        # make all other values 0
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict.update(
            {
                prefix + "k": self.k,
                prefix + "postact_fn": self.postact_fn.__class__.__name__,
            }
        )
        return state_dict

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> "TopK":
        """Creates an instance of the class from a state dictionary.

        Args:
            state_dict (dict[str, torch.Tensor]): A dictionary containing the state of the object.
            strict (bool, optional): Whether to strictly enforce that the keys in `state_dict`
            match the keys returned by this module's `state_dict` function. Defaults to True.

        Returns:
            TopK: An instance of the class with parameters loaded from the state dictionary.
        """
        k = state_dict["k"]
        postact_fn = ACTIVATIONS_CLASSES[state_dict["postact_fn"]]()
        return cls(k=k, postact_fn=postact_fn)


ACTIVATIONS_CLASSES = {
    "ReLU": nn.ReLU,
    "Identity": nn.Identity,
    "TopK": TopK,
}
