import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed
from utils import clustering, config, constants, data, helper_modules, models, sampler

set_seed(42)

helper_modules.clear_gpu_cache()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
if config.DEVICE == "cuda":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logger, log_file = helper_modules.setup_logging(config)
logger.info(f"Analysis results will be saved to: {log_file}")

if config.DEVICE == "cuda" and not torch.cuda.is_available():
    logger.warning("CUDA is not available. Falling back to CPU.")
    device = torch.device("cpu")
else:
    device = torch.device(config.DEVICE)

Path(constants.DESCRIPTIONS_PATH).mkdir(parents=True, exist_ok=True)

# Load target model and tokenizer
model = models.get_model(config.TARGET_MODEL_NAME)
model.to(device)
model.eval()

tokenizer = models.get_tokenizer(config.TARGET_MODEL_NAME)

# Load dataset (streaming mode)
dataset = data.get_dataset(
    data_name=config.TARGET_DATA,
    data_files=config.DATA_FILES,
    split=config.SPLIT,
    streaming=config.STREAMING,
)

# Tokenize, pad and truncate the dataset
dataset = dataset.map(
    lambda x: tokenizer(
        x["text"],
        truncation=True,
        padding="max_length",
        max_length=config.MAX_TEXT_LENGTH,
    ),
    batched=True,
)

dataset = dataset.with_format("torch")
data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)


""" Percentile sampling """
percentile_sampler = sampler.Sampler.collect_percentiles(
    model=model,
    data_loader=data_loader,
    agg_method=config.AGG_METHOD,
)

del model, dataset, data_loader

helper_modules.clear_gpu_cache()

# Determine candidates (samples / input IDs, activations) with percentile sampler
# 0-index: Assuming FEATURES_SIZE=1
best_samples = percentile_sampler.get_samples()["candidate_inputs"][0]
best_activations = percentile_sampler.get_samples()["candidate_activation_vectors"][0]
best_mean_activations = percentile_sampler.get_samples()["candidate_activation"][0]
# best_samples and best_activations both have shape: (11, 512)


""" Decode text for clustering & Highlight text for LLM input prompt """
candidate_inputs_decoded = []
highlighted_tokens = []
mean_sentence_activations = []

# Iterate jointly over tokens and activations
for i, (sample_input_ids, sample_activations, sample_mean_activations) in enumerate(
    tqdm(
        zip(best_samples, best_activations, best_mean_activations, strict=False),
        desc="Adding delimiters",
    )
):
    helper_modules.clear_gpu_cache()
    input_ids = sample_input_ids.detach().tolist()
    activations = sample_activations.detach().cpu().tolist()
    mean_activations = sample_mean_activations.detach().cpu().item()

    mean_sentence_activations.append(
        {
            "sentence_id": i,
            "mean_activation": mean_activations,
        }
    )

    # For each sample individually, determine minimum activation threshold based on the percentile given by the config
    activation_threshold = np.percentile(activations, config.ACTIVATION_PERCENTILE)

    marked_input_ids = []
    in_span = False

    # Include information about top-activating tokens
    # Add delimiters around top-activating spans,
    # following https://blog.eleuther.ai/autointerp/#generating-explanations
    for token_id, activation in zip(input_ids, activations, strict=False):
        if activation > activation_threshold and (
            activation > 0 or not config.FILTER_FOR_POSITIVE_ACTIVATIONS
        ):
            if not in_span:
                marked_input_ids.extend(tokenizer.encode("[", add_special_tokens=False))
                in_span = True
                highlighted_tokens.append(
                    {
                        "sentence_id": i,
                        "token": tokenizer.decode(token_id),
                        "activation": activation,
                    }
                )
        elif in_span:
            marked_input_ids.extend(tokenizer.encode("]", add_special_tokens=False))
            in_span = False
        marked_input_ids.append(token_id)

    # Close any open span at the end
    if in_span:
        marked_input_ids.extend(tokenizer.encode("]", add_special_tokens=False))

    # Decode input IDs into clear text
    decoded_text = tokenizer.decode(
        marked_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    candidate_inputs_decoded.append(decoded_text)

    # Get the original text
    original_text = tokenizer.decode(input_ids)


""" Apply clustering method """
# Run the embedder on the candidate inputs and apply the clustering method
# input text and activations(optional)
cluster_activations = (
    best_activations if config.CLUSTER_EMBEDDING_MODEL_NAME is None else None
)
clustered_sentences = clustering.apply(
    candidate_inputs_decoded, activations=cluster_activations
)

logger.info(f"Number of candidates: {len(candidate_inputs_decoded)}")


""" Generate a descriptive label for each cluster of texts/sentences """
# Initialize CSV file if it doesn't exist
result_path = (
    f"{constants.DESCRIPTIONS_PATH}/"
    f"{config.TEXT_GENERATOR_NAME.split('/')[-1].replace('.', '-')}/"
    f"{config.TARGET_MODEL_NAME.split('/')[-1].replace('.', '-')}"
)
Path(result_path).mkdir(parents=True, exist_ok=True)
csv_filename = f"{result_path}/{log_file[5:].replace('.log', '.csv')}"
if not Path(csv_filename).exists() or Path(csv_filename).stat().st_size == 0:
    with Path(csv_filename).open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["layer", "unit", "description", "mean_activation", "highlights"]
        )

generator_model, generator_tokenizer = models.get_text_generator(
    config.TEXT_GENERATOR_NAME
)

for i, sentence_cluster in enumerate(clustered_sentences):
    helper_modules.clear_gpu_cache()
    if len(sentence_cluster) > 0:
        prompt = config.PROMPT_INSTRUCTION

        highlights_summary = []
        activations_per_cluster = []
        cluster_sentence_mean_activations = [
            mean_sentence_activations[sc["sentence_id"]]["mean_activation"]
            for sc in sentence_cluster
        ]

        if len(sentence_cluster) > config.MAX_CLUSTER_SIZE:
            # Sort sentences in cluster by mean activation (highest first)
            sorted_indices = sorted(
                range(len(sentence_cluster)),
                key=lambda idx: cluster_sentence_mean_activations[idx],
                reverse=True,
            )
            # Take only the top MAX_CLUSTER_SIZE sentences with highest activations
            sentence_cluster = [
                sentence_cluster[idx]
                for idx in sorted_indices[: config.MAX_CLUSTER_SIZE]
            ]
            # Also update the cluster_sentence_mean_activations list to match
            cluster_sentence_mean_activations = [
                cluster_sentence_mean_activations[idx]
                for idx in sorted_indices[: config.MAX_CLUSTER_SIZE]
            ]

        for sc_idx, sc in enumerate(sentence_cluster):
            sentence_activations = []

            for htok in highlighted_tokens:
                if htok["sentence_id"] == sc["sentence_id"]:
                    highlights_summary.append(
                        f"Text #{sc_idx + 1}: {htok['token']} ({htok['activation']})"
                    )
                    sentence_activations.append(htok["activation"])
            # Compute mean activation per sentence, avoid empty lists
            if sentence_activations:
                activations_per_cluster.extend(sentence_activations)

        # Compute mean activation per cluster
        mean_activation_cluster = (
            np.mean(activations_per_cluster) if activations_per_cluster else None
        )

        prompt += "\n\n=== Summary of highlights ===\n" + "\n".join(highlights_summary)

        prompt += "\n\n" + "\n".join(
            [
                f"=== Text #{sc_idx + 1} ===\n> {sc['text']}\n"
                for sc_idx, sc in enumerate(sentence_cluster)
            ]
        )

        prompt += "\n\nDescription: "

        cluster_label = models.get_generated_text(
            prompt, config.TEXT_GENERATOR_NAME, generator_model, generator_tokenizer
        )

        logger.info(
            f"\n---\n> CLUSTER #{i} (Number of samples: {len(sentence_cluster)})\n"
        )
        logger.info(prompt)
        logger.info(cluster_label)
        # Save as CSV
        with Path(csv_filename).open(mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    config.LAYER_ID,
                    config.UNIT_ID,
                    cluster_label.replace("\n", ""),
                    mean_activation_cluster,
                    highlights_summary,
                ]
            )
        logger.info(f"Cluster #{i} saved to: {csv_filename}")
    else:
        logger.warning("Cluster contains no sentences.")

logger.info("=" * 80)
logger.info("ANALYSIS COMPLETED")
logger.info("=" * 80)
logger.info(f"Logs saved to: {log_file}")
logger.info(f"CSV saved to: {result_path}/{log_file[5:].replace('.log', '.csv')}")
