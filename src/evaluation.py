import csv
import gc
import os
import random
import re
import string
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import set_seed
from utils import config, constants, data, helper_modules, models

# SET UNIVERSAL SEED
set_seed(42)
random.seed(42)

# Environment settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if config.DEVICE == "cuda":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:128,expandable_segments:True"
    )
    torch.cuda.empty_cache()

logger, log_file = helper_modules.setup_logging(
    config_module_path=config,
    log_dir=constants.LOGS_PATH,
    evaluation=True,
    meta_evaluation=False,
)
logger.info(f"Analysis results will be saved to: {log_file}")

# Device setup
if config.DEVICE == "cuda" and not torch.cuda.is_available():
    logger.warning("CUDA is not available. Falling back to CPU.")
    device = torch.device("cpu")
else:
    device = torch.device(config.DEVICE)

# Create necessary directories at once to avoid multiple file system calls
for path in [
    constants.ACTIVATIONS_PATH,
    f"{constants.ACTIVATIONS_PATH}/{config.TARGET_MODEL_NAME}/",
    constants.RESULTS_PATH,
    f"{constants.GEN_TEXT_PATH}/{config.EVALUATION_TEXT_GENERATOR_NAME.split('/')[-1].replace('.', '-')}/",
]:
    Path(path).mkdir(parents=True, exist_ok=True)

model_activation_path = f"{constants.ACTIVATIONS_PATH}/{config.TARGET_MODEL_NAME}/"
# Clear cache at the beginning
helper_modules.clear_gpu_cache()

# Load models once
logger.info("Loading target model...")
model = models.get_model(config.TARGET_MODEL_NAME)
model.to(device)
model.eval()
logger.info("Model loaded successfully")

tokenizer = models.get_tokenizer(config.TARGET_MODEL_NAME)
logger.info("Tokenizer loaded successfully")


# Function to get or generate control activations for a specific layer
def get_control_activations(model, layer_id):
    control_tensor_path = (
        f"{model_activation_path}/control_{config.TARGET_MODEL_NAME}_"
        f"layer{layer_id}_{config.AGG_METHOD}_{config.CONTROL_DATA}_"
        f"{config.SUBSET_SIZE}.pt"
    )

    if Path(control_tensor_path).exists():
        return torch.load(control_tensor_path, weights_only=True, map_location="cpu")
    else:
        # Load control samples
        full_control_dataset = data.get_dataset(
            data_name=config.CONTROL_DATA,
            data_files=config.CONTROL_DATA_FILES,
            split=config.CONTROL_SPLIT,
            streaming=config.STREAMING,
        )
        subset_samples = data.collect_samples(full_control_dataset, config.SUBSET_SIZE)

        # Process in smaller batches
        control_dataset = Dataset.from_pandas(pd.DataFrame(subset_samples))
        # Free memory
        del subset_samples

        # Tokenize with batched processing
        control_dataset = control_dataset.map(
            lambda x: tokenizer(
                x["text"],
                truncation=True,
                padding="max_length",
                max_length=config.MAX_TEXT_LENGTH,
            ),
            batched=True,
        )
        control_dataset = control_dataset.with_format("torch")
        control_data_loader = DataLoader(
            control_dataset, batch_size=config.CONTROL_BATCH_SIZE, shuffle=False
        )
        # Get activation tensor for all units
        activations = models.get_activations(
            model=model,
            data_loader=control_data_loader,
            tensor_path=control_tensor_path,
            unit_id=None,
            hook_name=f"blocks.{layer_id}.{config.HOOK_ID}",
            agg_method=config.AGG_METHOD,
            save=config.SAVE_ACTIVATIONS,
        )
        # Move to CPU to free GPU memory
        activations = activations.cpu()

        # Free memory
        del control_dataset, control_data_loader
        gc.collect()

        return activations


# Load the dataframe with layer-unit pairs to process
# Assuming the dataframe has columns 'layer' and 'unit'
layer_unit_df = pd.read_csv(
    f"{constants.ASSETS_PATH}/explanations/{config.METHOD_NAME}/{config.EXPLAIN_FILE}"
)
logger.info(f"Loaded {len(layer_unit_df)} layer-unit pairs to process")

# Define result path
if config.MULTI_EVAL == True:
    csv_filename = (
        f"{constants.RESULTS_PATH}/cosy-evaluation_target-{config.TARGET_MODEL_NAME}_"
        f"textgen-{config.TEXT_GENERATOR_NAME.split('/')[-1].replace('.', '-')}_"
        f"{config.AGG_METHOD}_evalgen-{config.EVALUATION_TEXT_GENERATOR_NAME.split('/')[-1].replace('.', '-')}_"
        f"{config.CONTROL_DATA}_{config.SUBSET_SIZE}.csv"
    )
else:
    csv_filename = (
        f"{constants.RESULTS_PATH}/cosy-evaluation_{config.METHOD_NAME}_target-"
        f"{config.TARGET_MODEL_NAME}_textgen-{config.TEXT_GENERATOR_NAME.split('/')[-1].replace('.', '-')}_"
        f"{config.AGG_METHOD}_evalgen-{config.EVALUATION_TEXT_GENERATOR_NAME.split('/')[-1].replace('.', '-')}_"
        f"{config.CONTROL_DATA}_{config.SUBSET_SIZE}.csv"
    )


# Initialize CSV file if it doesn't exist
if not Path(csv_filename).exists() or Path(csv_filename).stat().st_size == 0:
    with Path(csv_filename).open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["layer", "unit", "explanation", "AUC", "U1", "p", "MAD"])

# Load text generator
logger.info("Loading text generator model...")
model_gen, tokenizer_gen = models.get_text_generator(
    config.EVALUATION_TEXT_GENERATOR_NAME
)
logger.info("Text generator loaded successfully")

try:
    # Process each layer-unit pair
    for idx, row in layer_unit_df.iterrows():
        helper_modules.clear_gpu_cache()
        gc.collect()
        layer_id = int(row["layer"])
        unit_id = int(row["unit"])

        logger.info(
            f"Processing layer {layer_id}, unit {unit_id} ({idx + 1}/{len(layer_unit_df)})"
        )

        # Get explanations for this layer-unit pair
        if config.MULTI_EVAL == True:
            folder_path = (
                f"{constants.DESCRIPTIONS_PATH}/"
                f"{config.TEXT_GENERATOR_NAME.split('/')[-1].replace('.', '-')}/"
                f"{config.TARGET_MODEL_NAME.split('/')[-1].replace('.', '-')}"
            )
            pattern = rf"layer-{layer_id}_unit-{unit_id}.*\.csv$"

            # Find the matching file
            matched_filename = None
            for filename in os.listdir(folder_path):
                if re.search(pattern, filename):
                    matched_filename = filename
                    break

            if not matched_filename:
                logger.warning(
                    f"No matching file found for layer {layer_id}, unit {unit_id}. Skipping."
                )
                continue

            # Read only the necessary column
            full_explanation_path = os.path.join(folder_path, matched_filename)
            explanation_list = pd.read_csv(
                full_explanation_path, usecols=["description"]
            )["description"].tolist()
        else:
            # Load single explanation
            df = pd.read_csv(
                f"{constants.ASSETS_PATH}/explanations/{config.METHOD_NAME}/{config.EXPLAIN_FILE}"
            )
            try:
                explanation = df.loc[
                    (df["layer"] == layer_id) & (df["unit"] == unit_id), "description"
                ].to_numpy()[0]
                explanation_list = [explanation]
            except IndexError:
                logger.warning(
                    f"No explanation found for layer {layer_id}, unit {unit_id}. Skipping."
                )
                continue

        # Process each explanation for this layer-unit pair
        for i, explanation in enumerate(explanation_list):
            # Clear cache at each iteration to prevent OOM
            helper_modules.clear_gpu_cache()

            if len(explanation_list) > 1:
                tensor_path = (
                    f"{model_activation_path}/explain_target-{config.TARGET_MODEL_NAME}_"
                    f"layer{layer_id}_{unit_id}_{i}_textgen-"
                    f"{config.TEXT_GENERATOR_NAME.split('/')[-1].replace('.', '-')}_"
                    f"{config.AGG_METHOD}_evalgen-"
                    f"{config.EVALUATION_TEXT_GENERATOR_NAME.split('/')[-1].replace('.', '-')}.pt"
                )
            else:
                tensor_path = (
                    f"{model_activation_path}/explain_{config.TARGET_MODEL_NAME}_"
                    f"layer{layer_id}_{unit_id}_textgen-"
                    f"{config.TEXT_GENERATOR_NAME.split('/')[-1].replace('.', '-')}_"
                    f"{config.AGG_METHOD}_evalgen-"
                    f"{config.EVALUATION_TEXT_GENERATOR_NAME.split('/')[-1].replace('.', '-')}.pt"
                )

            explain_name = explanation.translate(
                str.maketrans("", "", string.punctuation)
            ).replace(" ", "_")[:100]

            A_0 = get_control_activations(model, layer_id)

            # Process only for the unit we need
            activ_non_concept = A_0[:, unit_id]

            del A_0

            # Get concept activations
            while True:
                needs_generation = not Path(tensor_path).exists()
                prompt = f"{config.PROMPT_SAMPLE_INSTRUCTION} {explanation}"
                file_path = (
                    f"{constants.GEN_TEXT_PATH}/"
                    f"{config.EVALUATION_TEXT_GENERATOR_NAME.split('/')[-1].replace('.', '-')}/"
                )
                Path(file_path).mkdir(parents=True, exist_ok=True)
                text_filename = f"{file_path}{explain_name}.txt"

                # Handle text generation if needed
                if needs_generation:
                    # Check if file already exists
                    if not Path(text_filename).exists():
                        explain_samples = models.get_generated_text(
                            prompt,
                            config.EVALUATION_TEXT_GENERATOR_NAME,
                            model_gen,
                            tokenizer_gen,
                        )
                        with Path(text_filename).open("w", encoding="utf-8") as f:
                            f.write(explain_samples)
                        logger.info(f"Samples saved to: {text_filename}")
                    else:
                        logger.info(
                            f"File already exists, skipping generation: {text_filename}"
                        )

                # If tensor doesn't exist or needs regeneration, process the text data and get activations
                if needs_generation or not Path(tensor_path).exists():
                    # Load the text samples using generator-based loading
                    explain_dataset = data.ExplainDataset(
                        text_filename,
                        tokenizer=tokenizer,
                        max_length=config.MAX_TEXT_LENGTH,
                    )
                    explain_data_loader = DataLoader(
                        explain_dataset,
                        batch_size=config.EXPLAIN_BATCH_SIZE,
                        shuffle=False,
                    )

                    # Get activations
                    A_1 = models.get_activations(
                        model=model,
                        data_loader=explain_data_loader,
                        tensor_path=tensor_path,
                        unit_id=unit_id,
                        hook_name=f"blocks.{layer_id}.{config.HOOK_ID}",
                        agg_method=config.AGG_METHOD,
                        save=config.SAVE_ACTIVATIONS,
                    )
                    A_1.cpu()

                    # Free memory
                    del explain_dataset, explain_data_loader
                else:
                    # Load existing tensor
                    A_1 = torch.load(tensor_path, weights_only=True, map_location="cpu")

                # Check if activation shape is valid
                if A_1.shape[0] == config.N_SAMPLES:
                    logger.info(f"Activation shape valid: {A_1.shape[0]}")
                    break
                else:
                    # Invalid shape, remove files and try again
                    logger.warning(
                        f"Wrong activation shape: {A_1.shape[0]}. Removing this explanation."
                    )
                    if Path(tensor_path).exists():
                        os.remove(tensor_path)

                    # Remove the text file to force regeneration
                    if Path(text_filename).exists():
                        logger.info(f"Removing wrong generated file: {text_filename}")
                        os.remove(text_filename)

                    # Force regeneration on next loop iteration
                    needs_generation = True
                    logger.info("Regenerating explanation samples...")

            # Calculate metrics
            activ_concept = A_1

            # Calculate statistics with minimal memory
            with Path(csv_filename).open(
                mode="a", newline="", encoding="utf-8"
            ) as file:
                writer = csv.writer(file)

                # Create labels and concatenate datasets
                concept_labels = torch.cat(
                    (
                        torch.zeros([activ_non_concept.shape[0]]),
                        torch.ones([activ_concept.shape[0]]),
                    ),
                    0,
                )

                A_D = torch.cat((activ_non_concept.cpu(), activ_concept.cpu()), 0)

                # Score explanations
                auc_synthetic = roc_auc_score(concept_labels.cpu(), A_D.cpu())
                U1, p = mannwhitneyu(activ_non_concept.cpu(), activ_concept.cpu())

                if activ_non_concept.std().item() == 0:
                    mad = 0.0
                else:
                    mad = (
                        activ_concept.mean().item() - activ_non_concept.mean().item()
                    ) / activ_non_concept.std().item()

                writer.writerow(
                    [
                        layer_id,
                        unit_id,
                        explanation,
                        auc_synthetic,
                        U1.item(),
                        p.item(),
                        mad,
                    ]
                )

            # Free memory explicitly
            del activ_concept, activ_non_concept, A_D, concept_labels
            del A_1
            helper_modules.clear_gpu_cache()
            gc.collect()

            logger.info(
                f"Processed explanation {i + 1}/{len(explanation_list)} for layer {layer_id}, unit {unit_id}"
            )

    logger.info(f"All layer-unit pairs processed. Results saved to:\n {csv_filename}")

finally:
    # Clean up resources in all cases
    if model_gen is not None:
        del model_gen
    if tokenizer_gen is not None:
        del tokenizer_gen
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer

    # Final cache clearing
    helper_modules.clear_gpu_cache()
    gc.collect()
