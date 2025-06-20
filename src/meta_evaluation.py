import csv
import os
import random
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import set_seed
from utils import clustering, config, constants, helper_modules

# SET UNIVERSAL SEED
set_seed(42)
random.seed(42)

# Environment settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if config.DEVICE == "cuda":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logger, log_file = helper_modules.setup_logging(
    config_module_path=config,
    log_dir=constants.LOGS_PATH,
    evaluation=False,
    meta_evaluation=True,
)
logger.info(f"Analysis results will be saved to: {log_file}")

csv_filename = (
    f"{constants.RESULTS_PATH}/cosy-evaluation_target-{config.TARGET_MODEL_NAME}_textgen-"
    f"{config.TEXT_GENERATOR_NAME.split('/')[-1].replace('.', '-')}_"
    f"{config.AGG_METHOD}_evalgen-{config.EVALUATION_TEXT_GENERATOR_NAME.split('/')[-1].replace('.', '-')}_"
    f"{config.CONTROL_DATA}_{config.SUBSET_SIZE}.csv"
)
df = pd.read_csv(csv_filename)

result_filename = (
    f"{constants.RESULTS_PATH}/meta-evaluation_cosine-similarity_target-{config.TARGET_MODEL_NAME}_textgen-"
    f"{config.TEXT_GENERATOR_NAME.split('/')[-1].replace('.', '-')}_"
    f"{config.AGG_METHOD}_evalgen-{config.EVALUATION_TEXT_GENERATOR_NAME.split('/')[-1].replace('.', '-')}_"
    f"{config.CONTROL_DATA}_{config.SUBSET_SIZE}.csv"
)

if not Path(result_filename).exists() or Path(result_filename).stat().st_size == 0:
    with Path(result_filename).open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "layer",
                "unit",
                "cosine_similarity",
                "cosine_similarity_random",
                "max_auc",
                "max_mad",
            ]
        )

# Group by 'layer' and 'unit' to get the 5 pairs
groups = df.groupby(["layer", "unit"])

all_descriptions = df["explanation"].tolist()

embedder = SentenceTransformer(
    config.CLUSTER_EMBEDDING_MODEL_NAME, trust_remote_code=True
)
embedder.max_seq_length = config.CLUSTER_MAX_SEQ_LEN

# Process each group
for (layer, unit), group in groups:
    with Path(result_filename).open(mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        true_descriptions = group["explanation"].tolist()
        false_descriptions = [
            desc for desc in all_descriptions if desc not in true_descriptions
        ]
        random_samples = random.sample(false_descriptions, 5)

        sim_matrix, avg_sim = clustering.get_cosine_similarity(
            true_descriptions, embedder
        )
        sim_matrix_random, avg_sim_random = clustering.get_cosine_similarity(
            random_samples, embedder
        )

        max_auc = group["AUC"].max()
        max_mad = group["MAD"].max()

        logger.info(
            f"{layer}, {unit}:\n"
            f" {avg_sim} cosine similarity\n"
            f" {avg_sim_random} cosine similarity random\n"
            f" {max_auc} max AUC\n"
            f" {max_mad} max MAD"
        )

        writer.writerow([layer, unit, avg_sim, avg_sim_random, max_auc, max_mad])

logger.info(f"All layer-unit pairs processed. Results saved to:\n {result_filename}")
