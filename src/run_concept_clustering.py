import re
import subprocess

import pandas as pd
from utils import config, constants

neuron_descriptions_df = pd.read_csv(
    f"{constants.ASSETS_PATH}/explanations/{config.METHOD_NAME}/{config.EXPLAIN_FILE}"
)

experiment_settings = []
for i, row in neuron_descriptions_df.iterrows():
    experiment_settings.append(
        {
            "LAYER_ID": row["layer"],
            "UNIT_ID": row["unit"],
        }
    )

# Path to the config file
config_file_path = "src/utils/config.py"


def update_config(settings):
    """Modify only LAYER_ID and UNIT_ID in utils/config.py while keeping other settings untouched."""
    with open(config_file_path) as f:
        config_lines = f.readlines()

    updated_lines = []
    for line in config_lines:
        # Check and update only LAYER_ID and UNIT_ID, leave other lines unchanged
        if re.match(r"LAYER_ID\s*=", line):
            updated_lines.append(f"LAYER_ID = {settings['LAYER_ID']}\n")
        elif re.match(r"UNIT_ID\s*=", line):
            updated_lines.append(f"UNIT_ID = {settings['UNIT_ID']}\n")
        else:
            updated_lines.append(line)  # Keep other lines unchanged

    with open(config_file_path, "w") as f:
        f.writelines(updated_lines)


for i, params in enumerate(experiment_settings):
    print(f"Running experiment {i + 1} with settings: {params}")

    # Update the configuration file
    update_config(params)

    # Run concept_clustering.py
    process = subprocess.run(
        ["python", "src/concept_clustering.py"],
        stdout=None,  # Let output go directly to console
        stderr=None,
        check=False,
    )

    print()
