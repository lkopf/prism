"""Data utilities"""

import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

random.seed(42)


def get_dataset(
    data_name: str,
    path: str | None = None,
    name: str | None = None,
    data_dir: str | None = None,
    data_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None,
    split: str | None = None,
    streaming: bool = True,
) -> Any:
    """Load a dataset based on the provided parameters.

    Args:
        data_name (str): The name of the dataset to load. Can be "c4", "fineweb", "openwebtext", "cosmopedia" or None.
        path (str | None): The path to the dataset. Default is None.
        name (str | None): The name of the dataset configuration. Default is None.
        data_dir (str | None): The directory where the dataset is stored. Default is None.
        data_files (str | Sequence[str] | Mapping[str, str | Sequence[str]] | None): The files to load
            from the dataset. Default is None.
        split (str | None): The dataset split to load (e.g., "train", "test"). Default is None.
        streaming (bool): Whether to stream the dataset. Default is True.

    Returns:
        Any: The loaded dataset.
    """
    if data_name == "c4":
        data = load_dataset(
            path="allenai/c4",
            name="en",
            data_files=data_files,
            split=split,
            streaming=streaming,
            verification_mode="no_checks",
        )
    elif data_name == "fineweb":
        data = load_dataset(
            path="HuggingFaceFW/fineweb",
            name="CC-MAIN-2024-51",
            split="train",
            streaming=streaming,
            verification_mode="no_checks",
        )
    elif data_name == "openwebtext":
        data = load_dataset(
            path="openwebtext",
            split="train",
            streaming=streaming,
            trust_remote_code=True,
            verification_mode="no_checks",
        )
    elif data_name == "cosmopedia":
        data = load_dataset(
            path="HuggingFaceTB/cosmopedia-100k",
            split="train",
            streaming=streaming,
            verification_mode="no_checks",
        )
    elif data_name is None:
        data = load_dataset(
            path=path,
            name=name,
            data_dir=data_dir,
            data_files=data_files,
            split=split,
            streaming=streaming,
            verification_mode="no_checks",
        )
    return data


class ExplainDataset(Dataset):
    """Dataset class for loading and tokenizing text explanations from a text file.

    Args:
        text_file (str): Path to the text file containing text explanations.
        tokenizer (Any): Tokenizer to convert text to tokens.
    """

    def __init__(self, text_file: str, tokenizer: Any, max_length: int) -> None:
        # Read the text file and parse its contents
        with Path(text_file).open("r", encoding="utf-8") as f:
            self.texts = [line.strip().rstrip(",") for line in f if line.strip()]

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of text samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retrieve the tokenized text sample at the given index.

        Args:
            idx (int): The index of the text sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: The tokenized text sample.
        """
        text = self.texts[idx]

        # Tokenize the text
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )

        # Convert to tensors manually
        return {
            "text": text,
            "input_ids": torch.tensor(tokens["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tokens["attention_mask"], dtype=torch.long),
        }


def collect_samples(iterable_dataset, n_samples):
    """Collect n_samples from an IterableDataset using reservoir sampling"""
    print(f"Collecting {n_samples} samples...")

    # Initialize with first n_samples
    reservoir = []

    # Counter for progress reporting
    count = 0

    # Process each item
    for i, item in enumerate(iterable_dataset):
        count += 1

        # Fill the reservoir until we have n_samples
        if len(reservoir) < n_samples:
            reservoir.append(item)
        else:
            # Randomly replace items with decreasing probability
            j = random.randint(0, i)
            if j < n_samples:
                reservoir[j] = item

    print(f"Finished after processing {count} items.")
    return reservoir


def extract_and_save_samples(input_text, output_filename):
    """
    Extract text samples from the input text and save to a file.

    Args:
    input_text (str): The input text containing samples
    output_filename (str): Name of the output text file
    """
    # Find all samples using regex
    samples = re.findall(r"Sample \d+: (.*?)", input_text, re.DOTALL)

    # Write to file with samples comma-separated
    with open(output_filename, "w", encoding="utf-8") as file:
        # Join samples with comma and newline
        formatted_text = ",\n".join(f'"{sample}"' for sample in samples)
        file.write(formatted_text)

    return samples


if __name__ == "__main__":
    df_in = pd.read_csv(
        "../../assets/explanations/GPT-explain/GPT-explain_features.csv", header=None
    )
    data = defaultdict(list)
    for i, row in df_in.iterrows():
        unit_id = row[1].split("/")[-1]
        layer_num = row[1].split("/")[-3]
        data["layer"].append(layer_num)
        data["unit"].append(unit_id)
        data["description"].append(row[0])

    df_out = pd.DataFrame.from_dict(data)
    df_out.to_csv("../../assets/explanations/GPT-explain/gpt2-xl_all-layers.csv")

    print()
