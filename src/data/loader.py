"""Dataset loading and preprocessing for all supported tasks."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def load_dataset_from_config(config: dict[str, Any]) -> dict[str, Dataset]:
    """Load dataset based on configuration.

    Args:
        config: Full merged configuration dict.

    Returns:
        Dict with 'train', 'validation', 'test' splits as PyTorch Datasets.
    """
    task = config["model"]["task"]
    dataset_config = config["dataset"]

    if task == "language_modeling":
        return _load_lm_dataset(config)
    elif task == "text_classification":
        return _load_classification_dataset(config)
    elif task == "image_classification":
        return _load_image_dataset(config)
    else:
        raise ValueError(f"Unsupported task: {task}")


def _load_lm_dataset(config: dict) -> dict[str, Dataset]:
    """Load language modeling dataset (e.g., WikiText-2)."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds_config = config["dataset"]
    model_name = config["model"]["name"]
    max_length = ds_config.get("max_length", 512)

    logger.info(f"Loading LM dataset: {ds_config['name']}/{ds_config.get('subset', '')}")

    # Load dataset
    if ds_config.get("custom_path"):
        raw = load_dataset("text", data_files=ds_config["custom_path"])
    else:
        raw = load_dataset(ds_config["name"], ds_config.get("subset"))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_special_tokens_mask=True,
        )

    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw["train"].column_names if "train" in raw else raw[list(raw.keys())[0]].column_names,
        desc="Tokenizing",
    )

    # Group texts into chunks for language modeling
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // max_length) * max_length
        result = {
            k: [v[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, v in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    grouped = tokenized.map(group_texts, batched=True, desc="Grouping texts")
    grouped.set_format("torch")

    splits = {}
    split_mapping = {
        "train": ["train"],
        "validation": ["validation", "valid", "dev"],
        "test": ["test"],
    }

    for target_name, candidates in split_mapping.items():
        for candidate in candidates:
            if candidate in grouped:
                splits[target_name] = grouped[candidate]
                break

    logger.info(f"Dataset loaded: {', '.join(f'{k}: {len(v)}' for k, v in splits.items())}")
    return splits


def _load_classification_dataset(config: dict) -> dict[str, Dataset]:
    """Load text classification dataset (e.g., SST-2)."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds_config = config["dataset"]
    model_name = config["model"]["name"]
    max_length = ds_config.get("max_length", 128)

    logger.info(f"Loading classification dataset: {ds_config['name']}")

    # Load dataset
    if ds_config["name"] == "sst2":
        raw = load_dataset("glue", "sst2")
        text_column = "sentence"
        label_column = "label"
    elif ds_config.get("custom_path"):
        raw = load_dataset("csv", data_files=ds_config["custom_path"])
        text_column = "text"
        label_column = "label"
    else:
        raw = load_dataset(ds_config["name"])
        text_column = "text"
        label_column = "label"

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        result = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        result["labels"] = examples[label_column]
        return result

    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing",
    )
    tokenized.set_format("torch")

    splits = {}
    split_mapping = {
        "train": ["train"],
        "validation": ["validation", "valid", "dev"],
        "test": ["test"],
    }

    for target_name, candidates in split_mapping.items():
        for candidate in candidates:
            if candidate in tokenized:
                splits[target_name] = tokenized[candidate]
                break

    logger.info(f"Dataset loaded: {', '.join(f'{k}: {len(v)}' for k, v in splits.items())}")
    return splits


def _load_image_dataset(config: dict) -> dict[str, Dataset]:
    """Load image classification dataset (e.g., CIFAR-10)."""
    ds_config = config["dataset"]

    try:
        import torchvision
        import torchvision.transforms as T
    except ImportError:
        raise ImportError(
            "torchvision required for image tasks. "
            "Install with: pip install 'auto-augment-agent[cv]'"
        )

    image_size = ds_config.get("image_size", 32)
    dataset_name = ds_config["name"].lower()

    # Base transforms (always applied)
    base_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])

    if dataset_name == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(
            root="data_cache",
            train=True,
            download=True,
            transform=base_transform,
        )
        test_ds = torchvision.datasets.CIFAR10(
            root="data_cache",
            train=False,
            download=True,
            transform=base_transform,
        )
    elif dataset_name == "cifar100":
        train_ds = torchvision.datasets.CIFAR100(
            root="data_cache",
            train=True,
            download=True,
            transform=base_transform,
        )
        test_ds = torchvision.datasets.CIFAR100(
            root="data_cache",
            train=False,
            download=True,
            transform=base_transform,
        )
    else:
        raise ValueError(f"Unsupported image dataset: {dataset_name}")

    # Split train into train + validation (90/10)
    train_size = int(0.9 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_split, val_split = torch.utils.data.random_split(
        train_ds, [train_size, val_size]
    )

    # Wrap torchvision datasets to return dicts for consistency
    splits = {
        "train": _ImageDatasetWrapper(train_split),
        "validation": _ImageDatasetWrapper(val_split),
        "test": _ImageDatasetWrapper(test_ds),
    }

    logger.info(f"Dataset loaded: {', '.join(f'{k}: {len(v)}' for k, v in splits.items())}")
    return splits


class _ImageDatasetWrapper(Dataset):
    """Wraps torchvision datasets to return dicts with 'pixel_values' and 'labels'."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return {"pixel_values": image, "labels": label}


def create_dataloaders(
    splits: dict[str, Dataset],
    config: dict[str, Any],
) -> dict[str, DataLoader]:
    """Create DataLoaders from dataset splits."""
    batch_config = config.get("batch", {})
    train_bs = batch_config.get("train_batch_size", 8)
    eval_bs = batch_config.get("eval_batch_size", 16)
    num_workers = config.get("training", {}).get("dataloader_workers", 4)

    loaders = {}

    if "train" in splits:
        loaders["train"] = DataLoader(
            splits["train"],
            batch_size=train_bs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    for split_name in ["validation", "test"]:
        if split_name in splits:
            loaders[split_name] = DataLoader(
                splits[split_name],
                batch_size=eval_bs,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

    return loaders
