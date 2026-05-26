"""
dataset/preprocess.py (v2)
--------------------------
Task-aware preprocessing pipeline. Reads config.EXPERIMENT["task"] and
config.DATASET["format"] and returns (train, val, test) Dataset splits.
"""

import logging
from typing import Tuple

import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms

from config              import DATASET, EXPERIMENT, DATA_DIR
from dataset.formats     import load_dataset
from utils               import Timer


def get_datasets(
    logger: logging.Logger = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load, preprocess and split the dataset for the current task.

    Returns
    -------
    (train_dataset, val_dataset, test_dataset)
    """
    task   = EXPERIMENT["task"].lower()
    fmt    = DATASET["format"].lower()
    log    = logger.info if logger else print

    train_transform = _build_transform(augment=DATASET.get("augment", False))
    test_transform  = _build_transform(augment=False)

    with Timer("Loading training data", logger=logger):
        if fmt == "torchvision":
            full_train = load_dataset(task, fmt, DATASET, train_transform, split="train", logger=logger)
            test_ds    = load_dataset(task, fmt, DATASET, test_transform,  split="test",  logger=logger)
        else:
            full_train = load_dataset(task, fmt, DATASET, train_transform, split="train", logger=logger)
            # For non-torchvision, carve test from full dataset
            n_test     = int(len(full_train) * (1 - DATASET["train_fraction"] - DATASET["val_fraction"]))
            n_train    = len(full_train) - n_test
            generator  = torch.Generator().manual_seed(42)
            full_train, test_ds = random_split(
                full_train, [n_train, n_test], generator=generator
            )

    with Timer("Splitting train → train/val", logger=logger):
        total      = len(full_train)
        train_size = int(total * DATASET["train_fraction"] /
                         (DATASET["train_fraction"] + DATASET["val_fraction"]))
        val_size   = total - train_size
        generator  = torch.Generator().manual_seed(42)
        train_ds, val_ds = random_split(
            full_train, [train_size, val_size], generator=generator
        )

    log(f"Splits → train: {len(train_ds):,}  val: {len(val_ds):,}  test: {len(test_ds):,}")
    return train_ds, val_ds, test_ds


def _build_transform(augment: bool) -> transforms.Compose:
    """Build the preprocessing pipeline for images."""
    ops = [transforms.Resize((DATASET["image_size"], DATASET["image_size"]))]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(DATASET["image_size"], padding=4),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=DATASET["mean"], std=DATASET["std"]),
    ]
    return transforms.Compose(ops)