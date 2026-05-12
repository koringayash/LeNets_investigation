"""
dataset/preprocess.py
---------------------
Applies all preprocessing operations to the raw dataset and returns
three split datasets: train, validation, and test.

Operations applied (in order)
------------------------------
  1. Resize images to config.DATASET["image_size"] × image_size
  2. Optional augmentation (random flip + crop) if DATASET["augment"]=True
  3. Convert PIL Image → float32 tensor (values 0–1)
  4. Normalise with per-channel mean and std from config
  5. Split training data into train / val subsets

Beginners: Preprocessing turns raw images into clean, normalised tensors
that neural networks can learn from efficiently.
"""

import logging
from typing import Tuple

import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import datasets, transforms

from config import DATASET, DATA_DIR
from utils  import Timer


def get_datasets(
    logger: logging.Logger = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load, preprocess, and split the dataset into train, val, and test subsets.

    For torchvision sources, data is loaded directly from DATA_DIR.
    For local/url/github sources, torchvision's ImageFolder loader is used.

    Parameters
    ----------
    logger : logging.Logger, optional

    Returns
    -------
    (train_dataset, val_dataset, test_dataset)  Three PyTorch Dataset objects.
    """
    log    = logger.info if logger else print
    source = DATASET["source"].lower()

    train_transform = _build_transform(augment=DATASET.get("augment", False))
    test_transform  = _build_transform(augment=False)

    with Timer("Loading datasets from disk", logger=logger):
        if source == "torchvision":
            full_train, test_dataset = _load_torchvision(
                train_transform, test_transform
            )
        else:
            full_train, test_dataset = _load_image_folder(
                train_transform, test_transform, logger
            )

    with Timer("Splitting train → train / val", logger=logger):
        total      = len(full_train)
        train_size = int(total * DATASET["train_fraction"])
        val_size   = int(total * DATASET["val_fraction"])
        held_size  = total - train_size - val_size

        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset, _ = random_split(
            full_train,
            [train_size, val_size, held_size],
            generator=generator,
        )

    log(
        f"Split → train: {len(train_subset):,}  "
        f"val: {len(val_subset):,}  "
        f"test: {len(test_dataset):,}"
    )

    return train_subset, val_subset, test_dataset


def _load_torchvision(train_transform, test_transform):
    """Load a torchvision built-in dataset from DATA_DIR."""
    import torchvision.datasets as tvd

    name     = DATASET["name"].lower().replace("-", "")
    cls_name = {
        "mnist"       : "MNIST",
        "cifar10"     : "CIFAR10",
        "cifar100"    : "CIFAR100",
        "fashionmnist": "FashionMNIST",
        "svhn"        : "SVHN",
    }[name]
    cls = getattr(tvd, cls_name)

    if cls_name == "SVHN":
        full_train   = cls(str(DATA_DIR), split="train", download=False, transform=train_transform)
        test_dataset = cls(str(DATA_DIR), split="test",  download=False, transform=test_transform)
    else:
        full_train   = cls(str(DATA_DIR), train=True,  download=False, transform=train_transform)
        test_dataset = cls(str(DATA_DIR), train=False, download=False, transform=test_transform)

    return full_train, test_dataset


def _load_image_folder(train_transform, test_transform, logger=None):
    """
    Load a local/url/github dataset using ImageFolder.

    Expects structure: root/class_name/image.ext
    Splits the single folder into train (80%) and test (20%) by index
    since there is no separate test folder.
    """
    from torch.utils.data import Subset as _Subset
    import torchvision.datasets as tvd

    log        = logger.info if logger else print
    local_path = DATASET.get("local_path") or DATA_DIR

    full = tvd.ImageFolder(str(local_path), transform=train_transform)
    log(f"ImageFolder loaded: {len(full)} samples, {len(full.classes)} classes")

    # 80% train, 20% test (no separate official test set for custom data)
    n_test     = int(len(full) * 0.2)
    n_train    = len(full) - n_test
    generator  = torch.Generator().manual_seed(42)
    train_ds, test_ds = random_split(full, [n_train, n_test], generator=generator)

    # Apply test transform to test split
    test_ds.dataset.transform = test_transform

    return train_ds, test_ds


def _build_transform(augment: bool) -> transforms.Compose:
    """
    Build the preprocessing pipeline for one split.

    Parameters
    ----------
    augment : bool  If True, adds random horizontal flip and random crop.

    Returns
    -------
    transforms.Compose
    """
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