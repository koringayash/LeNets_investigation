"""
dataset/formats/classification_formats.py
------------------------------------------
Dataset loaders for the three supported classification input formats.

Format A — "imagefolder" : class_name/image.jpg directory structure
Format B — "csv"         : CSV file with columns image_path, label
Format C — "torchvision" : built-in datasets (MNIST, CIFAR-10, etc.)

All three return a standard PyTorch Dataset producing:
  (image_tensor, label_int)

Everything downstream (DataLoader, train loop, metrics) sees
only this unified format — it never knows which format was on disk.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Format A: ImageFolder
# ---------------------------------------------------------------------------

def load_imagefolder(
    root_path  : str,
    transform  : transforms.Compose,
    logger     : logging.Logger = None,
) -> Dataset:
    """
    Load a dataset from an ImageFolder-style directory.

    Expected structure
    ------------------
    root_path/
        class_a/image1.jpg
        class_a/image2.jpg
        class_b/image1.jpg
        ...

    Parameters
    ----------
    root_path : str   Root directory containing class subdirectories.
    transform : transforms.Compose  Preprocessing pipeline.
    logger    : logging.Logger, optional

    Returns
    -------
    Dataset  torchvision.datasets.ImageFolder instance.
    """
    log = logger.info if logger else print
    path = Path(root_path)

    if not path.exists():
        raise FileNotFoundError(f"ImageFolder path not found: {path}")

    dataset = datasets.ImageFolder(str(path), transform=transform)
    log(f"ImageFolder loaded: {len(dataset):,} samples, {len(dataset.classes)} classes")
    log(f"  Classes: {dataset.classes[:10]}"
        + ("..." if len(dataset.classes) > 10 else ""))

    return dataset


# ---------------------------------------------------------------------------
# Format B: CSV
# ---------------------------------------------------------------------------

class CSVDataset(Dataset):
    """
    Dataset loaded from a CSV file with columns: image_path, label.

    The CSV must have at least two columns:
      - image_path : relative or absolute path to the image file
      - label      : integer class index (0-indexed)

    Parameters
    ----------
    csv_path  : str   Path to the CSV file.
    transform : transforms.Compose  Preprocessing pipeline.
    root_dir  : str   Optional base directory prepended to relative paths.
    """

    def __init__(
        self,
        csv_path  : str,
        transform : transforms.Compose,
        root_dir  : str = "",
    ):
        import csv
        from PIL import Image

        self.transform = transform
        self.root_dir  = Path(root_dir)
        self.samples   = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = self.root_dir / row["image_path"]
                label    = int(row["label"])
                self.samples.append((str(img_path), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_csv_dataset(
    csv_path  : str,
    transform : transforms.Compose,
    root_dir  : str = "",
    logger    : logging.Logger = None,
) -> Dataset:
    """
    Load a classification dataset from a CSV file.

    Parameters
    ----------
    csv_path  : str  Path to CSV with columns: image_path, label.
    transform : transforms.Compose
    root_dir  : str  Optional base directory for relative image paths.
    logger    : logging.Logger, optional

    Returns
    -------
    CSVDataset
    """
    log = logger.info if logger else print
    dataset = CSVDataset(csv_path, transform, root_dir)
    log(f"CSV dataset loaded: {len(dataset):,} samples from {csv_path}")
    return dataset


# ---------------------------------------------------------------------------
# Format C: torchvision built-in
# ---------------------------------------------------------------------------

_TORCHVISION_MAP = {
    "mnist"       : datasets.MNIST,
    "cifar10"     : datasets.CIFAR10,
    "cifar100"    : datasets.CIFAR100,
    "fashionmnist": datasets.FashionMNIST,
    "svhn"        : datasets.SVHN,
}


def load_torchvision_dataset(
    name           : str,
    data_dir       : str,
    transform      : transforms.Compose,
    train          : bool = True,
    logger         : logging.Logger = None,
) -> Dataset:
    """
    Load a torchvision built-in dataset.

    Parameters
    ----------
    name      : str   Dataset name. One of: MNIST, CIFAR10, CIFAR100,
                      FashionMNIST, SVHN. Case-insensitive.
    data_dir  : str   Root directory where data is cached.
    transform : transforms.Compose
    train     : bool  True for training split, False for test.
    logger    : logging.Logger, optional

    Returns
    -------
    Dataset

    Raises
    ------
    ValueError  If name is not in the supported list.
    """
    log  = logger.info if logger else print
    key  = name.lower().replace("-", "")

    if key not in _TORCHVISION_MAP:
        raise ValueError(
            f"Unknown torchvision dataset '{name}'. "
            f"Supported: {list(_TORCHVISION_MAP.keys())}"
        )

    cls = _TORCHVISION_MAP[key]
    split_label = "train" if train else "test"

    if key == "svhn":
        dataset = cls(
            data_dir, split=split_label, download=False, transform=transform
        )
    else:
        dataset = cls(
            data_dir, train=train, download=False, transform=transform
        )

    log(f"torchvision {name} ({split_label}): {len(dataset):,} samples")
    return dataset