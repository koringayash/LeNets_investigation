"""
dataset.py
----------
Handles everything related to data: downloading, preprocessing, splitting,
and wrapping into PyTorch DataLoader objects ready for training.

Three public functions
----------------------
  download_mnist   : Downloads MNIST to disk (skips if already present).
  get_dataloaders  : Returns (train_loader, val_loader, test_loader).
  describe_dataset : Prints a quick summary of the loaded splits.

How the split works
-------------------
The official MNIST test set (10 000 samples) is kept completely untouched and
only used ONCE at the end of training to report the final test accuracy.

The official training set (60 000 samples) is split as follows:
  - 80% → training   (~48 000 samples)
  - 10% → validation (~6 000 samples)  used to tune and pick best checkpoint
  - 10% → held-out   (~6 000 samples)  treated the same as the test set for splits

Beginners: A DataLoader is like a conveyor belt — it feeds batches of images
to the model in random order during training and in a fixed order during eval.
"""

import logging
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from config import DATASET, TRAIN, DATA_DIR
from utils import Timer


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_mnist(logger: logging.Logger = None) -> None:
    """
    Download the MNIST dataset to the /Data directory.

    If the dataset is already present on disk, this function skips the
    download entirely (torchvision handles the check internally).

    The tqdm bar shows download progress for each file.

    Parameters
    ----------
    logger : logging.Logger, optional
        Where to write timing and status messages.

    Side effects
    ------------
    Creates files under DATA_DIR/MNIST/raw/
    """
    log = logger.info if logger else print

    transform = _build_transform()

    with Timer("Downloading MNIST (train split)", logger=logger):
        log("Checking / downloading MNIST training set …")
        datasets.MNIST(
            root      = str(DATA_DIR),
            train     = True,
            download  = True,
            transform = transform,
        )

    with Timer("Downloading MNIST (test split)", logger=logger):
        log("Checking / downloading MNIST test set …")
        datasets.MNIST(
            root      = str(DATA_DIR),
            train     = False,
            download  = True,
            transform = transform,
        )

    log(f"MNIST data ready at: {DATA_DIR}")


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def get_dataloaders(
    logger: logging.Logger = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build and return train, validation, and test DataLoaders.

    The function:
      1. Loads the dataset from disk (download_mnist must have been called first).
      2. Applies the same normalisation transform used during download.
      3. Splits the 60 000 training samples into train (80%) and val (10%)
         subsets, with an optional 10% held-out portion.
      4. Wraps each split in a DataLoader with the batch size from config.

    Parameters
    ----------
    logger : logging.Logger, optional
        Where to write timing and status messages.

    Returns
    -------
    train_loader : DataLoader  Shuffled batches for the training loop.
    val_loader   : DataLoader  Sequential batches for validation after each epoch.
    test_loader  : DataLoader  Sequential batches for the final one-time test eval.

    Example
    -------
    >>> train_loader, val_loader, test_loader = get_dataloaders(logger=my_logger)
    >>> for images, labels in train_loader:
    ...     ...  # images shape: (batch_size, 1, 32, 32)
    """
    log = logger.info if logger else print
    transform = _build_transform()

    # ---- Load raw datasets ------------------------------------------------
    with Timer("Loading MNIST from disk", logger=logger):
        full_train_dataset = datasets.MNIST(
            root      = str(DATA_DIR),
            train     = True,
            download  = False,   # must already exist; call download_mnist() first
            transform = transform,
        )
        test_dataset = datasets.MNIST(
            root      = str(DATA_DIR),
            train     = False,
            download  = False,
            transform = transform,
        )

    # ---- Split training set into train / val / held-out ------------------
    with Timer("Splitting dataset", logger=logger):
        total        = len(full_train_dataset)   # 60 000
        train_size   = int(total * DATASET["train_fraction"])   # 48 000
        val_size     = int(total * DATASET["val_fraction"])     # 6 000
        held_size    = total - train_size - val_size            # 6 000

        # random_split uses a fixed generator seed for reproducibility
        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset, _ = random_split(
            full_train_dataset,
            [train_size, val_size, held_size],
            generator=generator,
        )

        log(
            f"Split sizes — train: {len(train_subset):,}  "
            f"val: {len(val_subset):,}  "
            f"test: {len(test_dataset):,}"
        )

    # ---- Wrap in DataLoaders ----------------------------------------------
    with Timer("Creating DataLoaders", logger=logger):
        num_workers = TRAIN["num_workers"]

        train_loader = DataLoader(
            train_subset,
            batch_size  = TRAIN["batch_size"],
            shuffle     = True,    # randomise order every epoch
            num_workers = num_workers,
            pin_memory  = True,    # faster GPU transfer
        )
        val_loader = DataLoader(
            val_subset,
            batch_size  = TRAIN["batch_size"],
            shuffle     = False,   # fixed order for reproducible val metrics
            num_workers = num_workers,
            pin_memory  = True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size  = TRAIN["batch_size"],
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = True,
        )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Dataset description helper
# ---------------------------------------------------------------------------

def describe_dataset(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    test_loader:  DataLoader,
    logger: logging.Logger = None,
) -> None:
    """
    Print a short human-readable summary of the three data splits.

    Useful to quickly verify that data is loaded correctly and that tensor
    shapes match what the model expects.

    Parameters
    ----------
    train_loader, val_loader, test_loader : DataLoader
        The three loaders returned by get_dataloaders().
    logger : logging.Logger, optional
        Where to write the summary.

    Example output
    --------------
    ============================================================
      Dataset Summary: MNIST
    ============================================================
      Train batches : 750  |  samples : 48,000
      Val   batches : 94   |  samples : 6,016
      Test  batches : 157  |  samples : 10,000
      Image shape   : (1, 32, 32)
      Num classes   : 10
    ============================================================
    """
    log = logger.info if logger else print

    # Peek at one batch to get the image shape
    sample_images, _ = next(iter(train_loader))
    img_shape = tuple(sample_images.shape[1:])   # drop batch dim

    lines = [
        "=" * 60,
        f"  Dataset Summary: {DATASET['name']}",
        "=" * 60,
        f"  Train batches : {len(train_loader):<6} |  samples : {len(train_loader.dataset):,}",
        f"  Val   batches : {len(val_loader):<6} |  samples : {len(val_loader.dataset):,}",
        f"  Test  batches : {len(test_loader):<6} |  samples : {len(test_loader.dataset):,}",
        f"  Image shape   : {img_shape}",
        f"  Num classes   : {DATASET['num_classes']}",
        "=" * 60,
    ]
    log("\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_transform() -> transforms.Compose:
    """
    Build the preprocessing pipeline applied to every image.

    Steps
    -----
    1. Resize to the target image_size defined in config (32×32 for LeNet).
    2. Convert PIL Image → float32 tensor, values in [0, 1].
    3. Normalise with MNIST mean and std so values are roughly in [-1, 1].

    Returns
    -------
    transforms.Compose  A callable that accepts a PIL Image and returns a tensor.
    """
    return transforms.Compose([
        transforms.Resize((DATASET["image_size"], DATASET["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=DATASET["mean"], std=DATASET["std"]),
    ])
