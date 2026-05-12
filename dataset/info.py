"""
dataset/info.py
---------------
Prints a human-readable summary of the loaded dataset splits.

Useful for verifying that data is loaded correctly and that tensor
shapes, class counts, and split sizes match expectations before
starting a potentially long training run.
"""

import logging
from torch.utils.data import DataLoader

from config import DATASET


def print_dataset_summary(
    train_loader : DataLoader,
    val_loader   : DataLoader,
    test_loader  : DataLoader,
    logger       : logging.Logger = None,
) -> None:
    """
    Print a summary table of the three dataset splits.

    Parameters
    ----------
    train_loader, val_loader, test_loader : DataLoader
    logger : logging.Logger, optional

    Example output
    --------------
    ============================================================
      Dataset Summary: MNIST
    ============================================================
      Source        : torchvision
      Train samples : 48,000   |  batches: 750
      Val   samples :  6,000   |  batches:  94
      Test  samples : 10,000   |  batches: 157
      Image shape   : (1, 32, 32)
      Num classes   : 10
      Augmentation  : False
    ============================================================
    """
    log = logger.info if logger else print

    # Peek at one batch to get real image shape
    sample_images, _ = next(iter(train_loader))
    img_shape        = tuple(sample_images.shape[1:])

    lines = [
        "=" * 60,
        f"  Dataset Summary: {DATASET.get('name', 'Custom')}",
        "=" * 60,
        f"  Source        : {DATASET['source']}",
        f"  Train samples : {len(train_loader.dataset):>7,}   |  batches: {len(train_loader)}",
        f"  Val   samples : {len(val_loader.dataset):>7,}   |  batches: {len(val_loader)}",
        f"  Test  samples : {len(test_loader.dataset):>7,}   |  batches: {len(test_loader)}",
        f"  Image shape   : {img_shape}",
        f"  Num classes   : {DATASET['num_classes']}",
        f"  Augmentation  : {DATASET.get('augment', False)}",
        "=" * 60,
    ]
    log("\n" + "\n".join(lines))