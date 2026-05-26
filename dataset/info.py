"""
dataset/info.py (v2)
---------------------
Task-aware dataset summary printer.
"""

import logging
from torch.utils.data import DataLoader
from config import DATASET, EXPERIMENT


def print_dataset_summary(
    train_loader : DataLoader,
    val_loader   : DataLoader,
    test_loader  : DataLoader,
    logger       : logging.Logger = None,
) -> None:
    """
    Print a human-readable summary of the three dataset splits.

    Adapts output based on the current task (classification / detection /
    segmentation) since the sample format differs per task.

    Parameters
    ----------
    train_loader, val_loader, test_loader : DataLoader
    logger : logging.Logger, optional
    """
    log  = logger.info if logger else print
    task = EXPERIMENT["task"]

    # Peek at one batch to get sample shapes
    batch = next(iter(train_loader))
    images = batch[0]
    img_shape = tuple(images.shape[1:])

    # Build target info string per task
    if task == "classification":
        labels    = batch[1]
        tgt_info  = f"Labels shape : {tuple(labels.shape)} (class indices)"
    elif task == "detection":
        targets   = batch[1]   # list of dicts
        n_boxes   = sum(t["boxes"].shape[0] for t in targets)
        tgt_info  = (f"Targets      : list of {len(targets)} dicts "
                     f"(avg {n_boxes/len(targets):.1f} boxes/image)")
    elif task == "segmentation":
        masks     = batch[1]
        tgt_info  = f"Masks shape  : {tuple(masks.shape)} (H×W class indices)"
    else:
        tgt_info  = "Unknown task"

    lines = [
        "=" * 60,
        f"  Dataset Summary — Task: {task.upper()}",
        "=" * 60,
        f"  Source        : {DATASET['source']}",
        f"  Format        : {DATASET['format']}",
        f"  Train samples : {len(train_loader.dataset):>7,}  |  batches: {len(train_loader)}",
        f"  Val   samples : {len(val_loader.dataset):>7,}  |  batches: {len(val_loader)}",
        f"  Test  samples : {len(test_loader.dataset):>7,}  |  batches: {len(test_loader)}",
        f"  Image shape   : {img_shape}",
        f"  {tgt_info}",
        f"  Num classes   : {DATASET['num_classes']}",
        f"  Augmentation  : {DATASET.get('augment', False)}",
        "=" * 60,
    ]
    log("\n" + "\n".join(lines))