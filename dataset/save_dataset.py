"""
dataset/save_dataset.py (v2)
-----------------------------
Saves processed datasets to .pt files and creates task-aware DataLoaders.

Key addition over v1
---------------------
Custom collate functions per task. The default PyTorch collate_fn stacks
tensors along a new batch dimension. This works for classification (images
are all the same shape, labels are scalars) but breaks for detection because
each image has a different number of bounding boxes — you cannot stack
variable-length lists into a tensor.

Collate functions defined here
-------------------------------
  _collate_classification : default behaviour, explicit for clarity
  _collate_detection      : pads box tensors, keeps them as a list of dicts
  _collate_segmentation   : default behaviour (masks are always same H×W)
"""

import logging
from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from config    import DATASET, TRAIN, EXPERIMENT, DATA_DIR
from utils     import Timer


PROCESSED_DIR = DATA_DIR / "processed"


# ---------------------------------------------------------------------------
# Task-aware collate functions
# ---------------------------------------------------------------------------

def _collate_classification(batch):
    """Stack images and labels into tensors. Standard behaviour."""
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


def _collate_detection(batch):
    """
    Detection collate: stack images, keep targets as a list of dicts.

    Each image has a different number of bounding boxes (N_i boxes for
    image i). We cannot stack these into a single tensor because N_i
    varies. Instead we keep targets as a Python list of dicts:
      [{"boxes": Tensor(N_i, 4), "labels": Tensor(N_i,)}, ...]

    The training loop and loss function iterate over this list.
    """
    images  = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]   # list of {"boxes":..., "labels":...}
    return images, targets


def _collate_segmentation(batch):
    """Stack images and masks. Both are always the same spatial size."""
    images = torch.stack([item[0] for item in batch])
    masks  = torch.stack([item[1] for item in batch])
    return images, masks


def _get_collate_fn(task: str):
    """Return the correct collate function for the given task."""
    mapping = {
        "classification": _collate_classification,
        "detection"     : _collate_detection,
        "segmentation"  : _collate_segmentation,
    }
    if task not in mapping:
        raise ValueError(f"Unknown task '{task}'. Choose from {list(mapping.keys())}.")
    return mapping[task]


# ---------------------------------------------------------------------------
# Save processed datasets to .pt files
# ---------------------------------------------------------------------------

def save_processed_datasets(
    train_dataset : Dataset,
    val_dataset   : Dataset,
    test_dataset  : Dataset,
    logger        : logging.Logger = None,
) -> None:
    """
    Save processed datasets to .pt files on disk.

    Classification: saves tensors {"images": ..., "labels": ...}
    Detection/Segmentation: saves list of samples as-is (cannot tensor-stack
    variable-length detection targets or large mask tensors efficiently).

    Parameters
    ----------
    train_dataset, val_dataset, test_dataset : Dataset
    logger : logging.Logger, optional
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    task = EXPERIMENT["task"].lower()
    log  = logger.info if logger else print

    for split_name, dataset in [
        ("train", train_dataset),
        ("val",   val_dataset),
        ("test",  test_dataset),
    ]:
        dest = PROCESSED_DIR / f"{split_name}.pt"
        if dest.exists():
            log(f"Processed {split_name}.pt already exists — skipping.")
            continue

        with Timer(f"Saving {split_name} split", logger=logger):
            if task == "classification":
                _save_classification(dataset, dest, split_name, logger)
            else:
                # Detection and segmentation: save raw sample list
                # (boxes vary per image; masks are large — save per-sample)
                samples = [dataset[i] for i in
                           tqdm(range(len(dataset)),
                                desc=f"  Saving {split_name}",
                                leave=False)]
                torch.save(samples, dest)
                log(f"  Saved {split_name}.pt: {len(samples):,} samples")


def _save_classification(dataset, dest, split_name, logger=None):
    """Stack classification samples into image/label tensors and save."""
    log = logger.info if logger else print
    loader = DataLoader(
        dataset, batch_size=512, shuffle=False,
        num_workers=TRAIN["num_workers"],
        collate_fn=_collate_classification,
    )
    all_images, all_labels = [], []
    for images, labels in tqdm(loader, desc=f"  Saving {split_name}", leave=False):
        all_images.append(images)
        all_labels.append(labels)

    tensors = {
        "images": torch.cat(all_images, dim=0),
        "labels": torch.cat(all_labels, dim=0),
    }
    torch.save(tensors, dest)
    log(f"  Saved {split_name}.pt: {tensors['images'].shape[0]:,} samples, "
        f"shape {tuple(tensors['images'].shape[1:])}")


# ---------------------------------------------------------------------------
# Load DataLoaders from processed .pt files
# ---------------------------------------------------------------------------

def get_dataloaders(
    logger: logging.Logger = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load processed .pt files and return task-aware DataLoaders.

    Parameters
    ----------
    logger : logging.Logger, optional

    Returns
    -------
    (train_loader, val_loader, test_loader)

    Raises
    ------
    FileNotFoundError  If processed .pt files are not found.
    """
    log        = logger.info if logger else print
    task       = EXPERIMENT["task"].lower()
    collate_fn = _get_collate_fn(task)
    loaders    = {}

    for split_name in ("train", "val", "test"):
        pt_path = PROCESSED_DIR / f"{split_name}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {pt_path}\n"
                f"Run dataset phase first: python main.py --stage dataset"
            )

        with Timer(f"Loading {split_name}.pt", logger=logger):
            data = torch.load(pt_path, weights_only=False)

            if task == "classification":
                # data is {"images": Tensor, "labels": Tensor}
                ds = TensorDataset(data["images"], data["labels"])
            else:
                # data is a list of (image, target) tuples
                ds = _ListDataset(data)

            loader = DataLoader(
                ds,
                batch_size  = TRAIN["batch_size"],
                shuffle     = (split_name == "train"),
                num_workers = TRAIN["num_workers"],
                pin_memory  = True,
                collate_fn  = collate_fn,
            )
            loaders[split_name] = loader
            log(f"  {split_name}: {len(ds):,} samples, {len(loader)} batches")

    return loaders["train"], loaders["val"], loaders["test"]


def processed_files_exist() -> bool:
    """Return True if all three processed .pt files exist."""
    return all(
        (PROCESSED_DIR / f"{s}.pt").exists()
        for s in ("train", "val", "test")
    )


# ---------------------------------------------------------------------------
# Simple list-based Dataset wrapper
# ---------------------------------------------------------------------------

class _ListDataset(Dataset):
    """Wraps a list of (image, target) tuples as a PyTorch Dataset."""

    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]