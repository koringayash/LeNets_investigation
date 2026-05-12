"""
dataset/save_dataset.py
-----------------------
Saves processed dataset tensors to .pt files on disk and creates
PyTorch DataLoaders from them at training time.

Why save to .pt files?
-----------------------
Preprocessing (resize, normalise, augment) takes time. Saving the
processed tensors means we only preprocess once — future runs load
the fast binary format directly, skipping all image decoding and
transform steps.

Note: DataLoaders themselves cannot be serialised. This module saves
the processed Dataset tensors, then recreates DataLoaders from them
on demand.

Files created
-------------
  Data/processed/train.pt  — processed training tensors
  Data/processed/val.pt    — processed validation tensors
  Data/processed/test.pt   — processed test tensors

Each .pt file contains a dict: {"images": Tensor, "labels": Tensor}
"""

import logging
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from config import DATASET, TRAIN, DATA_DIR
from utils  import Timer


PROCESSED_DIR = DATA_DIR / "processed"


class ProcessedDataset(TensorDataset):
    """
    A TensorDataset loaded from a .pt file.

    Identical to torch.utils.data.TensorDataset but with a descriptive
    name so it's recognisable in summary output.
    """
    pass


def save_processed_datasets(
    train_dataset,
    val_dataset,
    test_dataset,
    logger: logging.Logger = None,
) -> None:
    """
    Convert three Dataset objects to tensors and save them as .pt files.

    This iterates over each dataset once, stacks all images and labels
    into tensors, and saves them. Progress is shown with tqdm.

    Parameters
    ----------
    train_dataset, val_dataset, test_dataset : Dataset
        The three splits returned by dataset/preprocess.py
    logger : logging.Logger, optional
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    log = logger.info if logger else print

    for split_name, dataset in [
        ("train", train_dataset),
        ("val",   val_dataset),
        ("test",  test_dataset),
    ]:
        dest = PROCESSED_DIR / f"{split_name}.pt"
        if dest.exists():
            log(f"Processed {split_name}.pt already exists — skipping.")
            continue

        with Timer(f"Saving {split_name} split to {dest.name}", logger=logger):
            loader = DataLoader(dataset, batch_size=512, shuffle=False,
                                num_workers=TRAIN["num_workers"])
            all_images, all_labels = [], []

            for images, labels in tqdm(
                loader,
                desc=f"  Saving {split_name}",
                unit="batch",
                leave=False,
            ):
                all_images.append(images)
                all_labels.append(labels)

            tensors = {
                "images": torch.cat(all_images, dim=0),
                "labels": torch.cat(all_labels, dim=0),
            }
            torch.save(tensors, dest)
            log(
                f"  Saved {split_name}.pt: "
                f"{tensors['images'].shape[0]:,} samples, "
                f"image shape {tuple(tensors['images'].shape[1:])}"
            )


def get_dataloaders(
    logger: logging.Logger = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load processed .pt files and return ready-to-use DataLoaders.

    Call this at the start of training after save_processed_datasets()
    has been run. If .pt files don't exist, raises FileNotFoundError
    with a clear message.

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
    log = logger.info if logger else print

    loaders = {}
    for split_name in ("train", "val", "test"):
        pt_path = PROCESSED_DIR / f"{split_name}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {pt_path}\n"
                f"Run the dataset phase first: python main.py --stage dataset"
            )

        with Timer(f"Loading {split_name}.pt", logger=logger):
            data   = torch.load(pt_path)
            ds     = TensorDataset(data["images"], data["labels"])
            loader = DataLoader(
                ds,
                batch_size  = TRAIN["batch_size"],
                shuffle     = (split_name == "train"),
                num_workers = TRAIN["num_workers"],
                pin_memory  = True,
            )
            loaders[split_name] = loader
            log(f"  {split_name}: {len(ds):,} samples, {len(loader)} batches")

    return loaders["train"], loaders["val"], loaders["test"]


def processed_files_exist() -> bool:
    """
    Return True if all three processed .pt files already exist on disk.

    Used by dataset/main.py to skip re-processing on resume.

    Returns
    -------
    bool
    """
    return all(
        (PROCESSED_DIR / f"{s}.pt").exists()
        for s in ("train", "val", "test")
    )