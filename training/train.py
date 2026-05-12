"""
training/train.py
-----------------
The core training loop for the CV framework.

Responsibilities
----------------
  - Initialise the model, optimiser, and loss function from config.
  - Load the latest checkpoint when resuming from a crashed run.
  - Run the train → validate → checkpoint cycle for every epoch.
  - Write per-epoch metrics to CSV and JSON via MetricWriter.
  - Save the best checkpoint (by val accuracy) and the latest checkpoint.
  - Write a training_manifest.json after training completes so the
    evaluation phase knows which checkpoint to load.

Checkpoint files saved
----------------------
  Checkpoint/{experiment}_best.pth    — best val accuracy so far
  Checkpoint/{experiment}_latest.pth  — always the most recent epoch
  Checkpoint/training_manifest.json   — handoff file for evaluation phase

Usage (called from training/main.py — not run directly)
------
>>> from training.train import run_training
>>> run_training(state, logger, resume=False)
"""

import json
import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config          import TRAIN, EXPERIMENT, CHECKPOINT_DIR, LOG_DIR
from pipeline_state  import PipelineState
from dataset.save_dataset import get_dataloaders
from training.model_factory import get_model
from utils           import Timer, MetricWriter


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Return the best available compute device based on config.TRAIN["device"].

    Returns
    -------
    torch.device  cuda if available and requested, else cpu.
    """
    pref = TRAIN["device"]
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model    : nn.Module,
    loader   : DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device   : torch.device,
    epoch    : int,
) -> Tuple[float, float]:
    """
    Run one full pass over the training data and update model weights.

    Parameters
    ----------
    model     : The CNN model being trained.
    loader    : DataLoader for the training split.
    criterion : Loss function (CrossEntropyLoss).
    optimiser : Optimiser attached to model parameters.
    device    : torch.device.
    epoch     : Current epoch number (shown in tqdm bar).

    Returns
    -------
    (avg_loss, accuracy)  Both floats. accuracy is in range [0, 1].
    """
    model.train()
    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    pbar = tqdm(
        loader,
        desc         = f"  Epoch {epoch:>3} [Train]",
        leave        = False,
        unit         = "batch",
        dynamic_ncols= True,
    )

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimiser.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimiser.step()

        total_loss    += loss.item() * images.size(0)
        correct       += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total_samples, correct / total_samples


# ---------------------------------------------------------------------------
# Evaluation (val or test)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model    : nn.Module,
    loader   : DataLoader,
    criterion: nn.Module,
    device   : torch.device,
    split    : str = "Val",
) -> Tuple[float, float]:
    """
    Evaluate the model on a dataset split without updating weights.

    The @torch.no_grad() decorator disables gradient tracking for the
    entire function — saves memory and speeds up evaluation.

    Parameters
    ----------
    model     : The CNN model (eval mode set internally).
    loader    : DataLoader for val or test split.
    criterion : Loss function.
    device    : torch.device.
    split     : Label shown in tqdm bar ("Val" or "Test").

    Returns
    -------
    (avg_loss, accuracy)  Both floats. accuracy is in range [0, 1].
    """
    model.eval()
    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    pbar = tqdm(
        loader,
        desc         = f"           [{split}]  ",
        leave        = False,
        unit         = "batch",
        dynamic_ncols= True,
    )

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        logits         = model(images)
        loss           = criterion(logits, labels)

        total_loss    += loss.item() * images.size(0)
        correct       += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, correct / total_samples


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model       : nn.Module,
    optimiser   : torch.optim.Optimizer,
    epoch       : int,
    val_acc     : float,
    path        : Path,
) -> None:
    """
    Save a full training checkpoint to disk.

    Saves model weights, optimiser state, and metadata so training can
    be resumed from exactly this point after a crash.

    Parameters
    ----------
    model     : The model whose weights to save.
    optimiser : The optimiser whose state to save (includes momentum buffers).
    epoch     : Current epoch number.
    val_acc   : Validation accuracy at this epoch.
    path      : Full file path to save to (e.g. Checkpoint/exp_best.pth).
    """
    torch.save({
        "epoch"          : epoch,
        "model_state"    : model.state_dict(),
        "optimiser_state": optimiser.state_dict(),
        "val_acc"        : val_acc,
        "experiment"     : EXPERIMENT["name"],
    }, path)


def _load_checkpoint(
    model     : nn.Module,
    optimiser : torch.optim.Optimizer,
    path      : Path,
    device    : torch.device,
    logger    : logging.Logger,
) -> int:
    """
    Load a checkpoint and restore model + optimiser state.

    Parameters
    ----------
    model, optimiser : Objects to restore state into.
    path             : Path to the .pth checkpoint file.
    device           : Device to map tensors to.
    logger           : For logging the resume message.

    Returns
    -------
    int  The epoch number stored in the checkpoint (training resumes
         from epoch + 1).
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimiser.load_state_dict(ckpt["optimiser_state"])
    epoch   = ckpt["epoch"]
    val_acc = ckpt.get("val_acc", 0.0)
    logger.info(
        f"Resumed from checkpoint: {path.name} "
        f"(epoch {epoch}, val_acc {val_acc:.4f})"
    )
    return epoch


def _write_training_manifest(best_path: Path, best_val_acc: float) -> None:
    """
    Write training_manifest.json — the handoff file for evaluation phase.

    Evaluation reads this file to know which checkpoint to load and
    which config was used, without the user needing to specify manually.

    Parameters
    ----------
    best_path    : Path to the best checkpoint file.
    best_val_acc : Best validation accuracy achieved during training.
    """
    from config import MODEL, DATASET, TRAIN as TRAIN_CFG

    manifest = {
        "experiment"     : EXPERIMENT["name"],
        "best_checkpoint": str(best_path),
        "best_val_acc"   : best_val_acc,
        "config_snapshot": {
            "model"   : MODEL,
            "dataset" : {k: v for k, v in DATASET.items()
                         if not isinstance(v, Path)},
            "train"   : TRAIN_CFG,
        },
    }
    manifest_path = CHECKPOINT_DIR / "training_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_training(
    state  : PipelineState,
    logger : logging.Logger,
    resume : bool = False,
) -> None:
    """
    Run the full training loop, including optional resume from checkpoint.

    Parameters
    ----------
    state  : PipelineState  Shared state manager (tracks epoch progress).
    logger : logging.Logger  Phase logger with [Training] prefix.
    resume : bool
        If True AND training stage is "done" in pipeline_state.json,
        skip this phase entirely.
        If True AND training was "in_progress", resume from last epoch.
        If False, always start from epoch 1 (fresh run).
    """
    # ---- Resume check — skip if already fully done -----------------------
    if resume and state.is_done("training"):
        logger.info("Training phase already complete — skipping (--resume)")
        return

    device      = get_device()
    exp_name    = EXPERIMENT["name"]
    best_path   = CHECKPOINT_DIR / f"{exp_name}_best.pth"
    latest_path = CHECKPOINT_DIR / f"{exp_name}_latest.pth"

    logger.info("=" * 60)
    logger.info("  TRAINING PHASE")
    logger.info("=" * 60)
    logger.info(f"Experiment : {exp_name}")
    logger.info(f"Device     : {device}")
    logger.info(f"Epochs     : {TRAIN['epochs']}")

    # ---- Data ------------------------------------------------------------
    with Timer("Loading DataLoaders", logger=logger):
        train_loader, val_loader, test_loader = get_dataloaders(logger=logger)

    # ---- Model, loss, optimiser ------------------------------------------
    model     = get_model(logger=logger).to(device)
    criterion = nn.CrossEntropyLoss()

    if TRAIN["optimizer"].lower() == "adam":
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr           = TRAIN["learning_rate"],
            weight_decay = TRAIN["weight_decay"],
        )
    elif TRAIN["optimizer"].lower() == "sgd":
        optimiser = torch.optim.SGD(
            model.parameters(),
            lr           = TRAIN["learning_rate"],
            momentum     = TRAIN["momentum"],
            weight_decay = TRAIN["weight_decay"],
        )
    else:
        raise ValueError(f"Unknown optimizer '{TRAIN['optimizer']}'. Choose 'adam' or 'sgd'.")

    # ---- Resume: load checkpoint if available ----------------------------
    start_epoch  = 1
    best_val_acc = 0.0

    if resume and latest_path.exists():
        start_epoch  = _load_checkpoint(model, optimiser, latest_path, device, logger) + 1
        # Restore best_val_acc from best checkpoint metadata if it exists
        if best_path.exists():
            ckpt         = torch.load(best_path, map_location=device)
            best_val_acc = ckpt.get("val_acc", 0.0)
        logger.info(f"Resuming from epoch {start_epoch}/{TRAIN['epochs']}")
    else:
        logger.info("Starting fresh training from epoch 1")

    # ---- Metric writer ---------------------------------------------------
    writer = MetricWriter(log_dir=str(LOG_DIR), experiment=exp_name)

    # ---- Training loop ---------------------------------------------------
    state.mark_started("training")

    with Timer(f"Total training — {TRAIN['epochs']} epochs", logger=logger):
        for epoch in range(start_epoch, TRAIN["epochs"] + 1):
            with Timer(f"Epoch {epoch}", logger=logger) as epoch_timer:

                train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimiser, device, epoch
                )
                val_loss, val_acc = evaluate(
                    model, val_loader, criterion, device, split="Val"
                )

            logger.info(
                f"Epoch {epoch:>3}/{TRAIN['epochs']} | "
                f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
                f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f} | "
                f"Time: {epoch_timer.elapsed:.1f}s"
            )

            # Save latest checkpoint (always)
            _save_checkpoint(model, optimiser, epoch, val_acc, latest_path)

            # Save best checkpoint (only if val_acc improved)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                _save_checkpoint(model, optimiser, epoch, val_acc, best_path)
                logger.info(
                    f"  ✓ New best val acc: {val_acc:.4f} → "
                    f"{best_path.name}"
                )

            # Write metrics to CSV + JSON
            writer.write(
                epoch        = epoch,
                train_loss   = train_loss,
                val_loss     = val_loss,
                val_acc      = val_acc,
                test_acc     = -1.0,
                lr           = TRAIN["learning_rate"],
                duration_sec = epoch_timer.elapsed,
            )

            # Update resume state after EVERY epoch
            state.mark_training_epoch(epoch=epoch, total_epochs=TRAIN["epochs"])

    # ---- Final test evaluation -------------------------------------------
    with Timer("Final test evaluation", logger=logger):
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            logger.info(f"Loaded best checkpoint (val acc: {best_val_acc:.4f})")

        _, test_acc = evaluate(model, test_loader, criterion, device, split="Test")

    logger.info(f"Final test accuracy: {test_acc:.4f}  ({test_acc * 100:.2f}%)")

    # Log final test result row
    writer.write(
        epoch        = TRAIN["epochs"],
        train_loss   = -1.0,
        val_loss     = -1.0,
        val_acc      = best_val_acc,
        test_acc     = test_acc,
        lr           = TRAIN["learning_rate"],
        duration_sec = 0.0,
    )

    # Write handoff manifest for evaluation phase
    _write_training_manifest(best_path, best_val_acc)
    logger.info(f"Training manifest saved → {CHECKPOINT_DIR / 'training_manifest.json'}")

    state.mark_done("training")
    logger.info("Training phase complete ✓")