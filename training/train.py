"""
training/train.py (v2)
-----------------------
Core training loop for all three tasks. The task selector from
task_loop.py is assigned ONCE before the epoch loop — the inner
loop always calls the same function with no branching overhead.

Identical structure to v1. Changes:
  - Imports task-aware step function from task_loop.py
  - Loss fetched via losses.get_loss() task selector
  - Metric label adapts to task (acc / obj_acc / pixel_acc)
"""

import json
import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from config              import TRAIN, EXPERIMENT, LOSS, CHECKPOINT_DIR, LOG_DIR
from pipeline_state      import PipelineState
from dataset.save_dataset import get_dataloaders
from training.model_factory import get_model
from training.task_loop  import get_step_function
from losses              import get_loss
from utils               import Timer, MetricWriter


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    pref = TRAIN["device"]
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


# ---------------------------------------------------------------------------
# Evaluation loop (no gradient updates)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model    : nn.Module,
    loader,
    criterion: nn.Module,
    device   : torch.device,
    step_fn,
    split    : str = "Val",
) -> Tuple[float, float]:
    """
    Evaluate the model on a dataset split.

    Uses the same step function as training but with no gradient updates.

    Parameters
    ----------
    model     : nn.Module
    loader    : DataLoader
    criterion : Task loss module
    device    : torch.device
    step_fn   : Callable from task_loop.get_step_function()
    split     : str  Label for tqdm bar.

    Returns
    -------
    (avg_loss, avg_metric)  Both floats.
    """
    model.eval()
    total_loss   = 0.0
    total_metric = 0.0
    n_batches    = 0

    pbar = tqdm(
        loader,
        desc         = f"           [{split}]",
        leave        = False,
        unit         = "batch",
        dynamic_ncols= True,
    )

    for batch in pbar:
        loss, metric = step_fn(model, batch, criterion, device)
        total_loss   += loss.item()
        total_metric += metric
        n_batches    += 1

    if n_batches == 0:
        return 0.0, 0.0
    return total_loss / n_batches, total_metric / n_batches


# ---------------------------------------------------------------------------
# Checkpoint helpers (identical to v1)
# ---------------------------------------------------------------------------

def _save_checkpoint(model, optimiser, epoch, val_metric, path):
    torch.save({
        "epoch"          : epoch,
        "model_state"    : model.state_dict(),
        "optimiser_state": optimiser.state_dict(),
        "val_metric"     : val_metric,
        "experiment"     : EXPERIMENT["name"],
        "task"           : EXPERIMENT["task"],
    }, path)


def _load_checkpoint(model, optimiser, path, device, logger):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimiser.load_state_dict(ckpt["optimiser_state"])
    epoch      = ckpt["epoch"]
    val_metric = ckpt.get("val_metric", 0.0)
    logger.info(f"Resumed checkpoint: {path.name} (epoch {epoch}, val_metric {val_metric:.4f})")
    return epoch


def _write_training_manifest(best_path, best_val_metric):
    manifest = {
        "experiment"     : EXPERIMENT["name"],
        "task"           : EXPERIMENT["task"],
        "best_checkpoint": str(best_path),
        "best_val_metric": best_val_metric,
    }
    manifest_path = CHECKPOINT_DIR / "training_manifest.json"
    with open(manifest_path, "w") as f:
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
    Full training loop for the configured task.

    Parameters
    ----------
    state  : PipelineState
    logger : logging.Logger  Phase logger with [Training] prefix.
    resume : bool
    """
    if resume and state.is_done("training"):
        logger.info("Training phase already complete — skipping (--resume)")
        return

    task        = EXPERIMENT["task"].lower()
    device      = get_device()
    exp_name    = EXPERIMENT["name"]
    best_path   = CHECKPOINT_DIR / f"{exp_name}_best.pth"
    latest_path = CHECKPOINT_DIR / f"{exp_name}_latest.pth"

    # Metric label varies by task
    metric_labels = {
        "classification": "acc",
        "detection"     : "obj_acc",
        "segmentation"  : "pixel_acc",
    }
    metric_label = metric_labels.get(task, "metric")

    logger.info("=" * 60)
    logger.info(f"  TRAINING PHASE  |  task={task}")
    logger.info("=" * 60)
    logger.info(f"Experiment : {exp_name}")
    logger.info(f"Device     : {device}")
    logger.info(f"Epochs     : {TRAIN['epochs']}")

    # ---- Data, model, loss -----------------------------------------------
    with Timer("Loading DataLoaders", logger=logger):
        train_loader, val_loader, test_loader = get_dataloaders(logger=logger)

    model     = get_model(logger=logger).to(device)
    criterion = get_loss(task=task, config=LOSS)

    if TRAIN["optimizer"].lower() == "adam":
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=TRAIN["learning_rate"], weight_decay=TRAIN["weight_decay"],
        )
    else:
        optimiser = torch.optim.SGD(
            model.parameters(),
            lr=TRAIN["learning_rate"],
            momentum=TRAIN["momentum"],
            weight_decay=TRAIN["weight_decay"],
        )

    # ---- Task step function (selected ONCE, called every batch) ----------
    step_fn = get_step_function(task)

    # ---- Resume ----------------------------------------------------------
    start_epoch      = 1
    best_val_metric  = 0.0

    if resume and latest_path.exists():
        start_epoch = _load_checkpoint(
            model, optimiser, latest_path, device, logger
        ) + 1
        if best_path.exists():
            ckpt            = torch.load(best_path, map_location=device,
                                         weights_only=False)
            best_val_metric = ckpt.get("val_metric", 0.0)
        logger.info(f"Resuming from epoch {start_epoch}/{TRAIN['epochs']}")
    else:
        logger.info("Fresh training from epoch 1")

    writer = MetricWriter(log_dir=str(LOG_DIR), experiment=exp_name)
    state.mark_started("training")

    # ---- Training loop ---------------------------------------------------
    with Timer(f"Total training — {TRAIN['epochs']} epochs", logger=logger):
        for epoch in range(start_epoch, TRAIN["epochs"] + 1):
            model.train()
            total_loss   = 0.0
            total_metric = 0.0
            n_batches    = 0

            pbar = tqdm(
                train_loader,
                desc         = f"  Epoch {epoch:>3} [Train]",
                leave        = False,
                unit         = "batch",
                dynamic_ncols= True,
            )

            with Timer(f"Epoch {epoch}", logger=logger) as epoch_timer:
                for batch in pbar:
                    optimiser.zero_grad()
                    loss, metric = step_fn(model, batch, criterion, device)
                    loss.backward()
                    optimiser.step()

                    total_loss   += loss.item()
                    total_metric += metric
                    n_batches    += 1
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

                train_loss   = total_loss   / max(n_batches, 1)
                train_metric = total_metric / max(n_batches, 1)

                val_loss, val_metric = evaluate(
                    model, val_loader, criterion, device, step_fn, split="Val"
                )

            logger.info(
                f"Epoch {epoch:>3}/{TRAIN['epochs']} | "
                f"Train loss: {train_loss:.4f}  {metric_label}: {train_metric:.4f} | "
                f"Val loss: {val_loss:.4f}  {metric_label}: {val_metric:.4f} | "
                f"Time: {epoch_timer.elapsed:.1f}s"
            )

            _save_checkpoint(model, optimiser, epoch, val_metric, latest_path)

            if val_metric > best_val_metric:
                best_val_metric = val_metric
                _save_checkpoint(model, optimiser, epoch, val_metric, best_path)
                logger.info(f"  ✓ New best {metric_label}: {val_metric:.4f} → {best_path.name}")

            writer.write(
                epoch        = epoch,
                train_loss   = train_loss,
                val_loss     = val_loss,
                val_acc      = val_metric,
                test_acc     = -1.0,
                lr           = TRAIN["learning_rate"],
                duration_sec = epoch_timer.elapsed,
            )
            state.mark_training_epoch(epoch=epoch, total_epochs=TRAIN["epochs"])

    # ---- Final test evaluation -------------------------------------------
    with Timer("Final test evaluation", logger=logger):
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            logger.info(f"Loaded best checkpoint (val {metric_label}: {best_val_metric:.4f})")

        _, test_metric = evaluate(
            model, test_loader, criterion, device, step_fn, split="Test"
        )

    logger.info(f"Final test {metric_label}: {test_metric:.4f}  ({test_metric*100:.2f}%)")

    writer.write(
        epoch=TRAIN["epochs"], train_loss=-1.0, val_loss=-1.0,
        val_acc=best_val_metric, test_acc=test_metric,
        lr=TRAIN["learning_rate"], duration_sec=0.0,
    )

    _write_training_manifest(best_path, best_val_metric)
    state.mark_done("training")
    logger.info("Training phase complete ✓")