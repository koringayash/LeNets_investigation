"""
train.py
--------
Orchestrates the full training pipeline for the LeNet empirical study.

What this script does
---------------------
  1. Parses command-line arguments.
  2. Prints system info (Python version, device, etc.).
  3. Downloads / loads the MNIST dataset once.
  4. Iterates over every experiment defined in config.EXPERIMENTS.
  5. For each experiment:
       a. Builds the LeNet model via CNNModel.
       b. Trains for the configured number of epochs.
       c. Tracks train/val loss and accuracy with a tqdm progress bar.
       d. Saves the best checkpoint (by val accuracy).
       e. Logs every epoch's metrics to the master CSV file.
       f. Evaluates final test accuracy once training completes.
  6. Prints a summary table of all experiments at the end.

Usage
-----
  # Run all experiments (defined in config.py)
  python train.py

  # Run a specific experiment only
  python train.py --experiment lenet_relu_maxpool

  # Sanity check: one batch, no checkpoint saved
  python train.py --dry-run

  # Override number of epochs
  python train.py --epochs 5

Beginners: The training loop is the heart of machine learning.
For each epoch:
  - We feed the whole training set to the model in mini-batches.
  - The model predicts labels, we compute the loss, and backpropagation
    adjusts the weights to reduce that loss.
  - After every epoch, we evaluate on the validation set (no weight updates)
    to see how well the model generalises.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DATASET, TRAIN, EXPERIMENTS,
    CHECKPOINT_DIR, RESULTS_CSV, LOG_DIR,
    build_lenet_config, get_experiment_name,
)
from dataset import download_mnist, get_dataloaders, describe_dataset
from network import CNNModel
from utils import Timer, get_logger, SystemInfo, save_metrics


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Return the best available device (CUDA GPU > CPU).

    If TRAIN['device'] is set to 'auto' in config, we pick GPU when available.
    You can override by setting it to 'cpu' or 'cuda' explicitly.

    Returns
    -------
    torch.device
    """
    pref = TRAIN["device"]
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    criterion:    nn.Module,
    optimiser:    torch.optim.Optimizer,
    device:       torch.device,
    epoch:        int,
    dry_run:      bool = False,
) -> Tuple[float, float]:
    """
    Run one full pass over the training data and update model weights.

    Parameters
    ----------
    model     : The CNN model being trained.
    loader    : DataLoader for the training split.
    criterion : Loss function (CrossEntropyLoss).
    optimiser : Adam optimiser attached to the model's parameters.
    device    : torch.device to move tensors to.
    epoch     : Current epoch number (1-indexed), shown in the progress bar.
    dry_run   : If True, stop after the very first batch (for quick sanity checks).

    Returns
    -------
    avg_loss : float  Mean cross-entropy loss over all batches.
    accuracy : float  Fraction of correctly classified training samples (0–1).
    """
    model.train()   # enable dropout, batchnorm training behaviour
    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    # tqdm wraps the DataLoader — it shows a live progress bar in the terminal.
    pbar = tqdm(
        loader,
        desc        = f"  Epoch {epoch:>3} [Train]",
        leave       = False,
        unit        = "batch",
        dynamic_ncols= True,
    )

    for batch_idx, (images, labels) in enumerate(pbar):
        # Move data to GPU / CPU
        images, labels = images.to(device), labels.to(device)

        # --- Forward pass --------------------------------------------------
        optimiser.zero_grad()          # clear gradients from the last batch
        logits = model(images)         # raw predictions (not softmax yet)
        loss   = criterion(logits, labels)

        # --- Backward pass -------------------------------------------------
        loss.backward()                # compute gradients
        optimiser.step()               # update weights

        # --- Accumulate metrics --------------------------------------------
        total_loss    += loss.item() * images.size(0)   # weighted by batch size
        preds          = logits.argmax(dim=1)
        correct       += (preds == labels).sum().item()
        total_samples += images.size(0)

        # Live loss display in the progress bar
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if dry_run:
            break   # exit after first batch for sanity-check runs

    avg_loss = total_loss / total_samples
    accuracy = correct   / total_samples
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Evaluation (validation or test)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    split:     str = "Val",
    dry_run:   bool = False,
) -> Tuple[float, float]:
    """
    Evaluate the model on a dataset split without updating weights.

    The @torch.no_grad() decorator turns off gradient tracking for the whole
    function, which saves memory and speeds up evaluation.

    Parameters
    ----------
    model     : The CNN model (in eval mode, set by caller).
    loader    : DataLoader for val or test split.
    criterion : Loss function (same as training).
    device    : torch.device.
    split     : Label shown in the progress bar ("Val" or "Test").
    dry_run   : If True, stop after the first batch.

    Returns
    -------
    avg_loss : float  Mean loss over all batches.
    accuracy : float  Fraction of correct predictions (0–1).
    """
    model.eval()   # disable dropout, use running stats for batchnorm
    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    pbar = tqdm(
        loader,
        desc        = f"           [{split}]  ",
        leave       = False,
        unit        = "batch",
        dynamic_ncols= True,
    )

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        logits         = model(images)
        loss           = criterion(logits, labels)

        total_loss    += loss.item() * images.size(0)
        preds          = logits.argmax(dim=1)
        correct       += (preds == labels).sum().item()
        total_samples += images.size(0)

        if dry_run:
            break

    avg_loss = total_loss / total_samples
    accuracy = correct   / total_samples
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    exp:          dict,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    test_loader:  DataLoader,
    device:       torch.device,
    args:         argparse.Namespace,
) -> dict:
    """
    Train one LeNet variant end-to-end and return its final metrics.

    Parameters
    ----------
    exp          : Experiment config dict, e.g. {"activation": "relu", "pooling": "max"}.
    train_loader : DataLoader for training.
    val_loader   : DataLoader for validation.
    test_loader  : DataLoader for final test evaluation.
    device       : torch.device.
    args         : Parsed CLI arguments (epochs, dry_run, etc.).

    Returns
    -------
    dict  {"name": str, "best_val_acc": float, "test_acc": float}
    """
    exp_name = get_experiment_name(exp)
    logger   = get_logger(exp_name, log_dir=str(LOG_DIR))

    logger.info("=" * 60)
    logger.info(f"  EXPERIMENT: {exp_name}")
    logger.info(f"  Activation : {exp['activation']}")
    logger.info(f"  Pooling    : {exp['pooling']}")
    logger.info(f"  Epochs     : {args.epochs}")
    logger.info(f"  Device     : {device}")
    logger.info("=" * 60)

    # ---- Build model ------------------------------------------------------
    with Timer("Building model", logger=logger):
        layer_cfg = build_lenet_config(exp["activation"], exp["pooling"])
        input_shape = (
            DATASET["in_channels"],
            DATASET["image_size"],
            DATASET["image_size"],
        )
        model = CNNModel(layer_cfg, input_shape=input_shape).to(device)
        model.summary()
        logger.info(f"Parameters  : {model.count_parameters():,}")
        logger.info(f"Model size  : {model.model_size_mb()} MB")

    # ---- Loss and optimiser -----------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr           = TRAIN["learning_rate"],
        weight_decay = TRAIN["weight_decay"],
    )

    # ---- Training loop ----------------------------------------------------
    best_val_acc  = 0.0
    best_ckpt_path = CHECKPOINT_DIR / f"{exp_name}_best.pth"

    with Timer(f"Total training for {exp_name}", logger=logger):
        for epoch in range(1, args.epochs + 1):
            with Timer(f"Epoch {epoch}", logger=logger) as epoch_timer:

                train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimiser,
                    device, epoch, dry_run=args.dry_run,
                )
                val_loss, val_acc = evaluate(
                    model, val_loader, criterion,
                    device, split="Val", dry_run=args.dry_run,
                )

            # Log to console / file
            logger.info(
                f"Epoch {epoch:>3}/{args.epochs} | "
                f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
                f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f} | "
                f"Time: {epoch_timer.elapsed:.1f}s"
            )

            # Save best checkpoint
            if val_acc > best_val_acc and not args.dry_run:
                best_val_acc = val_acc
                torch.save(
                    {
                        "epoch"      : epoch,
                        "model_state": model.state_dict(),
                        "val_acc"    : val_acc,
                        "experiment" : exp,
                    },
                    best_ckpt_path,
                )
                logger.info(
                    f"  ✓ New best val acc: {val_acc:.4f} — "
                    f"checkpoint saved → {best_ckpt_path.name}"
                )

            # Write metrics row to master CSV (-1.0 for test_acc mid-training)
            save_metrics(
                csv_path    = str(RESULTS_CSV),
                experiment  = exp_name,
                epoch       = epoch,
                train_loss  = train_loss,
                val_loss    = val_loss,
                val_acc     = val_acc,
                test_acc    = -1.0,
                duration_sec= epoch_timer.elapsed,
            )

            if args.dry_run:
                logger.info("Dry-run mode: stopping after first epoch.")
                break

    # ---- Final test evaluation --------------------------------------------
    with Timer("Final test evaluation", logger=logger):
        # Load best weights before testing (unless dry-run)
        if best_ckpt_path.exists() and not args.dry_run:
            checkpoint = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            logger.info(
                f"Loaded best checkpoint (val acc: "
                f"{checkpoint['val_acc']:.4f}, epoch {checkpoint['epoch']})"
            )

        _, test_acc = evaluate(
            model, test_loader, criterion,
            device, split="Test", dry_run=args.dry_run,
        )

    logger.info(f"Final test accuracy: {test_acc:.4f}  ({test_acc*100:.2f}%)")

    # Append a dedicated test-result row to the CSV
    save_metrics(
        csv_path    = str(RESULTS_CSV),
        experiment  = exp_name,
        epoch       = args.epochs,    # mark as the final epoch
        train_loss  = -1.0,
        val_loss    = -1.0,
        val_acc     = best_val_acc,
        test_acc    = test_acc,
        duration_sec= 0.0,
    )

    return {
        "name"        : exp_name,
        "best_val_acc": best_val_acc,
        "test_acc"    : test_acc,
    }


# ---------------------------------------------------------------------------
# Final summary printer
# ---------------------------------------------------------------------------

def print_summary(results: list, logger) -> None:
    """
    Print a comparison table of all experiments after all runs finish.

    Parameters
    ----------
    results : list of dict  Each dict has "name", "best_val_acc", "test_acc".
    logger  : logging.Logger
    """
    # Sort by test accuracy descending so the best experiment is on top
    results_sorted = sorted(results, key=lambda r: r["test_acc"], reverse=True)

    lines = [
        "",
        "=" * 60,
        "  EMPIRICAL STUDY RESULTS",
        "=" * 60,
        f"  {'Rank':<5} {'Experiment':<35} {'Val Acc':>8} {'Test Acc':>9}",
        "-" * 60,
    ]
    for rank, r in enumerate(results_sorted, start=1):
        lines.append(
            f"  {rank:<5} {r['name']:<35} "
            f"{r['best_val_acc']:>7.4f}  {r['test_acc']:>8.4f}"
        )
    lines += ["=" * 60, f"  Results CSV → {RESULTS_CSV}", "=" * 60, ""]

    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Define and parse command-line arguments.

    Returns
    -------
    argparse.Namespace  Object with attributes: experiment, epochs, dry_run.
    """
    parser = argparse.ArgumentParser(
        description="LeNet Empirical Study — train and compare CNN variants on MNIST.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        type    = str,
        default = None,
        help    = (
            "Name of a single experiment to run, e.g. 'lenet_relu_maxpool'. "
            "If omitted, all experiments in config.EXPERIMENTS are run."
        ),
    )
    parser.add_argument(
        "--epochs",
        type    = int,
        default = TRAIN["epochs"],
        help    = "Number of training epochs (overrides config.TRAIN['epochs']).",
    )
    parser.add_argument(
        "--dry-run",
        action  = "store_true",
        help    = (
            "Sanity-check mode: run only one batch per phase, "
            "no checkpoints saved. Use to verify the pipeline works end-to-end."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Main entry point: parse args → setup → download data → run experiments.
    """
    args   = parse_args()
    device = get_device()

    # ---- Root logger (shared across all experiments) ----------------------
    root_logger = get_logger("main", log_dir=str(LOG_DIR))
    SystemInfo.print(logger=root_logger)

    # ---- Data (downloaded once, shared across all experiments) ------------
    with Timer("Data preparation", logger=root_logger):
        download_mnist(logger=root_logger)
        train_loader, val_loader, test_loader = get_dataloaders(logger=root_logger)
        describe_dataset(train_loader, val_loader, test_loader, logger=root_logger)

    # ---- Filter experiments if --experiment flag is set -------------------
    if args.experiment:
        experiments = [
            e for e in EXPERIMENTS
            if get_experiment_name(e) == args.experiment
        ]
        if not experiments:
            root_logger.error(
                f"Experiment '{args.experiment}' not found. "
                f"Valid names: {[get_experiment_name(e) for e in EXPERIMENTS]}"
            )
            sys.exit(1)
    else:
        experiments = EXPERIMENTS

    root_logger.info(
        f"Running {len(experiments)} experiment(s) "
        f"| {args.epochs} epoch(s) each "
        f"| device: {device}"
        + (" | DRY RUN" if args.dry_run else "")
    )

    # ---- Run all experiments ----------------------------------------------
    results = []
    with Timer("All experiments combined", logger=root_logger):
        for i, exp in enumerate(experiments, start=1):
            root_logger.info(
                f"\n{'─' * 60}\n"
                f"  Experiment {i}/{len(experiments)}: {get_experiment_name(exp)}\n"
                f"{'─' * 60}"
            )
            result = run_experiment(
                exp, train_loader, val_loader, test_loader, device, args
            )
            results.append(result)

    # ---- Final summary ----------------------------------------------------
    if len(results) > 1:
        print_summary(results, logger=root_logger)
    else:
        r = results[0]
        root_logger.info(
            f"Done. Best val acc: {r['best_val_acc']:.4f} | "
            f"Test acc: {r['test_acc']:.4f}"
        )


if __name__ == "__main__":
    main()
