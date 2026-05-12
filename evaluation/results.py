"""
evaluation/results.py
---------------------
Reads logs/results.json and logs/eval_metrics.json and generates all
visualisation plots saved to the plots/ directory.

Plots generated
---------------
  1. val_accuracy_curves.png  — val accuracy over epochs
  2. train_loss_curves.png    — training loss over epochs
  3. confusion_matrix.png     — heatmap of the confusion matrix

Usage
-----
>>> from evaluation.results import generate_all_plots
>>> generate_all_plots(logger=logger)
"""

import json
import logging
from pathlib import Path

import numpy as np

from config import LOG_DIR, PLOTS_DIR, EXPERIMENT


def generate_all_plots(logger: logging.Logger = None) -> None:
    """
    Generate all visualisation plots from the log files.

    Reads
    -----
    logs/results.json      — per-epoch training metrics
    logs/eval_metrics.json — final evaluation metrics + confusion matrix

    Writes
    ------
    plots/val_accuracy_curves.png
    plots/train_loss_curves.png
    plots/confusion_matrix.png

    Parameters
    ----------
    logger : logging.Logger, optional
    """
    log = logger.info if logger else print

    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — safe for VMs
        import matplotlib.pyplot as plt
    except ImportError:
        log("WARNING: matplotlib not installed — skipping plots. "
            "Install with: pip install matplotlib")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load training history -------------------------------------------
    results_path = LOG_DIR / "results.json"
    if not results_path.exists():
        log(f"WARNING: {results_path} not found — skipping training curves.")
    else:
        with open(results_path, "r") as f:
            history = json.load(f)

        # Filter to real training rows (train_loss != -1)
        train_rows = [r for r in history if r.get("train_loss", -1) != -1.0]

        if train_rows:
            epochs    = [r["epoch"]      for r in train_rows]
            val_accs  = [r["val_acc"]    for r in train_rows]
            t_losses  = [r["train_loss"] for r in train_rows]

            _plot_line(
                x=epochs, y=[v * 100 for v in val_accs],
                title="Validation Accuracy over Epochs",
                xlabel="Epoch", ylabel="Val Accuracy (%)",
                path=PLOTS_DIR / "val_accuracy_curves.png",
                experiment=EXPERIMENT["name"],
                logger=logger,
            )
            _plot_line(
                x=epochs, y=t_losses,
                title="Training Loss over Epochs",
                xlabel="Epoch", ylabel="Cross-Entropy Loss",
                path=PLOTS_DIR / "train_loss_curves.png",
                experiment=EXPERIMENT["name"],
                logger=logger,
            )

    # ---- Load eval metrics (confusion matrix) ----------------------------
    eval_path = LOG_DIR / "eval_metrics.json"
    if not eval_path.exists():
        log(f"WARNING: {eval_path} not found — skipping confusion matrix.")
    else:
        with open(eval_path, "r") as f:
            eval_data = json.load(f)

        cm = np.array(eval_data["confusion_matrix"])
        _plot_confusion_matrix(
            cm         = cm,
            path       = PLOTS_DIR / "confusion_matrix.png",
            experiment = EXPERIMENT["name"],
            logger     = logger,
        )

    log(f"All plots saved → {PLOTS_DIR}/")


# ---------------------------------------------------------------------------
# Private plotting helpers
# ---------------------------------------------------------------------------

def _plot_line(
    x, y, title, xlabel, ylabel, path, experiment, logger=None
) -> None:
    """Plot a simple line chart and save it."""
    import matplotlib.pyplot as plt

    log = logger.info if logger else print
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x, y, color="#2196F3", linewidth=2, marker="o", markersize=4,
            label=experiment)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved → {path.name}")


def _plot_confusion_matrix(cm, path, experiment, logger=None) -> None:
    """
    Plot a confusion matrix heatmap.

    Cells are colour-coded by count. Each cell shows the raw count.
    Rows = true labels, Columns = predicted labels.
    """
    import matplotlib.pyplot as plt

    log = logger.info if logger else print
    n   = cm.shape[0]

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_title(f"Confusion Matrix — {experiment}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label",      fontsize=11)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    # Annotate each cell with its count
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=max(6, 12 - n // 3),
            )

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved → {path.name}")