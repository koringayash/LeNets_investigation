"""
evaluation/results.py (v2)
---------------------------
Reads logs/results.json + logs/eval_metrics.json and generates
task-appropriate plots in the plots/ directory.
"""

import json
import logging
import numpy as np
from pathlib import Path

from config import LOG_DIR, PLOTS_DIR, EXPERIMENT


def generate_all_plots(logger: logging.Logger = None) -> None:
    """Generate training curves and task-specific evaluation plots."""
    log = logger.info if logger else print
    task = EXPERIMENT["task"].lower()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("WARNING: matplotlib not installed — skipping plots.")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Training curves (all tasks) ------------------------------------
    results_path = LOG_DIR / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            history = json.load(f)
        rows = [r for r in history if r.get("train_loss", -1) != -1.0]
        if rows:
            epochs   = [r["epoch"]     for r in rows]
            val_accs = [r["val_acc"]   for r in rows]
            t_losses = [r["train_loss"] for r in rows]
            _plot_line(epochs, [v*100 for v in val_accs],
                       "Validation Metric over Epochs", "Epoch", "Val Metric (%)",
                       PLOTS_DIR / "val_metric_curves.png", logger)
            _plot_line(epochs, t_losses,
                       "Training Loss over Epochs", "Epoch", "Loss",
                       PLOTS_DIR / "train_loss_curves.png", logger)

    # ---- Task-specific evaluation plots ---------------------------------
    eval_path = LOG_DIR / "eval_metrics.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_data = json.load(f)

        if task == "classification" and "confusion_matrix" in eval_data:
            cm = np.array(eval_data["confusion_matrix"])
            _plot_confusion_matrix(cm, PLOTS_DIR / "confusion_matrix.png",
                                   EXPERIMENT["name"], logger)

        elif task == "detection" and "per_class_ap" in eval_data:
            _plot_bar(eval_data["per_class_ap"],
                      "Per-Class AP@0.5", "Class", "AP",
                      PLOTS_DIR / "per_class_ap.png", logger)

        elif task == "segmentation" and "per_class_iou" in eval_data:
            _plot_bar(eval_data["per_class_iou"],
                      "Per-Class IoU", "Class", "IoU",
                      PLOTS_DIR / "per_class_iou.png", logger)

    log(f"All plots saved → {PLOTS_DIR}/")


def _plot_line(x, y, title, xlabel, ylabel, path, logger=None):
    import matplotlib.pyplot as plt
    log = logger.info if logger else print
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, color="#2196F3", linewidth=2, marker="o", markersize=4)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig); log(f"  Saved → {Path(path).name}")


def _plot_bar(data_dict, title, xlabel, ylabel, path, logger=None):
    import matplotlib.pyplot as plt
    log = logger.info if logger else print
    labels = list(data_dict.keys())
    values = [data_dict[k] * 100 for k in labels]
    fig, ax = plt.subplots(figsize=(max(8, len(labels)), 5))
    ax.bar(labels, values, color="#4CAF50", edgecolor="white")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel(ylabel + " (%)", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig); log(f"  Saved → {Path(path).name}")


def _plot_confusion_matrix(cm, path, experiment, logger=None):
    import matplotlib.pyplot as plt
    log = logger.info if logger else print
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(6,n), max(5,n-1)))
    im = ax.imshow(cm, cmap="Blues"); fig.colorbar(im, ax=ax)
    ax.set_title(f"Confusion Matrix — {experiment}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black",
                    fontsize=max(6, 12 - n//3))
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig); log(f"  Saved → {Path(path).name}")