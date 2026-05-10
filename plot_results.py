"""
plot_results.py
---------------
Reads logs/results.csv and produces 5 publication-quality figures
that together tell the complete story of the LeNet empirical study.

Generated plots (saved to plots/)
----------------------------------
  1. val_accuracy_curves.png  — Val accuracy over epochs for all 8 experiments
  2. train_loss_curves.png    — Training loss over epochs for all 8 experiments
  3. final_test_accuracy.png  — Ranked bar chart of final test accuracy
  4. activation_vs_pooling.png— Grouped bar chart: activation × pooling comparison
  5. convergence_speed.png    — Val accuracy at epoch 1 (how fast each variant learns)

Usage
-----
  python plot_results.py                        # reads logs/results.csv by default
  python plot_results.py --csv path/to/file.csv # custom CSV path
  python plot_results.py --out custom_dir/      # custom output directory

Beginners: Each plot is saved as a high-resolution PNG so you can drop
them straight into a report or README. The script is self-contained —
it only needs matplotlib and the CSV file.
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Style config — change these to restyle all plots at once
# ---------------------------------------------------------------------------

# One distinct colour per experiment, chosen for colour-blind friendliness
PALETTE = {
    "lenet_relu_maxpool"      : "#2196F3",   # blue
    "lenet_sigmoid_maxpool"   : "#FF9800",   # orange
    "lenet_tanh_maxpool"      : "#4CAF50",   # green
    "lenet_leakyrelu_maxpool" : "#F44336",   # red
    "lenet_relu_avgpool"      : "#9C27B0",   # purple
    "lenet_sigmoid_avgpool"   : "#795548",   # brown
    "lenet_tanh_avgpool"      : "#00BCD4",   # cyan
    "lenet_leakyrelu_avgpool" : "#E91E63",   # pink
}

# Line style: MaxPool = solid, AvgPool = dashed  → easy to tell them apart
LINESTYLE = {
    "max": "-",
    "avg": "--",
}

# Marker: one per activation so lines are readable even in greyscale print
MARKER = {
    "relu"      : "o",
    "sigmoid"   : "s",
    "tanh"      : "^",
    "leakyrelu" : "D",
}

DPI        = 150      # resolution of saved PNGs
FIG_SIZE   = (12, 6)  # width × height in inches (most plots)
FONT_TITLE = 14
FONT_LABEL = 12
FONT_TICK  = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(csv_path: str) -> Dict[str, List[dict]]:
    """
    Parse the results CSV and return per-experiment training histories.

    Dry-run rows (duration_sec < 1.0 and val_acc near 0) are automatically
    filtered out — they come from the pipeline sanity-check in Run.sh and
    should not appear in any plot.

    Parameters
    ----------
    csv_path : str  Path to logs/results.csv

    Returns
    -------
    dict  Keys are experiment names. Values are lists of row dicts,
          one per epoch, sorted by epoch number.
          Special "final" key holds the last test-accuracy row per experiment.

    Example
    -------
    >>> data = load_csv("logs/results.csv")
    >>> data["lenet_relu_maxpool"][0]
    {"epoch": 1, "train_loss": 0.27, "val_loss": 0.097, "val_acc": 0.970, ...}
    """
    raw: Dict[str, List[dict]] = defaultdict(list)
    finals: Dict[str, dict]    = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            exp          = row["experiment"]
            train_loss   = float(row["train_loss"])
            val_loss     = float(row["val_loss"])
            val_acc      = float(row["val_acc"])
            test_acc     = float(row["test_acc"])
            duration_sec = float(row["duration_sec"])
            epoch        = int(row["epoch"])

            # --- Final test-accuracy row (train_loss == -1) ----------------
            if train_loss == -1.0 and test_acc != -1.0:
                # Keep only the row where val_acc > 0 (the real run, not dry-run)
                if val_acc > 0.5:
                    finals[exp] = {
                        "experiment": exp,
                        "best_val_acc": val_acc,
                        "test_acc": test_acc,
                    }
                continue

            # --- Training row ----------------------------------------------
            # Skip dry-run rows: they have duration < 1 s and near-random accuracy
            if duration_sec < 1.0:
                continue
            if train_loss == -1.0:
                continue

            raw[exp].append({
                "epoch"      : epoch,
                "train_loss" : train_loss,
                "val_loss"   : val_loss,
                "val_acc"    : val_acc,
                "duration_sec": duration_sec,
            })

    # Sort each experiment's rows by epoch
    training: Dict[str, List[dict]] = {}
    for exp, rows in raw.items():
        training[exp] = sorted(rows, key=lambda r: r["epoch"])

    return training, finals


# ---------------------------------------------------------------------------
# Helper: extract activation and pooling from experiment name
# ---------------------------------------------------------------------------

def parse_name(exp_name: str):
    """
    Split 'lenet_relu_maxpool' into ('relu', 'max').

    Returns
    -------
    (activation, pooling)  both lowercase strings
    """
    # Format: lenet_{activation}_{pooling}pool
    parts      = exp_name.split("_")   # ['lenet', 'relu', 'maxpool']
    pooling    = "max" if "maxpool" in parts[-1] else "avg"
    activation = "_".join(parts[1:-1]) # handles 'leakyrelu' with underscore
    return activation, pooling


# ---------------------------------------------------------------------------
# Plot 1: Validation accuracy curves
# ---------------------------------------------------------------------------

def plot_val_accuracy(training: dict, out_dir: Path) -> None:
    """
    Line plot: validation accuracy vs epoch for all 8 experiments.

    MaxPool variants use solid lines; AvgPool variants use dashed lines.
    Each activation gets a unique colour AND marker so the chart is readable
    in both colour and greyscale (e.g. printed papers).
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for exp, rows in sorted(training.items()):
        activation, pooling = parse_name(exp)
        epochs   = [r["epoch"]  for r in rows]
        val_accs = [r["val_acc"] * 100 for r in rows]   # convert to %

        ax.plot(
            epochs, val_accs,
            color     = PALETTE[exp],
            linestyle = LINESTYLE[pooling],
            marker    = MARKER[activation],
            markersize= 5,
            linewidth = 1.8,
            label     = exp.replace("lenet_", "").replace("_", " | "),
        )

    ax.set_title("Validation Accuracy over Epochs — All Experiments", fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=FONT_LABEL)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.set_xlim(1, max(r["epoch"] for rows in training.values() for r in rows))
    ax.set_ylim(93, 100)   # zoom in — all experiments are above 93% after ep1
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, ncol=2, loc="lower right", framealpha=0.9)

    # Annotate final val acc for best and worst
    _add_linestyle_legend(ax)

    fig.tight_layout()
    path = out_dir / "val_accuracy_curves.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Plot 2: Training loss curves
# ---------------------------------------------------------------------------

def plot_train_loss(training: dict, out_dir: Path) -> None:
    """
    Line plot: training loss vs epoch for all 8 experiments.

    Sigmoid experiments stand out here — their epoch-1 loss is much higher
    (~1.0) because the sigmoid gradient saturates early, slowing convergence.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for exp, rows in sorted(training.items()):
        activation, pooling = parse_name(exp)
        epochs      = [r["epoch"]      for r in rows]
        train_losses= [r["train_loss"] for r in rows]

        ax.plot(
            epochs, train_losses,
            color     = PALETTE[exp],
            linestyle = LINESTYLE[pooling],
            marker    = MARKER[activation],
            markersize= 5,
            linewidth = 1.8,
            label     = exp.replace("lenet_", "").replace("_", " | "),
        )

    ax.set_title("Training Loss over Epochs — All Experiments", fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=FONT_LABEL)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, ncol=2, loc="upper right", framealpha=0.9)
    _add_linestyle_legend(ax)

    fig.tight_layout()
    path = out_dir / "train_loss_curves.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Plot 3: Final test accuracy bar chart (ranked)
# ---------------------------------------------------------------------------

def plot_final_test_accuracy(finals: dict, out_dir: Path) -> None:
    """
    Horizontal bar chart of final test accuracy, sorted best → worst.

    Each bar is coloured by its experiment colour from PALETTE, with the
    exact accuracy printed at the end of each bar for easy reading.
    """
    # Sort by test accuracy descending
    sorted_finals = sorted(finals.values(), key=lambda r: r["test_acc"], reverse=True)
    names   = [r["experiment"].replace("lenet_", "").replace("_", "\n") for r in sorted_finals]
    test_acc= [r["test_acc"] * 100 for r in sorted_finals]
    colors  = [PALETTE[r["experiment"]] for r in sorted_finals]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, test_acc, color=colors, edgecolor="white", height=0.6)

    # Print value labels at the end of each bar
    for bar, val in zip(bars, test_acc):
        ax.text(
            bar.get_width() - 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%",
            va="center", ha="right",
            fontsize=10, fontweight="bold", color="white",
        )

    ax.set_title("Final Test Accuracy — Ranked", fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlabel("Test Accuracy (%)", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    # Zoom x-axis to highlight differences
    min_acc = min(test_acc)
    ax.set_xlim(min_acc - 0.3, max(test_acc) + 0.1)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax.invert_yaxis()   # best at top

    fig.tight_layout()
    path = out_dir / "final_test_accuracy.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Plot 4: Grouped bar chart — activation × pooling
# ---------------------------------------------------------------------------

def plot_activation_vs_pooling(finals: dict, out_dir: Path) -> None:
    """
    Grouped bar chart: one group per activation, two bars per group
    (MaxPool vs AvgPool). Makes it very easy to see whether MaxPool or
    AvgPool consistently wins for a given activation.
    """
    activations = ["relu", "sigmoid", "tanh", "leakyrelu"]
    poolings    = ["max", "avg"]
    pool_colors = {"max": "#2196F3", "avg": "#FF5722"}

    x          = np.arange(len(activations))
    bar_width  = 0.35
    fig, ax    = plt.subplots(figsize=(10, 6))

    for i, pooling in enumerate(poolings):
        accs = []
        for act in activations:
            exp_name = f"lenet_{act}_{pooling}pool"
            accs.append(finals.get(exp_name, {}).get("test_acc", 0) * 100)

        offset = (i - 0.5) * bar_width
        bars = ax.bar(
            x + offset, accs,
            width      = bar_width,
            color      = pool_colors[pooling],
            label      = f"{pooling.capitalize()}Pool",
            edgecolor  = "white",
            linewidth  = 0.8,
        )

        # Value labels on top of each bar
        for bar, val in zip(bars, accs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.2f}%",
                ha="center", va="bottom",
                fontsize=8.5, color="#333333",
            )

    ax.set_title("Test Accuracy: Activation Function × Pooling Type", fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlabel("Activation Function", fontsize=FONT_LABEL)
    ax.set_ylabel("Test Accuracy (%)", fontsize=FONT_LABEL)
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in activations], fontsize=FONT_TICK)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    # Zoom y-axis
    all_accs = [finals[k]["test_acc"] * 100 for k in finals]
    ax.set_ylim(min(all_accs) - 0.3, max(all_accs) + 0.2)

    fig.tight_layout()
    path = out_dir / "activation_vs_pooling.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Plot 5: Convergence speed — val accuracy at epoch 1
# ---------------------------------------------------------------------------

def plot_convergence_speed(training: dict, out_dir: Path) -> None:
    """
    Bar chart of validation accuracy at epoch 1 only.

    This shows how quickly each variant starts learning — ReLU/Tanh/LeakyReLU
    all reach ~96-97% in the very first epoch, while Sigmoid lags behind at
    ~90-93% because its gradient saturates for large or small activations.
    """
    ep1_data = {}
    for exp, rows in training.items():
        if rows:
            ep1_data[exp] = rows[0]["val_acc"] * 100

    sorted_data = sorted(ep1_data.items(), key=lambda x: x[1], reverse=True)
    names  = [k.replace("lenet_", "").replace("_", "\n") for k, _ in sorted_data]
    values = [v for _, v in sorted_data]
    colors = [PALETTE[k] for k, _ in sorted_data]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, values, color=colors, edgecolor="white", width=0.6)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.1f}%",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )

    ax.set_title("Convergence Speed — Validation Accuracy at Epoch 1", fontsize=FONT_TITLE, fontweight="bold")
    ax.set_ylabel("Val Accuracy at Epoch 1 (%)", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=9)
    ax.set_ylim(min(values) - 2, 100)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Highlight sigmoid bars with a note
    ax.axhline(y=95, color="red", linestyle="--", linewidth=1, alpha=0.5, label="95% threshold")
    ax.legend(fontsize=10)

    fig.tight_layout()
    path = out_dir / "convergence_speed.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Helper: add a linestyle legend (solid = MaxPool, dashed = AvgPool)
# ---------------------------------------------------------------------------

def _add_linestyle_legend(ax) -> None:
    """
    Add a small secondary legend showing that solid = MaxPool, dashed = AvgPool.
    Placed in the top-left so it doesn't overlap the main legend.
    """
    handles = [
        mpatches.Patch(facecolor="grey", label="— MaxPool (solid)"),
        mpatches.Patch(facecolor="grey", label="-- AvgPool (dashed)"),
    ]
    # We use a plain text annotation instead of a second legend box
    ax.annotate(
        "— solid: MaxPool    -- dashed: AvgPool",
        xy=(0.01, 0.02), xycoords="axes fraction",
        fontsize=8, color="grey",
    )


# ---------------------------------------------------------------------------
# Print summary table to console
# ---------------------------------------------------------------------------

def print_summary_table(finals: dict, training: dict) -> None:
    """Print a clean results table to the console after all plots are saved."""
    sorted_f = sorted(finals.values(), key=lambda r: r["test_acc"], reverse=True)

    print()
    print("=" * 72)
    print("  EMPIRICAL STUDY RESULTS SUMMARY")
    print("=" * 72)
    print(f"  {'Rank':<5} {'Experiment':<30} {'Best Val':>9} {'Test Acc':>9} {'Conv.Ep1':>9}")
    print("-" * 72)

    for rank, r in enumerate(sorted_f, 1):
        exp  = r["experiment"]
        rows = training.get(exp, [])
        ep1  = rows[0]["val_acc"] * 100 if rows else 0.0
        print(
            f"  {rank:<5} {exp:<30} "
            f"{r['best_val_acc']*100:>8.2f}%  "
            f"{r['test_acc']*100:>8.2f}%  "
            f"{ep1:>8.1f}%"
        )

    print("=" * 72)
    print()
    print("  Key findings:")
    best = sorted_f[0]
    worst= sorted_f[-1]
    print(f"  • Best  : {best['experiment']}  ({best['test_acc']*100:.2f}% test acc)")
    print(f"  • Worst : {worst['experiment']}  ({worst['test_acc']*100:.2f}% test acc)")
    gap = (best['test_acc'] - worst['test_acc']) * 100
    print(f"  • Gap   : {gap:.2f} percentage points between best and worst")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# CLI and entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot LeNet empirical study results from the CSV log file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", default="logs/results.csv",
                   help="Path to the results CSV file.")
    p.add_argument("--out", default="plots",
                   help="Output directory for PNG files.")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nReading  : {args.csv}")
    print(f"Saving to: {out_dir}/\n")

    training, finals = load_csv(args.csv)

    if not training:
        print("ERROR: No training rows found. Check the CSV path or content.")
        return

    print("Generating plots …")
    plot_val_accuracy      (training, out_dir)
    plot_train_loss        (training, out_dir)
    plot_final_test_accuracy(finals,  out_dir)
    plot_activation_vs_pooling(finals,out_dir)
    plot_convergence_speed (training, out_dir)

    print_summary_table(finals, training)
    print(f"Done. All plots saved to '{out_dir}/'")


if __name__ == "__main__":
    main()