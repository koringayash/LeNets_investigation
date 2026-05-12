"""
utils/metrics.py
----------------
Saves per-epoch training metrics to both CSV and JSON formats
simultaneously. Having two formats means:

  - CSV  → easy to open in Excel / pandas / LibreOffice for analysis
  - JSON → easy to parse programmatically (e.g. in results.py for plots)

Both files are appended to on every call — one row/entry per epoch —
so they grow incrementally throughout training. If training is resumed
after a crash, new rows are simply appended after the existing ones.

Usage
-----
>>> from utils.metrics import MetricWriter
>>> writer = MetricWriter(log_dir="logs", experiment="cifar10_resnet18")
>>> writer.write(epoch=1, train_loss=0.35, val_loss=0.28, val_acc=0.92,
...              test_acc=-1.0, lr=0.001, duration_sec=45.2)
"""

import csv
import json
from pathlib import Path
from datetime import datetime


# Columns written to the CSV (order is preserved)
_FIELDS = [
    "experiment", "epoch", "train_loss", "val_loss",
    "val_acc", "test_acc", "lr", "duration_sec", "timestamp",
]


class MetricWriter:
    """
    Writes training metrics to CSV and JSON log files in one call.

    Both files are created on the first write if they don't exist yet.
    Subsequent writes append new rows/entries without overwriting history.

    Parameters
    ----------
    log_dir    : str   Directory where log files are saved.
    experiment : str   Experiment name used as the filename base and
                       stored in every row so multiple experiments can
                       share one file if needed.

    Example
    -------
    >>> writer = MetricWriter("logs", "cifar10_resnet18")
    >>> writer.write(epoch=1, train_loss=0.42, val_loss=0.31,
    ...              val_acc=0.91, test_acc=-1.0, lr=1e-3,
    ...              duration_sec=47.3)
    """

    def __init__(self, log_dir: str, experiment: str):
        self.experiment = experiment
        log_path        = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        self.csv_path  = log_path / "results.csv"
        self.json_path = log_path / "results.json"

    def write(
        self,
        epoch       : int,
        train_loss  : float,
        val_loss    : float,
        val_acc     : float,
        test_acc    : float,
        lr          : float,
        duration_sec: float,
    ) -> None:
        """
        Append one epoch's metrics to both the CSV and JSON log files.

        Parameters
        ----------
        epoch        : int    Current epoch number (1-indexed).
        train_loss   : float  Average training loss for this epoch.
        val_loss     : float  Average validation loss for this epoch.
        val_acc      : float  Validation accuracy (0.0 – 1.0).
        test_acc     : float  Final test accuracy. Pass -1.0 during training;
                              only meaningful on the last epoch.
        lr           : float  Learning rate used this epoch (useful if you
                              use a scheduler that changes LR over time).
        duration_sec : float  Wall-clock seconds this epoch took.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        row = {
            "experiment"  : self.experiment,
            "epoch"       : epoch,
            "train_loss"  : round(train_loss,   6),
            "val_loss"    : round(val_loss,     6),
            "val_acc"     : round(val_acc,      6),
            "test_acc"    : round(test_acc,     6),
            "lr"          : lr,
            "duration_sec": round(duration_sec, 2),
            "timestamp"   : timestamp,
        }

        self._write_csv(row)
        self._write_json(row)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _write_csv(self, row: dict) -> None:
        """Append one row to the CSV file, writing the header if needed."""
        file_exists = self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _write_json(self, row: dict) -> None:
        """
        Append one entry to the JSON file.

        The JSON file is structured as a list of dicts. We read the existing
        list (if any), append the new row, and write the whole list back.
        This keeps the file valid JSON at all times.

        Note: for very long training runs (1000+ epochs) this is slightly
        inefficient. For typical use (10–200 epochs) it is perfectly fine.
        """
        if self.json_path.exists():
            with open(self.json_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []   # recover from a corrupted file
        else:
            data = []

        data.append(row)

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)