"""
utils.py
--------
Shared utility classes and functions used across the entire project.

Contains:
  - Timer        : A context manager to measure and log how long any code block takes.
  - get_logger   : Sets up a logger that writes to both the console and a log file.
  - SystemInfo   : Prints Python, PyTorch, and hardware info at startup.
  - save_metrics : Appends one row of training metrics to the master CSV file.

Beginners: Think of this file as a toolbox. Every other file in the project
imports from here so we don't repeat the same helper code everywhere.
"""

import csv
import logging
import os
import platform
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    """
    A context manager that measures elapsed time for any block of code.

    Usage
    -----
    >>> with Timer("Downloading dataset", logger=my_logger) as t:
    ...     download_something()
    # Console/log output: [Downloading dataset] completed in 3.42s

    Parameters
    ----------
    label : str
        A human-readable name for the phase being timed (e.g. "Epoch 1 training").
    logger : logging.Logger, optional
        If provided, the timing message is written to this logger.
        If None, the message is printed to the console with print().

    Attributes
    ----------
    elapsed : float
        Seconds elapsed. Available after the `with` block exits.
    """

    def __init__(self, label: str, logger: logging.Logger = None):
        self.label   = label
        self.logger  = logger
        self.elapsed = 0.0
        self._start  = None

    def __enter__(self):
        """Called when entering the `with` block. Records the start time."""
        self._start = time.perf_counter()
        return self  # lets the caller do `as t` and later read t.elapsed

    def __exit__(self, *args):
        """Called when leaving the `with` block. Computes and reports elapsed time."""
        self.elapsed = time.perf_counter() - self._start
        msg = f"[{self.label}] completed in {self.elapsed:.2f}s"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Create (or retrieve) a logger that writes to both the console and a file.

    Each experiment gets its own log file, e.g. logs/lenet_relu_maxpool.log.
    If the logger with this name already exists (happens when a module is
    imported twice), we return the existing one unchanged to avoid duplicate
    log lines.

    Parameters
    ----------
    name    : str
        Unique name for this logger, typically the experiment name.
        Also used as the log filename (spaces replaced with underscores).
    log_dir : str
        Directory where log files are saved. Created automatically if missing.

    Returns
    -------
    logging.Logger
        Configured logger ready to use.

    Example
    -------
    >>> logger = get_logger("lenet_relu_maxpool", log_dir="logs")
    >>> logger.info("Training started")
    """
    # Reuse existing logger to prevent duplicate handlers on re-import
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # capture everything; handlers filter by level

    # ---- Formatter -------------------------------------------------------
    # Example output: 2024-05-10 14:23:01 | INFO  | Training started
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---- Console handler (INFO and above) --------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # ---- File handler (DEBUG and above — captures everything) ------------
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{name.replace(' ', '_')}.log"
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialised → {log_file}")
    return logger


# ---------------------------------------------------------------------------
# SystemInfo
# ---------------------------------------------------------------------------

class SystemInfo:
    """
    Prints a summary of the current hardware and software environment.

    Call this once at the very start of a training run so that log files
    contain enough context for debugging later (especially useful on VMs
    where you might forget which instance type you used).

    Example
    -------
    >>> SystemInfo.print(logger=my_logger)
    """

    @staticmethod
    def print(logger: logging.Logger = None):
        """
        Collect and log system information.

        Parameters
        ----------
        logger : logging.Logger, optional
            Where to write the info. Falls back to print() if None.
        """
        # Lazy import so torch isn't required just to use Timer/Logger
        try:
            import torch
            torch_version = torch.__version__
            if torch.cuda.is_available():
                device_info = (
                    f"CUDA {torch.version.cuda} | "
                    f"GPU: {torch.cuda.get_device_name(0)} | "
                    f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
                )
            else:
                device_info = "CPU only (no CUDA GPU detected)"
        except ImportError:
            torch_version = "not installed"
            device_info   = "unknown"

        lines = [
            "=" * 60,
            "  SYSTEM INFORMATION",
            "=" * 60,
            f"  OS          : {platform.system()} {platform.release()}",
            f"  Python      : {platform.python_version()}",
            f"  PyTorch     : {torch_version}",
            f"  Device      : {device_info}",
            "=" * 60,
        ]

        output = "\n".join(lines)
        if logger:
            logger.info("\n" + output)
        else:
            print(output)


# ---------------------------------------------------------------------------
# CSV metrics helper
# ---------------------------------------------------------------------------

def save_metrics(
    csv_path: str,
    experiment: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_acc: float,
    test_acc: float,
    duration_sec: float,
) -> None:
    """
    Append one row of training metrics to the master CSV file.

    If the file does not exist yet, the header row is written first.
    This means every experiment, every epoch produces exactly one row,
    making it easy to load the CSV into pandas or a spreadsheet for analysis.

    Parameters
    ----------
    csv_path     : str   Path to the CSV file (e.g. "logs/results.csv").
    experiment   : str   Name of the current experiment (e.g. "lenet_relu_maxpool").
    epoch        : int   Current epoch number (1-indexed).
    train_loss   : float Average training loss for this epoch.
    val_loss     : float Average validation loss for this epoch.
    val_acc      : float Validation accuracy (0.0 – 1.0).
    test_acc     : float Final test accuracy — only meaningful on the last epoch; 
                         pass -1.0 for intermediate epochs.
    duration_sec : float Wall-clock seconds this epoch took.

    Example
    -------
    >>> save_metrics("logs/results.csv", "lenet_relu_maxpool", 1,
    ...              train_loss=0.35, val_loss=0.28, val_acc=0.92,
    ...              test_acc=-1.0, duration_sec=45.2)
    """
    fieldnames = [
        "experiment", "epoch", "train_loss",
        "val_loss", "val_acc", "test_acc", "duration_sec",
    ]

    file_exists = Path(csv_path).exists()
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  # write column names on first call only
        writer.writerow({
            "experiment"  : experiment,
            "epoch"       : epoch,
            "train_loss"  : round(train_loss,  6),
            "val_loss"    : round(val_loss,    6),
            "val_acc"     : round(val_acc,     6),
            "test_acc"    : round(test_acc,    6),
            "duration_sec": round(duration_sec, 2),
        })
