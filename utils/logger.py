"""
utils/logger.py
---------------
Sets up loggers that write to both the console and a log file, with a
[Phase] prefix tag on every message so you always know which part of
the pipeline produced a given log line.

Usage
-----
>>> from utils.logger import get_logger
>>> logger = get_logger("training", phase="Training", log_dir="logs")
>>> logger.info("Epoch 1 started")
# Console + file: 2024-05-10 14:23:01 | INFO  | [Training] Epoch 1 started
"""

import logging
from pathlib import Path


def get_logger(
    name    : str,
    phase   : str = "",
    log_dir : str = "logs",
) -> logging.Logger:
    """
    Create (or retrieve) a logger that writes to both console and a log file.

    Each unique `name` gets its own log file under log_dir/. If a logger
    with this name was already created (e.g. on re-import), the existing
    one is returned unchanged — this prevents duplicate log lines.

    Parameters
    ----------
    name    : str
        Unique identifier for this logger. Also used as the log filename.
        e.g. "training" → logs/training.log
    phase   : str
        Short label prepended to every message in square brackets.
        e.g. "Dataset" → every message reads "[Dataset] ..."
        Leave empty for a general-purpose logger with no prefix.
    log_dir : str
        Directory where log files are saved. Created if it does not exist.

    Returns
    -------
    logging.Logger
        Fully configured logger, ready to use.

    Example
    -------
    >>> logger = get_logger("dataset", phase="Dataset", log_dir="logs")
    >>> logger.info("Download started")
    # → 2024-05-10 14:23:01 | INFO  | [Dataset] Download started
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if this logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ---- Formatter -------------------------------------------------------
    # Adds [Phase] prefix if a phase label was given
    prefix = f"[{phase}] " if phase else ""

    class PhaseFormatter(logging.Formatter):
        """Custom formatter that injects the phase prefix into every message."""
        def format(self, record):
            record.msg = f"{prefix}{record.msg}"
            return super().format(record)

    fmt = PhaseFormatter(
        fmt     = "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    # ---- Console handler (INFO and above) --------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # ---- File handler (DEBUG and above) ----------------------------------
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{name.replace(' ', '_')}.log"
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialised → {log_file}")
    return logger