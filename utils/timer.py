"""
utils/timer.py
--------------
A context manager that measures elapsed wall-clock time for any block
of code and reports it to a logger or the console.

Usage
-----
>>> from utils.timer import Timer
>>> with Timer("Downloading dataset", logger=my_logger) as t:
...     download_something()
# Output: [Downloading dataset] completed in 3.42s
>>> print(t.elapsed)   # access programmatically after the block
3.42
"""

import time
import logging


class Timer:
    """
    Measures elapsed time for any block of code using a context manager.

    Parameters
    ----------
    label  : str
        Human-readable name for the phase being timed.
        Shown in the completion message, e.g. "Epoch 3 training".
    logger : logging.Logger, optional
        If provided, the completion message is written via logger.info().
        If None, the message is printed to the console with print().

    Attributes
    ----------
    elapsed : float
        Seconds elapsed. Available after the `with` block exits.

    Example
    -------
    >>> with Timer("Preprocessing", logger=logger) as t:
    ...     preprocess_data()
    # logs: [Preprocessing] completed in 12.30s
    >>> t.elapsed
    12.30
    """

    def __init__(self, label: str, logger: logging.Logger = None):
        self.label   = label
        self.logger  = logger
        self.elapsed = 0.0
        self._start  = None

    def __enter__(self):
        """Record start time when entering the `with` block."""
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        """Compute elapsed time and report it when leaving the `with` block."""
        self.elapsed = time.perf_counter() - self._start
        msg = f"[{self.label}] completed in {self.elapsed:.2f}s"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)