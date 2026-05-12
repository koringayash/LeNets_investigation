"""
utils/seed.py
-------------
Ensures every run is fully reproducible by seeding all random number
generators that PyTorch and Python use internally.

Why do we need this?
--------------------
Deep learning involves randomness at many levels:
  - Weight initialisation    (torch)
  - Data shuffling           (torch DataLoader)
  - Dropout masks            (torch)
  - NumPy operations         (numpy)
  - Python's own RNG         (random module)
  - GPU operations           (torch.cuda)

If ANY of these are not seeded, two runs with identical configs will
produce different results. set_seed() fixes all of them in one call.

Usage
-----
>>> from utils.seed import set_seed
>>> set_seed(42)   # call this before ANYTHING else in main.py
"""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Seed every random number generator used by the framework.

    Call this once at the very start of main.py, before dataset loading,
    model creation, or any other operation.

    Parameters
    ----------
    seed : int
        The seed value. Any integer works. The project default is 42
        (set in config.py under EXPERIMENT["seed"]).

    What gets seeded
    ----------------
    - Python built-in  : random.seed()
    - NumPy            : numpy.random.seed()
    - PyTorch CPU      : torch.manual_seed()
    - PyTorch GPU      : torch.cuda.manual_seed_all()
    - cuDNN backend    : deterministic mode + disabled benchmarking

    Note on cuDNN
    -------------
    Setting deterministic=True can slightly slow down training because
    cuDNN picks the fastest algorithm at runtime by default. The trade-off
    is exact reproducibility, which is worth it for research/experiments.

    Example
    -------
    >>> set_seed(42)
    >>> # Now every run with seed=42 produces identical results
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Force cuDNN to use deterministic algorithms only
    torch.backends.cudnn.deterministic = True
    # Disable cuDNN's auto-tuner (it picks different algorithms each run)
    torch.backends.cudnn.benchmark = False