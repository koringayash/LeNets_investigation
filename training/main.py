"""
training/main.py
----------------
Orchestrates the training phase. Thin wrapper that sets up the
phase logger and delegates to training/train.py.

Usage (called from main.py — not run directly)
------
>>> from training.main import run_training_phase
>>> run_training_phase(state, logger=root_logger, resume=False)
"""

import logging
from pipeline_state  import PipelineState
from training.train  import run_training
from utils           import get_logger, Timer
from config          import LOG_DIR


def run_training_phase(
    state      : PipelineState,
    logger     : logging.Logger,
    resume     : bool = False,
) -> None:
    """
    Entry point for the training phase called by main.py.

    Sets up a dedicated [Training] phase logger, then calls run_training().

    Parameters
    ----------
    state  : PipelineState  Shared pipeline state manager.
    logger : logging.Logger  Root logger (used for phase boundary messages).
    resume : bool            Whether to resume from last checkpoint.
    """
    phase_logger = get_logger("training", phase="Training", log_dir=str(LOG_DIR))

    with Timer("Training phase (total wall time)", logger=logger):
        run_training(state=state, logger=phase_logger, resume=resume)