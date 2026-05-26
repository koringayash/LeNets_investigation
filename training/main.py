"""training/main.py (v2) — identical structure to v1."""
import logging
from pipeline_state import PipelineState
from training.train import run_training
from utils          import get_logger, Timer
from config         import LOG_DIR


def run_training_phase(
    state  : PipelineState,
    logger : logging.Logger,
    resume : bool = False,
) -> None:
    phase_logger = get_logger("training", phase="Training", log_dir=str(LOG_DIR))
    with Timer("Training phase (total wall time)", logger=logger):
        run_training(state=state, logger=phase_logger, resume=resume)