"""
dataset/main.py
---------------
Orchestrates the complete dataset preparation pipeline:

  Step 1 — Download   : fetch data from torchvision / local / url / github
  Step 2 — Preprocess : resize, normalise, split into train/val/test
  Step 3 — Save       : write processed tensors to Data/processed/*.pt
  Step 4 — Info       : print dataset summary

Resume behaviour
----------------
If called with resume=True AND the pipeline state shows dataset as "done",
the entire phase is skipped. Otherwise it runs fresh from Step 1.

If resume=True but processed .pt files already exist, Step 3 is skipped
(no re-processing), but Steps 1 and 4 still run.

Usage (called from main.py — not run directly)
------
>>> from dataset.main import run_dataset_phase
>>> run_dataset_phase(state, logger, resume=False)
"""

import logging

from pipeline_state import PipelineState
from dataset.download    import download_dataset
from dataset.preprocess  import get_datasets
from dataset.save_dataset import save_processed_datasets, get_dataloaders, processed_files_exist
from dataset.info        import print_dataset_summary
from utils               import Timer


def run_dataset_phase(
    state  : PipelineState,
    logger : logging.Logger,
    resume : bool = False,
) -> None:
    """
    Run the full dataset preparation pipeline.

    Parameters
    ----------
    state  : PipelineState  Shared pipeline state manager (for resume tracking).
    logger : logging.Logger  Phase logger — messages prefixed with [Dataset].
    resume : bool
        If True AND dataset stage is already "done" in pipeline_state.json,
        skip this entire phase.
        If True but not done, run the phase from scratch (safe restart).
        If False, always run the phase fresh.
    """
    # ---- Resume check -----------------------------------------------------
    if resume and state.is_done("dataset"):
        logger.info("Dataset phase already complete — skipping (--resume)")
        return

    logger.info("=" * 60)
    logger.info("  DATASET PHASE")
    logger.info("=" * 60)

    state.mark_started("dataset")

    with Timer("Total dataset phase", logger=logger):

        # Step 1: Download
        with Timer("Step 1: Download", logger=logger):
            download_dataset(logger=logger)

        # Step 2 & 3: Preprocess + Save (skip if .pt files already exist)
        if processed_files_exist():
            logger.info("Processed .pt files found — skipping preprocess & save steps.")
        else:
            # Step 2: Preprocess
            with Timer("Step 2: Preprocess + split", logger=logger):
                train_ds, val_ds, test_ds = get_datasets(logger=logger)

            # Step 3: Save to .pt files
            with Timer("Step 3: Save processed tensors", logger=logger):
                save_processed_datasets(train_ds, val_ds, test_ds, logger=logger)

        # Step 4: Info summary (always runs — loads from .pt files)
        with Timer("Step 4: Dataset summary", logger=logger):
            train_loader, val_loader, test_loader = get_dataloaders(logger=logger)
            print_dataset_summary(train_loader, val_loader, test_loader, logger=logger)

    state.mark_done("dataset")
    logger.info("Dataset phase complete ✓")