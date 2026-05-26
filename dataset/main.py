"""
dataset/main.py (v2)
---------------------
Orchestrates the full dataset preparation pipeline for all three tasks.
Identical structure to v1 — just imports the v2 versions of each module.
"""

import logging
from pipeline_state       import PipelineState
from dataset.download     import download_dataset
from dataset.preprocess   import get_datasets
from dataset.save_dataset import (
    save_processed_datasets, get_dataloaders, processed_files_exist
)
from dataset.info         import print_dataset_summary
from utils                import Timer


def run_dataset_phase(
    state  : PipelineState,
    logger : logging.Logger,
    resume : bool = False,
) -> None:
    """
    Run the full dataset preparation pipeline.

    Steps
    -----
    1. Download   — fetch data from configured source
    2. Preprocess — task-aware format loading + normalisation + split
    3. Save       — write processed data to .pt files
    4. Info       — print dataset summary

    Parameters
    ----------
    state  : PipelineState
    logger : logging.Logger
    resume : bool  If True and stage is "done", skip entirely.
    """
    if resume and state.is_done("dataset"):
        logger.info("Dataset phase already complete — skipping (--resume)")
        return

    logger.info("=" * 60)
    logger.info("  DATASET PHASE")
    logger.info("=" * 60)

    state.mark_started("dataset")

    with Timer("Total dataset phase", logger=logger):

        with Timer("Step 1: Download", logger=logger):
            download_dataset(logger=logger)

        if processed_files_exist():
            logger.info("Processed .pt files found — skipping preprocess & save.")
        else:
            with Timer("Step 2: Preprocess + split", logger=logger):
                train_ds, val_ds, test_ds = get_datasets(logger=logger)

            with Timer("Step 3: Save processed data", logger=logger):
                save_processed_datasets(train_ds, val_ds, test_ds, logger=logger)

        with Timer("Step 4: Summary", logger=logger):
            train_loader, val_loader, test_loader = get_dataloaders(logger=logger)
            print_dataset_summary(train_loader, val_loader, test_loader, logger=logger)

    state.mark_done("dataset")
    logger.info("Dataset phase complete ✓")