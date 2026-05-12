"""
main.py
-------
Master entry point for the CV Framework.

This file ties together every phase of the pipeline. It:
  1. Parses CLI arguments.
  2. Seeds all random number generators for reproducibility.
  3. Prints system information.
  4. Initialises (or loads) the pipeline state file.
  5. Runs the requested phases in order: dataset → training → evaluation.

CLI usage
---------
  # Run the full pipeline from scratch (all 3 phases)
  python main.py

  # Run a single phase from scratch
  python main.py --stage dataset
  python main.py --stage training
  python main.py --stage evaluation

  # Resume a specific phase from where it stopped
  python main.py --stage training --resume
  python main.py --resume             # resumes all incomplete phases

  # Override the number of epochs for this run only
  python main.py --epochs 5

How --resume works
------------------
  --resume tells each phase to check pipeline_state.json before running.
  - "done"        → phase is skipped entirely
  - "in_progress" → dataset/eval restarts; training resumes from last epoch
  - "pending"     → phase runs fresh

  Without --resume every phase always runs from scratch, even if
  pipeline_state.json shows it as "done".

Beginners: This file is like the manager of a factory. It reads your
config (the order form), checks what has already been done (state file),
and tells each department (dataset, training, evaluation) what to do.
"""

import argparse
import sys

from config         import EXPERIMENT, TRAIN, LOG_DIR, STATE_FILE
from pipeline_state import PipelineState
from utils          import set_seed, get_logger, SystemInfo, Timer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace  Object with attributes: stage, resume, epochs.
    """
    parser = argparse.ArgumentParser(
        description     = "CV Framework — modular deep learning pipeline.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--stage",
        type    = str,
        default = None,
        choices = ["dataset", "training", "evaluation"],
        help    = (
            "Run a single pipeline stage only. "
            "If omitted, all stages listed in config.EXPERIMENT['stages'] are run."
        ),
    )

    parser.add_argument(
        "--resume",
        action  = "store_true",
        help    = (
            "Resume from the last completed point. "
            "Skips stages marked 'done' in pipeline_state.json. "
            "Resumes training from the last completed epoch. "
            "Without this flag, every stage always runs fresh."
        ),
    )

    parser.add_argument(
        "--epochs",
        type    = int,
        default = None,
        help    = (
            "Override the number of training epochs from config.py. "
            "Useful for quick tests without editing the config."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point: parse args → setup → run phases.
    """
    args = parse_args()

    # ---- Override epochs if provided via CLI ------------------------------
    if args.epochs is not None:
        TRAIN["epochs"] = args.epochs

    # ---- Root logger (shared startup messages) ----------------------------
    logger = get_logger("main", phase="Main", log_dir=str(LOG_DIR))

    # ---- Banner -----------------------------------------------------------
    banner = [
        "",
        "╔══════════════════════════════════════════════════════╗",
        f"║  CV Framework  |  {EXPERIMENT['name'][:36]:<36}  ║",
        "╚══════════════════════════════════════════════════════╝",
        "",
    ]
    logger.info("\n".join(banner))

    # ---- System info ------------------------------------------------------
    SystemInfo.print(logger=logger)

    # ---- Seed all RNGs ----------------------------------------------------
    set_seed(EXPERIMENT["seed"])
    logger.info(f"Random seed set: {EXPERIMENT['seed']}")

    # ---- Pipeline state ---------------------------------------------------
    state = PipelineState(
        state_file = str(STATE_FILE),
        experiment = EXPERIMENT["name"],
        logger     = logger,
    )

    if args.resume:
        logger.info("--resume flag detected")
        state.print_status()
    else:
        # Fresh run: reset state so completed stages don't get skipped
        state.reset()
        logger.info("Fresh run — pipeline state reset")

    # ---- Determine which stages to run ------------------------------------
    if args.stage:
        # Single stage requested via CLI
        stages_to_run = [args.stage]
        logger.info(f"Running single stage: {args.stage}")
    else:
        # All stages listed in config
        stages_to_run = EXPERIMENT["stages"]
        logger.info(f"Running all stages: {stages_to_run}")

    # ---- Run phases in order ----------------------------------------------
    with Timer("Total pipeline", logger=logger):

        if "dataset" in stages_to_run:
            from dataset.main import run_dataset_phase
            logger.info("\n" + "─" * 60)
            logger.info("  PHASE 1 / DATASET")
            logger.info("─" * 60)
            run_dataset_phase(state=state, logger=logger, resume=args.resume)

        if "training" in stages_to_run:
            from training.main import run_training_phase
            logger.info("\n" + "─" * 60)
            logger.info("  PHASE 2 / TRAINING")
            logger.info("─" * 60)
            run_training_phase(state=state, logger=logger, resume=args.resume)

        if "evaluation" in stages_to_run:
            from evaluation.main import run_evaluation_phase
            logger.info("\n" + "─" * 60)
            logger.info("  PHASE 3 / EVALUATION")
            logger.info("─" * 60)
            run_evaluation_phase(state=state, logger=logger, resume=args.resume)

    # ---- Final summary ----------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Logs        → {LOG_DIR}/")
    logger.info(f"  Checkpoints → {STATE_FILE.parent / 'Checkpoint/'}")
    logger.info(f"  Plots       → {STATE_FILE.parent / 'plots/'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()