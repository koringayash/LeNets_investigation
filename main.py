"""
main.py (v2)
------------
Master entry point for CV Framework v2.

Reads config.EXPERIMENT["task"] at startup and prints it in the banner.
Everything else is handled by the task selectors in lower layers.

CLI usage
---------
  python main.py                               # full pipeline, all stages
  python main.py --stage dataset               # single stage, fresh
  python main.py --stage training --resume     # resume training only
  python main.py --resume                      # resume all incomplete stages
  python main.py --epochs 5                    # override epoch count
"""

import argparse
import sys

from config         import EXPERIMENT, TRAIN, LOG_DIR, STATE_FILE
from pipeline_state import PipelineState
from utils          import set_seed, get_logger, SystemInfo, Timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description     = "CV Framework v2 — Classification / Detection / Segmentation",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage", type=str, default=None,
        choices=["dataset", "training", "evaluation"],
        help="Run a single pipeline stage only.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume from last completed point. Skips stages marked 'done'. "
            "Resumes training from last completed epoch. "
            "Without this flag, every stage runs fresh."
        ),
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override training epoch count from config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.epochs is not None:
        TRAIN["epochs"] = args.epochs

    logger = get_logger("main", phase="Main", log_dir=str(LOG_DIR))

    task = EXPERIMENT["task"].upper()
    banner = [
        "",
        "╔══════════════════════════════════════════════════════╗",
        f"║  CV Framework v2  |  Task: {task:<27}  ║",
        f"║  Experiment: {EXPERIMENT['name'][:41]:<41}  ║",
        "╚══════════════════════════════════════════════════════╝",
        "",
    ]
    logger.info("\n".join(banner))

    SystemInfo.print(logger=logger)
    set_seed(EXPERIMENT["seed"])
    logger.info(f"Random seed : {EXPERIMENT['seed']}")
    logger.info(f"Task        : {EXPERIMENT['task']}")

    state = PipelineState(
        state_file = str(STATE_FILE),
        experiment = EXPERIMENT["name"],
        logger     = logger,
    )

    if args.resume:
        logger.info("--resume flag detected")
        state.print_status()
    else:
        state.reset()
        logger.info("Fresh run — pipeline state reset")

    stages_to_run = [args.stage] if args.stage else EXPERIMENT["stages"]
    logger.info(f"Stages to run : {stages_to_run}")

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

    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Logs        → {LOG_DIR}/")
    logger.info(f"  Checkpoints → {STATE_FILE.parent / 'Checkpoint/'}")
    logger.info(f"  Plots       → {STATE_FILE.parent / 'plots/'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()