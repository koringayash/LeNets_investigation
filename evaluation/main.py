"""
evaluation/main.py
------------------
Orchestrates the complete evaluation pipeline:

  Step 1 — Inference : run predictions over the test set
  Step 2 — Metrics   : compute accuracy, precision, recall, F1, confusion matrix
  Step 3 — Plots     : generate and save all visualisations

Resume behaviour
----------------
If called with resume=True AND pipeline_state shows evaluation as "done",
the phase is skipped. Otherwise it always runs fresh (evaluation is fast,
so resuming mid-evaluation is not worth the complexity).

Usage (called from main.py — not run directly)
------
>>> from evaluation.main import run_evaluation_phase
>>> run_evaluation_phase(state, logger, resume=False)
"""

import logging

from pipeline_state       import PipelineState
from evaluation.inference import Predictor
from evaluation.evaluate  import compute_metrics, print_metrics, save_metrics_json
from evaluation.results   import generate_all_plots
from dataset.save_dataset import get_dataloaders
from utils                import Timer, get_logger
from config               import LOG_DIR, EXPERIMENT, DATASET


def run_evaluation_phase(
    state  : PipelineState,
    logger : logging.Logger,
    resume : bool = False,
) -> None:
    """
    Run the full evaluation phase.

    Parameters
    ----------
    state  : PipelineState  Shared pipeline state manager.
    logger : logging.Logger  Root logger passed from main.py.
    resume : bool
        If True AND evaluation is already "done", skip the phase.
        Otherwise always run fresh.
    """
    # ---- Resume check ----------------------------------------------------
    if resume and state.is_done("evaluation"):
        logger.info("Evaluation phase already complete — skipping (--resume)")
        return

    phase_logger = get_logger("evaluation", phase="Evaluation", log_dir=str(LOG_DIR))

    phase_logger.info("=" * 60)
    phase_logger.info("  EVALUATION PHASE")
    phase_logger.info("=" * 60)

    state.mark_started("evaluation")

    with Timer("Total evaluation phase", logger=phase_logger):

        # Step 1: Load model + run inference on test set
        with Timer("Step 1: Inference", logger=phase_logger):
            predictor = Predictor(logger=phase_logger)
            _, _, test_loader = get_dataloaders(logger=phase_logger)
            all_preds, all_labels = predictor.predict_batch(test_loader)
            phase_logger.info(
                f"Inference complete: {len(all_preds):,} predictions"
            )

        # Step 2: Compute metrics
        with Timer("Step 2: Computing metrics", logger=phase_logger):
            metrics = compute_metrics(
                preds       = all_preds,
                labels      = all_labels,
                num_classes = DATASET["num_classes"],
            )
            print_metrics(metrics, logger=phase_logger)

            # Save eval metrics to JSON for results.py and future reference
            eval_json_path = str(LOG_DIR / "eval_metrics.json")
            save_metrics_json(
                metrics     = metrics,
                output_path = eval_json_path,
                experiment  = EXPERIMENT["name"],
            )
            phase_logger.info(f"Metrics saved → {eval_json_path}")

        # Step 3: Generate plots
        with Timer("Step 3: Generating plots", logger=phase_logger):
            generate_all_plots(logger=phase_logger)

    state.mark_done("evaluation")
    phase_logger.info("Evaluation phase complete ✓")