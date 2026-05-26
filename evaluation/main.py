"""evaluation/main.py (v2) — task-aware evaluation phase orchestrator."""
import logging
from pipeline_state       import PipelineState
from evaluation.inference import Predictor
from evaluation.evaluate  import run_evaluation, save_eval_metrics
from evaluation.results   import generate_all_plots
from dataset.save_dataset import get_dataloaders
from utils                import Timer, get_logger
from config               import LOG_DIR, EXPERIMENT


def run_evaluation_phase(
    state  : PipelineState,
    logger : logging.Logger,
    resume : bool = False,
) -> None:
    if resume and state.is_done("evaluation"):
        logger.info("Evaluation phase already complete — skipping (--resume)")
        return

    phase_logger = get_logger("evaluation", phase="Evaluation", log_dir=str(LOG_DIR))
    phase_logger.info("=" * 60)
    phase_logger.info(f"  EVALUATION PHASE  |  task={EXPERIMENT['task'].upper()}")
    phase_logger.info("=" * 60)

    state.mark_started("evaluation")

    with Timer("Total evaluation phase", logger=phase_logger):
        with Timer("Step 1: Inference", logger=phase_logger):
            predictor = Predictor(logger=phase_logger)
            _, _, test_loader = get_dataloaders(logger=phase_logger)
            all_preds, all_labels = predictor.predict_batch(test_loader)
            phase_logger.info(f"Inference complete: {len(all_preds):,} predictions")

        with Timer("Step 2: Computing metrics", logger=phase_logger):
            metrics = run_evaluation(all_preds, all_labels, logger=phase_logger)
            save_eval_metrics(metrics, logger=phase_logger)

        with Timer("Step 3: Generating plots", logger=phase_logger):
            generate_all_plots(logger=phase_logger)

    state.mark_done("evaluation")
    phase_logger.info("Evaluation phase complete ✓")