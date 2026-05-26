"""
evaluation/evaluate.py (v2)
----------------------------
Task-aware evaluation: routes to the correct metrics module and
saves results to JSON for later use by results.py.
"""

import json
import logging
from pathlib import Path
from typing import Dict

from config import EVAL, EXPERIMENT, DATASET, LOG_DIR
from evaluation.metrics import compute_metrics, print_metrics


def run_evaluation(
    all_preds : list,
    all_labels: list,
    logger    : logging.Logger = None,
) -> Dict:
    """
    Compute and log metrics for the current task.

    Parameters
    ----------
    all_preds  : list  Task-specific predictions from Predictor.predict_batch()
    all_labels : list  Task-specific ground truth from Predictor.predict_batch()
    logger     : logging.Logger, optional

    Returns
    -------
    dict  Computed metrics.
    """
    log        = logger.info if logger else print
    task       = EXPERIMENT["task"].lower()
    num_classes= DATASET["num_classes"]
    task_cfg   = EVAL.get(task, {})

    metrics = compute_metrics(
        task        = task,
        preds       = all_preds,
        labels      = all_labels,
        num_classes = num_classes,
        config      = task_cfg,
    )

    print_metrics(metrics, task=task, logger=logger)
    return metrics


def save_eval_metrics(metrics: Dict, logger: logging.Logger = None) -> None:
    """
    Save evaluation metrics to logs/eval_metrics.json.

    Parameters
    ----------
    metrics : dict  Returned by run_evaluation().
    logger  : logging.Logger, optional
    """
    log       = logger.info if logger else print
    out_path  = LOG_DIR / "eval_metrics.json"

    # Convert numpy arrays to lists for JSON serialisation
    def _make_serialisable(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    payload = {
        "experiment": EXPERIMENT["name"],
        "task"      : EXPERIMENT["task"],
        **_make_serialisable(metrics),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    log(f"Eval metrics saved → {out_path}")