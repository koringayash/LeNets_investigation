"""
evaluation/metrics/__init__.py
--------------------------------
Task-aware metric selector. One call returns the right metrics dict.
"""

from typing import Dict, List
import numpy as np


def compute_metrics(
    task       : str,
    preds      : list,
    labels     : list,
    num_classes: int,
    config     : dict = None,
) -> Dict:
    """
    Compute evaluation metrics for the given task.

    Parameters
    ----------
    task        : str   "classification" | "detection" | "segmentation"
    preds       : list  Task-specific predictions (see per-module docs).
    labels      : list  Task-specific ground truth.
    num_classes : int
    config      : dict  EVAL[task] section from config.py (for IoU thresholds etc.)

    Returns
    -------
    dict  Task-specific metrics dictionary.
    """
    task   = task.lower()
    config = config or {}

    if task == "classification":
        from evaluation.metrics.classification_metrics import compute_classification_metrics
        return compute_classification_metrics(preds, labels, num_classes)

    elif task == "detection":
        from evaluation.metrics.detection_metrics import compute_detection_metrics
        iou_thresholds = config.get("iou_thresholds",
                                    [round(t, 2) for t in np.arange(0.50, 1.00, 0.05)])
        return compute_detection_metrics(preds, labels, num_classes, iou_thresholds)

    elif task == "segmentation":
        from evaluation.metrics.segmentation_metrics import compute_segmentation_metrics
        return compute_segmentation_metrics(preds, labels, num_classes)

    else:
        raise ValueError(
            f"Unknown task '{task}'. "
            f"Choose from: 'classification', 'detection', 'segmentation'."
        )


def print_metrics(metrics: Dict, task: str, logger=None) -> None:
    """Print a clean formatted summary of computed metrics."""
    log = logger.info if logger else print

    lines = ["=" * 60, f"  Evaluation Metrics — {task.upper()}", "=" * 60]

    if task == "classification":
        lines += [
            f"  Accuracy  : {metrics['accuracy']        * 100:.2f}%",
            f"  Precision : {metrics['precision_macro'] * 100:.2f}%",
            f"  Recall    : {metrics['recall_macro']    * 100:.2f}%",
            f"  F1 Score  : {metrics['f1_macro']        * 100:.2f}%",
        ]
    elif task == "detection":
        lines += [
            f"  mAP@0.5      : {metrics['mAP_50']    * 100:.2f}%",
            f"  mAP@0.5:0.95 : {metrics['mAP_50_95'] * 100:.2f}%",
            f"  AR           : {metrics['AR']         * 100:.2f}%",
        ]
    elif task == "segmentation":
        lines += [
            f"  Pixel Accuracy : {metrics['pixel_accuracy'] * 100:.2f}%",
            f"  mIoU           : {metrics['miou']            * 100:.2f}%",
            f"  Mean Dice      : {metrics['mean_dice']       * 100:.2f}%",
        ]

    lines.append("=" * 60)
    log("\n" + "\n".join(lines))


__all__ = ["compute_metrics", "print_metrics"]