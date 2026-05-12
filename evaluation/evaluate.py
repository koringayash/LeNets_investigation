"""
evaluation/evaluate.py
----------------------
Computes evaluation metrics on the test set using predictions from
the Predictor class in inference.py.

Metrics computed (all macro-averaged across classes)
-----------------------------------------------------
  - Accuracy         : fraction of correct predictions
  - Precision (macro): mean precision across all classes
  - Recall (macro)   : mean recall across all classes
  - F1 score (macro) : mean F1 across all classes
  - Confusion matrix : N×N matrix of true vs predicted labels

All metrics are computed from scratch using only NumPy — no sklearn
required — so the dependency list stays minimal.

Usage
-----
>>> from evaluation.evaluate import compute_metrics, print_metrics
>>> metrics = compute_metrics(all_preds, all_labels, num_classes=10)
>>> print_metrics(metrics, logger=logger)
"""

import logging
from typing import Dict, List

import numpy as np


def compute_metrics(
    preds      : List[int],
    labels     : List[int],
    num_classes: int,
) -> Dict:
    """
    Compute all evaluation metrics from prediction and ground-truth lists.

    Parameters
    ----------
    preds       : list of int  Predicted class indices (from Predictor).
    labels      : list of int  True class indices (from DataLoader).
    num_classes : int          Total number of classes.

    Returns
    -------
    dict with keys:
        "accuracy"         : float
        "precision_macro"  : float
        "recall_macro"     : float
        "f1_macro"         : float
        "confusion_matrix" : np.ndarray  shape (num_classes, num_classes)

    Example
    -------
    >>> metrics = compute_metrics(preds, labels, num_classes=10)
    >>> print(metrics["accuracy"])
    0.9876
    """
    preds_arr  = np.array(preds)
    labels_arr = np.array(labels)
    n          = len(labels_arr)

    # ---- Accuracy ---------------------------------------------------------
    accuracy = float((preds_arr == labels_arr).sum() / n)

    # ---- Confusion matrix -------------------------------------------------
    # cm[true_class][pred_class] = count
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels_arr, preds_arr):
        cm[t][p] += 1

    # ---- Per-class precision, recall, F1 ----------------------------------
    # True positives  for class c: cm[c, c]
    # False positives for class c: cm[:, c].sum() - cm[c, c]
    # False negatives for class c: cm[c, :].sum() - cm[c, c]

    precisions, recalls, f1s = [], [], []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy"        : accuracy,
        "precision_macro" : float(np.mean(precisions)),
        "recall_macro"    : float(np.mean(recalls)),
        "f1_macro"        : float(np.mean(f1s)),
        "confusion_matrix": cm,
    }


def print_metrics(
    metrics : Dict,
    logger  : logging.Logger = None,
) -> None:
    """
    Print a clean summary of all computed metrics.

    Parameters
    ----------
    metrics : dict  Returned by compute_metrics().
    logger  : logging.Logger, optional

    Example output
    --------------
    ============================================================
      Evaluation Metrics (macro-averaged)
    ============================================================
      Accuracy  : 98.76%
      Precision : 98.74%
      Recall    : 98.77%
      F1 Score  : 98.75%
    ============================================================
    """
    log = logger.info if logger else print

    lines = [
        "=" * 60,
        "  Evaluation Metrics (macro-averaged)",
        "=" * 60,
        f"  Accuracy  : {metrics['accuracy']        * 100:.2f}%",
        f"  Precision : {metrics['precision_macro'] * 100:.2f}%",
        f"  Recall    : {metrics['recall_macro']    * 100:.2f}%",
        f"  F1 Score  : {metrics['f1_macro']        * 100:.2f}%",
        "=" * 60,
    ]
    log("\n" + "\n".join(lines))


def save_metrics_json(
    metrics    : Dict,
    output_path: str,
    experiment : str,
) -> None:
    """
    Save evaluation metrics to a JSON file for later use by results.py.

    The confusion matrix is converted to a nested list so it can be
    serialised to JSON (numpy arrays are not JSON serialisable by default).

    Parameters
    ----------
    metrics     : dict    Returned by compute_metrics().
    output_path : str     Path to write the JSON file.
    experiment  : str     Experiment name stored in the file for reference.

    Example
    -------
    >>> save_metrics_json(metrics, "logs/eval_metrics.json", "mnist_lenet5")
    """
    import json
    from pathlib import Path

    payload = {
        "experiment"      : experiment,
        "accuracy"        : metrics["accuracy"],
        "precision_macro" : metrics["precision_macro"],
        "recall_macro"    : metrics["recall_macro"],
        "f1_macro"        : metrics["f1_macro"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)