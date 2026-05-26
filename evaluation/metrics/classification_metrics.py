"""
evaluation/metrics/classification_metrics.py
---------------------------------------------
Macro-averaged classification metrics: accuracy, precision, recall,
F1, and confusion matrix. Computed from scratch with NumPy.
(Moved from v1's evaluate.py into this sub-package.)
"""

import numpy as np
from typing import Dict, List


def compute_classification_metrics(
    preds      : List[int],
    labels     : List[int],
    num_classes: int,
) -> Dict:
    """
    Compute classification metrics from prediction and label lists.

    Parameters
    ----------
    preds       : list of int  Predicted class indices.
    labels      : list of int  True class indices.
    num_classes : int

    Returns
    -------
    dict  accuracy, precision_macro, recall_macro, f1_macro, confusion_matrix
    """
    preds_arr  = np.array(preds)
    labels_arr = np.array(labels)

    accuracy = float((preds_arr == labels_arr).sum() / len(labels_arr))

    # Confusion matrix: cm[true][pred]
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels_arr, preds_arr):
        cm[t][p] += 1

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
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