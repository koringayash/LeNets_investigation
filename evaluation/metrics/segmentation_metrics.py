"""
evaluation/metrics/segmentation_metrics.py
-------------------------------------------
mIoU, Dice coefficient, and Pixel Accuracy for semantic segmentation.
All computed from scratch with NumPy from a single confusion matrix.
"""

import numpy as np
from typing import Dict, List


def compute_segmentation_metrics(
    preds      : List[np.ndarray],   # per-image (H, W) predicted class maps
    labels     : List[np.ndarray],   # per-image (H, W) ground truth class maps
    num_classes: int,
    ignore_idx : int = 255,          # pixels with this label are ignored (e.g. Cityscapes void)
) -> Dict:
    """
    Compute segmentation metrics from per-image prediction and ground truth masks.

    Parameters
    ----------
    preds       : list of np.ndarray  Predicted class index maps (H, W).
    labels      : list of np.ndarray  Ground truth class index maps (H, W).
    num_classes : int
    ignore_idx  : int  Class index to ignore (masked out before computation).

    Returns
    -------
    dict  pixel_accuracy, miou, mean_dice, per_class_iou, per_class_dice
    """
    # Build global confusion matrix across all images
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for pred, label in zip(preds, labels):
        pred  = pred.flatten()
        label = label.flatten()

        # Mask out ignored pixels
        valid  = label != ignore_idx
        pred   = pred[valid]
        label  = label[valid]

        # Only consider valid class indices
        valid2 = (label >= 0) & (label < num_classes) & \
                 (pred  >= 0) & (pred  < num_classes)
        pred   = pred[valid2]
        label  = label[valid2]

        np.add.at(cm, (label, pred), 1)

    # ---- Pixel accuracy --------------------------------------------------
    pixel_accuracy = float(np.diag(cm).sum() / (cm.sum() + 1e-7))

    # ---- Per-class IoU (Jaccard index) -----------------------------------
    # IoU_c = TP_c / (TP_c + FP_c + FN_c)
    # TP_c = cm[c, c]
    # FP_c = cm[:, c].sum() - cm[c, c]
    # FN_c = cm[c, :].sum() - cm[c, c]
    intersection = np.diag(cm)
    union        = cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)
    iou_per_cls  = intersection / (union + 1e-7)

    # Only average over classes that have at least one GT pixel
    valid_classes = cm.sum(axis=1) > 0
    miou          = float(iou_per_cls[valid_classes].mean()) if valid_classes.any() else 0.0

    # ---- Per-class Dice coefficient --------------------------------------
    # Dice_c = 2*TP_c / (2*TP_c + FP_c + FN_c)
    #        = 2 * intersection / (predicted + actual)
    predicted = cm.sum(axis=0)
    actual    = cm.sum(axis=1)
    dice_per_cls = (2 * intersection) / (predicted + actual + 1e-7)
    mean_dice    = float(dice_per_cls[valid_classes].mean()) if valid_classes.any() else 0.0

    return {
        "pixel_accuracy": pixel_accuracy,
        "miou"          : miou,
        "mean_dice"     : mean_dice,
        "per_class_iou" : {f"class_{c}": float(iou_per_cls[c])
                           for c in range(num_classes) if valid_classes[c]},
        "per_class_dice": {f"class_{c}": float(dice_per_cls[c])
                           for c in range(num_classes) if valid_classes[c]},
    }