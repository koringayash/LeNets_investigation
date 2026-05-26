"""
evaluation/metrics/detection_metrics.py
----------------------------------------
Full COCO-style detection metrics from scratch using only NumPy.

Metrics computed
----------------
  mAP@0.5      — mean Average Precision at IoU threshold 0.50
  mAP@0.5:0.95 — mean AP averaged over IoU thresholds [0.50, 0.55, …, 0.95]
  AR           — Average Recall at max 100 detections per image

How mAP is computed
--------------------
For each class c and IoU threshold t:
  1. Collect all predicted boxes for class c across ALL images,
     sorted by confidence score (descending).
  2. For each predicted box (in confidence order), check if it matches
     any unmatched ground truth box with IoU >= t.
     If yes → True Positive. If no → False Positive.
  3. Compute cumulative precision and recall at each rank.
  4. AP = area under the Precision-Recall curve (interpolated at 101 points).
mAP = mean of AP across all classes.
mAP@0.5:0.95 = mean of mAP across all IoU thresholds.
"""

import numpy as np
from typing import List, Dict


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

def box_iou_np(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU between one box and an array of boxes (NumPy).

    Parameters
    ----------
    box   : np.ndarray  Shape (4,) — [x1, y1, x2, y2]
    boxes : np.ndarray  Shape (N, 4)

    Returns
    -------
    np.ndarray  Shape (N,) — IoU values
    """
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])

    iw  = np.maximum(ix2 - ix1, 0)
    ih  = np.maximum(iy2 - iy1, 0)
    inter = iw * ih

    box_area   = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union      = box_area + boxes_area - inter + 1e-7

    return inter / union


# ---------------------------------------------------------------------------
# Per-class AP at one IoU threshold
# ---------------------------------------------------------------------------

def compute_ap(
    pred_boxes   : List[np.ndarray],   # per-image list of (N_i, 4) arrays
    pred_scores  : List[np.ndarray],   # per-image list of (N_i,) confidence scores
    gt_boxes     : List[np.ndarray],   # per-image list of (M_i, 4) arrays
    iou_threshold: float,
) -> float:
    """
    Compute Average Precision for one class at one IoU threshold.

    Parameters
    ----------
    pred_boxes    : list of np.ndarray  Predicted boxes per image.
    pred_scores   : list of np.ndarray  Confidence scores per image.
    gt_boxes      : list of np.ndarray  Ground truth boxes per image.
    iou_threshold : float               e.g. 0.50

    Returns
    -------
    float  Average Precision (area under PR curve, 0–1).
    """
    # Flatten all predictions across images, keeping image index
    all_scores, all_pred_boxes, all_img_idx = [], [], []
    for img_idx, (boxes, scores) in enumerate(zip(pred_boxes, pred_scores)):
        for box, score in zip(boxes, scores):
            all_scores.append(score)
            all_pred_boxes.append(box)
            all_img_idx.append(img_idx)

    if len(all_scores) == 0:
        return 0.0

    # Sort by confidence descending
    order        = np.argsort(all_scores)[::-1]
    all_scores   = [all_scores[i]    for i in order]
    all_pred_b   = [all_pred_boxes[i] for i in order]
    all_img_idx  = [all_img_idx[i]   for i in order]

    # Track which GT boxes have been matched
    gt_matched = [np.zeros(len(gt), dtype=bool) for gt in gt_boxes]
    n_gt       = sum(len(gt) for gt in gt_boxes)

    if n_gt == 0:
        return 0.0

    tp_list, fp_list = [], []

    for pred_box, img_idx in zip(all_pred_b, all_img_idx):
        gt = gt_boxes[img_idx]

        if len(gt) == 0:
            fp_list.append(1); tp_list.append(0)
            continue

        ious     = box_iou_np(pred_box, gt)
        best_iou = ious.max()
        best_idx = ious.argmax()

        if best_iou >= iou_threshold and not gt_matched[img_idx][best_idx]:
            tp_list.append(1); fp_list.append(0)
            gt_matched[img_idx][best_idx] = True
        else:
            tp_list.append(0); fp_list.append(1)

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)

    recalls    = tp_cum / (n_gt + 1e-7)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-7)

    # 101-point interpolation (COCO style)
    ap = 0.0
    for thr in np.linspace(0, 1, 101):
        prec_at_thr = precisions[recalls >= thr]
        ap += prec_at_thr.max() if len(prec_at_thr) > 0 else 0.0
    ap /= 101.0

    return float(ap)


# ---------------------------------------------------------------------------
# Average Recall
# ---------------------------------------------------------------------------

def compute_ar(
    pred_boxes   : List[List[np.ndarray]],   # [class][image] → (N,4)
    pred_scores  : List[List[np.ndarray]],   # [class][image] → (N,)
    gt_boxes     : List[List[np.ndarray]],   # [class][image] → (M,4)
    iou_thresholds: List[float],
    max_dets     : int = 100,
) -> float:
    """
    Compute Average Recall across IoU thresholds and classes.

    AR = mean recall at max_dets detections per image,
         averaged over all IoU thresholds and classes.

    Parameters
    ----------
    pred_boxes     : [class][image] → predicted box array
    pred_scores    : [class][image] → score array
    gt_boxes       : [class][image] → GT box array
    iou_thresholds : list of float
    max_dets       : int  Maximum detections per image to consider.

    Returns
    -------
    float  Average Recall (0–1).
    """
    num_classes = len(gt_boxes)
    recalls     = []

    for c in range(num_classes):
        for iou_thr in iou_thresholds:
            n_gt = sum(len(gt) for gt in gt_boxes[c])
            if n_gt == 0:
                continue

            tp_total = 0
            for img_idx in range(len(gt_boxes[c])):
                gt     = gt_boxes[c][img_idx]
                preds  = pred_boxes[c][img_idx]
                scores = pred_scores[c][img_idx]

                if len(preds) == 0 or len(gt) == 0:
                    continue

                # Limit to max_dets highest confidence
                if len(scores) > max_dets:
                    top_idx = np.argsort(scores)[::-1][:max_dets]
                    preds   = preds[top_idx]

                matched = np.zeros(len(gt), dtype=bool)
                for pred_box in preds:
                    ious     = box_iou_np(pred_box, gt)
                    best_idx = ious.argmax()
                    if ious[best_idx] >= iou_thr and not matched[best_idx]:
                        tp_total += 1
                        matched[best_idx] = True

            recalls.append(tp_total / (n_gt + 1e-7))

    return float(np.mean(recalls)) if recalls else 0.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_detection_metrics(
    all_predictions : List[Dict],   # per-image: {"boxes":arr, "scores":arr, "labels":arr}
    all_targets     : List[Dict],   # per-image: {"boxes":arr, "labels":arr}
    num_classes     : int,
    iou_thresholds  : List[float] = None,
) -> Dict:
    """
    Compute full COCO-style detection metrics.

    Parameters
    ----------
    all_predictions : list of dict per image.
    all_targets     : list of dict per image.
    num_classes     : int
    iou_thresholds  : list of float  Default: [0.50, 0.55, …, 0.95]

    Returns
    -------
    dict  mAP_50, mAP_50_95, AR, per_class_ap
    """
    if iou_thresholds is None:
        iou_thresholds = [round(t, 2) for t in np.arange(0.50, 1.00, 0.05)]

    n_img = len(all_predictions)

    # Organise by class: pred_boxes[c][img_idx] = array of boxes
    pred_boxes_cls  = [[np.zeros((0,4)) for _ in range(n_img)] for _ in range(num_classes)]
    pred_scores_cls = [[np.zeros(0)     for _ in range(n_img)] for _ in range(num_classes)]
    gt_boxes_cls    = [[np.zeros((0,4)) for _ in range(n_img)] for _ in range(num_classes)]

    for img_idx, (pred, tgt) in enumerate(zip(all_predictions, all_targets)):
        pred_b  = np.array(pred["boxes"])   if len(pred["boxes"])  > 0 else np.zeros((0,4))
        pred_s  = np.array(pred["scores"])  if len(pred["scores"]) > 0 else np.zeros(0)
        pred_l  = np.array(pred["labels"])  if len(pred["labels"]) > 0 else np.zeros(0, dtype=int)
        gt_b    = np.array(tgt["boxes"])    if len(tgt["boxes"])   > 0 else np.zeros((0,4))
        gt_l    = np.array(tgt["labels"])   if len(tgt["labels"])  > 0 else np.zeros(0, dtype=int)

        for c in range(num_classes):
            p_mask = pred_l == c
            g_mask = gt_l   == c
            if p_mask.any():
                pred_boxes_cls[c][img_idx]  = pred_b[p_mask]
                pred_scores_cls[c][img_idx] = pred_s[p_mask]
            if g_mask.any():
                gt_boxes_cls[c][img_idx]    = gt_b[g_mask]

    # mAP@0.5
    ap_50 = [
        compute_ap(pred_boxes_cls[c], pred_scores_cls[c], gt_boxes_cls[c], 0.50)
        for c in range(num_classes)
    ]
    map_50 = float(np.mean(ap_50))

    # mAP@0.5:0.95
    ap_all_thresholds = []
    for thr in iou_thresholds:
        ap_thr = [
            compute_ap(pred_boxes_cls[c], pred_scores_cls[c], gt_boxes_cls[c], thr)
            for c in range(num_classes)
        ]
        ap_all_thresholds.append(np.mean(ap_thr))
    map_50_95 = float(np.mean(ap_all_thresholds))

    # AR
    ar = compute_ar(pred_boxes_cls, pred_scores_cls, gt_boxes_cls, iou_thresholds)

    return {
        "mAP_50"      : map_50,
        "mAP_50_95"   : map_50_95,
        "AR"          : ar,
        "per_class_ap": {f"class_{c}": float(ap_50[c]) for c in range(num_classes)},
    }