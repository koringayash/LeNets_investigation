"""
losses/detection_loss.py
------------------------
Combined detection loss: CIoU box regression + Focal classification
+ Binary Cross Entropy objectness score.

This is the most complex loss in the framework. Detection models output
three things per anchor at every spatial location:
  1. Box coordinates  (x, y, w, h) — regressed continuously
  2. Class scores     (C values)   — which object class
  3. Objectness score (1 value)    — does this anchor contain any object?

Each requires a different loss. The three are weighted and summed:
  total_loss = lambda_box * L_ciou
             + lambda_class * L_focal
             + lambda_obj * L_bce_obj

CIoU Loss (Complete IoU — Zheng et al., 2020)
----------------------------------------------
Improves over plain IoU loss by penalising three things:
  1. Overlap area      (like IoU)
  2. Centre point distance
  3. Aspect ratio difference
This means it provides gradient even when predicted and target boxes
don't overlap at all, which plain IoU cannot do.

Usage
-----
>>> from losses.detection_loss import get_detection_loss
>>> criterion = get_detection_loss(config=LOSS["detection"])
>>> loss, components = criterion(predictions, targets)
>>> # components = {"box": 0.42, "class": 0.31, "obj": 0.18}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import math


def get_detection_loss(config: Dict[str, Any]) -> "DetectionLoss":
    """
    Return the combined detection loss configured from config.py.

    Parameters
    ----------
    config : dict  LOSS["detection"] section from config.py.

    Returns
    -------
    DetectionLoss  Combined loss module.
    """
    return DetectionLoss(
        box_loss    = config.get("box_loss",     "ciou"),
        class_loss  = config.get("class_loss",   "focal"),
        focal_gamma = config.get("focal_gamma",  2.0),
        focal_alpha = config.get("focal_alpha",  0.25),
        lambda_box  = config.get("lambda_box",   5.0),
        lambda_class= config.get("lambda_class", 1.0),
        lambda_obj  = config.get("lambda_obj",   1.0),
    )


# ---------------------------------------------------------------------------
# CIoU geometry helpers
# ---------------------------------------------------------------------------

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.

    Parameters
    ----------
    boxes1, boxes2 : Tensor  Shape (N, 4) in [x1, y1, x2, y2] format.

    Returns
    -------
    Tensor  Shape (N,) — IoU for each pair.
    """
    # Intersection
    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union
    area1    = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2    = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union    = area1 + area2 - inter_area + 1e-7   # epsilon prevents div-by-zero

    return inter_area / union


def ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute CIoU loss between predicted and target bounding boxes.

    CIoU = IoU - (rho² / c²) - alpha * v
    Where:
      rho = Euclidean distance between box centres
      c   = diagonal of the smallest enclosing box
      v   = aspect ratio consistency term
      alpha = weight balancing IoU and aspect ratio

    Parameters
    ----------
    pred_boxes   : Tensor  Shape (N, 4) predicted boxes [x1, y1, x2, y2].
    target_boxes : Tensor  Shape (N, 4) ground truth boxes [x1, y1, x2, y2].

    Returns
    -------
    Tensor  Shape (N,) — CIoU loss per box pair (1 - CIoU).
    """
    iou = box_iou(pred_boxes, target_boxes)

    # Centre points
    pred_cx   = (pred_boxes[:, 0]   + pred_boxes[:, 2])   / 2
    pred_cy   = (pred_boxes[:, 1]   + pred_boxes[:, 3])   / 2
    target_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2

    # rho² = squared Euclidean distance between centres
    rho2 = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

    # c² = squared diagonal of the smallest enclosing box
    enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-7

    # Aspect ratio consistency term v
    pred_w   = (pred_boxes[:, 2]   - pred_boxes[:, 0]).clamp(min=1e-7)
    pred_h   = (pred_boxes[:, 3]   - pred_boxes[:, 1]).clamp(min=1e-7)
    target_w = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=1e-7)
    target_h = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=1e-7)

    v     = (4 / (math.pi ** 2)) * (
        torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)
    ) ** 2

    # alpha balances IoU and aspect ratio (detached — not differentiated)
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    ciou  = iou - rho2 / c2 - alpha * v
    return 1.0 - ciou   # loss = 1 - CIoU (lower is better)


# ---------------------------------------------------------------------------
# Combined Detection Loss
# ---------------------------------------------------------------------------

class DetectionLoss(nn.Module):
    """
    Combined loss for object detection models.

    Combines three component losses:
      1. Box regression loss   (CIoU or GIoU or Smooth L1)
      2. Classification loss   (Focal or CrossEntropy)
      3. Objectness BCE loss   (binary: object present or not)

    The final loss is a weighted sum:
      L = lambda_box * L_box + lambda_class * L_class + lambda_obj * L_obj

    Parameters
    ----------
    box_loss    : str    Box regression loss. "ciou" | "smooth_l1".
    class_loss  : str    Classification loss. "focal" | "cross_entropy".
    focal_gamma : float  Focal loss gamma parameter.
    focal_alpha : float  Focal loss alpha parameter.
    lambda_box  : float  Weight for box loss. Default 5.0 (YOLO convention).
    lambda_class: float  Weight for class loss. Default 1.0.
    lambda_obj  : float  Weight for objectness loss. Default 1.0.

    Expected input format
    ---------------------
    predictions : dict with keys:
        "boxes"      : Tensor (N, 4)  predicted boxes [x1,y1,x2,y2] (matched)
        "scores"     : Tensor (N, C)  class logits
        "objectness" : Tensor (N, 1)  objectness logits

    targets : dict with keys:
        "boxes"      : Tensor (N, 4)  ground truth boxes [x1,y1,x2,y2]
        "labels"     : Tensor (N,)    ground truth class indices
        "objectness" : Tensor (N, 1)  1.0 where object exists, 0.0 background

    Note: N here means matched anchor-target pairs (after anchor assignment).
    The model_factory and task_loop handle this matching before calling the loss.
    """

    def __init__(
        self,
        box_loss    : str   = "ciou",
        class_loss  : str   = "focal",
        focal_gamma : float = 2.0,
        focal_alpha : float = 0.25,
        lambda_box  : float = 5.0,
        lambda_class: float = 1.0,
        lambda_obj  : float = 1.0,
    ):
        super().__init__()
        self.box_loss     = box_loss.lower()
        self.class_loss   = class_loss.lower()
        self.focal_gamma  = focal_gamma
        self.focal_alpha  = focal_alpha
        self.lambda_box   = lambda_box
        self.lambda_class = lambda_class
        self.lambda_obj   = lambda_obj

        # Objectness loss is always BCE (binary: object or background)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets    : Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the combined detection loss.

        Parameters
        ----------
        predictions : dict  Model outputs (boxes, scores, objectness).
        targets     : dict  Ground truth (boxes, labels, objectness).

        Returns
        -------
        total_loss  : Tensor  Scalar. Backpropagate this.
        components  : dict    {"box": float, "class": float, "obj": float}
                              Individual component values for logging.
        """
        pred_boxes  = predictions["boxes"]        # (N, 4)
        pred_scores = predictions["scores"]       # (N, C)
        pred_obj    = predictions["objectness"]   # (N, 1)

        tgt_boxes   = targets["boxes"]            # (N, 4)
        tgt_labels  = targets["labels"]           # (N,)
        tgt_obj     = targets["objectness"]       # (N, 1)

        # ---- Box regression loss -----------------------------------------
        if self.box_loss == "ciou":
            l_box = ciou_loss(pred_boxes, tgt_boxes).mean()
        else:   # smooth_l1 fallback
            l_box = F.smooth_l1_loss(pred_boxes, tgt_boxes)

        # ---- Classification loss (only on positive anchors w/ objects) ---
        # Mask: only compute class loss where an object actually exists
        obj_mask   = (tgt_obj.squeeze(1) > 0.5)

        if obj_mask.sum() > 0:
            pos_scores = pred_scores[obj_mask]
            pos_labels = tgt_labels[obj_mask]

            if self.class_loss == "focal":
                l_class = self._focal_loss(pos_scores, pos_labels)
            else:
                l_class = F.cross_entropy(pos_scores, pos_labels)
        else:
            l_class = torch.tensor(0.0, device=pred_scores.device)

        # ---- Objectness loss (all anchors — positive and negative) --------
        l_obj = self.bce(pred_obj, tgt_obj)

        # ---- Weighted sum -------------------------------------------------
        total = (
            self.lambda_box   * l_box   +
            self.lambda_class * l_class +
            self.lambda_obj   * l_obj
        )

        components = {
            "box"  : l_box.item(),
            "class": l_class.item(),
            "obj"  : l_obj.item(),
        }

        return total, components

    def _focal_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Internal focal loss for the classification head."""
        ce     = F.cross_entropy(logits, labels, reduction="none")
        p_t    = torch.exp(-ce)
        fl     = self.focal_alpha * (1 - p_t) ** self.focal_gamma * ce
        return fl.mean()