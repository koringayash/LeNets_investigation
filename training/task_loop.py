"""
training/task_loop.py
---------------------
Task-specific forward pass and loss computation functions.

Each function handles ONE batch for ONE task:
  classification_step — forward → CrossEntropy/Focal/LabelSmoothing → loss
  detection_step      — forward → CIoU + Focal + BCE objectness → combined loss
  segmentation_step   — forward → BCE + Dice → combined loss

train.py selects the correct step function ONCE at the start of training
and calls it in the inner loop. No if/else branches inside the hot loop.

Why this file exists
---------------------
The three tasks have fundamentally different batch formats and loss
computations. Putting all the branching logic here — in one dedicated
file — keeps train.py clean and readable. Each step function has a
clear contract: (model, batch, criterion, device) → (loss, metric_value).

Detection anchor matching
--------------------------
Full detection requires matching predicted anchors to ground truth boxes
(IoU-based assignment). We implement a simplified version here:
for each ground truth box, we assign it to the nearest predicted anchor
by centre-point distance. This is sufficient for training and is the
approach used by early YOLO versions.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Classification step
# ---------------------------------------------------------------------------

def classification_step(
    model    : nn.Module,
    batch    : tuple,
    criterion: nn.Module,
    device   : torch.device,
) -> Tuple[torch.Tensor, float]:
    """
    Forward pass + loss for one classification batch.

    Parameters
    ----------
    model     : CNNModel
    batch     : (images, labels) — both Tensors
    criterion : Classification loss (CrossEntropy / Focal / LabelSmoothing)
    device    : torch.device

    Returns
    -------
    (loss, accuracy)
      loss     : Tensor  Scalar. Call .backward() on this.
      accuracy : float   Fraction of correct predictions in this batch.
    """
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)

    logits   = model(images)                        # (B, num_classes)
    loss     = criterion(logits, labels)
    preds    = logits.argmax(dim=1)
    accuracy = (preds == labels).float().mean().item()

    return loss, accuracy


# ---------------------------------------------------------------------------
# Detection step
# ---------------------------------------------------------------------------

def detection_step(
    model    : nn.Module,
    batch    : tuple,
    criterion: nn.Module,
    device   : torch.device,
) -> Tuple[torch.Tensor, float]:
    """
    Forward pass + combined loss for one detection batch.

    Parameters
    ----------
    model     : DetectionModel
    batch     : (images, targets) where targets is a list of dicts:
                [{"boxes": Tensor(N_i,4), "labels": Tensor(N_i,)}, ...]
    criterion : DetectionLoss (CIoU + Focal + BCE objectness)
    device    : torch.device

    Returns
    -------
    (total_loss, obj_accuracy)
      total_loss   : Tensor  Combined weighted loss. Call .backward() on this.
      obj_accuracy : float   Fraction of anchors with correct objectness pred.
    """
    images, targets = batch
    images = images.to(device)

    # Move targets to device
    targets = [
        {k: v.to(device) for k, v in t.items()}
        for t in targets
    ]

    # Forward: returns {"boxes": ..., "scores": ..., "objectness": ...}
    predictions = model(images)

    # Match predictions to targets for loss computation
    matched_preds, matched_targets = _match_anchors_to_targets(
        predictions, targets, device
    )

    if matched_preds["boxes"].shape[0] == 0:
        # No valid matches in this batch — return zero loss
        zero = torch.tensor(0.0, requires_grad=True, device=device)
        return zero, 0.0

    total_loss, components = criterion(matched_preds, matched_targets)

    # Simple objectness accuracy: did we correctly predict object vs background?
    obj_pred    = (matched_preds["objectness"].squeeze(1) > 0.0)
    obj_true    = (matched_targets["objectness"].squeeze(1) > 0.5)
    obj_acc     = (obj_pred == obj_true).float().mean().item()

    return total_loss, obj_acc


def _match_anchors_to_targets(
    predictions: Dict[str, torch.Tensor],
    targets    : list,
    device     : torch.device,
) -> Tuple[Dict, Dict]:
    """
    Simplified anchor-to-target matching for detection loss computation.

    For each ground truth box in each image, we find the predicted anchor
    closest to it by centre-point distance and mark it as a positive.
    All other anchors are marked as negatives (background).

    Parameters
    ----------
    predictions : dict  Model output from DetectionModel.forward()
                        {"boxes": (B,A,4), "scores": (B,A,C), "objectness": (B,A,1)}
                        where A = num_anchors * H * W
    targets     : list  List of B dicts, each with "boxes" (N_i,4) and "labels" (N_i,)
    device      : torch.device

    Returns
    -------
    (matched_preds, matched_targets)
    Both are flat dicts of Tensors with all matched anchor pairs concatenated.
    """
    all_pred_boxes  = []
    all_pred_scores = []
    all_pred_obj    = []
    all_tgt_boxes   = []
    all_tgt_labels  = []
    all_tgt_obj     = []

    B = predictions["boxes"].shape[0]

    for b in range(B):
        pred_boxes  = predictions["boxes"][b]       # (A, 4)
        pred_scores = predictions["scores"][b]      # (A, C)
        pred_obj    = predictions["objectness"][b]  # (A, 1)

        tgt_boxes   = targets[b]["boxes"]           # (N, 4)
        tgt_labels  = targets[b]["labels"]          # (N,)

        if tgt_boxes.shape[0] == 0:
            continue

        A = pred_boxes.shape[0]
        N = tgt_boxes.shape[0]

        # Compute centres of predicted anchors
        pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2   # (A,)
        pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2   # (A,)

        # Compute centres of ground truth boxes
        tgt_cx  = (tgt_boxes[:, 0] + tgt_boxes[:, 2]) / 2    # (N,)
        tgt_cy  = (tgt_boxes[:, 1] + tgt_boxes[:, 3]) / 2    # (N,)

        # Pairwise distances: (A, N)
        dx   = pred_cx.unsqueeze(1) - tgt_cx.unsqueeze(0)
        dy   = pred_cy.unsqueeze(1) - tgt_cy.unsqueeze(0)
        dist = dx**2 + dy**2

        # For each GT box, find the closest predicted anchor
        best_anchor_per_gt = dist.argmin(dim=0)   # (N,)

        # Build objectness target: 1.0 for matched anchors, 0.0 for background
        obj_target = torch.zeros(A, 1, device=device)
        obj_target[best_anchor_per_gt] = 1.0

        # Collect matched pairs
        matched_indices = best_anchor_per_gt

        all_pred_boxes .append(pred_boxes[matched_indices])
        all_pred_scores.append(pred_scores[matched_indices])
        all_pred_obj   .append(pred_obj)
        all_tgt_boxes  .append(tgt_boxes)
        all_tgt_labels .append(tgt_labels)
        all_tgt_obj    .append(obj_target)

    if not all_pred_boxes:
        empty = torch.zeros(0, device=device)
        return (
            {"boxes": empty.view(0,4), "scores": empty.view(0,1), "objectness": empty.view(0,1)},
            {"boxes": empty.view(0,4), "labels": empty.view(0).long(), "objectness": empty.view(0,1)},
        )

    matched_preds = {
        "boxes"      : torch.cat(all_pred_boxes,  dim=0),
        "scores"     : torch.cat(all_pred_scores, dim=0),
        "objectness" : torch.cat(all_pred_obj,    dim=0),
    }
    matched_targets = {
        "boxes"      : torch.cat(all_tgt_boxes,  dim=0),
        "labels"     : torch.cat(all_tgt_labels, dim=0),
        "objectness" : torch.cat(all_tgt_obj,    dim=0),
    }

    return matched_preds, matched_targets


# ---------------------------------------------------------------------------
# Segmentation step
# ---------------------------------------------------------------------------

def segmentation_step(
    model    : nn.Module,
    batch    : tuple,
    criterion: nn.Module,
    device   : torch.device,
) -> Tuple[torch.Tensor, float]:
    """
    Forward pass + combined BCE+Dice loss for one segmentation batch.

    Parameters
    ----------
    model     : EncoderDecoderModel
    batch     : (images, masks) — both Tensors
                images : (B, C, H, W)
                masks  : (B, H, W) long tensor of class indices
    criterion : BCEDiceLoss or DiceLoss or PixelCrossEntropyLoss
    device    : torch.device

    Returns
    -------
    (loss, pixel_accuracy)
      loss           : Tensor  Scalar. Call .backward() on this.
      pixel_accuracy : float   Fraction of pixels correctly classified.
    """
    images, masks = batch
    images = images.to(device)
    masks  = masks.to(device)

    pred_logits = model(images)    # (B, num_classes, H, W)

    # Resize prediction to match mask spatial size if they differ
    if pred_logits.shape[2:] != masks.shape[1:]:
        import torch.nn.functional as F
        pred_logits = F.interpolate(
            pred_logits,
            size         = masks.shape[1:],
            mode         = "bilinear",
            align_corners= False,
        )

    loss = criterion(pred_logits, masks)

    # Pixel accuracy: argmax over class dim
    preds        = pred_logits.argmax(dim=1)      # (B, H, W)
    pixel_acc    = (preds == masks).float().mean().item()

    return loss, pixel_acc


# ---------------------------------------------------------------------------
# Task selector — called ONCE at start of training
# ---------------------------------------------------------------------------

def get_step_function(task: str):
    """
    Return the step function for the given task.

    Called once at the start of training. The returned function is then
    called every batch in the inner loop with no branching overhead.

    Parameters
    ----------
    task : str  "classification" | "detection" | "segmentation"

    Returns
    -------
    Callable  (model, batch, criterion, device) → (loss, metric)

    Example
    -------
    >>> step_fn = get_step_function("classification")
    >>> loss, acc = step_fn(model, batch, criterion, device)
    """
    mapping = {
        "classification": classification_step,
        "detection"     : detection_step,
        "segmentation"  : segmentation_step,
    }
    if task not in mapping:
        raise ValueError(
            f"Unknown task '{task}'. "
            f"Choose from: {list(mapping.keys())}"
        )
    return mapping[task]