"""
losses/segmentation_loss.py
----------------------------
Loss functions for semantic segmentation tasks.

Supported losses
----------------
  BCEDiceLoss       — BCE + Dice combined (default, safest choice)
  DiceLoss          — pure Dice, good for foreground/background imbalance
  PixelCrossEntropy — standard pixel-wise cross entropy

Why BCE + Dice?
---------------
BCE (Binary Cross Entropy) gives stable, well-behaved gradients everywhere.
Dice Loss directly optimises the overlap metric (Dice coefficient), which
handles class imbalance well since it focuses on the ratio of intersection
to total predicted+target pixels rather than counting each pixel equally.
Combining them gets the best of both: stability from BCE, balance from Dice.

Usage
-----
>>> from losses.segmentation_loss import get_segmentation_loss
>>> criterion = get_segmentation_loss(config=LOSS["segmentation"])
>>> loss = criterion(pred_masks, target_masks)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


def get_segmentation_loss(config: Dict[str, Any]) -> nn.Module:
    """
    Return the segmentation loss function specified in config.

    Parameters
    ----------
    config : dict  LOSS["segmentation"] section from config.py.

    Returns
    -------
    nn.Module  Loss function.

    Raises
    ------
    ValueError  If loss name is not recognised.
    """
    name = config["name"].lower()

    if name == "bce_dice":
        return BCEDiceLoss(
            dice_weight = config.get("dice_weight", 0.5),
            bce_weight  = config.get("bce_weight",  0.5),
        )
    elif name == "dice":
        return DiceLoss()
    elif name == "cross_entropy":
        return PixelCrossEntropyLoss()
    else:
        raise ValueError(
            f"Unknown segmentation loss '{name}'. "
            f"Choose from: 'bce_dice', 'dice', 'cross_entropy'."
        )


# ---------------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation (Milletari et al., 2016 — V-Net paper).

    Dice coefficient measures mask overlap:
        Dice = 2 * |A ∩ B| / (|A| + |B|)

    Dice Loss = 1 - Dice coefficient

    Works well when foreground pixels are rare (e.g. tumour in MRI),
    because it directly optimises overlap rather than counting each pixel.

    Parameters
    ----------
    smooth : float  Small constant added to numerator and denominator to
                    prevent division by zero. Default 1.0.

    Expects
    -------
    pred   : Tensor  Shape (B, C, H, W) — raw logits (before sigmoid/softmax).
    target : Tensor  Shape (B, H, W)    — integer class labels per pixel.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        pred  : torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        num_classes = pred.shape[1]

        # Convert logits to probabilities
        pred_prob = torch.softmax(pred, dim=1)   # (B, C, H, W)

        # One-hot encode target: (B, H, W) → (B, C, H, W)
        target_one_hot = F.one_hot(target, num_classes)          # (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Dice per class
        intersection = (pred_prob * target_one_hot).sum(dim=(2, 3))
        union        = pred_prob.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice_per_cls = (2 * intersection + self.smooth) / (union + self.smooth)

        # Average over classes and batch
        return 1.0 - dice_per_cls.mean()


# ---------------------------------------------------------------------------
# BCE + Dice Combined Loss
# ---------------------------------------------------------------------------

class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy + Dice Loss.

    total_loss = bce_weight * BCE + dice_weight * Dice

    This is the most practical default for segmentation:
    - BCE: provides stable gradients everywhere, fast convergence
    - Dice: handles class imbalance, directly optimises overlap quality

    Parameters
    ----------
    dice_weight : float  Weight for Dice component. Default 0.5.
    bce_weight  : float  Weight for BCE component.  Default 0.5.

    Expects
    -------
    pred   : Tensor  Shape (B, C, H, W) — raw logits.
    target : Tensor  Shape (B, H, W)    — integer class labels.
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight
        self.dice_loss   = DiceLoss()

    def forward(
        self,
        pred  : torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # Pixel-wise cross entropy (works natively with (B,C,H,W) and (B,H,W))
        bce  = F.cross_entropy(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice


# ---------------------------------------------------------------------------
# Pixel-wise Cross Entropy (wrapper for clarity)
# ---------------------------------------------------------------------------

class PixelCrossEntropyLoss(nn.Module):
    """
    Standard pixel-wise cross entropy loss for segmentation.

    Equivalent to F.cross_entropy applied to (B, C, H, W) predictions
    and (B, H, W) integer targets. Each pixel is treated as an independent
    classification problem.

    Simplest choice. Works well when class distribution is balanced.
    Use DiceLoss or BCEDiceLoss when classes are imbalanced.

    Expects
    -------
    pred   : Tensor  Shape (B, C, H, W) — raw logits.
    target : Tensor  Shape (B, H, W)    — integer class labels.
    """

    def forward(
        self,
        pred  : torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return F.cross_entropy(pred, target)