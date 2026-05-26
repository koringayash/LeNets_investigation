"""
losses/classification_loss.py
------------------------------
Loss functions for image classification tasks.

Supported losses
----------------
  CrossEntropyLoss   — standard, works for balanced datasets
  FocalLoss          — down-weights easy examples, good for class imbalance
  LabelSmoothingLoss — prevents overconfidence, improves generalisation

Usage
-----
>>> from losses.classification_loss import get_classification_loss
>>> criterion = get_classification_loss(config=LOSS["classification"])
>>> loss = criterion(logits, labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


def get_classification_loss(config: Dict[str, Any]) -> nn.Module:
    """
    Return the classification loss function specified in config.

    Parameters
    ----------
    config : dict  LOSS["classification"] section from config.py.

    Returns
    -------
    nn.Module  Loss function ready to call as criterion(logits, labels).

    Raises
    ------
    ValueError  If the loss name is not recognised.
    """
    name = config["name"].lower()

    if name == "cross_entropy":
        return nn.CrossEntropyLoss()

    elif name == "focal":
        return FocalLoss(
            gamma=config.get("focal_gamma", 2.0),
            alpha=config.get("focal_alpha", 0.25),
        )

    elif name == "label_smoothing":
        return LabelSmoothingLoss(
            smoothing   = config.get("label_smoothing", 0.1),
            num_classes = None,   # inferred from logits at call time
        )

    else:
        raise ValueError(
            f"Unknown classification loss '{name}'. "
            f"Choose from: 'cross_entropy', 'focal', 'label_smoothing'."
        )


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017 — RetinaNet paper).

    Addresses class imbalance by down-weighting the loss contribution from
    easy, well-classified examples and focusing on hard, misclassified ones.

    Loss formula
    ------------
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where p_t is the model's estimated probability for the correct class.
    - gamma > 0 reduces the relative loss for well-classified examples.
    - alpha balances the importance of positive/negative examples.

    Parameters
    ----------
    gamma : float  Focusing parameter. 0 = standard cross entropy. Default 2.
    alpha : float  Weighting factor for the rare class. Default 0.25.

    Example
    -------
    >>> criterion = FocalLoss(gamma=2.0, alpha=0.25)
    >>> loss = criterion(logits, labels)   # logits: (B, C), labels: (B,)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Parameters
        ----------
        logits : Tensor  Shape (batch, num_classes). Raw model outputs.
        labels : Tensor  Shape (batch,). Ground-truth class indices.

        Returns
        -------
        Tensor  Scalar loss value.
        """
        # Standard cross entropy gives us -log(p_t)
        ce_loss = F.cross_entropy(logits, labels, reduction="none")

        # p_t = probability assigned to the correct class
        p_t = torch.exp(-ce_loss)

        # Focal weight: (1 - p_t)^gamma — small when p_t is high (easy example)
        focal_weight = (1.0 - p_t) ** self.gamma

        # Alpha weighting
        focal_loss = self.alpha * focal_weight * ce_loss

        return focal_loss.mean()


# ---------------------------------------------------------------------------
# Label Smoothing Loss
# ---------------------------------------------------------------------------

class LabelSmoothingLoss(nn.Module):
    """
    Cross Entropy with label smoothing (Szegedy et al., 2016).

    Instead of hard one-hot labels (1.0 for correct, 0.0 for others),
    this distributes a small amount of probability mass to incorrect
    classes. This prevents the model from becoming overconfident and
    generally improves generalisation.

    Smooth label formula
    --------------------
    y_smooth = (1 - smoothing) * y_hard + smoothing / num_classes

    Parameters
    ----------
    smoothing   : float  Amount of probability mass to redistribute. 
                         0 = standard cross entropy. Typical: 0.1.
    num_classes : int or None
                  Number of classes. If None, inferred from logits shape.

    Example
    -------
    >>> criterion = LabelSmoothingLoss(smoothing=0.1)
    >>> loss = criterion(logits, labels)
    """

    def __init__(self, smoothing: float = 0.1, num_classes: int = None):
        super().__init__()
        self.smoothing   = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : Tensor  Shape (batch, num_classes).
        labels : Tensor  Shape (batch,). Ground-truth class indices.

        Returns
        -------
        Tensor  Scalar loss value.
        """
        n_classes = self.num_classes or logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)

        # Build smoothed target distribution
        # Start with all-zeros, then fill in the smoothing floor
        smooth_labels = torch.full_like(log_probs, self.smoothing / n_classes)

        # Add (1 - smoothing) weight to the correct class
        smooth_labels.scatter_(
            dim   = 1,
            index = labels.unsqueeze(1),
            value = 1.0 - self.smoothing + self.smoothing / n_classes,
        )

        # KL-divergence style loss: -sum(y_smooth * log_probs)
        loss = -(smooth_labels * log_probs).sum(dim=1)
        return loss.mean()