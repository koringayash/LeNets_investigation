"""
losses/__init__.py
------------------
Task-aware loss selector. One call returns the right loss for the task.

Usage
-----
>>> from losses import get_loss
>>> criterion = get_loss(task="classification", config=LOSS)
>>> loss = criterion(predictions, targets)
"""

import torch.nn as nn
from typing import Dict, Any

from losses.classification_loss import get_classification_loss
from losses.detection_loss      import get_detection_loss
from losses.segmentation_loss   import get_segmentation_loss


def get_loss(task: str, config: Dict[str, Any]) -> nn.Module:
    """
    Return the correct loss module for the given task.

    Parameters
    ----------
    task   : str   "classification" | "detection" | "segmentation"
    config : dict  Full LOSS dict from config.py.

    Returns
    -------
    nn.Module  Ready to call as criterion(predictions, targets).

    Raises
    ------
    ValueError  If task is not recognised.

    Example
    -------
    >>> from losses import get_loss
    >>> from config import LOSS, EXPERIMENT
    >>> criterion = get_loss(task=EXPERIMENT["task"], config=LOSS)
    """
    task = task.lower()

    if task == "classification":
        return get_classification_loss(config["classification"])
    elif task == "detection":
        return get_detection_loss(config["detection"])
    elif task == "segmentation":
        return get_segmentation_loss(config["segmentation"])
    else:
        raise ValueError(
            f"Unknown task '{task}'. "
            f"Choose from: 'classification', 'detection', 'segmentation'."
        )


__all__ = ["get_loss"]