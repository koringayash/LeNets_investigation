"""
config.py
---------
Single source of truth for the entire project.

Every path, hyperparameter, and experiment variant is defined here.
No other file should contain hardcoded values — they all import from this module.

Beginners: If you want to change the number of epochs, batch size, learning
rate, or add a new experiment variant, THIS is the only file you need to edit.
"""

import os
from pathlib import Path
from typing import Dict, Any, List


# ---------------------------------------------------------------------------
# Root paths  (all relative to the project directory)
# ---------------------------------------------------------------------------

ROOT_DIR        = Path(__file__).parent.resolve()
DATA_DIR        = ROOT_DIR / "Data"
CHECKPOINT_DIR  = ROOT_DIR / "Checkpoint"
LOG_DIR         = ROOT_DIR / "logs"
RESULTS_CSV     = LOG_DIR  / "results.csv"

# Create directories now so every module can assume they exist
for _d in [DATA_DIR, CHECKPOINT_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DATASET: Dict[str, Any] = {
    "name"          : "MNIST",
    # MNIST images are 28×28 grayscale; LeNet expects 32×32, so we pad/resize.
    "image_size"    : 32,          # final H and W fed into the network
    "num_classes"   : 10,          # digits 0-9
    "in_channels"   : 1,           # grayscale = 1 channel
    # Normalization constants computed from the MNIST training set
    "mean"          : (0.1307,),
    "std"           : (0.3081,),
    # Split fractions (applied to the official 60 000 training samples)
    # The official 10 000 test samples are kept separate and only used ONCE
    # at the very end to report final accuracy.
    "train_fraction": 0.8,         # 48 000 samples for training
    "val_fraction"  : 0.1,         # 6 000 samples for validation  (tune hyperparams)
    # test_fraction is implicitly 0.1 (6 000 samples) — not used during training
}


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

TRAIN: Dict[str, Any] = {
    "epochs"           : 15,
    "batch_size"       : 64,
    "learning_rate"    : 1e-3,
    "optimizer"        : "adam",   # fixed to Adam per project spec
    "weight_decay"     : 0.0,      # L2 regularisation; 0 = disabled
    "num_workers"      : 2,        # parallel data-loading workers
    "device"           : "auto",   # "auto" → GPU if available, else CPU
    "save_best_only"   : True,     # only keep the checkpoint with best val accuracy
}


# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------
# Each dict defines ONE experiment run.
# train.py iterates over this list and runs them sequentially.
#
# To add a new experiment, just append another dict here.
# get_experiment_name() will auto-generate a unique run name from it.
# ---------------------------------------------------------------------------

EXPERIMENTS: List[Dict[str, Any]] = [
    {"activation": "relu",       "pooling": "max"},
    {"activation": "sigmoid",    "pooling": "max"},
    {"activation": "tanh",       "pooling": "max"},
    {"activation": "leakyrelu",  "pooling": "max"},
    {"activation": "relu",       "pooling": "avg"},
    {"activation": "sigmoid",    "pooling": "avg"},
    {"activation": "tanh",       "pooling": "avg"},
    {"activation": "leakyrelu",  "pooling": "avg"},
]


# ---------------------------------------------------------------------------
# LeNet-5 layer factory
# ---------------------------------------------------------------------------

def build_lenet_config(activation: str, pooling: str) -> List[Dict[str, Any]]:
    """
    Return the layer-config list for a LeNet-5 variant.

    LeNet-5 (LeCun et al., 1998) has the following structure:
        Conv → Pool → Conv → Pool → Flatten → FC → FC → FC(output)

    This function injects the caller's choice of activation function and
    pooling type so that every experiment shares the same topology and only
    the activation / pooling differ.

    Parameters
    ----------
    activation : str
        One of: "relu", "sigmoid", "tanh", "leakyrelu"
    pooling : str
        One of: "max", "avg"

    Returns
    -------
    list of dict
        Ready to pass directly to CNNModel(layer_configs, input_shape).

    Example
    -------
    >>> cfg = build_lenet_config("relu", "max")
    >>> model = CNNModel(cfg, input_shape=(1, 32, 32))
    """
    act = {"type": "activation", "name": activation}
    pool = {
        "type"       : "pool",
        "name"       : pooling,
        "kernel_size": 2,
        "stride"     : 2,
    }

    return [
        # ---- Block 1: 1 → 6 feature maps, 32×32 → 28×28 → 14×14 ----------
        {"type": "conv", "out_channels": 6,  "kernel_size": 5, "stride": 1, "padding": 0},
        act,
        pool,

        # ---- Block 2: 6 → 16 feature maps, 14×14 → 10×10 → 5×5 -----------
        {"type": "conv", "out_channels": 16, "kernel_size": 5, "stride": 1, "padding": 0},
        act,
        pool,

        # ---- Classifier: flatten → 120 → 84 → 10 --------------------------
        {"type": "flatten"},
        {"type": "linear", "out_features": 120},
        act,
        {"type": "linear", "out_features": 84},
        act,
        {"type": "linear", "out_features": DATASET["num_classes"]},
    ]


# ---------------------------------------------------------------------------
# Experiment name helper
# ---------------------------------------------------------------------------

def get_experiment_name(exp: Dict[str, Any]) -> str:
    """
    Build a clean, filesystem-safe string that uniquely identifies one
    experiment configuration.

    Parameters
    ----------
    exp : dict
        One entry from the EXPERIMENTS list,
        e.g. {"activation": "relu", "pooling": "max"}.

    Returns
    -------
    str
        e.g. "lenet_relu_maxpool"

    Example
    -------
    >>> get_experiment_name({"activation": "relu", "pooling": "max"})
    'lenet_relu_maxpool'
    """
    return f"lenet_{exp['activation']}_{exp['pooling']}pool"
