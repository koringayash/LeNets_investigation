"""
config.py
---------
Single source of truth for the entire CV framework.

This is the ONLY file a user needs to edit to run a completely
different experiment — different dataset, different model, different
hyperparameters, different evaluation metrics.

How to use this file
--------------------
1. Set EXPERIMENT["name"] to something descriptive.
2. Set DATASET["source"] and fill in the matching fields.
3. Set MODEL["type"] and either MODEL["name"] (predefined) or
   MODEL["layer_configs"] (custom).
4. Adjust TRAIN hyperparameters as needed.
5. Run: python main.py

Beginners: Think of this file as the control panel for the entire
project. Every other file reads from here — nothing is hardcoded
anywhere else.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

ROOT_DIR       = Path(__file__).parent.resolve()
DATA_DIR       = ROOT_DIR / "Data"
CHECKPOINT_DIR = ROOT_DIR / "Checkpoint"
LOG_DIR        = ROOT_DIR / "logs"
PLOTS_DIR      = ROOT_DIR / "plots"
STATE_FILE     = ROOT_DIR / "pipeline_state.json"

for _d in [DATA_DIR, CHECKPOINT_DIR, LOG_DIR, PLOTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

EXPERIMENT = {
    # A unique name for this run. Used to name checkpoints, logs, and plots.
    # Change this every time you start a meaningfully different experiment
    # so results don't overwrite each other.
    "name"  : "mnist_lenet_relu_maxpool",

    # Random seed for full reproducibility across runs.
    # Any integer works — 42 is the project default.
    "seed"  : 42,

    # Which phases to run. Remove a phase name to skip it entirely.
    # Order matters: dataset must come before training, training before evaluation.
    "stages": ["dataset", "training", "evaluation"],
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DATASET = {
    # Where the data comes from. Choose ONE of:
    #   "torchvision" → built-in datasets (MNIST, CIFAR10, etc.)
    #   "local"       → a folder on your machine (ImageFolder format)
    #   "url"         → direct link to a .zip or .tar.gz file
    #   "github"      → a GitHub folder URL, release URL, or full repo URL
    "source"         : "torchvision",

    # ---- torchvision source fields ----------------------------------------
    # Name of the torchvision dataset. Case-sensitive.
    # Supported: "MNIST", "CIFAR10", "CIFAR100", "FashionMNIST", "SVHN"
    "name"           : "MNIST",

    # ---- local source fields ----------------------------------------------
    # Absolute or relative path to your dataset folder.
    # Expected structure: dataset_root/class_name/image.jpg
    "local_path"     : None,

    # ---- url / github source fields ---------------------------------------
    # Direct URL to a zip/tar.gz file  OR  a GitHub folder/repo URL.
    "url"            : None,

    # ---- Common fields (all sources) -------------------------------------
    # Final image size fed into the network (H and W, square images only).
    "image_size"     : 32,

    # Number of output classes.
    "num_classes"    : 10,

    # Number of colour channels: 1 = grayscale, 3 = RGB.
    "in_channels"    : 1,

    # Normalisation constants (mean and std per channel).
    # MNIST defaults shown. For CIFAR-10 RGB use:
    #   mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
    "mean"           : (0.1307,),
    "std"            : (0.3081,),

    # Fraction of the training split used for training (remainder → validation).
    # The official test set is always kept separate.
    "train_fraction" : 0.8,
    "val_fraction"   : 0.1,
    # Implicit: remaining 10% is a held-out portion (not used in training loop)

    # Set True to apply random horizontal flip and random crop augmentation.
    # Useful for CIFAR-10 and natural image datasets. Keep False for MNIST.
    "augment"        : False,
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL = {
    # "predefined" → use a standard architecture by name (see MODEL["name"])
    # "custom"     → define your own architecture via MODEL["layer_configs"]
    "type"         : "predefined",

    # ---- Predefined model -------------------------------------------------
    # Used when type = "predefined". Supported names:
    #   "lenet5"    → LeNet-5  (LeCun 1998)
    #   "alexnet"   → AlexNet  (Krizhevsky 2012) — expects image_size >= 64
    #   "vgg11"     → VGG-11   (Simonyan 2014)   — expects image_size >= 32
    #   "vgg16"     → VGG-16   (Simonyan 2014)   — expects image_size >= 32
    #   "resnet18"  → ResNet-18 (He 2015)         — expects image_size >= 32
    #   "resnet34"  → ResNet-34 (He 2015)         — expects image_size >= 32
    "name"         : "lenet5",

    # ---- Custom model (layer-config list) ---------------------------------
    # Used when type = "custom".
    # Each dict describes one layer. Supported types:
    #
    #   {"type": "conv",            "out_channels": 32, "kernel_size": 3,
    #    "stride": 1, "padding": 1}
    #
    #   {"type": "pool",            "name": "max",  "kernel_size": 2, "stride": 2}
    #   {"type": "pool",            "name": "avg",  "kernel_size": 2, "stride": 2}
    #
    #   {"type": "activation",      "name": "relu"}
    #   {"type": "activation",      "name": "leakyrelu", "negative_slope": 0.01}
    #   {"type": "activation",      "name": "tanh"}
    #   {"type": "activation",      "name": "sigmoid"}
    #
    #   {"type": "batchnorm"}       # auto-selects BatchNorm2d or BatchNorm1d
    #   {"type": "dropout",         "p": 0.5}
    #   {"type": "flatten"}
    #   {"type": "linear",          "out_features": 128}
    #
    #   {"type": "residual_block",  "out_channels": 64, "stride": 1,
    #    "activation": "relu"}      # stride=2 for downsampling blocks
    #
    "layer_configs": None,
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

TRAIN = {
    "epochs"        : 15,
    "batch_size"    : 64,
    "learning_rate" : 1e-3,

    # Optimiser. Currently supported: "adam", "sgd"
    "optimizer"     : "adam",

    # SGD-specific: momentum and weight decay (L2 regularisation)
    "momentum"      : 0.9,
    "weight_decay"  : 0.0,

    # Number of parallel workers for data loading.
    # Set to 0 on Windows if you hit multiprocessing errors.
    "num_workers"   : 2,

    # "auto" → GPU if available, else CPU.
    # Override with "cuda" or "cpu" to force a specific device.
    "device"        : "auto",

    # Only keep the checkpoint with the highest validation accuracy.
    # If False, every epoch's checkpoint is saved (uses more disk space).
    "save_best_only": True,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

EVAL = {
    # Which metrics to compute on the test set.
    # All are macro-averaged across classes.
    "metrics": ["accuracy", "precision", "recall", "f1", "confusion_matrix"],
}