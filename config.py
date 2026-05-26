"""
config.py
---------
Single source of truth for the entire CV Framework v2.

This is the ONLY file a user needs to edit to switch between tasks,
datasets, models, and hyperparameters.

Three task modes
----------------
  "classification" → image classification (single label per image)
  "detection"      → object detection (bounding boxes + labels per image)
  "segmentation"   → semantic segmentation (per-pixel class labels)

How to use
----------
1. Set EXPERIMENT["task"] to your task.
2. Set DATASET["source"] and the matching source fields.
3. Set MODEL["name"] (predefined) or MODEL["layer_configs"] (custom).
4. Adjust TRAIN and the task-specific loss/metric sections.
5. Run: python main.py

Beginners: Think of this as the control panel for the entire project.
Every other file reads from here — nothing is hardcoded anywhere else.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths  (auto-created at import time)
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
# Experiment  ← START HERE
# ---------------------------------------------------------------------------

EXPERIMENT = {
    # -----------------------------------------------------------------------
    # TASK — the most important field in the entire config.
    # Everything else (loss, metrics, dataset format, model type) is driven
    # by this single value. Change it and the whole pipeline adapts.
    #
    # Choices: "classification" | "detection" | "segmentation"
    # -----------------------------------------------------------------------
    "task"  : "classification",

    # Unique name for this run. Used for checkpoint names, log filenames,
    # and plot titles. Change this for every meaningfully different experiment.
    "name"  : "mnist_lenet5_relu",

    # Random seed. Any integer. 42 is the project default.
    "seed"  : 42,

    # Which pipeline phases to run. Remove a name to skip that phase.
    # Order matters: dataset → training → evaluation.
    "stages": ["dataset", "training", "evaluation"],
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DATASET = {
    # Where the data comes from. Choose ONE:
    #   "torchvision" → built-in datasets (MNIST, CIFAR10, etc.)
    #   "local"       → a folder on your machine
    #   "url"         → direct link to a .zip or .tar.gz file
    #   "github"      → GitHub folder URL, release asset, or full repo URL
    "source"         : "torchvision",

    # ---- torchvision source -----------------------------------------------
    # Supported: "MNIST", "CIFAR10", "CIFAR100", "FashionMNIST", "SVHN"
    "name"           : "MNIST",

    # ---- local / url / github source fields --------------------------------
    "local_path"     : None,   # path to dataset root folder
    "url"            : None,   # direct download or GitHub URL

    # ---- Format (ignored for torchvision — format is fixed) ---------------
    # Classification formats : "imagefolder" | "csv" | "torchvision"
    # Detection formats      : "yolo"        | "coco_json" | "pascal_voc"
    # Segmentation formats   : "image_mask"  | "coco_json" | "cityscapes"
    #
    # Set this to match YOUR dataset's on-disk format. The framework
    # converts everything to a unified internal format before training.
    "format"         : "torchvision",

    # ---- For CSV classification format ------------------------------------
    # Path to CSV file with columns: image_path, label
    "csv_path"       : None,

    # ---- For segmentation image+mask format --------------------------------
    # Folder containing mask images (parallel to image folder)
    "mask_dir"       : None,

    # ---- Common fields (all sources + tasks) ------------------------------
    "image_size"     : 32,       # H and W (square). LeNet needs 32, most others 224+
    "num_classes"    : 10,       # number of output classes
    "in_channels"    : 1,        # 1 = grayscale, 3 = RGB

    # Normalisation: MNIST defaults shown.
    # CIFAR-10: mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616)
    "mean"           : (0.1307,),
    "std"            : (0.3081,),

    # Train/val split fractions (applied to official training split).
    # Official test set is always kept separate.
    "train_fraction" : 0.8,
    "val_fraction"   : 0.1,

    # Random horizontal flip + random crop augmentation.
    # Keep False for MNIST. Recommended True for CIFAR/natural images.
    "augment"        : False,
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL = {
    # "predefined" → use a named architecture (see MODEL["name"] below)
    # "custom"     → define your own via MODEL["layer_configs"]
    "type"          : "predefined",

    # ---- Predefined model names -------------------------------------------
    # Classification : "lenet5" | "alexnet" | "vgg11" | "vgg16"
    #                  "resnet18" | "resnet34"
    # Detection      : "yolov1" | "ssd_mobilenet" | "faster_rcnn"
    # Segmentation   : "unet" | "fcn" | "deeplabv3"
    "name"          : "lenet5",

    # ---- LeNet / classification-specific options --------------------------
    "activation"    : "relu",   # relu | sigmoid | tanh | leakyrelu
    "pooling"       : "max",    # max | avg

    # ---- Detection-specific options ---------------------------------------
    # Number of anchors per spatial location (used by detection heads)
    "num_anchors"   : 5,

    # ---- Custom architecture (layer-config list) --------------------------
    # Used when type = "custom". Full layer list as dicts.
    # See architectures/base.py for all supported layer types including:
    #   conv, pool, activation, linear, flatten, batchnorm, dropout,
    #   residual_block, depthwise_conv, dilated_conv, transposed_conv,
    #   bilinear_upsample, aspp_block
    "layer_configs" : None,
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

TRAIN = {
    "epochs"         : 15,
    "batch_size"     : 64,
    "learning_rate"  : 1e-3,

    # Optimiser: "adam" | "sgd"
    "optimizer"      : "adam",
    "momentum"       : 0.9,       # SGD only
    "weight_decay"   : 0.0,

    "num_workers"    : 2,
    "device"         : "auto",    # "auto" | "cuda" | "cpu"
    "save_best_only" : True,
}


# ---------------------------------------------------------------------------
# Loss configuration  (task-specific)
# ---------------------------------------------------------------------------

LOSS = {
    # ---- Classification loss ----------------------------------------------
    # Options: "cross_entropy" | "focal" | "label_smoothing"
    "classification" : {
        "name"            : "cross_entropy",
        "label_smoothing" : 0.1,    # used only when name="label_smoothing"
        "focal_gamma"     : 2.0,    # used only when name="focal"
        "focal_alpha"     : 0.25,   # used only when name="focal"
    },

    # ---- Detection loss ---------------------------------------------------
    # Box regression : "ciou" | "giou" | "smooth_l1"
    # Class loss     : "focal" | "cross_entropy"
    # The three component losses are weighted and summed.
    "detection" : {
        "box_loss"        : "ciou",
        "class_loss"      : "focal",
        "focal_gamma"     : 2.0,
        "focal_alpha"     : 0.25,
        # Weights for combining the three loss components
        "lambda_box"      : 5.0,   # weight on box regression loss
        "lambda_class"    : 1.0,   # weight on classification loss
        "lambda_obj"      : 1.0,   # weight on objectness (BCE) loss
    },

    # ---- Segmentation loss ------------------------------------------------
    # "bce_dice"     → BCE + Dice combined (default, safest)
    # "cross_entropy"→ pixel-wise cross entropy only
    # "dice"         → Dice only (good for imbalanced foreground/background)
    "segmentation" : {
        "name"            : "bce_dice",
        "dice_weight"     : 0.5,   # weight of Dice in the combined loss
        "bce_weight"      : 0.5,   # weight of BCE in the combined loss
    },
}


# ---------------------------------------------------------------------------
# Evaluation configuration  (task-specific)
# ---------------------------------------------------------------------------

EVAL = {
    # ---- Classification metrics ------------------------------------------
    # All macro-averaged across classes.
    "classification" : {
        "metrics": ["accuracy", "precision", "recall", "f1", "confusion_matrix"],
    },

    # ---- Detection metrics -----------------------------------------------
    # mAP computed at multiple IoU thresholds (COCO-style).
    "detection" : {
        # IoU thresholds for mAP computation (0.50 to 0.95, step 0.05)
        "iou_thresholds"  : [0.50, 0.55, 0.60, 0.65, 0.70,
                             0.75, 0.80, 0.85, 0.90, 0.95],
        # Primary reported metrics
        "metrics"         : ["mAP_50", "mAP_50_95", "AR"],
        # Confidence threshold for filtering predictions before metric calc
        "conf_threshold"  : 0.25,
        # IoU threshold for NMS (non-maximum suppression) post-processing
        "nms_iou_threshold": 0.45,
    },

    # ---- Segmentation metrics --------------------------------------------
    "segmentation" : {
        "metrics": ["miou", "dice", "pixel_accuracy"],
    },
}