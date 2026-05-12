"""
architectures/alexnet.py
------------------------
Layer-config builder for AlexNet (Krizhevsky et al., 2012).

AlexNet won ImageNet 2012 and popularised deep CNNs. It introduced ReLU,
dropout regularisation, and GPU training at scale.

Original design expects 224×224 RGB input. We support smaller inputs
(minimum 64×64) for experimentation on CIFAR-style datasets.

Architecture
------------
Input → Conv(96,11,s=4) → Pool → Conv(256,5,p=2) → Pool
      → Conv(384,3,p=1) → Conv(384,3,p=1) → Conv(256,3,p=1) → Pool
      → Flatten → FC(4096,drop) → FC(4096,drop) → FC(num_classes)

Usage
-----
>>> from architectures.alexnet import build_alexnet_config
>>> cfg   = build_alexnet_config(num_classes=10)
>>> model = CNNModel(cfg, input_shape=(3, 64, 64))
"""


def build_alexnet_config(num_classes: int = 10) -> list:
    """
    Return the layer-config list for AlexNet.

    Parameters
    ----------
    num_classes : int  Number of output classes.

    Returns
    -------
    list of dict  Ready to pass to CNNModel(layer_configs, input_shape).

    Note
    ----
    Original AlexNet uses LRN (Local Response Normalisation) layers.
    These are omitted here as they are not used in modern practice.
    BatchNorm is NOT added either — this preserves the original design.
    Use image_size >= 64 in config for stable spatial dimensions.
    """
    relu = {"type": "activation", "name": "relu"}
    pool = {"type": "pool", "name": "max", "kernel_size": 3, "stride": 2}

    return [
        # Feature extraction
        {"type": "conv", "out_channels": 96,  "kernel_size": 11, "stride": 4, "padding": 2},
        relu, pool,

        {"type": "conv", "out_channels": 256, "kernel_size": 5,  "stride": 1, "padding": 2},
        relu, pool,

        {"type": "conv", "out_channels": 384, "kernel_size": 3,  "stride": 1, "padding": 1},
        relu,
        {"type": "conv", "out_channels": 384, "kernel_size": 3,  "stride": 1, "padding": 1},
        relu,
        {"type": "conv", "out_channels": 256, "kernel_size": 3,  "stride": 1, "padding": 1},
        relu, pool,

        # Classifier (with dropout for regularisation)
        {"type": "flatten"},
        {"type": "dropout", "p": 0.5},
        {"type": "linear",  "out_features": 4096}, relu,
        {"type": "dropout", "p": 0.5},
        {"type": "linear",  "out_features": 4096}, relu,
        {"type": "linear",  "out_features": num_classes},
    ]