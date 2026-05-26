"""
architectures/classification_configs.py
----------------------------------------
Layer-config builders for all supported classification architectures.

Supported models
----------------
  lenet5    — LeNet-5 (LeCun 1998)        — lightweight, MNIST/CIFAR
  alexnet   — AlexNet (Krizhevsky 2012)   — image_size >= 64
  vgg11     — VGG-11 (Simonyan 2014)      — image_size >= 32
  vgg16     — VGG-16 (Simonyan 2014)      — image_size >= 32
  resnet18  — ResNet-18 (He 2015)         — image_size >= 32
  resnet34  — ResNet-34 (He 2015)         — image_size >= 32

All return a flat list of layer-config dicts for CNNModel (classifier.py).
"""


# ---------------------------------------------------------------------------
# LeNet-5
# ---------------------------------------------------------------------------

def build_lenet_config(
    activation : str = "relu",
    pooling    : str = "max",
    num_classes: int = 10,
) -> list:
    """
    LeNet-5 variant with configurable activation and pooling.

    Parameters
    ----------
    activation  : str  "relu" | "sigmoid" | "tanh" | "leakyrelu"
    pooling     : str  "max" | "avg"
    num_classes : int

    Returns
    -------
    list of dict  For CNNModel(layer_configs, input_shape=(1,32,32)).
    """
    act  = {"type": "activation", "name": activation}
    pool = {"type": "pool", "name": pooling, "kernel_size": 2, "stride": 2}

    return [
        {"type": "conv", "out_channels": 6,  "kernel_size": 5}, act, pool,
        {"type": "conv", "out_channels": 16, "kernel_size": 5}, act, pool,
        {"type": "flatten"},
        {"type": "linear", "out_features": 120}, act,
        {"type": "linear", "out_features": 84},  act,
        {"type": "linear", "out_features": num_classes},
    ]


# ---------------------------------------------------------------------------
# AlexNet
# ---------------------------------------------------------------------------

def build_alexnet_config(num_classes: int = 10) -> list:
    """
    AlexNet — image_size >= 64 recommended.

    Parameters
    ----------
    num_classes : int

    Returns
    -------
    list of dict  For CNNModel(layer_configs, input_shape=(3,64,64)).
    """
    relu = {"type": "activation", "name": "relu"}
    pool = {"type": "pool", "name": "max", "kernel_size": 3, "stride": 2}

    return [
        {"type": "conv", "out_channels": 96,  "kernel_size": 11, "stride": 4, "padding": 2},
        relu, pool,
        {"type": "conv", "out_channels": 256, "kernel_size": 5,  "padding": 2}, relu, pool,
        {"type": "conv", "out_channels": 384, "kernel_size": 3,  "padding": 1}, relu,
        {"type": "conv", "out_channels": 384, "kernel_size": 3,  "padding": 1}, relu,
        {"type": "conv", "out_channels": 256, "kernel_size": 3,  "padding": 1}, relu, pool,
        {"type": "flatten"},
        {"type": "dropout", "p": 0.5},
        {"type": "linear", "out_features": 4096}, relu,
        {"type": "dropout", "p": 0.5},
        {"type": "linear", "out_features": 4096}, relu,
        {"type": "linear", "out_features": num_classes},
    ]


# ---------------------------------------------------------------------------
# VGG helpers
# ---------------------------------------------------------------------------

def _vgg_conv_block(out_channels: int, num_convs: int) -> list:
    relu = {"type": "activation", "name": "relu"}
    pool = {"type": "pool", "name": "max", "kernel_size": 2, "stride": 2}
    layers = []
    for _ in range(num_convs):
        layers += [
            {"type": "conv", "out_channels": out_channels,
             "kernel_size": 3, "stride": 1, "padding": 1},
            {"type": "batchnorm"},
            relu,
        ]
    layers.append(pool)
    return layers


def build_vgg11_config(num_classes: int = 10) -> list:
    """VGG-11 (configuration A) — 8 conv + 3 FC = 11 learnable layers."""
    relu = {"type": "activation", "name": "relu"}
    return [
        *_vgg_conv_block(64,  1), *_vgg_conv_block(128, 1),
        *_vgg_conv_block(256, 2), *_vgg_conv_block(512, 2),
        *_vgg_conv_block(512, 2),
        {"type": "flatten"},
        {"type": "dropout", "p": 0.5},
        {"type": "linear", "out_features": 4096}, relu,
        {"type": "dropout", "p": 0.5},
        {"type": "linear", "out_features": 4096}, relu,
        {"type": "linear", "out_features": num_classes},
    ]


def build_vgg16_config(num_classes: int = 10) -> list:
    """VGG-16 (configuration D) — 13 conv + 3 FC = 16 learnable layers."""
    relu = {"type": "activation", "name": "relu"}
    return [
        *_vgg_conv_block(64,  2), *_vgg_conv_block(128, 2),
        *_vgg_conv_block(256, 3), *_vgg_conv_block(512, 3),
        *_vgg_conv_block(512, 3),
        {"type": "flatten"},
        {"type": "dropout", "p": 0.5},
        {"type": "linear", "out_features": 4096}, relu,
        {"type": "dropout", "p": 0.5},
        {"type": "linear", "out_features": 4096}, relu,
        {"type": "linear", "out_features": num_classes},
    ]


# ---------------------------------------------------------------------------
# ResNet helpers
# ---------------------------------------------------------------------------

def _resnet_layer(out_channels: int, num_blocks: int, stride: int = 1) -> list:
    blocks = []
    for i in range(num_blocks):
        blocks.append({
            "type"        : "residual_block",
            "out_channels": out_channels,
            "stride"      : stride if i == 0 else 1,
            "activation"  : "relu",
        })
    return blocks


def build_resnet18_config(num_classes: int = 10, input_size: int = 32) -> list:
    """ResNet-18 — 2 blocks per group, 16 conv layers total."""
    gap = max(1, input_size // 32)
    return [
        {"type": "conv", "out_channels": 64, "kernel_size": 7, "stride": 2, "padding": 3},
        {"type": "batchnorm"}, {"type": "activation", "name": "relu"},
        {"type": "pool", "name": "max", "kernel_size": 3, "stride": 2, "padding": 1},
        *_resnet_layer(64,  2, stride=1),
        *_resnet_layer(128, 2, stride=2),
        *_resnet_layer(256, 2, stride=2),
        *_resnet_layer(512, 2, stride=2),
        {"type": "pool", "name": "avg", "kernel_size": gap, "stride": 1},
        {"type": "flatten"},
        {"type": "linear", "out_features": num_classes},
    ]


def build_resnet34_config(num_classes: int = 10, input_size: int = 32) -> list:
    """ResNet-34 — [3,4,6,3] blocks per group, 32 conv layers total."""
    gap = max(1, input_size // 32)
    return [
        {"type": "conv", "out_channels": 64, "kernel_size": 7, "stride": 2, "padding": 3},
        {"type": "batchnorm"}, {"type": "activation", "name": "relu"},
        {"type": "pool", "name": "max", "kernel_size": 3, "stride": 2, "padding": 1},
        *_resnet_layer(64,  3, stride=1),
        *_resnet_layer(128, 4, stride=2),
        *_resnet_layer(256, 6, stride=2),
        *_resnet_layer(512, 3, stride=2),
        {"type": "pool", "name": "avg", "kernel_size": gap, "stride": 1},
        {"type": "flatten"},
        {"type": "linear", "out_features": num_classes},
    ]