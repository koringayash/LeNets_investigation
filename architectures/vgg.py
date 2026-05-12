"""
architectures/vgg.py
--------------------
Layer-config builders for VGG-11 and VGG-16 (Simonyan & Zisserman, 2014).

VGG showed that network depth using small (3×3) convolutions is the key
factor for achieving strong performance. The design is fully sequential —
no skip connections — making it a natural fit for our dict-based builder.

Both variants use:
  - 3×3 convolutions with padding=1 (preserves spatial size)
  - 2×2 MaxPool with stride=2 (halves spatial size)
  - Three FC layers at the end
  - BatchNorm after every Conv (modern convention; original VGG didn't have it)

Minimum recommended image size: 32×32 (gives 1×1 feature maps before FC).
Original VGG used 224×224.

Usage
-----
>>> from architectures.vgg import build_vgg11_config, build_vgg16_config
>>> cfg   = build_vgg11_config(num_classes=10)
>>> model = CNNModel(cfg, input_shape=(3, 32, 32))
"""


def _conv_block(out_channels: int, num_convs: int) -> list:
    """
    Build one VGG convolutional block: num_convs × (Conv → BN → ReLU) → MaxPool.

    Parameters
    ----------
    out_channels : int  Number of output channels for every conv in this block.
    num_convs    : int  How many conv layers in this block (1, 2, or 3).

    Returns
    -------
    list of dict
    """
    relu = {"type": "activation", "name": "relu"}
    pool = {"type": "pool", "name": "max", "kernel_size": 2, "stride": 2}

    layers = []
    for _ in range(num_convs):
        layers += [
            {"type": "conv",      "out_channels": out_channels,
             "kernel_size": 3, "stride": 1, "padding": 1},
            {"type": "batchnorm"},
            relu,
        ]
    layers.append(pool)
    return layers


def build_vgg11_config(num_classes: int = 10) -> list:
    """
    Return the layer-config list for VGG-11 (configuration A in the paper).

    VGG-11 has 8 conv layers + 3 FC layers = 11 learnable layers.

    Parameters
    ----------
    num_classes : int  Number of output classes.

    Returns
    -------
    list of dict  Ready to pass to CNNModel(layer_configs, input_shape).
    """
    relu = {"type": "activation", "name": "relu"}

    return [
        *_conv_block(64,  num_convs=1),   # block 1
        *_conv_block(128, num_convs=1),   # block 2
        *_conv_block(256, num_convs=2),   # block 3
        *_conv_block(512, num_convs=2),   # block 4
        *_conv_block(512, num_convs=2),   # block 5

        {"type": "flatten"},
        {"type": "dropout", "p": 0.5},
        {"type": "linear",  "out_features": 4096}, relu,
        {"type": "dropout", "p": 0.5},
        {"type": "linear",  "out_features": 4096}, relu,
        {"type": "linear",  "out_features": num_classes},
    ]


def build_vgg16_config(num_classes: int = 10) -> list:
    """
    Return the layer-config list for VGG-16 (configuration D in the paper).

    VGG-16 has 13 conv layers + 3 FC layers = 16 learnable layers.

    Parameters
    ----------
    num_classes : int  Number of output classes.

    Returns
    -------
    list of dict  Ready to pass to CNNModel(layer_configs, input_shape).
    """
    relu = {"type": "activation", "name": "relu"}

    return [
        *_conv_block(64,  num_convs=2),   # block 1
        *_conv_block(128, num_convs=2),   # block 2
        *_conv_block(256, num_convs=3),   # block 3
        *_conv_block(512, num_convs=3),   # block 4
        *_conv_block(512, num_convs=3),   # block 5

        {"type": "flatten"},
        {"type": "dropout", "p": 0.5},
        {"type": "linear",  "out_features": 4096}, relu,
        {"type": "dropout", "p": 0.5},
        {"type": "linear",  "out_features": 4096}, relu,
        {"type": "linear",  "out_features": num_classes},
    ]