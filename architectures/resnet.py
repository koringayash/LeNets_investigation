"""
architectures/resnet.py
-----------------------
Layer-config builders for ResNet-18 and ResNet-34 (He et al., 2015).

ResNets introduced skip (residual) connections that let gradients flow
directly through the network, enabling training of very deep models
without vanishing gradients.

Each "residual_block" dict is handled by the ResidualBlock class in
architectures/base.py, which internally implements:
  main path  : Conv → BN → ReLU → Conv → BN
  skip path  : Identity  OR  Conv(1×1) → BN  (when dimensions change)
  output     : ReLU(main + skip)

ResNet layer groups
-------------------
  Stem     : Conv(64, 7×7, s=2) → BN → ReLU → MaxPool(3×3, s=2)
  Layer 1  : 2 (ResNet-18) or 3 (ResNet-34) residual blocks, 64 channels
  Layer 2  : blocks, 128 channels, first block has stride=2 (downsampling)
  Layer 3  : blocks, 256 channels, first block has stride=2
  Layer 4  : blocks, 512 channels, first block has stride=2
  Head     : GlobalAvgPool → Flatten → FC(num_classes)

GlobalAvgPool is implemented as AvgPool with kernel_size matching the
spatial size after all blocks. For 32×32 input (CIFAR) this is 1×1;
for 224×224 input (ImageNet) this is 7×7.

Usage
-----
>>> from architectures.resnet import build_resnet18_config
>>> cfg   = build_resnet18_config(num_classes=10, input_size=32)
>>> model = CNNModel(cfg, input_shape=(3, 32, 32))
"""


def _resnet_layer(out_channels: int, num_blocks: int, stride: int = 1) -> list:
    """
    Build one ResNet layer group (a sequence of residual blocks).

    The first block in the group uses the given stride (often 2 for
    spatial downsampling). All subsequent blocks use stride=1.

    Parameters
    ----------
    out_channels : int  Output channels for every block in this group.
    num_blocks   : int  Number of residual blocks in this group.
    stride       : int  Stride for the FIRST block only.

    Returns
    -------
    list of dict
    """
    blocks = []
    for i in range(num_blocks):
        blocks.append({
            "type"        : "residual_block",
            "out_channels": out_channels,
            "stride"      : stride if i == 0 else 1,
            "activation"  : "relu",
        })
    return blocks


def build_resnet18_config(
    num_classes: int = 10,
    input_size : int = 32,
) -> list:
    """
    Return the layer-config list for ResNet-18.

    ResNet-18 uses 2 residual blocks per layer group (4 groups = 8 blocks
    total = 16 conv layers + 1 stem conv + 1 FC = 18 learnable layers).

    Parameters
    ----------
    num_classes : int  Number of output classes.
    input_size  : int  Spatial size of input images (H=W assumed square).
                       32 for CIFAR-style, 224 for ImageNet-style.

    Returns
    -------
    list of dict  Ready to pass to CNNModel(layer_configs, input_shape).
    """
    # After stem (7×7 conv, s=2) + MaxPool(3×3, s=2): size ≈ input_size // 4
    # After 4 layer groups (each halves): size ≈ input_size // 32
    # For input_size=32: 32//32 = 1 → GlobalAvgPool kernel = 1
    # For input_size=224: 224//32 = 7 → GlobalAvgPool kernel = 7
    gap_kernel = max(1, input_size // 32)

    return [
        # ---- Stem ----------------------------------------------------------
        {"type": "conv",      "out_channels": 64, "kernel_size": 7,
         "stride": 2, "padding": 3},
        {"type": "batchnorm"},
        {"type": "activation", "name": "relu"},
        {"type": "pool",       "name": "max", "kernel_size": 3, "stride": 2, "padding": 1},

        # ---- Layer groups --------------------------------------------------
        *_resnet_layer(64,  num_blocks=2, stride=1),   # layer1 — no downsampling
        *_resnet_layer(128, num_blocks=2, stride=2),   # layer2
        *_resnet_layer(256, num_blocks=2, stride=2),   # layer3
        *_resnet_layer(512, num_blocks=2, stride=2),   # layer4

        # ---- Global Average Pool + Classifier ------------------------------
        {"type": "pool",    "name": "avg", "kernel_size": gap_kernel, "stride": 1},
        {"type": "flatten"},
        {"type": "linear",  "out_features": num_classes},
    ]


def build_resnet34_config(
    num_classes: int = 10,
    input_size : int = 32,
) -> list:
    """
    Return the layer-config list for ResNet-34.

    ResNet-34 uses [3, 4, 6, 3] residual blocks per group
    = 16 blocks total = 32 conv + 1 stem + 1 FC = 34 learnable layers.

    Parameters
    ----------
    num_classes : int  Number of output classes.
    input_size  : int  Spatial size of input images.

    Returns
    -------
    list of dict  Ready to pass to CNNModel(layer_configs, input_shape).
    """
    gap_kernel = max(1, input_size // 32)

    return [
        # ---- Stem ----------------------------------------------------------
        {"type": "conv",      "out_channels": 64, "kernel_size": 7,
         "stride": 2, "padding": 3},
        {"type": "batchnorm"},
        {"type": "activation", "name": "relu"},
        {"type": "pool",       "name": "max", "kernel_size": 3, "stride": 2, "padding": 1},

        # ---- Layer groups --------------------------------------------------
        *_resnet_layer(64,  num_blocks=3, stride=1),   # layer1
        *_resnet_layer(128, num_blocks=4, stride=2),   # layer2
        *_resnet_layer(256, num_blocks=6, stride=2),   # layer3
        *_resnet_layer(512, num_blocks=3, stride=2),   # layer4

        # ---- Global Average Pool + Classifier ------------------------------
        {"type": "pool",    "name": "avg", "kernel_size": gap_kernel, "stride": 1},
        {"type": "flatten"},
        {"type": "linear",  "out_features": num_classes},
    ]