"""
architectures/lenet.py
----------------------
Layer-config builder for LeNet-5 (LeCun et al., 1998).

LeNet-5 was the first successful deep CNN, designed for handwritten digit
recognition on 28×28 (padded to 32×32) grayscale images.

Architecture
------------
Input → Conv(6, 5×5) → Pool → Conv(16, 5×5) → Pool
      → Flatten → FC(120) → FC(84) → FC(num_classes)

Usage
-----
>>> from architectures.lenet import build_lenet_config
>>> cfg   = build_lenet_config("relu", "max", num_classes=10)
>>> model = CNNModel(cfg, input_shape=(1, 32, 32))
"""


def build_lenet_config(
    activation : str = "relu",
    pooling    : str = "max",
    num_classes: int = 10,
) -> list:
    """
    Return the layer-config list for a LeNet-5 variant.

    Parameters
    ----------
    activation  : str  Activation function for all non-output layers.
                       One of: "relu", "sigmoid", "tanh", "leakyrelu"
    pooling     : str  Pooling strategy. One of: "max", "avg"
    num_classes : int  Number of output classes (final Linear size).

    Returns
    -------
    list of dict  Ready to pass to CNNModel(layer_configs, input_shape).
    """
    act  = {"type": "activation", "name": activation}
    pool = {"type": "pool", "name": pooling, "kernel_size": 2, "stride": 2}

    return [
        # Block 1: 1→6 feature maps, spatial 32→28→14
        {"type": "conv", "out_channels": 6,  "kernel_size": 5, "stride": 1, "padding": 0},
        act, pool,

        # Block 2: 6→16 feature maps, spatial 14→10→5
        {"type": "conv", "out_channels": 16, "kernel_size": 5, "stride": 1, "padding": 0},
        act, pool,

        # Classifier
        {"type": "flatten"},
        {"type": "linear", "out_features": 120}, act,
        {"type": "linear", "out_features": 84},  act,
        {"type": "linear", "out_features": num_classes},
    ]