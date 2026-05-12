"""
architectures/base.py
---------------------
The core CNNModel class — a configuration-driven sequential CNN builder
extended from v1 with two major additions:

  1. ResidualBlock support — a new "residual_block" layer type that
     implements skip connections for ResNet-style architectures.
  2. Full (C, H, W) shape tracking through _build_all, so the first
     Linear layer after Flatten always gets the correct in_features.

All architecture helpers (lenet.py, alexnet.py, etc.) return a list of
layer-config dicts that get passed directly to CNNModel.

Beginners: This is the LEGO factory. You describe the bricks (layer dicts)
and CNNModel assembles them into a working PyTorch model, validates the
config, and gives you handy inspection tools like summary().
"""

import math
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple


# ---------------------------------------------------------------------------
# Supported options (used for validation)
# ---------------------------------------------------------------------------

_SUPPORTED_ACTIVATIONS = {"relu", "leakyrelu", "sigmoid", "tanh", "softmax"}
_SUPPORTED_POOLS       = {"max", "avg"}
_REQUIRED_KEYS: Dict[str, List[str]] = {
    "conv"          : ["out_channels", "kernel_size"],
    "pool"          : ["name", "kernel_size"],
    "activation"    : ["name"],
    "dropout"       : ["p"],
    "flatten"       : [],
    "linear"        : ["out_features"],
    "batchnorm"     : [],
    "residual_block": ["out_channels"],
}


# ---------------------------------------------------------------------------
# ResidualBlock — a self-contained nn.Module for skip connections
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    A single residual block as used in ResNet architectures (He et al. 2015).

    Architecture
    ------------
    Main path  : Conv(3×3, stride) → BN → Act → Conv(3×3, stride=1) → BN
    Skip path  : Identity           (if in_channels == out_channels and stride == 1)
                 Conv(1×1, stride) → BN  (if channels or spatial size changes)
    Output     : Act(main_path + skip_path)

    The 1×1 convolution on the skip path is called a "projection shortcut"
    — it matches the dimensions so the addition is valid.

    Parameters
    ----------
    in_channels  : int   Number of input feature maps.
    out_channels : int   Number of output feature maps.
    stride       : int   Stride for the first conv. Use 2 to halve H and W.
    activation   : str   Activation function name ("relu", "leakyrelu", etc.)
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : int = 1,
        activation  : str = "relu",
    ):
        super().__init__()

        # ---- Main path ----------------------------------------------------
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.act1  = _make_activation(activation)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # ---- Skip (shortcut) path -----------------------------------------
        # Needed when dimensions change (different channels or spatial stride)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()  # no parameters, just passes input through

        self.act_out = _make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute main path + skip path, then activate.

        Parameters
        ----------
        x : torch.Tensor  Shape (batch, in_channels, H, W)

        Returns
        -------
        torch.Tensor  Shape (batch, out_channels, H', W')
                      H' = H // stride,  W' = W // stride
        """
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.act_out(out + identity)   # the skip connection addition
        return out


# ---------------------------------------------------------------------------
# Activation factory (used by ResidualBlock and _build_layer)
# ---------------------------------------------------------------------------

def _make_activation(name: str, cfg: dict = None) -> nn.Module:
    """
    Return the nn.Module for the requested activation function.

    Parameters
    ----------
    name : str   Activation name (lowercase).
    cfg  : dict  Optional full layer config dict (for extra params like
                 negative_slope for LeakyReLU).

    Returns
    -------
    nn.Module  The activation layer.
    """
    cfg = cfg or {}
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=cfg.get("negative_slope", 0.01), inplace=True)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "softmax":
        return nn.Softmax(dim=cfg.get("dim", 1))
    else:
        raise ValueError(f"Unknown activation '{name}'. Choose from {_SUPPORTED_ACTIVATIONS}.")


# ---------------------------------------------------------------------------
# Single-layer builder
# ---------------------------------------------------------------------------

def _build_layer(cfg: Dict[str, Any], in_channels: int) -> nn.Module:
    """
    Translate one layer-config dict into an nn.Module.

    Parameters
    ----------
    cfg         : dict  One entry from the layer_configs list.
    in_channels : int   Channels / features coming into this layer.

    Returns
    -------
    nn.Module

    Raises
    ------
    ValueError  Unknown layer type or missing required keys.
    """
    ltype = cfg["type"].lower()

    if ltype == "conv":
        return nn.Conv2d(
            in_channels  = in_channels,
            out_channels = cfg["out_channels"],
            kernel_size  = cfg["kernel_size"],
            stride       = cfg.get("stride",  1),
            padding      = cfg.get("padding", 0),
        )

    elif ltype == "pool":
        name        = cfg["name"].lower()
        kernel_size = cfg["kernel_size"]
        stride      = cfg.get("stride", kernel_size)
        padding     = cfg.get("padding", 0)
        if name == "max":
            return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        elif name == "avg":
            return nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            raise ValueError(f"Unknown pool type '{name}'.")

    elif ltype == "activation":
        return _make_activation(cfg["name"], cfg)

    elif ltype == "dropout":
        p       = cfg.get("p", 0.5)
        spatial = cfg.get("spatial", False)
        return nn.Dropout2d(p=p) if spatial else nn.Dropout(p=p)

    elif ltype == "flatten":
        return nn.Flatten(start_dim=1)

    elif ltype == "linear":
        return nn.Linear(in_features=in_channels, out_features=cfg["out_features"])

    elif ltype == "batchnorm":
        is_flat = cfg.get("_is_flat", False)
        return nn.BatchNorm1d(in_channels) if is_flat else nn.BatchNorm2d(in_channels)

    elif ltype == "residual_block":
        return ResidualBlock(
            in_channels  = in_channels,
            out_channels = cfg["out_channels"],
            stride       = cfg.get("stride",     1),
            activation   = cfg.get("activation", "relu"),
        )

    else:
        raise ValueError(
            f"Unknown layer type '{ltype}'. "
            f"Supported: {list(_REQUIRED_KEYS.keys())}"
        )


# ---------------------------------------------------------------------------
# CNNModel
# ---------------------------------------------------------------------------

class CNNModel(nn.Module):
    """
    A configuration-driven sequential CNN that supports all standard layer
    types including residual blocks for ResNet-style architectures.

    Parameters
    ----------
    layer_configs : list of dict
        Ordered list of layer descriptions. Each dict must have a "type" key.
    input_shape   : tuple (C, H, W)
        Shape of a single input sample — no batch dimension.

    Example
    -------
    >>> from architectures.lenet import build_lenet_config
    >>> cfg   = build_lenet_config("relu", "max")
    >>> model = CNNModel(cfg, input_shape=(1, 32, 32))
    >>> model.summary()
    """

    def __init__(
        self,
        layer_configs: List[Dict[str, Any]],
        input_shape  : Tuple[int, int, int],
    ):
        super().__init__()
        self._input_shape   = input_shape
        self._layer_cfgs    = layer_configs
        self._output_shapes : List[Tuple] = []

        self._validate(layer_configs)
        layers, self._layer_names = self._build_all(layer_configs, input_shape)
        self.model = nn.Sequential(*layers)
        self._trace_shapes(input_shape)

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def _validate(self, layer_configs: List[Dict[str, Any]]) -> None:
        """Validate every config dict before any PyTorch objects are created."""
        for i, cfg in enumerate(layer_configs):
            if not isinstance(cfg, dict):
                raise TypeError(f"Layer {i}: expected dict, got {type(cfg).__name__}.")
            if "type" not in cfg:
                raise ValueError(f"Layer {i}: missing required key 'type'.")
            ltype = cfg["type"].lower()
            if ltype not in _REQUIRED_KEYS:
                raise ValueError(f"Layer {i}: unknown type '{ltype}'.")
            for key in _REQUIRED_KEYS[ltype]:
                if key not in cfg:
                    raise ValueError(f"Layer {i} (type='{ltype}'): missing key '{key}'.")
            if ltype == "activation" and cfg["name"].lower() not in _SUPPORTED_ACTIVATIONS:
                raise ValueError(f"Layer {i}: unknown activation '{cfg['name']}'.")
            if ltype == "pool" and cfg["name"].lower() not in _SUPPORTED_POOLS:
                raise ValueError(f"Layer {i}: unknown pool '{cfg['name']}'.")
            if ltype == "dropout":
                p = cfg.get("p", 0.5)
                if not (0.0 <= p < 1.0):
                    raise ValueError(f"Layer {i}: dropout p must be in [0,1), got {p}.")

    # -----------------------------------------------------------------------
    # Build layers with full (C, H, W) shape tracking
    # -----------------------------------------------------------------------

    def _build_all(
        self,
        layer_configs: List[Dict[str, Any]],
        input_shape  : Tuple[int, int, int],
    ) -> Tuple[List[nn.Module], List[str]]:
        """
        Convert every config dict into an nn.Module, tracking (C, H, W)
        throughout so Linear layers always get the correct in_features.
        """
        layers      : List[nn.Module] = []
        layer_names : List[str]       = []

        C, H, W = input_shape
        is_flat  = False

        for cfg in layer_configs:
            ltype = cfg["type"].lower()
            current_features = C if not is_flat else C  # C holds flat count after flatten

            if ltype == "batchnorm":
                cfg = {**cfg, "_is_flat": is_flat}

            layer = _build_layer(cfg, in_channels=current_features)
            layers.append(layer)

            # Update running shape and build human-readable name
            if ltype == "conv":
                k, s, p = cfg["kernel_size"], cfg.get("stride", 1), cfg.get("padding", 0)
                H = math.floor((H + 2*p - k) / s) + 1
                W = math.floor((W + 2*p - k) / s) + 1
                name = f"Conv2d({C}→{cfg['out_channels']}, k={k}, s={s}, p={p})"
                C    = cfg["out_channels"]

            elif ltype == "pool":
                k = cfg["kernel_size"]
                s = cfg.get("stride", k)
                p = cfg.get("padding", 0)
                H = math.floor((H + 2*p - k) / s) + 1
                W = math.floor((W + 2*p - k) / s) + 1
                name = f"{cfg['name'].capitalize()}Pool2d(k={k}, s={s})"

            elif ltype == "residual_block":
                # ResidualBlock: kernel=3, padding=1 → H,W only change with stride
                s    = cfg.get("stride", 1)
                H    = math.ceil(H / s)
                W    = math.ceil(W / s)
                name = (f"ResidualBlock({C}→{cfg['out_channels']}, "
                        f"s={s}, act={cfg.get('activation','relu')})")
                C    = cfg["out_channels"]

            elif ltype == "flatten":
                flat_size = C * H * W
                name      = f"Flatten [{C}×{H}×{W} → {flat_size}]"
                C, H, W   = flat_size, 1, 1
                is_flat   = True

            elif ltype == "linear":
                out = cfg["out_features"]
                name = f"Linear({C}→{out})"
                C    = out

            elif ltype == "activation":
                name = cfg["name"].capitalize()

            elif ltype == "dropout":
                name = f"Dropout(p={cfg.get('p', 0.5)})"

            elif ltype == "batchnorm":
                name = f"BatchNorm({'1d' if is_flat else '2d'})"

            else:
                name = ltype.capitalize()

            layer_names.append(name)

        return layers, layer_names

    # -----------------------------------------------------------------------
    # Shape tracing
    # -----------------------------------------------------------------------

    def _trace_shapes(self, input_shape: Tuple[int, int, int]) -> None:
        """Run a dummy forward pass to record exact output shapes per layer."""
        self._output_shapes = []
        dummy = torch.zeros(1, *input_shape)
        with torch.no_grad():
            x = dummy
            for layer in self.model:
                x = layer(x)
                self._output_shapes.append(tuple(x.shape[1:]))

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # -----------------------------------------------------------------------
    # Public inspection methods
    # -----------------------------------------------------------------------

    def summary(self) -> None:
        """Print a Keras-style summary table with real output shapes."""
        param_counts = [
            sum(p.numel() for p in layer.parameters() if p.requires_grad)
            for layer in self.model
        ]
        total = sum(param_counts)
        col_w = [5, 42, 16, 10]
        sep   = "-" * (sum(col_w) + 9)
        thick = "=" * (sum(col_w) + 9)

        print(thick)
        print("  CNN Architecture Summary")
        print(thick)
        print(f"  Input shape  : {self._input_shape}")
        print(sep)
        print(f"  {'#':<{col_w[0]}}| {'Layer':<{col_w[1]}}| {'Output shape':<{col_w[2]}}| {'Params':>{col_w[3]}}")
        print(sep)
        for i, (name, shape, params) in enumerate(
            zip(self._layer_names, self._output_shapes, param_counts)
        ):
            print(f"  {i:<{col_w[0]}}| {name:<{col_w[1]}}| {str(shape):<{col_w[2]}}| {params:>{col_w[3]},}")
        print(thick)
        print(f"  Total trainable parameters: {total:,}")
        print(f"  Estimated size            : {self.model_size_mb()} MB")
        print(thick)

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_input_shape(self) -> Tuple:
        """Return expected input shape (C, H, W) — no batch dim."""
        return self._input_shape

    def get_output_shape(self) -> Tuple:
        """Return the model's final output shape for one sample."""
        return self._output_shapes[-1] if self._output_shapes else None

    def get_layer_output_shapes(self) -> List[Tuple]:
        """Return output shapes for every layer — useful for debugging."""
        return list(self._output_shapes)

    def model_size_mb(self) -> float:
        """Approximate model size in MB (float32 = 4 bytes per parameter)."""
        return round(self.count_parameters() * 4 / (1024 ** 2), 4)