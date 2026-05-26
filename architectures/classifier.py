"""
architectures/classifier.py
----------------------------
CNNModel — sequential CNN for image classification tasks.

This is the same model class from v1, now living in its own file.
It supports any architecture that can be expressed as a straight
sequential pipeline: conv → pool → ... → flatten → linear.

Includes ResNet support via the "residual_block" dict type, which
handles skip connections internally inside each block.

Usage
-----
>>> from architectures.classifier import CNNModel
>>> from architectures.classification_configs import build_lenet_config
>>> cfg   = build_lenet_config("relu", "max", num_classes=10)
>>> model = CNNModel(cfg, input_shape=(1, 32, 32))
>>> model.summary()
"""

import math
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple

from architectures.base import (
    REQUIRED_KEYS, SUPPORTED_ACTIVATIONS, SUPPORTED_POOLS,
    build_layer, compute_spatial_size,
)


class CNNModel(nn.Module):
    """
    Configuration-driven sequential CNN for image classification.

    Accepts a list of layer-config dicts and builds an nn.Sequential model.
    Validates config at construction time and traces real output shapes via
    a dummy forward pass so summary() always shows correct dimensions.

    Parameters
    ----------
    layer_configs : list of dict  Ordered layer descriptions.
    input_shape   : tuple (C, H, W)  Single input sample shape.

    Example
    -------
    >>> model = CNNModel([
    ...     {"type": "conv",       "out_channels": 6,  "kernel_size": 5},
    ...     {"type": "activation", "name": "relu"},
    ...     {"type": "pool",       "name": "max", "kernel_size": 2},
    ...     {"type": "flatten"},
    ...     {"type": "linear",     "out_features": 10},
    ... ], input_shape=(1, 32, 32))
    """

    def __init__(
        self,
        layer_configs: List[Dict[str, Any]],
        input_shape  : Tuple[int, int, int],
    ):
        super().__init__()
        self._input_shape   = input_shape
        self._output_shapes : List[Tuple] = []

        self._validate(layer_configs)
        layers, self._layer_names = self._build_all(layer_configs, input_shape)
        self.model = nn.Sequential(*layers)
        self._trace_shapes(input_shape)

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def _validate(self, cfgs: List[Dict]) -> None:
        for i, cfg in enumerate(cfgs):
            if not isinstance(cfg, dict):
                raise TypeError(f"Layer {i}: expected dict, got {type(cfg).__name__}.")
            if "type" not in cfg:
                raise ValueError(f"Layer {i}: missing 'type' key.")
            ltype = cfg["type"].lower()
            if ltype not in REQUIRED_KEYS:
                raise ValueError(f"Layer {i}: unknown type '{ltype}'.")
            for key in REQUIRED_KEYS[ltype]:
                if key not in cfg:
                    raise ValueError(f"Layer {i} ({ltype}): missing key '{key}'.")
            if ltype == "activation" and cfg["name"].lower() not in SUPPORTED_ACTIVATIONS:
                raise ValueError(f"Layer {i}: unknown activation '{cfg['name']}'.")
            if ltype == "pool" and cfg["name"].lower() not in SUPPORTED_POOLS:
                raise ValueError(f"Layer {i}: unknown pool '{cfg['name']}'.")
            if ltype == "dropout":
                p = cfg.get("p", 0.5)
                if not (0.0 <= p < 1.0):
                    raise ValueError(f"Layer {i}: dropout p must be in [0,1).")

    # -----------------------------------------------------------------------
    # Build all layers with full (C, H, W) shape tracking
    # -----------------------------------------------------------------------

    def _build_all(
        self,
        layer_configs: List[Dict],
        input_shape  : Tuple[int, int, int],
    ) -> Tuple[List[nn.Module], List[str]]:
        layers, names = [], []
        C, H, W = input_shape
        is_flat = False

        for cfg in layer_configs:
            ltype = cfg["type"].lower()
            current = C if not is_flat else C

            if ltype == "batchnorm":
                cfg = {**cfg, "_is_flat": is_flat}

            layer = build_layer(cfg, in_channels=current)
            layers.append(layer)

            # Update shape + build human-readable name
            if ltype == "conv":
                k, s, p = cfg["kernel_size"], cfg.get("stride",1), cfg.get("padding",0)
                H = compute_spatial_size(H, k, s, p)
                W = compute_spatial_size(W, k, s, p)
                name = f"Conv2d({C}→{cfg['out_channels']}, k={k}, s={s}, p={p})"
                C = cfg["out_channels"]

            elif ltype in ("depthwise_conv",):
                k, s, p = cfg["kernel_size"], cfg.get("stride",1), cfg.get("padding",0)
                H = compute_spatial_size(H, k, s, p)
                W = compute_spatial_size(W, k, s, p)
                name = f"DepthwiseSepConv({C}→{cfg['out_channels']}, k={k})"
                C = cfg["out_channels"]

            elif ltype == "dilated_conv":
                d = cfg["dilation"]
                k, s = cfg["kernel_size"], cfg.get("stride",1)
                p = cfg.get("padding", d)
                H = compute_spatial_size(H, k + (k-1)*(d-1), s, p)
                W = compute_spatial_size(W, k + (k-1)*(d-1), s, p)
                name = f"DilatedConv({C}→{cfg['out_channels']}, d={d})"
                C = cfg["out_channels"]

            elif ltype == "pool":
                k = cfg["kernel_size"]; s = cfg.get("stride", k); p = cfg.get("padding",0)
                H = compute_spatial_size(H, k, s, p)
                W = compute_spatial_size(W, k, s, p)
                name = f"{cfg['name'].capitalize()}Pool2d(k={k})"

            elif ltype == "residual_block":
                s = cfg.get("stride", 1)
                H = math.ceil(H / s); W = math.ceil(W / s)
                name = f"ResidualBlock({C}→{cfg['out_channels']}, s={s})"
                C = cfg["out_channels"]

            elif ltype == "flatten":
                flat = C * H * W
                name = f"Flatten [{C}×{H}×{W}→{flat}]"
                C, H, W = flat, 1, 1
                is_flat = True

            elif ltype == "linear":
                name = f"Linear({C}→{cfg['out_features']})"
                C = cfg["out_features"]

            elif ltype == "activation":
                name = cfg["name"].capitalize()
            elif ltype == "dropout":
                name = f"Dropout(p={cfg.get('p',0.5)})"
            elif ltype == "batchnorm":
                name = f"BatchNorm({'1d' if is_flat else '2d'})"
            elif ltype == "aspp_block":
                name = f"ASPP({C}→{cfg['out_channels']})"
                C = cfg["out_channels"]
            else:
                name = ltype.capitalize()

            names.append(name)

        return layers, names

    # -----------------------------------------------------------------------
    # Shape tracing
    # -----------------------------------------------------------------------

    def _trace_shapes(self, input_shape: Tuple[int, int, int]) -> None:
        self._output_shapes = []
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            for layer in self.model:
                x = layer(x)
                self._output_shapes.append(tuple(x.shape[1:]))

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # -----------------------------------------------------------------------
    # Inspection methods
    # -----------------------------------------------------------------------

    def summary(self) -> None:
        """Print a Keras-style layer summary table."""
        param_counts = [
            sum(p.numel() for p in l.parameters() if p.requires_grad)
            for l in self.model
        ]
        total = sum(param_counts)
        col   = [5, 42, 16, 10]
        sep   = "-" * (sum(col) + 9)
        thick = "=" * (sum(col) + 9)
        print(thick)
        print("  CNN Classification Model Summary")
        print(thick)
        print(f"  Input  : {self._input_shape}")
        print(sep)
        print(f"  {'#':<{col[0]}}| {'Layer':<{col[1]}}| {'Output':<{col[2]}}| {'Params':>{col[3]}}")
        print(sep)
        for i, (n, s, p) in enumerate(
            zip(self._layer_names, self._output_shapes, param_counts)
        ):
            print(f"  {i:<{col[0]}}| {n:<{col[1]}}| {str(s):<{col[2]}}| {p:>{col[3]},}")
        print(thick)
        print(f"  Total trainable parameters : {total:,}")
        print(f"  Estimated size             : {self.model_size_mb()} MB")
        print(thick)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_input_shape(self)  -> Tuple: return self._input_shape
    def get_output_shape(self) -> Tuple: return self._output_shapes[-1] if self._output_shapes else None
    def get_layer_output_shapes(self) -> List[Tuple]: return list(self._output_shapes)
    def model_size_mb(self) -> float:
        return round(self.count_parameters() * 4 / (1024**2), 4)