"""
architectures/base.py
---------------------
Shared layer-building primitives used by all three model classes:
  classifier.py  (CNNModel — sequential)
  segmentor.py   (EncoderDecoderModel — U-Net style)
  detector.py    (DetectionModel — backbone + neck + head)

This file contains
------------------
  1. All supported layer type definitions and validation constants.
  2. Individual layer builder functions (_build_layer).
  3. ResidualBlock nn.Module.
  4. New v2 layer types: DepthwiseSeparableConv, DilatedConv,
     TransposedConv, BilinearUpsample, ASPPBlock.
  5. Shared shape-tracking utility (compute_conv_output_size).

What was removed from v1
------------------------
  CNNModel is now in architectures/classifier.py.
  This file only provides building blocks — no model class.

Beginners: This is the parts warehouse. The three model files
(classifier, segmentor, detector) are the assembly lines that
use these parts to build complete networks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List


# ---------------------------------------------------------------------------
# Supported options (validation constants)
# ---------------------------------------------------------------------------

SUPPORTED_ACTIVATIONS = {"relu", "leakyrelu", "sigmoid", "tanh", "softmax"}
SUPPORTED_POOLS       = {"max", "avg"}

# Required keys per layer type — used by all model classes for validation
REQUIRED_KEYS: Dict[str, List[str]] = {
    "conv"              : ["out_channels", "kernel_size"],
    "pool"              : ["name", "kernel_size"],
    "activation"        : ["name"],
    "dropout"           : ["p"],
    "flatten"           : [],
    "linear"            : ["out_features"],
    "batchnorm"         : [],
    "residual_block"    : ["out_channels"],
    "depthwise_conv"    : ["out_channels", "kernel_size"],
    "dilated_conv"      : ["out_channels", "kernel_size", "dilation"],
    "transposed_conv"   : ["out_channels", "kernel_size"],
    "bilinear_upsample" : ["scale_factor"],
    "aspp_block"        : ["out_channels"],
}


# ---------------------------------------------------------------------------
# Activation factory
# ---------------------------------------------------------------------------

def make_activation(name: str, cfg: dict = None) -> nn.Module:
    """
    Return the nn.Module for the requested activation function.

    Parameters
    ----------
    name : str   Activation name (lowercase).
    cfg  : dict  Optional full layer config (for extra params).

    Returns
    -------
    nn.Module
    """
    cfg  = cfg or {}
    name = name.lower()

    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "leakyrelu":
        return nn.LeakyReLU(
            negative_slope=cfg.get("negative_slope", 0.01), inplace=True
        )
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "softmax":
        return nn.Softmax(dim=cfg.get("dim", 1))
    else:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from {SUPPORTED_ACTIVATIONS}."
        )


# ---------------------------------------------------------------------------
# ResidualBlock  (v1 — unchanged)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Standard residual block with two 3×3 convolutions and a skip connection.

    Main path  : Conv(3×3,s) → BN → Act → Conv(3×3,1) → BN
    Skip path  : Identity  OR  Conv(1×1,s) → BN  (when dims change)
    Output     : Act(main + skip)

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    stride       : int   Use 2 to halve spatial dimensions.
    activation   : str
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : int = 1,
        activation  : str = "relu",
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,  out_channels, 3, stride,  1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.act1  = make_activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1,       1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.act_out = make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out      = self.act1(self.bn1(self.conv1(x)))
        out      = self.bn2(self.conv2(out))
        return self.act_out(out + identity)


# ---------------------------------------------------------------------------
# New v2 layer types
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution (Howard et al., 2017 — MobileNet).

    Splits a standard convolution into two steps:
      1. Depthwise conv: one filter per input channel (spatial filtering)
      2. Pointwise conv: 1×1 conv to combine channels (channel mixing)

    This reduces computation by roughly 8-9× vs a standard conv of the
    same kernel size, with minimal accuracy loss.

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    kernel_size  : int
    stride       : int  Default 1.
    padding      : int  Default 0.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int,
        stride      : int = 1,
        padding     : int = 0,
    ):
        super().__init__()
        # Depthwise: groups=in_channels means one filter per channel
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding,
            groups=in_channels, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU(inplace=True)

        # Pointwise: 1×1 conv to mix channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2  = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.depthwise(x)))
        x = self.act2(self.bn2(self.pointwise(x)))
        return x


class DilatedConv(nn.Module):
    """
    Dilated (Atrous) Convolution.

    A standard convolution with gaps in the filter, controlled by the
    dilation rate. A dilation of 2 means the filter covers a 5×5 area
    using only a 3×3 filter's parameters — larger receptive field for free.

    Used in segmentation (DeepLab) and detection for multi-scale context.

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    kernel_size  : int
    dilation     : int  Gap size between filter elements. 1 = standard conv.
    padding      : int  Usually set to dilation to preserve spatial size.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int,
        dilation    : int,
        stride      : int = 1,
        padding     : int = None,   # auto = dilation for same-size output
    ):
        super().__init__()
        if padding is None:
            padding = dilation  # keeps H and W the same when kernel_size=3

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=False,
        )
        self.bn  = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class TransposedConv(nn.Module):
    """
    Transposed Convolution (Deconvolution) for upsampling.

    Learnable upsampling — the network learns how to "spread" values to
    a larger spatial resolution. Used in the decoder path of segmentation
    models (U-Net, FCN) and generative networks.

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    kernel_size  : int  Usually 2 (doubles spatial size with stride=2).
    stride       : int  Controls upsampling factor. stride=2 doubles H,W.
    padding      : int  Default 0.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int,
        stride      : int = 2,
        padding     : int = 0,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        self.bn  = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class BilinearUpsample(nn.Module):
    """
    Non-learnable bilinear upsampling followed by an optional conv.

    Simpler and faster than TransposedConv. Often used in the decoder
    path of DeepLab and similar architectures: upsample spatially with
    interpolation, then refine with a conv layer.

    Parameters
    ----------
    scale_factor : float  Multiplicative upsampling factor. 2.0 doubles H,W.
    out_channels : int    If provided, adds a 1×1 conv after upsampling.
                          Pass None to skip the conv (pure upsampling only).
    in_channels  : int    Required only when out_channels is not None.
    """

    def __init__(
        self,
        scale_factor : float,
        out_channels : int  = None,
        in_channels  : int  = None,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if out_channels is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor = self.scale_factor,
            mode         = "bilinear",
            align_corners= False,
        )
        if self.conv is not None:
            x = self.conv(x)
        return x


class ASPPBlock(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (Chen et al., 2017 — DeepLab v3).

    Runs multiple dilated convolutions with different dilation rates in
    parallel and concatenates their outputs. This captures context at
    multiple scales simultaneously without losing spatial resolution.

    Structure
    ---------
    Branch 1: 1×1 conv (no dilation)
    Branch 2: 3×3 dilated conv, dilation=6
    Branch 3: 3×3 dilated conv, dilation=12
    Branch 4: 3×3 dilated conv, dilation=18
    Branch 5: Global Average Pooling → 1×1 conv → upsample to input size
    Output  : concat all 5 → 1×1 conv to out_channels

    Parameters
    ----------
    in_channels  : int
    out_channels : int  Output channels after the final 1×1 projection.
    dilations    : list of int  Dilation rates for the 3 dilated branches.
                               Default: [6, 12, 18] (DeepLab v3 defaults).
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        dilations   : List[int] = None,
    ):
        super().__init__()
        dilations = dilations or [6, 12, 18]
        mid_ch    = out_channels // 4   # intermediate channels per branch

        # Branch 1: 1×1 conv
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
        )

        # Branches 2-4: dilated convolutions
        self.dilated_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, mid_ch, 3,
                          padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            )
            for d in dilations
        ])

        # Branch 5: Global Average Pooling
        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
        )

        # Final projection: 5 branches × mid_ch → out_channels
        total_channels = mid_ch * (1 + len(dilations) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]

        b1   = self.b1(x)
        bds  = [b(x) for b in self.dilated_branches]
        gap  = F.interpolate(
            self.gap_branch(x), size=(h, w),
            mode="bilinear", align_corners=False,
        )

        out = torch.cat([b1, *bds, gap], dim=1)
        return self.project(out)


# ---------------------------------------------------------------------------
# Single-layer builder (used by all three model classes)
# ---------------------------------------------------------------------------

def build_layer(cfg: Dict[str, Any], in_channels: int) -> nn.Module:
    """
    Translate one layer-config dict into an nn.Module.

    Called by classifier.py, segmentor.py, and detector.py.

    Parameters
    ----------
    cfg         : dict  One entry from a layer_configs list.
    in_channels : int   Input channels / features for this layer.

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
            in_channels, cfg["out_channels"], cfg["kernel_size"],
            stride=cfg.get("stride", 1), padding=cfg.get("padding", 0),
        )
    elif ltype == "pool":
        name = cfg["name"].lower()
        k, s = cfg["kernel_size"], cfg.get("stride", cfg["kernel_size"])
        p    = cfg.get("padding", 0)
        if name == "max":
            return nn.MaxPool2d(k, s, p)
        return nn.AvgPool2d(k, s, p)

    elif ltype == "activation":
        return make_activation(cfg["name"], cfg)

    elif ltype == "dropout":
        return nn.Dropout2d(cfg["p"]) if cfg.get("spatial") else nn.Dropout(cfg["p"])

    elif ltype == "flatten":
        return nn.Flatten(start_dim=1)

    elif ltype == "linear":
        return nn.Linear(in_channels, cfg["out_features"])

    elif ltype == "batchnorm":
        return (
            nn.BatchNorm1d(in_channels)
            if cfg.get("_is_flat")
            else nn.BatchNorm2d(in_channels)
        )

    elif ltype == "residual_block":
        return ResidualBlock(
            in_channels, cfg["out_channels"],
            stride=cfg.get("stride", 1),
            activation=cfg.get("activation", "relu"),
        )

    elif ltype == "depthwise_conv":
        return DepthwiseSeparableConv(
            in_channels, cfg["out_channels"], cfg["kernel_size"],
            stride=cfg.get("stride", 1), padding=cfg.get("padding", 0),
        )

    elif ltype == "dilated_conv":
        return DilatedConv(
            in_channels, cfg["out_channels"], cfg["kernel_size"],
            dilation=cfg["dilation"],
            stride=cfg.get("stride", 1),
            padding=cfg.get("padding", None),
        )

    elif ltype == "transposed_conv":
        return TransposedConv(
            in_channels, cfg["out_channels"], cfg["kernel_size"],
            stride=cfg.get("stride", 2), padding=cfg.get("padding", 0),
        )

    elif ltype == "bilinear_upsample":
        return BilinearUpsample(
            scale_factor = cfg["scale_factor"],
            out_channels = cfg.get("out_channels"),
            in_channels  = in_channels if cfg.get("out_channels") else None,
        )

    elif ltype == "aspp_block":
        return ASPPBlock(
            in_channels  = in_channels,
            out_channels = cfg["out_channels"],
            dilations    = cfg.get("dilations", [6, 12, 18]),
        )

    else:
        raise ValueError(
            f"Unknown layer type '{ltype}'. "
            f"Supported: {list(REQUIRED_KEYS.keys())}"
        )


# ---------------------------------------------------------------------------
# Shape tracking utility
# ---------------------------------------------------------------------------

def compute_spatial_size(h: int, k: int, s: int, p: int) -> int:
    """
    Compute output spatial size for conv or pool along one axis.

    Formula: floor((H + 2*p - k) / s) + 1

    Parameters
    ----------
    h : int  Input size.
    k : int  Kernel size.
    s : int  Stride.
    p : int  Padding.

    Returns
    -------
    int  Output size.
    """
    return math.floor((h + 2 * p - k) / s) + 1