"""
architectures/detection_configs.py
------------------------------------
Layer-config builders for supported detection architectures.

Supported models
----------------
  yolov1        — YOLOv1-style (Redmon 2016) single-stage, no anchors
  ssd_mobilenet — SSD with MobileNet-style depthwise sep convs (multi-scale)
  faster_rcnn   — Faster R-CNN-style two-stage (VGG backbone + RPN head)

All return a dict with "backbone", "neck", "head" keys for DetectionModel.

Note on image sizes
-------------------
  yolov1        : expects 448×448
  ssd_mobilenet : expects 300×300
  faster_rcnn   : expects 600×600 (or any large size)
"""


def build_yolov1_config(
    num_classes: int = 20,
    num_anchors: int = 5,
) -> dict:
    """
    YOLOv1-style single-stage detector.

    Architecture (simplified from the original paper)
    -------------------------------------------------
    Backbone : 24 conv layers (alternating 1×1 and 3×3, with pooling)
    Neck     : none (YOLOv1 has no FPN)
    Head     : 2 FC layers → reshape to (S, S, num_anchors*(5+C)) grid

    In our framework, the head produces flat anchor predictions that
    DetectionModel reshapes to (B, num_anchors*H*W, 4/C/1).

    Parameters
    ----------
    num_classes : int  Number of object classes.
    num_anchors : int  Anchors per grid cell.

    Returns
    -------
    dict  {"backbone": [...], "neck": [], "head": [...]}
    """
    relu = {"type": "activation", "name": "leakyrelu", "negative_slope": 0.1}

    backbone = [
        # Block 1
        {"type": "conv", "out_channels": 64,  "kernel_size": 7, "stride": 2, "padding": 3},
        {"type": "batchnorm"}, relu,
        {"type": "pool", "name": "max", "kernel_size": 2, "stride": 2},

        # Block 2
        {"type": "conv", "out_channels": 192, "kernel_size": 3, "padding": 1},
        {"type": "batchnorm"}, relu,
        {"type": "pool", "name": "max", "kernel_size": 2, "stride": 2},

        # Block 3
        {"type": "conv", "out_channels": 128, "kernel_size": 1}, relu,
        {"type": "conv", "out_channels": 256, "kernel_size": 3, "padding": 1}, relu,
        {"type": "conv", "out_channels": 256, "kernel_size": 1}, relu,
        {"type": "conv", "out_channels": 512, "kernel_size": 3, "padding": 1},
        {"type": "batchnorm"}, relu,
        {"type": "pool", "name": "max", "kernel_size": 2, "stride": 2},

        # Block 4 (4 repeated 1×1 + 3×3 pairs)
        *[layer for _ in range(4) for layer in [
            {"type": "conv", "out_channels": 256, "kernel_size": 1}, relu,
            {"type": "conv", "out_channels": 512, "kernel_size": 3, "padding": 1}, relu,
        ]],
        {"type": "conv", "out_channels": 512,  "kernel_size": 1}, relu,
        {"type": "conv", "out_channels": 1024, "kernel_size": 3, "padding": 1},
        {"type": "batchnorm"}, relu,
        {"type": "pool", "name": "max", "kernel_size": 2, "stride": 2},

        # Block 5
        {"type": "conv", "out_channels": 512,  "kernel_size": 1}, relu,
        {"type": "conv", "out_channels": 1024, "kernel_size": 3, "padding": 1}, relu,
        {"type": "conv", "out_channels": 512,  "kernel_size": 1}, relu,
        {"type": "conv", "out_channels": 1024, "kernel_size": 3, "padding": 1},
        {"type": "batchnorm"}, relu,
    ]

    head = [
        {"type": "conv", "out_channels": 1024, "kernel_size": 3, "padding": 1},
        {"type": "batchnorm"}, relu,
        {"type": "conv", "out_channels": 1024, "kernel_size": 3, "stride": 2, "padding": 1},
        {"type": "batchnorm"}, relu,
        {"type": "conv", "out_channels": 1024, "kernel_size": 3, "padding": 1},
        {"type": "batchnorm"}, relu,
        {"type": "conv", "out_channels": 1024, "kernel_size": 3, "padding": 1},
        {"type": "batchnorm"}, relu,
    ]

    return {"backbone": backbone, "neck": [], "head": head}


def build_ssd_mobilenet_config(
    num_classes: int = 21,
    num_anchors: int = 6,
) -> dict:
    """
    SSD-style detector with MobileNet depthwise separable conv backbone.

    Uses lightweight depthwise separable convolutions throughout the
    backbone for efficient inference — a common production choice.
    The neck applies additional downsampling convolutions for multi-scale
    feature extraction at different spatial resolutions.

    Parameters
    ----------
    num_classes : int  Number of classes (including background for SSD).
    num_anchors : int  Anchors per spatial location.

    Returns
    -------
    dict  {"backbone": [...], "neck": [...], "head": [...]}
    """
    relu = {"type": "activation", "name": "relu"}

    def dw_block(out_ch, stride=1):
        """Depthwise separable conv block."""
        return [
            {"type": "depthwise_conv", "out_channels": out_ch,
             "kernel_size": 3, "stride": stride, "padding": 1},
            relu,
        ]

    backbone = [
        # Standard conv stem
        {"type": "conv", "out_channels": 32, "kernel_size": 3,
         "stride": 2, "padding": 1},
        {"type": "batchnorm"}, relu,

        # Depthwise separable blocks (MobileNet v1 style)
        *dw_block(64),
        *dw_block(128, stride=2),
        *dw_block(128),
        *dw_block(256, stride=2),
        *dw_block(256),
        *dw_block(512, stride=2),
        *[layer for _ in range(5) for layer in dw_block(512)],  # 5 repeated
        *dw_block(1024, stride=2),
        *dw_block(1024),
    ]

    # SSD extra feature layers (neck) — additional downsampling for multi-scale
    neck = [
        {"type": "conv", "out_channels": 256, "kernel_size": 1}, relu,
        {"type": "conv", "out_channels": 512, "kernel_size": 3,
         "stride": 2, "padding": 1}, relu,
        {"type": "conv", "out_channels": 128, "kernel_size": 1}, relu,
        {"type": "conv", "out_channels": 256, "kernel_size": 3,
         "stride": 2, "padding": 1}, relu,
    ]

    head = [
        {"type": "conv", "out_channels": 256, "kernel_size": 3, "padding": 1},
        {"type": "batchnorm"}, relu,
        {"type": "conv", "out_channels": 256, "kernel_size": 3, "padding": 1},
        {"type": "batchnorm"}, relu,
    ]

    return {"backbone": backbone, "neck": neck, "head": head}


def build_faster_rcnn_config(
    num_classes: int = 21,
    num_anchors: int = 9,
) -> dict:
    """
    Faster R-CNN-style detector with VGG-like backbone and RPN head.

    Faster R-CNN is a two-stage detector:
      Stage 1 (RPN): proposes candidate object regions
      Stage 2 (RoI): classifies each proposed region

    In our single-pass framework, we approximate this with a deep backbone
    and a dense prediction head — the full two-stage RoI processing would
    require custom CUDA ops (torchvision.ops.roi_align) beyond our scope.

    Parameters
    ----------
    num_classes : int
    num_anchors : int  9 is standard (3 scales × 3 aspect ratios).

    Returns
    -------
    dict  {"backbone": [...], "neck": [...], "head": [...]}
    """
    relu = {"type": "activation", "name": "relu"}

    def conv_block(out_ch, num_convs):
        layers = []
        for _ in range(num_convs):
            layers += [
                {"type": "conv", "out_channels": out_ch,
                 "kernel_size": 3, "padding": 1},
                {"type": "batchnorm"}, relu,
            ]
        layers.append({"type": "pool", "name": "max",
                        "kernel_size": 2, "stride": 2})
        return layers

    backbone = [
        *conv_block(64,  2),   # block 1
        *conv_block(128, 2),   # block 2
        *conv_block(256, 3),   # block 3
        *conv_block(512, 3),   # block 4
        # Block 5 — no pooling (keep spatial resolution for RPN)
        {"type": "conv", "out_channels": 512, "kernel_size": 3, "padding": 1},
        {"type": "batchnorm"}, relu,
        {"type": "conv", "out_channels": 512, "kernel_size": 3, "padding": 1},
        {"type": "batchnorm"}, relu,
        {"type": "conv", "out_channels": 512, "kernel_size": 3, "padding": 1},
        {"type": "batchnorm"}, relu,
    ]

    # RPN intermediate layer (neck)
    neck = [
        {"type": "conv", "out_channels": 512, "kernel_size": 3, "padding": 1},
        relu,
    ]

    # RPN prediction head
    head = [
        {"type": "conv", "out_channels": 512, "kernel_size": 3, "padding": 1},
        relu,
    ]

    return {"backbone": backbone, "neck": neck, "head": head}