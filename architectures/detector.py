"""
architectures/detector.py
--------------------------
DetectionModel — backbone + neck + head for object detection.

Unlike classification (one vector out), detection models output three
things per spatial location per anchor:
  1. Bounding box coordinates  (x1, y1, x2, y2)
  2. Class scores              (num_classes values)
  3. Objectness score          (1 value: object present?)

Config format
-------------
The config dict must have three named sections:

  "backbone" : [list of layer dicts]  — feature extractor
  "neck"     : [list of layer dicts]  — multi-scale fusion (FPN, optional)
  "head"     : [list of layer dicts]  — prediction layers

Output format
-------------
The forward pass returns a dict:
  {
    "boxes"      : Tensor (B, num_anchors * H * W, 4)
    "scores"     : Tensor (B, num_anchors * H * W, num_classes)
    "objectness" : Tensor (B, num_anchors * H * W, 1)
  }

This dict format is what DetectionLoss and the detection metrics expect.

Usage
-----
>>> from architectures.detector import DetectionModel
>>> from architectures.detection_configs import build_yolov1_config
>>> cfg   = build_yolov1_config(num_classes=20, num_anchors=5)
>>> model = DetectionModel(cfg, input_shape=(3, 448, 448), num_anchors=5)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple

from architectures.base import build_layer, REQUIRED_KEYS


class DetectionModel(nn.Module):
    """
    Detection model with three explicit stages: backbone → neck → head.

    Parameters
    ----------
    configs     : dict  Keys: "backbone", "neck" (optional), "head".
                        Each value is a list of layer-config dicts.
    input_shape : tuple (C, H, W)
    num_classes : int   Number of object classes (excluding background).
    num_anchors : int   Anchors per spatial location. Default 5.

    Forward output
    --------------
    dict with keys "boxes", "scores", "objectness" — see module docstring.
    """

    def __init__(
        self,
        configs     : Dict[str, List[Dict[str, Any]]],
        input_shape : Tuple[int, int, int],
        num_classes : int,
        num_anchors : int = 5,
    ):
        super().__init__()
        self._input_shape = input_shape
        self.num_classes  = num_classes
        self.num_anchors  = num_anchors

        if "backbone" not in configs:
            raise ValueError("DetectionModel requires a 'backbone' key in configs.")
        if "head" not in configs:
            raise ValueError("DetectionModel requires a 'head' key in configs.")

        # Build backbone
        backbone_layers, self._backbone_names = \
            self._build_section(configs["backbone"], input_shape[0], "Backbone")
        self.backbone = nn.Sequential(*backbone_layers)

        # Trace backbone output channels for neck/head input size
        backbone_out_ch = self._trace_output_channels(self.backbone, input_shape)

        # Build neck (optional — may be empty list)
        neck_cfgs = configs.get("neck", [])
        if neck_cfgs:
            neck_layers, self._neck_names = \
                self._build_section(neck_cfgs, backbone_out_ch, "Neck")
            self.neck = nn.Sequential(*neck_layers)
            neck_out_ch = self._trace_output_channels(
                self.neck,
                (backbone_out_ch, input_shape[1] // 16, input_shape[2] // 16)
            )
        else:
            self.neck = nn.Identity()
            self._neck_names = []
            neck_out_ch = backbone_out_ch

        # Build head feature layers (all layers except the final prediction)
        head_cfgs = configs.get("head", [])
        head_layers, self._head_names = \
            self._build_section(head_cfgs, neck_out_ch, "Head")
        self.head = nn.Sequential(*head_layers)
        head_out_ch = self._trace_output_channels(
            self.head,
            (neck_out_ch, input_shape[1] // 32, input_shape[2] // 32),
        )

        # Prediction heads — three parallel 1×1 convolutions
        # Each outputs num_anchors predictions per spatial location
        self.box_head = nn.Conv2d(head_out_ch, num_anchors * 4, kernel_size=1)
        self.cls_head = nn.Conv2d(head_out_ch, num_anchors * num_classes, kernel_size=1)
        self.obj_head = nn.Conv2d(head_out_ch, num_anchors * 1, kernel_size=1)

    # -----------------------------------------------------------------------
    # Build helpers
    # -----------------------------------------------------------------------

    def _build_section(
        self,
        layer_configs: List[Dict],
        in_channels  : int,
        label        : str,
    ) -> Tuple[List[nn.Module], List[str]]:
        """Build a list of layers tracking channel count through each."""
        layers, names = [], []
        C = in_channels

        for cfg in layer_configs:
            ltype = cfg["type"].lower()
            layer = build_layer(cfg, in_channels=C)
            layers.append(layer)

            if ltype in ("conv", "depthwise_conv", "dilated_conv"):
                names.append(f"{label}-{ltype.capitalize()}({C}→{cfg['out_channels']})")
                C = cfg["out_channels"]
            elif ltype == "residual_block":
                names.append(f"{label}-ResBlock({C}→{cfg['out_channels']})")
                C = cfg["out_channels"]
            elif ltype == "aspp_block":
                names.append(f"{label}-ASPP({C}→{cfg['out_channels']})")
                C = cfg["out_channels"]
            elif ltype == "batchnorm":
                names.append(f"{label}-BN2d")
            elif ltype == "activation":
                names.append(f"{label}-{cfg['name'].capitalize()}")
            elif ltype == "pool":
                names.append(f"{label}-{cfg['name'].capitalize()}Pool")
            else:
                names.append(f"{label}-{ltype.capitalize()}")

        return layers, names

    @staticmethod
    def _trace_output_channels(
        module      : nn.Module,
        input_shape : Tuple[int, int, int],
    ) -> int:
        """Run a dummy forward pass to find the output channel count."""
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            out = module(x)
        return out.shape[1]

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through backbone → neck → head → prediction heads.

        Parameters
        ----------
        x : Tensor  Shape (B, C, H, W)

        Returns
        -------
        dict:
          "boxes"      : Tensor (B, num_anchors*H'*W', 4)
          "scores"     : Tensor (B, num_anchors*H'*W', num_classes)
          "objectness" : Tensor (B, num_anchors*H'*W', 1)

        Where H', W' are the spatial dims of the final feature map.
        The task_loop is responsible for matching these predictions to
        ground truth targets before computing the detection loss.
        """
        feat = self.backbone(x)
        feat = self.neck(feat)
        feat = self.head(feat)

        B, _, H, W = feat.shape

        # boxes: (B, num_anchors*4, H, W) → (B, num_anchors*H*W, 4)
        boxes = self.box_head(feat)
        boxes = boxes.permute(0, 2, 3, 1).contiguous()
        boxes = boxes.view(B, -1, 4)

        # scores: (B, num_anchors*C, H, W) → (B, num_anchors*H*W, C)
        scores = self.cls_head(feat)
        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(B, -1, self.num_classes)

        # objectness: (B, num_anchors, H, W) → (B, num_anchors*H*W, 1)
        obj = self.obj_head(feat)
        obj = obj.permute(0, 2, 3, 1).contiguous()
        obj = obj.view(B, -1, 1)

        return {"boxes": boxes, "scores": scores, "objectness": obj}

    # -----------------------------------------------------------------------
    # Inspection
    # -----------------------------------------------------------------------

    def summary(self) -> None:
        thick = "=" * 60
        print(thick)
        print("  Detection Model Summary")
        print(thick)
        print(f"  Input shape  : {self._input_shape}")
        print(f"  Num classes  : {self.num_classes}")
        print(f"  Num anchors  : {self.num_anchors}")
        for section, names in [
            ("BACKBONE", self._backbone_names),
            ("NECK",     self._neck_names),
            ("HEAD",     self._head_names),
        ]:
            print(f"\n  {section}")
            print("-" * 40)
            for i, n in enumerate(names):
                print(f"  {i:<4} {n}")
        print(f"\n  Prediction heads:")
        print(f"    Box      : Conv2d → (num_anchors*4) per location")
        print(f"    Class    : Conv2d → (num_anchors*{self.num_classes}) per location")
        print(f"    Obj      : Conv2d → (num_anchors*1) per location")
        total = self.count_parameters()
        print(thick)
        print(f"  Total trainable parameters : {total:,}")
        print(f"  Estimated size             : {self.model_size_mb()} MB")
        print(thick)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        return round(self.count_parameters() * 4 / (1024**2), 4)