"""
architectures/segmentor.py
--------------------------
EncoderDecoderModel — U-Net style encoder-decoder for semantic segmentation.

Unlike classification (one label per image), segmentation assigns a class
to every single pixel. This requires:
  1. Encoder path  — progressively downsamples to extract features
  2. Decoder path  — progressively upsamples back to original resolution
  3. Skip connections — pass encoder feature maps to the matching decoder
     level so fine spatial details lost during downsampling are restored

Config format
-------------
The layer_configs list is split into two named sections:

  "encoder" : [list of layer dicts]  — downsampling path
  "decoder" : [list of layer dicts]  — upsampling path

Skip connections are matched by index: encoder block N feeds into
decoder block N (counted from the bottom of each path).

Example (simplified U-Net)
--------------------------
>>> configs = {
...   "encoder": [
...     {"type": "conv", "out_channels": 64,  "kernel_size": 3, "padding": 1},
...     {"type": "pool", "name": "max", "kernel_size": 2, "stride": 2},
...     {"type": "conv", "out_channels": 128, "kernel_size": 3, "padding": 1},
...     {"type": "pool", "name": "max", "kernel_size": 2, "stride": 2},
...   ],
...   "decoder": [
...     {"type": "transposed_conv", "out_channels": 64, "kernel_size": 2, "stride": 2},
...     {"type": "conv", "out_channels": 64, "kernel_size": 3, "padding": 1},
...     {"type": "transposed_conv", "out_channels": 32, "kernel_size": 2, "stride": 2},
...     {"type": "conv", "out_channels": 10, "kernel_size": 1},  # final: num_classes
...   ],
... }
>>> model = EncoderDecoderModel(configs, input_shape=(3, 256, 256))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple

from architectures.base import build_layer, REQUIRED_KEYS


class EncoderDecoderModel(nn.Module):
    """
    Encoder-decoder segmentation model with U-Net-style skip connections.

    Parameters
    ----------
    configs     : dict  Must have keys "encoder" and "decoder", each a
                        list of layer-config dicts.
    input_shape : tuple (C, H, W)  Shape of one input sample.

    How skip connections work
    -------------------------
    During the forward pass, the encoder stores feature maps after every
    pooling layer (downsampling step). The decoder receives these stored
    maps and concatenates them channel-wise before each upsampling step.
    This gives the decoder access to high-resolution spatial details that
    were lost during downsampling.

    Output
    ------
    Tensor  Shape (B, num_classes, H, W) — raw logits per pixel.
    Apply argmax over the class dimension to get the predicted mask.
    """

    def __init__(
        self,
        configs     : Dict[str, List[Dict[str, Any]]],
        input_shape : Tuple[int, int, int],
    ):
        super().__init__()
        self._input_shape = input_shape

        if "encoder" not in configs or "decoder" not in configs:
            raise ValueError(
                "EncoderDecoderModel requires configs with 'encoder' and "
                "'decoder' keys, each containing a list of layer dicts."
            )

        # Build encoder and decoder as ModuleLists so PyTorch tracks parameters
        enc_layers, self._enc_names, self._skip_indices = \
            self._build_encoder(configs["encoder"], input_shape)

        dec_layers, self._dec_names = \
            self._build_decoder(configs["decoder"], enc_layers, input_shape)

        self.encoder = nn.ModuleList(enc_layers)
        self.decoder = nn.ModuleList(dec_layers)

    # -----------------------------------------------------------------------
    # Build encoder
    # -----------------------------------------------------------------------

    def _build_encoder(
        self,
        layer_configs: List[Dict],
        input_shape  : Tuple[int, int, int],
    ) -> Tuple[List[nn.Module], List[str], List[int]]:
        """
        Build encoder layers and record which indices are pool (skip) layers.

        Returns
        -------
        enc_layers    : list of nn.Module
        enc_names     : list of str
        skip_indices  : list of int  — layer indices where skip maps are stored
        """
        layers, names, skip_indices = [], [], []
        C, H, W = input_shape

        for i, cfg in enumerate(layer_configs):
            ltype = cfg["type"].lower()
            layer = build_layer(cfg, in_channels=C)
            layers.append(layer)

            if ltype == "conv":
                k = cfg["kernel_size"]; s = cfg.get("stride",1); p = cfg.get("padding",0)
                name = f"Enc-Conv2d({C}→{cfg['out_channels']}, k={k})"
                C = cfg["out_channels"]

            elif ltype == "pool":
                k = cfg["kernel_size"]; s = cfg.get("stride", k)
                H = H // s; W = W // s
                skip_indices.append(i)   # mark this as a skip connection point
                name = f"Enc-{cfg['name'].capitalize()}Pool(k={k})"

            elif ltype == "residual_block":
                s = cfg.get("stride", 1)
                import math; H = math.ceil(H/s); W = math.ceil(W/s)
                name = f"Enc-ResBlock({C}→{cfg['out_channels']}, s={s})"
                C = cfg["out_channels"]
                if s > 1:
                    skip_indices.append(i)

            elif ltype == "batchnorm":
                name = "Enc-BN2d"
            elif ltype == "activation":
                name = f"Enc-{cfg['name'].capitalize()}"
            elif ltype == "aspp_block":
                name = f"Enc-ASPP({C}→{cfg['out_channels']})"
                C = cfg["out_channels"]
            else:
                name = f"Enc-{ltype.capitalize()}"

            names.append(name)

        return layers, names, skip_indices

    # -----------------------------------------------------------------------
    # Build decoder
    # -----------------------------------------------------------------------

    def _build_decoder(
        self,
        layer_configs: List[Dict],
        enc_layers   : List[nn.Module],
        input_shape  : Tuple[int, int, int],
    ) -> Tuple[List[nn.Module], List[str]]:
        """
        Build decoder layers. Channel counts are tracked from config.
        Skip connections will double in_channels at each upsampling step.
        """
        layers, names = [], []

        # Start from the last encoder output channel count
        # We trace this by running a dummy pass through the encoder
        C = self._trace_encoder_output_channels(enc_layers, input_shape)

        for cfg in layer_configs:
            ltype = cfg["type"].lower()
            layer = build_layer(cfg, in_channels=C)
            layers.append(layer)

            if ltype == "transposed_conv":
                name = f"Dec-TransConv({C}→{cfg['out_channels']}, k={cfg['kernel_size']})"
                C = cfg["out_channels"]
            elif ltype == "bilinear_upsample":
                name = f"Dec-Upsample(x{cfg['scale_factor']})"
                if cfg.get("out_channels"):
                    C = cfg["out_channels"]
            elif ltype == "conv":
                name = f"Dec-Conv2d({C}→{cfg['out_channels']}, k={cfg['kernel_size']})"
                C = cfg["out_channels"]
            elif ltype == "batchnorm":
                name = "Dec-BN2d"
            elif ltype == "activation":
                name = f"Dec-{cfg['name'].capitalize()}"
            else:
                name = f"Dec-{ltype.capitalize()}"

            names.append(name)

        return layers, names

    def _trace_encoder_output_channels(
        self,
        enc_layers  : List[nn.Module],
        input_shape : Tuple[int, int, int],
    ) -> int:
        """Run a dummy pass through encoder to find its output channel count."""
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            for layer in enc_layers:
                x = layer(x)
        return x.shape[1]

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connections.

        Encoder stores feature maps before each pooling step.
        Decoder receives stored maps (resized if needed) concatenated
        channel-wise at the matching depth level.

        Parameters
        ----------
        x : Tensor  Shape (B, C, H, W)

        Returns
        -------
        Tensor  Shape (B, num_classes, H, W) — raw logits per pixel.
        """
        skip_maps = []

        # ---- Encoder pass — store feature maps before each pool -----------
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
                skip_maps.append(x)   # save BEFORE pooling
            x = layer(x)

        # ---- Decoder pass — inject skip maps at each upsample step --------
        skip_maps = skip_maps[::-1]   # reverse: shallowest last → first now
        skip_idx  = 0

        for layer in self.decoder:
            if isinstance(layer, (nn.ConvTranspose2d,)):
                x = layer(x)
                # After upsampling, concatenate the matching skip map
                if skip_idx < len(skip_maps):
                    skip = skip_maps[skip_idx]
                    # Resize skip map if spatial dims don't match exactly
                    if skip.shape[2:] != x.shape[2:]:
                        skip = F.interpolate(
                            skip, size=x.shape[2:],
                            mode="bilinear", align_corners=False,
                        )
                    x = torch.cat([x, skip], dim=1)
                    skip_idx += 1
            else:
                x = layer(x)

        return x

    # -----------------------------------------------------------------------
    # Inspection
    # -----------------------------------------------------------------------

    def summary(self) -> None:
        """Print encoder and decoder layer names."""
        thick = "=" * 60
        print(thick)
        print("  Encoder-Decoder Segmentation Model Summary")
        print(thick)
        print(f"  Input shape : {self._input_shape}")
        print("\n  ENCODER")
        print("-" * 40)
        for i, name in enumerate(self._enc_names):
            print(f"  {i:<4} {name}")
        print("\n  DECODER")
        print("-" * 40)
        for i, name in enumerate(self._dec_names):
            print(f"  {i:<4} {name}")
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(thick)
        print(f"  Total trainable parameters : {total:,}")
        print(thick)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        return round(self.count_parameters() * 4 / (1024**2), 4)