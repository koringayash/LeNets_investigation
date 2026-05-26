"""
architectures/segmentation_configs.py
---------------------------------------
Layer-config builders for supported segmentation architectures.

Supported models
----------------
  unet      — U-Net (Ronneberger 2015) — encoder-decoder with skip connections
  fcn       — FCN-32s (Long 2015)      — fully convolutional, no skip connections
  deeplabv3 — DeepLab v3 (Chen 2017)  — dilated convs + ASPP

All return a dict with "encoder" and "decoder" keys for EncoderDecoderModel,
EXCEPT fcn which uses CNNModel (it's fully sequential with no decoder skip connections).
FCN is still placed here for consistency — model_factory handles the routing.

Minimum recommended image size: 256×256 for U-Net, 512×512 for DeepLab.
"""


def build_unet_config(
    num_classes: int = 2,
    in_channels: int = 3,
    base_channels: int = 64,
) -> dict:
    """
    U-Net for semantic segmentation (Ronneberger et al., 2015).

    The most widely-used segmentation architecture, especially in
    medical imaging. Uses max-pool downsampling in the encoder and
    transposed conv upsampling in the decoder, with skip connections
    at every depth level preserving fine spatial details.

    Encoder depth: 4 downsampling steps (image / 16 at bottleneck).
    Decoder depth: 4 upsampling steps back to original resolution.

    Parameters
    ----------
    num_classes   : int  Number of output segmentation classes.
    in_channels   : int  Input image channels (1=grayscale, 3=RGB).
    base_channels : int  Starting channel count. Doubles at each level.
                         Default 64 matches the original U-Net paper.

    Returns
    -------
    dict  {"encoder": [...], "decoder": [...]}
          For EncoderDecoderModel(configs, input_shape=(in_channels,H,W)).
    """
    b = base_channels   # 64 default
    relu = {"type": "activation", "name": "relu"}
    pool = {"type": "pool", "name": "max", "kernel_size": 2, "stride": 2}

    def enc_block(out_ch):
        """Two conv layers followed by BN+ReLU, then max pool."""
        return [
            {"type": "conv", "out_channels": out_ch, "kernel_size": 3, "padding": 1},
            {"type": "batchnorm"}, relu,
            {"type": "conv", "out_channels": out_ch, "kernel_size": 3, "padding": 1},
            {"type": "batchnorm"}, relu,
            pool,
        ]

    def dec_block(out_ch):
        """Transposed conv upsample + two conv refinement layers."""
        return [
            {"type": "transposed_conv", "out_channels": out_ch,
             "kernel_size": 2, "stride": 2},
            # After skip concat, in_channels = out_ch * 2
            # The conv below handles this because EncoderDecoderModel
            # concatenates skip features after the transposed conv
            {"type": "conv", "out_channels": out_ch, "kernel_size": 3, "padding": 1},
            {"type": "batchnorm"}, relu,
            {"type": "conv", "out_channels": out_ch, "kernel_size": 3, "padding": 1},
            {"type": "batchnorm"}, relu,
        ]

    encoder = [
        # Initial conv (no pool — keeps full resolution for first skip)
        {"type": "conv", "out_channels": b,    "kernel_size": 3, "padding": 1},
        {"type": "batchnorm"}, relu,
        {"type": "conv", "out_channels": b,    "kernel_size": 3, "padding": 1},
        {"type": "batchnorm"}, relu,

        # Downsampling blocks
        *enc_block(b*2),    # 64→128
        *enc_block(b*4),    # 128→256
        *enc_block(b*8),    # 256→512

        # Bottleneck (deepest, no pool after)
        {"type": "conv", "out_channels": b*16, "kernel_size": 3, "padding": 1},
        {"type": "batchnorm"}, relu,
        {"type": "conv", "out_channels": b*16, "kernel_size": 3, "padding": 1},
        {"type": "batchnorm"}, relu,
    ]

    decoder = [
        *dec_block(b*8),    # 1024→512
        *dec_block(b*4),    # 512→256
        *dec_block(b*2),    # 256→128
        *dec_block(b),      # 128→64

        # Final 1×1 conv: maps to num_classes
        {"type": "conv", "out_channels": num_classes, "kernel_size": 1},
    ]

    return {"encoder": encoder, "decoder": decoder}


def build_fcn_config(
    num_classes : int = 21,
    in_channels : int = 3,
) -> dict:
    """
    FCN-32s — Fully Convolutional Network (Long et al., 2015).

    Converts a VGG-16 classifier into a fully convolutional network by:
      1. Replacing FC layers with 1×1 convolutions
      2. Adding a final bilinear upsample to restore original resolution

    FCN-32s uses a single 32× upsample from the last feature map.
    It's simpler than U-Net (no skip connections) but less sharp at
    object boundaries. A good baseline for segmentation tasks.

    Returns
    -------
    dict  {"encoder": [...], "decoder": [...]}
          Encoder is VGG-like feature extractor.
          Decoder is a single large upsample + 1×1 conv.

    Note: FCN is sequential — no skip connections. EncoderDecoderModel
    will handle it correctly (skip_maps list stays unused in forward pass).
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

    encoder = [
        *conv_block(64,  2),
        *conv_block(128, 2),
        *conv_block(256, 3),
        *conv_block(512, 3),
        *conv_block(512, 3),
        # Replace FC with 1×1 convs (FCN adaptation)
        {"type": "conv", "out_channels": 4096, "kernel_size": 1}, relu,
        {"type": "dropout", "p": 0.5},
        {"type": "conv", "out_channels": 4096, "kernel_size": 1}, relu,
        {"type": "dropout", "p": 0.5},
        {"type": "conv", "out_channels": num_classes, "kernel_size": 1},
    ]

    # Upsample 32× back to input resolution
    decoder = [
        {"type": "bilinear_upsample", "scale_factor": 32.0},
    ]

    return {"encoder": encoder, "decoder": decoder}


def build_deeplabv3_config(
    num_classes : int = 21,
    in_channels : int = 3,
) -> dict:
    """
    DeepLab v3 (Chen et al., 2017).

    Uses dilated convolutions throughout to maintain spatial resolution
    without losing receptive field. ASPP (Atrous Spatial Pyramid Pooling)
    captures multi-scale context at the bottleneck. Bilinear upsampling
    restores full resolution in the decoder.

    Key features
    ------------
    - Rate=2 dilation in backbone blocks 3 & 4 (no spatial shrinkage)
    - ASPP module at bottleneck with rates [6, 12, 18]
    - Simple bilinear 8× or 16× upsample decoder (no skip connections)

    Returns
    -------
    dict  {"encoder": [...], "decoder": [...]}
    """
    relu = {"type": "activation", "name": "relu"}

    def resnet_block_dilated(out_ch, dilation):
        """Residual block using dilated convolutions."""
        return [
            {"type": "dilated_conv", "out_channels": out_ch,
             "kernel_size": 3, "dilation": dilation},
            {"type": "batchnorm"}, relu,
            {"type": "dilated_conv", "out_channels": out_ch,
             "kernel_size": 3, "dilation": dilation},
            {"type": "batchnorm"}, relu,
        ]

    encoder = [
        # Stem: standard conv + pool (only downsampling steps)
        {"type": "conv", "out_channels": 64, "kernel_size": 7,
         "stride": 2, "padding": 3},
        {"type": "batchnorm"}, relu,
        {"type": "pool", "name": "max", "kernel_size": 3, "stride": 2, "padding": 1},

        # ResNet block 1 (stride 1 — no spatial change)
        *[{"type": "residual_block", "out_channels": 64,  "stride": 1}] * 2,
        # ResNet block 2 (stride 2 — halve spatial size)
        {"type": "residual_block", "out_channels": 128, "stride": 2},
        {"type": "residual_block", "out_channels": 128, "stride": 1},

        # Blocks 3 & 4: use dilated convolutions instead of stride
        # (maintains spatial resolution while expanding receptive field)
        *resnet_block_dilated(256, dilation=2),
        *resnet_block_dilated(512, dilation=4),

        # ASPP bottleneck: captures multi-scale context
        {"type": "aspp_block", "out_channels": 256, "dilations": [6, 12, 18]},
        {"type": "batchnorm"}, relu,
        {"type": "dropout", "p": 0.5},

        # Project to num_classes
        {"type": "conv", "out_channels": num_classes, "kernel_size": 1},
    ]

    # Simple bilinear upsample (16× to restore to near-input resolution)
    decoder = [
        {"type": "bilinear_upsample", "scale_factor": 16.0},
    ]

    return {"encoder": encoder, "decoder": decoder}