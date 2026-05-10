"""
network.py
----------
A flexible, config-driven CNN builder for PyTorch.

The main class `CNNModel` takes a plain Python list of layer-config
dictionaries and builds a full `nn.Module` from them. This lets you define
any sequential CNN architecture (LeNet, AlexNet, custom) without writing
PyTorch boilerplate every time — just change the list of dicts.

Supported layer types
---------------------
  "conv"       - 2-D convolution
  "pool"       - 2-D pooling (max or avg)
  "activation" - element-wise activation function
  "dropout"    - spatial (2-D) or standard dropout
  "flatten"    - flattens spatial dims so Linear layers can follow
  "linear"     - fully-connected (dense) layer
  "batchnorm"  - batch normalisation (2-D or 1-D, chosen automatically)

Beginners: Think of this file as a LEGO factory. You describe the bricks
you want (the list of dicts) and CNNModel assembles them into a working
neural network, validates the config, and gives you handy inspection tools.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple


# ---------------------------------------------------------------------------
# Supported options (used for validation)
# ---------------------------------------------------------------------------

_SUPPORTED_ACTIVATIONS = {"relu", "leakyrelu", "sigmoid", "tanh", "softmax"}
_SUPPORTED_POOLS       = {"max", "avg"}
_REQUIRED_KEYS: Dict[str, List[str]] = {
    "conv"      : ["out_channels", "kernel_size"],
    "pool"      : ["name", "kernel_size"],
    "activation": ["name"],
    "dropout"   : ["p"],
    "flatten"   : [],
    "linear"    : ["out_features"],
    "batchnorm" : [],
}


# ---------------------------------------------------------------------------
# Helper: build a single nn.Module from one layer-config dict
# ---------------------------------------------------------------------------

def _build_layer(layer_cfg: Dict[str, Any], in_channels: int) -> nn.Module:
    """
    Translate one layer-config dictionary into the matching nn.Module.

    This function is called internally by CNNModel for every entry in the
    layer_configs list. You normally don't call it directly.

    Parameters
    ----------
    layer_cfg   : dict  One entry from the layer_configs list.
    in_channels : int   Number of feature maps / channels coming *into* this layer.
                        Only relevant for conv layers.

    Returns
    -------
    nn.Module  The constructed PyTorch layer.

    Raises
    ------
    ValueError  If the layer type is unknown or required keys are missing.
    """
    ltype = layer_cfg["type"].lower()

    # ---- Convolution -------------------------------------------------------
    if ltype == "conv":
        return nn.Conv2d(
            in_channels  = in_channels,
            out_channels = layer_cfg["out_channels"],
            kernel_size  = layer_cfg["kernel_size"],
            stride       = layer_cfg.get("stride",  1),
            padding      = layer_cfg.get("padding", 0),
        )

    # ---- Pooling -----------------------------------------------------------
    elif ltype == "pool":
        name        = layer_cfg["name"].lower()
        kernel_size = layer_cfg["kernel_size"]
        stride      = layer_cfg.get("stride", kernel_size)  # default: non-overlapping
        padding     = layer_cfg.get("padding", 0)
        if name == "max":
            return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        elif name == "avg":
            return nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            raise ValueError(f"Unknown pool type '{name}'. Choose from {_SUPPORTED_POOLS}.")

    # ---- Activation --------------------------------------------------------
    elif ltype == "activation":
        name = layer_cfg["name"].lower()
        if name == "relu":
            return nn.ReLU(inplace=True)
        elif name == "leakyrelu":
            return nn.LeakyReLU(
                negative_slope=layer_cfg.get("negative_slope", 0.01),
                inplace=True,
            )
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "softmax":
            return nn.Softmax(dim=layer_cfg.get("dim", 1))
        else:
            raise ValueError(
                f"Unknown activation '{name}'. Choose from {_SUPPORTED_ACTIVATIONS}."
            )

    # ---- Dropout -----------------------------------------------------------
    elif ltype == "dropout":
        p = layer_cfg.get("p", 0.5)
        # Dropout2d zeros entire feature-map channels (better for conv layers).
        # Standard Dropout zeros individual values (used after flatten/linear).
        spatial = layer_cfg.get("spatial", False)
        return nn.Dropout2d(p=p) if spatial else nn.Dropout(p=p)

    # ---- Flatten -----------------------------------------------------------
    elif ltype == "flatten":
        # start_dim=1 keeps the batch dimension (dim 0) intact.
        return nn.Flatten(start_dim=1)

    # ---- Linear (fully-connected) -----------------------------------------
    elif ltype == "linear":
        # in_channels is repurposed here to mean "in_features" after flatten.
        return nn.Linear(
            in_features  = in_channels,
            out_features = layer_cfg["out_features"],
        )

    # ---- Batch Normalisation -----------------------------------------------
    elif ltype == "batchnorm":
        # CNNModel will pass is_flat=True after a flatten layer so we know
        # whether to use BatchNorm1d or BatchNorm2d.
        is_flat = layer_cfg.get("_is_flat", False)
        if is_flat:
            return nn.BatchNorm1d(in_channels)
        else:
            return nn.BatchNorm2d(in_channels)

    else:
        raise ValueError(
            f"Unknown layer type '{ltype}'. "
            f"Supported types: {list(_REQUIRED_KEYS.keys())}."
        )


# ---------------------------------------------------------------------------
# CNNModel
# ---------------------------------------------------------------------------

class CNNModel(nn.Module):
    """
    A configuration-driven sequential CNN that can represent any architecture
    described by a list of layer-config dictionaries.

    Calling `CNNModel(layer_configs, input_shape)` will:
      1. Validate every config dict (catch typos / missing keys early).
      2. Build the `nn.Sequential` model.
      3. Run one silent dummy forward pass to record exact output shapes
         at every layer (so `summary()` shows real numbers).

    Parameters
    ----------
    layer_configs : list of dict
        Ordered list of layer descriptions. Each dict must have a "type" key.
        See module docstring for all supported types and their required keys.

    input_shape : tuple of int  (C, H, W)
        Shape of a **single** input sample (channels, height, width).
        Do NOT include the batch dimension.

    Example — LeNet-5 with ReLU and MaxPool
    ----------------------------------------
    >>> layer_configs = [
    ...     {"type": "conv",       "out_channels": 6,  "kernel_size": 5},
    ...     {"type": "activation", "name": "relu"},
    ...     {"type": "pool",       "name": "max", "kernel_size": 2, "stride": 2},
    ...     {"type": "conv",       "out_channels": 16, "kernel_size": 5},
    ...     {"type": "activation", "name": "relu"},
    ...     {"type": "pool",       "name": "max", "kernel_size": 2, "stride": 2},
    ...     {"type": "flatten"},
    ...     {"type": "linear",     "out_features": 120},
    ...     {"type": "activation", "name": "relu"},
    ...     {"type": "linear",     "out_features": 84},
    ...     {"type": "activation", "name": "relu"},
    ...     {"type": "linear",     "out_features": 10},
    ... ]
    >>> model = CNNModel(layer_configs, input_shape=(1, 32, 32))
    >>> model.summary()
    """

    def __init__(
        self,
        layer_configs: List[Dict[str, Any]],
        input_shape:   Tuple[int, int, int],
    ):
        super().__init__()

        self._input_shape  = input_shape   # (C, H, W) — no batch dim
        self._layer_cfgs   = layer_configs
        self._output_shapes: List[Tuple] = []  # filled by _trace_shapes()

        # Step 1: Validate all configs before touching PyTorch
        self._validate(layer_configs)

        # Step 2: Build the layers and store them as an nn.Sequential
        layers, layer_names = self._build_all(layer_configs, input_shape)
        self._layer_names   = layer_names
        self.model          = nn.Sequential(*layers)

        # Step 3: Trace shapes via a dummy forward pass (no gradients needed)
        self._trace_shapes(input_shape)

    # -----------------------------------------------------------------------
    # Private: validation
    # -----------------------------------------------------------------------

    def _validate(self, layer_configs: List[Dict[str, Any]]) -> None:
        """
        Walk through every layer config and raise an informative ValueError
        if anything looks wrong. Called before any PyTorch objects are created.

        Checks performed
        ----------------
        - 'type' key exists in every dict.
        - Layer type is one of the supported options.
        - All required keys for that type are present.
        - Activation / pool names are from the supported set.
        - 'p' for dropout is between 0 and 1.
        """
        for i, cfg in enumerate(layer_configs):
            # Each config must be a dict
            if not isinstance(cfg, dict):
                raise TypeError(f"Layer {i}: expected a dict, got {type(cfg).__name__}.")

            # Must have a 'type' key
            if "type" not in cfg:
                raise ValueError(f"Layer {i}: missing required key 'type'.")

            ltype = cfg["type"].lower()

            # Must be a known type
            if ltype not in _REQUIRED_KEYS:
                raise ValueError(
                    f"Layer {i}: unknown type '{ltype}'. "
                    f"Supported types: {list(_REQUIRED_KEYS.keys())}."
                )

            # All required keys must be present
            for key in _REQUIRED_KEYS[ltype]:
                if key not in cfg:
                    raise ValueError(
                        f"Layer {i} (type='{ltype}'): missing required key '{key}'."
                    )

            # Activation name check
            if ltype == "activation" and cfg["name"].lower() not in _SUPPORTED_ACTIVATIONS:
                raise ValueError(
                    f"Layer {i}: unknown activation '{cfg['name']}'. "
                    f"Choose from {_SUPPORTED_ACTIVATIONS}."
                )

            # Pool name check
            if ltype == "pool" and cfg["name"].lower() not in _SUPPORTED_POOLS:
                raise ValueError(
                    f"Layer {i}: unknown pool '{cfg['name']}'. "
                    f"Choose from {_SUPPORTED_POOLS}."
                )

            # Dropout probability range
            if ltype == "dropout":
                p = cfg.get("p", 0.5)
                if not (0.0 <= p < 1.0):
                    raise ValueError(
                        f"Layer {i}: dropout 'p' must be in [0, 1), got {p}."
                    )

    # -----------------------------------------------------------------------
    # Private: build layers
    # -----------------------------------------------------------------------

    def _build_all(
        self,
        layer_configs: List[Dict[str, Any]],
        input_shape:   Tuple[int, int, int],
    ) -> Tuple[List[nn.Module], List[str]]:
        """
        Convert every config dict into an nn.Module and return the list.

        Root cause of the original bug
        --------------------------------
        Tracking only `current_channels` (e.g. 16 after the last Conv) was not
        enough to build a Linear layer after Flatten. The correct in_features
        for the first Linear is C × H × W (e.g. 16 × 5 × 5 = 400).

        Fix: we track the full spatial shape (C, H, W) through every layer.
        Conv and Pool layers shrink H and W according to standard formulas.
        When we reach Flatten, we multiply C × H × W to get the exact
        in_features value and pass it to Linear. No dummy tensor needed here
        (the dummy pass in _trace_shapes is only for recording output shapes
        in the summary table).

        Spatial output size formulas
        ----------------------------
        Conv2d : out = floor((in + 2*padding - kernel_size) / stride) + 1
        Pool2d : out = floor((in - kernel_size) / stride) + 1  (padding=0)

        Returns
        -------
        layers       : list of nn.Module  (passed to nn.Sequential)
        layer_names  : list of str        (human-readable labels for summary)
        """
        import math

        layers:       List[nn.Module] = []
        layer_names:  List[str]       = []

        # Track full spatial shape so we always know the true in_features
        # for Linear layers that follow a Flatten.
        C, H, W = input_shape          # e.g. (1, 32, 32) for MNIST with padding
        is_flat  = False               # True once we've passed a Flatten layer

        for i, cfg in enumerate(layer_configs):
            ltype = cfg["type"].lower()

            # current_channels means different things before/after Flatten:
            #   before → number of feature-map channels (C)
            #   after  → number of flat features (C × H × W)
            current_channels = C if not is_flat else C * H * W

            # Inject _is_flat so batchnorm picks BatchNorm1d vs BatchNorm2d
            if ltype == "batchnorm":
                cfg = {**cfg, "_is_flat": is_flat}

            layer = _build_layer(cfg, in_channels=current_channels)
            layers.append(layer)

            # ------------------------------------------------------------------
            # Update the running shape tracker AND build the human-readable name
            # ------------------------------------------------------------------
            if ltype == "conv":
                k  = cfg["kernel_size"]
                s  = cfg.get("stride",  1)
                p  = cfg.get("padding", 0)
                # Standard Conv2d output formula (same for H and W when square)
                H_out = math.floor((H + 2 * p - k) / s) + 1
                W_out = math.floor((W + 2 * p - k) / s) + 1
                name  = (
                    f"Conv2d({C}→{cfg['out_channels']}, "
                    f"k={k}, s={s}, p={p})"
                )
                C, H, W = cfg["out_channels"], H_out, W_out

            elif ltype == "pool":
                k      = cfg["kernel_size"]
                s      = cfg.get("stride",  k)   # default stride = kernel_size (non-overlapping)
                p      = cfg.get("padding", 0)
                H_out  = math.floor((H + 2 * p - k) / s) + 1
                W_out  = math.floor((W + 2 * p - k) / s) + 1
                name   = f"{cfg['name'].capitalize()}Pool2d(k={k}, s={s})"
                H, W   = H_out, W_out          # C is unchanged by pooling

            elif ltype == "flatten":
                # After flatten the "channel" count becomes the total flat size.
                # We keep H=1, W=1 as sentinels so C holds the full flat count.
                flat_size = C * H * W
                name      = f"Flatten  [{C}×{H}×{W} → {flat_size}]"
                C, H, W   = flat_size, 1, 1
                is_flat   = True

            elif ltype == "linear":
                in_f  = current_channels          # correct flat size thanks to tracking
                out_f = cfg["out_features"]
                name  = f"Linear({in_f}→{out_f})"
                C     = out_f                     # update C for next layer

            elif ltype == "activation":
                name = cfg["name"].capitalize()   # shape unchanged

            elif ltype == "dropout":
                name = f"Dropout(p={cfg.get('p', 0.5)})"   # shape unchanged

            elif ltype == "batchnorm":
                name = f"BatchNorm({'1d' if is_flat else '2d'})"   # shape unchanged

            else:
                name = ltype.capitalize()

            layer_names.append(name)

        return layers, layer_names

    # -----------------------------------------------------------------------
    # Private: shape tracing
    # -----------------------------------------------------------------------

    def _trace_shapes(self, input_shape: Tuple[int, int, int]) -> None:
        """
        Run a single dummy forward pass (no gradients) to record the exact
        output tensor shape after every layer.

        Why? Calculating shapes analytically from kernel sizes and strides is
        error-prone. A real forward pass is always correct. This fills
        `self._output_shapes` which is read by `summary()`.

        Parameters
        ----------
        input_shape : (C, H, W)  Shape of one input sample.
        """
        self._output_shapes = []
        # Add batch dim of 1 → (1, C, H, W)
        dummy = torch.zeros(1, *input_shape)

        with torch.no_grad():
            x = dummy
            for layer in self.model:
                x = layer(x)
                # Store shape WITHOUT the batch dimension
                self._output_shapes.append(tuple(x.shape[1:]))

    # -----------------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard PyTorch forward pass.

        Parameters
        ----------
        x : torch.Tensor  Shape (batch_size, C, H, W).

        Returns
        -------
        torch.Tensor  Shape (batch_size, num_classes).
        """
        return self.model(x)

    # -----------------------------------------------------------------------
    # Public inspection methods
    # -----------------------------------------------------------------------

    def summary(self) -> None:
        """
        Print a human-readable table of every layer, its config, and its
        exact output shape — similar to Keras model.summary().

        Output columns
        --------------
        Index | Layer description | Output shape | Param count

        Example output
        --------------
        ============================================================
          LeNet-5 Architecture Summary
        ============================================================
          Input shape : (1, 32, 32)
        ------------------------------------------------------------
          #  | Layer                          | Output shape  | Params
        ------------------------------------------------------------
          0  | Conv2d(1→6, k=5, s=1, p=0)    | (6, 28, 28)   | 156
          1  | Relu                           | (6, 28, 28)   | 0
          ...
        ============================================================
          Total trainable parameters: 61,706
        ============================================================
        """
        # Compute per-layer parameter counts
        param_counts = []
        for layer in self.model:
            p = sum(par.numel() for par in layer.parameters() if par.requires_grad)
            param_counts.append(p)

        total_params = sum(param_counts)

        col_w = [5, 36, 16, 10]  # column widths
        sep   = "-" * (sum(col_w) + 9)
        thick = "=" * (sum(col_w) + 9)

        print(thick)
        print("  CNN Architecture Summary")
        print(thick)
        print(f"  Input shape  : {self._input_shape}")
        print(sep)
        header = (
            f"  {'#':<{col_w[0]}}| "
            f"{'Layer':<{col_w[1]}}| "
            f"{'Output shape':<{col_w[2]}}| "
            f"{'Params':>{col_w[3]}}"
        )
        print(header)
        print(sep)

        for i, (name, shape, params) in enumerate(
            zip(self._layer_names, self._output_shapes, param_counts)
        ):
            print(
                f"  {i:<{col_w[0]}}| "
                f"{name:<{col_w[1]}}| "
                f"{str(shape):<{col_w[2]}}| "
                f"{params:>{col_w[3]},}"
            )

        print(thick)
        print(f"  Total trainable parameters: {total_params:,}")
        print(thick)

    def count_parameters(self) -> int:
        """
        Return the total number of trainable parameters in the model.

        Returns
        -------
        int  Total trainable parameter count.

        Example
        -------
        >>> model.count_parameters()
        61706
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_input_shape(self) -> Tuple[int, int, int]:
        """
        Return the expected input shape for a single sample (no batch dim).

        Returns
        -------
        tuple  (C, H, W)

        Example
        -------
        >>> model.get_input_shape()
        (1, 32, 32)
        """
        return self._input_shape

    def get_output_shape(self) -> Tuple:
        """
        Return the shape of the model's final output for a single sample.

        Returns
        -------
        tuple  e.g. (10,) for a 10-class classifier.

        Example
        -------
        >>> model.get_output_shape()
        (10,)
        """
        return self._output_shapes[-1] if self._output_shapes else None

    def get_layer_output_shapes(self) -> List[Tuple]:
        """
        Return a list of output shapes, one per layer (no batch dim).

        Useful for debugging dimension issues — you can see exactly where
        the spatial size becomes 0 or unexpectedly small.

        Returns
        -------
        list of tuple  e.g. [(6,28,28), (6,14,14), ..., (10,)]
        """
        return list(self._output_shapes)

    def model_size_mb(self) -> float:
        """
        Estimate the model size in megabytes (parameters stored as float32).

        Returns
        -------
        float  Approximate size in MB.

        Example
        -------
        >>> model.model_size_mb()
        0.24
        """
        total_bytes = self.count_parameters() * 4  # float32 = 4 bytes
        return round(total_bytes / (1024 ** 2), 4)