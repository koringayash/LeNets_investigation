"""
training/model_factory.py
--------------------------
Reads config.MODEL and builds the correct CNNModel instance.

This is the single bridge between config.py and the architectures/
package. No other file needs to know about the registry or individual
architecture builders — they just call get_model() and receive a
fully-constructed, ready-to-train CNNModel.

Two modes
---------
  "predefined" → looks up MODEL["name"] in the architecture registry,
                 calls the matching builder, passes result to CNNModel.

  "custom"     → passes MODEL["layer_configs"] directly to CNNModel.
                 The user defines the full layer list in config.py.

Usage
-----
>>> from training.model_factory import get_model
>>> model = get_model()
>>> model.summary()
"""

import logging

from config        import MODEL, DATASET, TRAIN
from architectures import build_model_config
from architectures.base import CNNModel
from utils         import Timer


def get_model(logger: logging.Logger = None) -> CNNModel:
    """
    Build and return a CNNModel based on config.MODEL settings.

    Parameters
    ----------
    logger : logging.Logger, optional
        Where to write build-time messages (parameter count, size, etc.)

    Returns
    -------
    CNNModel  Fully constructed model, not yet moved to a device.
              Call .to(device) on the returned model in train.py.

    Raises
    ------
    ValueError  If MODEL["type"] is not "predefined" or "custom".
    ValueError  If MODEL["type"] is "custom" but layer_configs is None.
    ValueError  If MODEL["name"] is not found in the registry.

    Example
    -------
    >>> model = get_model(logger=logger)
    >>> model.summary()
    >>> print(model.count_parameters())
    """
    log        = logger.info if logger else print
    model_type = MODEL["type"].lower()

    input_shape = (
        DATASET["in_channels"],
        DATASET["image_size"],
        DATASET["image_size"],
    )

    with Timer("Building model", logger=logger):
        if model_type == "predefined":
            layer_configs = _build_predefined(log)

        elif model_type == "custom":
            layer_configs = MODEL.get("layer_configs")
            if not layer_configs:
                raise ValueError(
                    "MODEL['layer_configs'] must be set when MODEL['type'] = 'custom'.\n"
                    "Define your architecture as a list of layer dicts in config.py."
                )
            log(f"Using custom architecture ({len(layer_configs)} layers)")

        else:
            raise ValueError(
                f"Unknown MODEL['type'] = '{MODEL['type']}'. "
                f"Choose 'predefined' or 'custom'."
            )

        model = CNNModel(layer_configs, input_shape=input_shape)

    # Log key model stats
    model.summary()
    log(f"Trainable parameters : {model.count_parameters():,}")
    log(f"Estimated model size : {model.model_size_mb()} MB")
    log(f"Input shape          : {model.get_input_shape()}")
    log(f"Output shape         : {model.get_output_shape()}")

    return model


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _build_predefined(log) -> list:
    """
    Look up the predefined model name in the registry and build its config.

    Passes architecture-specific kwargs from config where applicable:
      - lenet5    : activation, pooling, num_classes
      - alexnet   : num_classes
      - vgg11/16  : num_classes
      - resnet18/34: num_classes, input_size

    Returns
    -------
    list of dict  Layer configs for CNNModel.
    """
    name        = MODEL["name"].lower()
    num_classes = DATASET["num_classes"]
    image_size  = DATASET["image_size"]

    log(f"Building predefined model: {name}")

    # Architecture-specific kwargs
    if name == "lenet5":
        kwargs = {
            "num_classes": num_classes,
            "activation" : MODEL.get("activation", "relu"),
            "pooling"    : MODEL.get("pooling",    "max"),
        }
    elif name in ("resnet18", "resnet34"):
        kwargs = {
            "num_classes": num_classes,
            "input_size" : image_size,
        }
    else:
        # alexnet, vgg11, vgg16
        kwargs = {"num_classes": num_classes}

    return build_model_config(name, **kwargs)