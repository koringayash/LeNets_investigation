"""
training/model_factory.py (v2)
--------------------------------
Reads config.MODEL and config.EXPERIMENT["task"] and returns the
correct model instance (CNNModel / EncoderDecoderModel / DetectionModel).

Single entry point: get_model(). No other file needs to know about the
architecture registry or individual model classes.
"""

import logging
from config    import MODEL, DATASET, TRAIN, EXPERIMENT
from architectures import build_model
from architectures.base import REQUIRED_KEYS
from utils     import Timer


def get_model(logger: logging.Logger = None):
    """
    Build and return the correct model for the current task and config.

    Parameters
    ----------
    logger : logging.Logger, optional

    Returns
    -------
    nn.Module  CNNModel | EncoderDecoderModel | DetectionModel
               Not yet moved to a device — call .to(device) in train.py.

    Raises
    ------
    ValueError  If task, model type, or model name is unrecognised.
    """
    log        = logger.info if logger else print
    task       = EXPERIMENT["task"].lower()
    model_type = MODEL["type"].lower()

    input_shape = (
        DATASET["in_channels"],
        DATASET["image_size"],
        DATASET["image_size"],
    )
    num_classes = DATASET["num_classes"]
    num_anchors = MODEL.get("num_anchors", 5)

    log(f"Building model | task={task} | type={model_type}")

    with Timer("Building model", logger=logger):

        if model_type == "predefined":
            name   = MODEL["name"].lower()
            kwargs = _get_builder_kwargs(task, name)
            model  = build_model(
                task        = task,
                name        = name,
                input_shape = input_shape,
                num_classes = num_classes,
                num_anchors = num_anchors,
                **kwargs,
            )

        elif model_type == "custom":
            layer_configs = MODEL.get("layer_configs")
            if not layer_configs:
                raise ValueError(
                    "MODEL['layer_configs'] must be set when MODEL['type']='custom'."
                )
            # Custom architectures use CNNModel for classification/sequential,
            # EncoderDecoderModel for segmentation, DetectionModel for detection.
            if task == "classification":
                from architectures.classifier import CNNModel
                model = CNNModel(layer_configs, input_shape=input_shape)
            elif task == "segmentation":
                from architectures.segmentor import EncoderDecoderModel
                model = EncoderDecoderModel(layer_configs, input_shape=input_shape)
            elif task == "detection":
                from architectures.detector import DetectionModel
                model = DetectionModel(
                    layer_configs, input_shape=input_shape,
                    num_classes=num_classes, num_anchors=num_anchors,
                )
            else:
                raise ValueError(f"Unknown task '{task}'.")
        else:
            raise ValueError(
                f"Unknown MODEL['type'] = '{MODEL['type']}'. "
                f"Choose 'predefined' or 'custom'."
            )

    model.summary()
    log(f"Parameters   : {model.count_parameters():,}")
    log(f"Model size   : {model.model_size_mb()} MB")

    return model


def _get_builder_kwargs(task: str, name: str) -> dict:
    """
    Return architecture-specific kwargs for the config builder.

    Different model families accept different keyword arguments.
    This function maps (task, name) to the correct kwargs from config.
    """
    if task == "classification":
        if name == "lenet5":
            return {
                "activation": MODEL.get("activation", "relu"),
                "pooling"   : MODEL.get("pooling",    "max"),
            }
        elif name in ("resnet18", "resnet34"):
            return {"input_size": DATASET["image_size"]}
        else:
            return {}   # alexnet, vgg11, vgg16 only need num_classes (passed by build_model)

    elif task in ("segmentation", "detection"):
        return {}   # all segmentation/detection builders only need num_classes + num_anchors

    return {}