"""
architectures/__init__.py
--------------------------
Three-task architecture registry.

Maps (task, model_name) pairs to their config-builder functions and the
correct model class (CNNModel / EncoderDecoderModel / DetectionModel).

How to add a new architecture
------------------------------
1. Write a build_mymodel_config() in the appropriate config file.
2. Add one entry to REGISTRY below.
3. Set MODEL["name"] = "mymodel" in config.py. Done.

Usage
-----
>>> from architectures import build_model
>>> model = build_model(task="classification", name="lenet5",
...                     input_shape=(1,32,32), num_classes=10)
"""

from architectures.classifier  import CNNModel
from architectures.segmentor   import EncoderDecoderModel
from architectures.detector    import DetectionModel

from architectures.classification_configs import (
    build_lenet_config, build_alexnet_config,
    build_vgg11_config, build_vgg16_config,
    build_resnet18_config, build_resnet34_config,
)
from architectures.detection_configs import (
    build_yolov1_config, build_ssd_mobilenet_config, build_faster_rcnn_config,
)
from architectures.segmentation_configs import (
    build_unet_config, build_fcn_config, build_deeplabv3_config,
)


# ---------------------------------------------------------------------------
# Registry  (task → model_name → builder function)
# ---------------------------------------------------------------------------

REGISTRY = {
    "classification": {
        "lenet5"   : build_lenet_config,
        "alexnet"  : build_alexnet_config,
        "vgg11"    : build_vgg11_config,
        "vgg16"    : build_vgg16_config,
        "resnet18" : build_resnet18_config,
        "resnet34" : build_resnet34_config,
    },
    "detection": {
        "yolov1"        : build_yolov1_config,
        "ssd_mobilenet" : build_ssd_mobilenet_config,
        "faster_rcnn"   : build_faster_rcnn_config,
    },
    "segmentation": {
        "unet"      : build_unet_config,
        "fcn"       : build_fcn_config,
        "deeplabv3" : build_deeplabv3_config,
    },
}


def build_model(
    task        : str,
    name        : str,
    input_shape : tuple,
    num_classes : int,
    num_anchors : int = 5,
    **kwargs,
):
    """
    Build and return the correct model instance for the given task.

    Parameters
    ----------
    task        : str    "classification" | "detection" | "segmentation"
    name        : str    Model name (e.g. "lenet5", "unet", "yolov1").
    input_shape : tuple  (C, H, W) — single sample shape.
    num_classes : int    Number of output classes.
    num_anchors : int    Used by detection models only. Default 5.
    **kwargs            Passed to the config builder (e.g. activation, pooling).

    Returns
    -------
    nn.Module  CNNModel | EncoderDecoderModel | DetectionModel

    Raises
    ------
    ValueError  If task or model name is not in the registry.

    Example
    -------
    >>> from architectures import build_model
    >>> model = build_model("classification", "resnet18",
    ...                     input_shape=(3,32,32), num_classes=10)
    >>> model = build_model("segmentation", "unet",
    ...                     input_shape=(3,256,256), num_classes=2)
    >>> model = build_model("detection", "yolov1",
    ...                     input_shape=(3,448,448), num_classes=20)
    """
    task = task.lower()
    name = name.lower()

    if task not in REGISTRY:
        raise ValueError(
            f"Unknown task '{task}'. "
            f"Choose from: {list(REGISTRY.keys())}"
        )

    task_registry = REGISTRY[task]
    if name not in task_registry:
        raise ValueError(
            f"Unknown model '{name}' for task '{task}'. "
            f"Available: {list(task_registry.keys())}\n"
            f"To use a custom architecture, set MODEL['type']='custom' "
            f"and provide MODEL['layer_configs'] in config.py."
        )

    builder = task_registry[name]

    # ---- Build layer configs ----------------------------------------------
    if task == "classification":
        layer_configs = builder(num_classes=num_classes, **kwargs)
        return CNNModel(layer_configs, input_shape=input_shape)

    elif task == "segmentation":
        configs = builder(num_classes=num_classes, **kwargs)
        return EncoderDecoderModel(configs, input_shape=input_shape)

    elif task == "detection":
        configs = builder(num_classes=num_classes, num_anchors=num_anchors)
        return DetectionModel(
            configs,
            input_shape = input_shape,
            num_classes = num_classes,
            num_anchors = num_anchors,
        )


__all__ = ["REGISTRY", "build_model", "CNNModel", "EncoderDecoderModel", "DetectionModel"]