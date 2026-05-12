"""
architectures/__init__.py
-------------------------
The architecture registry — maps model name strings (as written in config.py)
to their config-builder functions.

How to add a new architecture
------------------------------
1. Create a new file, e.g. architectures/squeezenet.py
2. Write a build_squeezenet_config() function that returns a list of dicts
3. Import it here and add one entry to REGISTRY

That's it. The rest of the framework (model_factory.py) picks it up
automatically — no other file needs to change.

Usage
-----
>>> from architectures import REGISTRY, build_model_config
>>> cfg = build_model_config("resnet18", num_classes=10, input_size=32)
"""

from architectures.lenet   import build_lenet_config
from architectures.alexnet import build_alexnet_config
from architectures.vgg     import build_vgg11_config, build_vgg16_config
from architectures.resnet  import build_resnet18_config, build_resnet34_config


# ---------------------------------------------------------------------------
# Registry: model name → builder function
# ---------------------------------------------------------------------------

REGISTRY = {
    "lenet5"   : build_lenet_config,
    "alexnet"  : build_alexnet_config,
    "vgg11"    : build_vgg11_config,
    "vgg16"    : build_vgg16_config,
    "resnet18" : build_resnet18_config,
    "resnet34" : build_resnet34_config,
}


def build_model_config(name: str, **kwargs) -> list:
    """
    Look up a model by name and return its layer-config list.

    Parameters
    ----------
    name   : str   Model name as defined in config.MODEL["name"].
                   Case-insensitive.
    kwargs : dict  Passed directly to the builder function.
                   Common kwargs: num_classes, input_size, activation, pooling.

    Returns
    -------
    list of dict  Layer configs ready for CNNModel().

    Raises
    ------
    ValueError  If the name is not found in the registry.

    Example
    -------
    >>> from architectures import build_model_config
    >>> cfg = build_model_config("resnet18", num_classes=10, input_size=32)
    >>> cfg = build_model_config("lenet5",   num_classes=10,
    ...                          activation="relu", pooling="max")
    """
    name_lower = name.lower()
    if name_lower not in REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available models: {sorted(REGISTRY.keys())}\n"
            f"To use a custom architecture, set MODEL['type'] = 'custom' "
            f"and provide MODEL['layer_configs'] in config.py."
        )
    return REGISTRY[name_lower](**kwargs)


__all__ = ["REGISTRY", "build_model_config"]