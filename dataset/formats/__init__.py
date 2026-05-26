"""
dataset/formats/__init__.py
----------------------------
Task + format aware dataset loader selector.

Reads config.EXPERIMENT["task"] and config.DATASET["format"] and
returns the correct Dataset object. Everything downstream never
needs to know which format was on disk.
"""

import logging
from torch.utils.data import Dataset
from torchvision import transforms


def load_dataset(
    task      : str,
    fmt       : str,
    config    : dict,
    transform : transforms.Compose,
    split     : str = "train",
    logger    : logging.Logger = None,
) -> Dataset:
    """
    Load a dataset for the given task and format.

    Parameters
    ----------
    task      : str  "classification" | "detection" | "segmentation"
    fmt       : str  Format string from config.DATASET["format"].
    config    : dict Full config.DATASET dict.
    transform : transforms.Compose  Preprocessing pipeline.
    split     : str  "train" | "val" | "test" (used by torchvision/cityscapes).
    logger    : logging.Logger, optional

    Returns
    -------
    Dataset  Unified dataset regardless of source format.
    """
    from config import DATA_DIR
    task = task.lower()
    fmt  = fmt.lower()

    # ---- Classification ---------------------------------------------------
    if task == "classification":
        from dataset.formats.classification_formats import (
            load_imagefolder, load_csv_dataset, load_torchvision_dataset,
        )
        if fmt == "imagefolder":
            return load_imagefolder(config["local_path"], transform, logger)
        elif fmt == "csv":
            return load_csv_dataset(config["csv_path"], transform, logger=logger)
        elif fmt == "torchvision":
            return load_torchvision_dataset(
                config["name"], str(DATA_DIR), transform,
                train=(split == "train"), logger=logger,
            )
        else:
            raise ValueError(
                f"Unknown classification format '{fmt}'. "
                f"Choose: 'imagefolder', 'csv', 'torchvision'."
            )

    # ---- Detection --------------------------------------------------------
    elif task == "detection":
        from dataset.formats.detection_formats import (
            load_yolo_dataset, load_coco_detection_dataset,
            load_pascal_voc_dataset,
        )
        if fmt == "yolo":
            return load_yolo_dataset(
                config["local_path"] + "/images",
                config["local_path"] + "/labels",
                transform, logger,
            )
        elif fmt == "coco_json":
            ann = config.get("annotation_path",
                             config["local_path"] + "/annotations.json")
            return load_coco_detection_dataset(
                config["local_path"] + "/images", ann, transform, logger,
            )
        elif fmt == "pascal_voc":
            return load_pascal_voc_dataset(
                config["local_path"] + "/images",
                config["local_path"] + "/annotations",
                config.get("class_names", []),
                transform, logger,
            )
        else:
            raise ValueError(
                f"Unknown detection format '{fmt}'. "
                f"Choose: 'yolo', 'coco_json', 'pascal_voc'."
            )

    # ---- Segmentation -----------------------------------------------------
    elif task == "segmentation":
        from dataset.formats.segmentation_formats import (
            load_image_mask_dataset, load_coco_segmentation_dataset,
            load_cityscapes_dataset,
        )
        mask_size = config.get("image_size", 256)
        if fmt == "image_mask":
            return load_image_mask_dataset(
                config["local_path"],
                config["mask_dir"],
                transform, mask_size, logger,
            )
        elif fmt == "coco_json":
            ann = config.get("annotation_path",
                             config["local_path"] + "/annotations.json")
            return load_coco_segmentation_dataset(
                config["local_path"] + "/images",
                ann, transform, mask_size, logger,
            )
        elif fmt == "cityscapes":
            return load_cityscapes_dataset(
                config["local_path"], split, transform, mask_size, logger,
            )
        else:
            raise ValueError(
                f"Unknown segmentation format '{fmt}'. "
                f"Choose: 'image_mask', 'coco_json', 'cityscapes'."
            )

    else:
        raise ValueError(
            f"Unknown task '{task}'. "
            f"Choose: 'classification', 'detection', 'segmentation'."
        )


__all__ = ["load_dataset"]