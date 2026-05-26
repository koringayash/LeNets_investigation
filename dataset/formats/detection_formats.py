"""
dataset/formats/detection_formats.py
--------------------------------------
Dataset loaders for the three supported detection input formats.

Format A — "yolo"      : one .txt per image, normalised [cx, cy, w, h, class]
Format B — "coco_json" : COCO-style single JSON annotation file
Format C — "pascal_voc": one .xml per image with <object> tags

All three return a Dataset producing:
  (image_tensor, {"boxes": Tensor(N,4), "labels": Tensor(N,)})

Boxes are in [x1, y1, x2, y2] format (absolute pixel coords) after loading.
The training loop and loss function always receive this unified format.
"""

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

class DetectionDataset(Dataset):
    """
    Base class for detection datasets.

    Subclasses implement _load_annotations() which fills self.samples:
    a list of (image_path, boxes, labels) tuples.
    boxes : np.ndarray or list of [x1, y1, x2, y2] in absolute pixel coords.
    labels: list of int class indices.
    """

    def __init__(self, transform: transforms.Compose = None):
        self.transform = transform
        self.samples: List[Tuple] = []   # filled by subclass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, boxes, labels = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        if self.transform:
            image = self.transform(image)

        # Scale boxes if image was resized by the transform
        if self.transform and hasattr(self.transform, 'transforms'):
            new_size = None
            for t in self.transform.transforms:
                if isinstance(t, transforms.Resize):
                    new_size = t.size if isinstance(t.size, int) else t.size[0]
            if new_size:
                scale_x = new_size / orig_w
                scale_y = new_size / orig_h
                boxes   = [
                    [b[0]*scale_x, b[1]*scale_y, b[2]*scale_x, b[3]*scale_y]
                    for b in boxes
                ]

        target = {
            "boxes" : torch.tensor(boxes,  dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return image, target


# ---------------------------------------------------------------------------
# Format A: YOLO txt
# ---------------------------------------------------------------------------

class YOLODataset(DetectionDataset):
    """
    YOLO-format detection dataset.

    Expected structure
    ------------------
    images/
        img001.jpg
        img002.jpg
    labels/
        img001.txt    ← one line per object: class cx cy w h (normalised 0-1)
        img002.txt

    Parameters
    ----------
    images_dir : str   Path to images folder.
    labels_dir : str   Path to labels folder.
    transform  : transforms.Compose
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        transform : transforms.Compose = None,
    ):
        super().__init__(transform)
        self._load_annotations(images_dir, labels_dir)

    def _load_annotations(self, images_dir: str, labels_dir: str) -> None:
        img_dir = Path(images_dir)
        lbl_dir = Path(labels_dir)

        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            # Read image size to convert normalised coords → absolute
            with Image.open(img_path) as im:
                w, h = im.size

            boxes, labels = [], []
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
                    # Convert cx,cy,w,h (normalised) → x1,y1,x2,y2 (absolute)
                    x1 = (cx - bw/2) * w
                    y1 = (cy - bh/2) * h
                    x2 = (cx + bw/2) * w
                    y2 = (cy + bh/2) * h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls)

            if boxes:
                self.samples.append((str(img_path), boxes, labels))


def load_yolo_dataset(
    images_dir: str,
    labels_dir: str,
    transform : transforms.Compose,
    logger    : logging.Logger = None,
) -> Dataset:
    """Load a YOLO-format detection dataset."""
    log     = logger.info if logger else print
    dataset = YOLODataset(images_dir, labels_dir, transform)
    log(f"YOLO dataset loaded: {len(dataset):,} images from {images_dir}")
    return dataset


# ---------------------------------------------------------------------------
# Format B: COCO JSON
# ---------------------------------------------------------------------------

class COCODetectionDataset(DetectionDataset):
    """
    COCO-format detection dataset.

    Expected: a single JSON file with COCO annotation format and
    an images directory.

    Parameters
    ----------
    images_dir   : str   Path to directory containing image files.
    annotation   : str   Path to COCO-format JSON annotation file.
    transform    : transforms.Compose
    """

    def __init__(
        self,
        images_dir : str,
        annotation : str,
        transform  : transforms.Compose = None,
    ):
        super().__init__(transform)
        self._load_annotations(images_dir, annotation)

    def _load_annotations(self, images_dir: str, annotation: str) -> None:
        import json
        from collections import defaultdict

        with open(annotation) as f:
            coco = json.load(f)

        # Build image_id → annotations mapping
        img_to_anns = defaultdict(list)
        for ann in coco.get("annotations", []):
            img_to_anns[ann["image_id"]].append(ann)

        for img_info in coco.get("images", []):
            img_id   = img_info["id"]
            img_path = Path(images_dir) / img_info["file_name"]

            if not img_path.exists():
                continue

            anns   = img_to_anns.get(img_id, [])
            boxes, labels = [], []

            for ann in anns:
                x, y, w, h = ann["bbox"]    # COCO: [x, y, width, height]
                boxes.append([x, y, x + w, y + h])   # convert to [x1,y1,x2,y2]
                labels.append(ann["category_id"])

            if boxes:
                self.samples.append((str(img_path), boxes, labels))


def load_coco_detection_dataset(
    images_dir : str,
    annotation : str,
    transform  : transforms.Compose,
    logger     : logging.Logger = None,
) -> Dataset:
    """Load a COCO-format detection dataset."""
    log     = logger.info if logger else print
    dataset = COCODetectionDataset(images_dir, annotation, transform)
    log(f"COCO detection dataset loaded: {len(dataset):,} images")
    return dataset


# ---------------------------------------------------------------------------
# Format C: Pascal VOC XML
# ---------------------------------------------------------------------------

class PascalVOCDataset(DetectionDataset):
    """
    Pascal VOC XML-format detection dataset.

    Expected structure
    ------------------
    images/
        img001.jpg
    annotations/
        img001.xml    ← standard VOC XML with <object> tags

    Parameters
    ----------
    images_dir      : str   Path to images directory.
    annotations_dir : str   Path to XML annotations directory.
    class_names     : list  List of class name strings (defines label indices).
    transform       : transforms.Compose
    """

    def __init__(
        self,
        images_dir      : str,
        annotations_dir : str,
        class_names     : List[str],
        transform       : transforms.Compose = None,
    ):
        super().__init__(transform)
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self._load_annotations(images_dir, annotations_dir)

    def _load_annotations(self, images_dir: str, annotations_dir: str) -> None:
        ann_dir = Path(annotations_dir)
        img_dir = Path(images_dir)

        for xml_path in sorted(ann_dir.glob("*.xml")):
            tree = ET.parse(xml_path)
            root = tree.getroot()

            filename = root.findtext("filename", default=xml_path.stem + ".jpg")
            img_path = img_dir / filename
            if not img_path.exists():
                continue

            boxes, labels = [], []
            for obj in root.findall("object"):
                name  = obj.findtext("name", "")
                if name not in self.class_to_idx:
                    continue
                label = self.class_to_idx[name]
                bbox  = obj.find("bndbox")
                x1 = float(bbox.findtext("xmin"))
                y1 = float(bbox.findtext("ymin"))
                x2 = float(bbox.findtext("xmax"))
                y2 = float(bbox.findtext("ymax"))
                boxes.append([x1, y1, x2, y2])
                labels.append(label)

            if boxes:
                self.samples.append((str(img_path), boxes, labels))


def load_pascal_voc_dataset(
    images_dir      : str,
    annotations_dir : str,
    class_names     : List[str],
    transform       : transforms.Compose,
    logger          : logging.Logger = None,
) -> Dataset:
    """Load a Pascal VOC XML-format detection dataset."""
    log     = logger.info if logger else print
    dataset = PascalVOCDataset(images_dir, annotations_dir, class_names, transform)
    log(f"Pascal VOC dataset loaded: {len(dataset):,} images")
    return dataset