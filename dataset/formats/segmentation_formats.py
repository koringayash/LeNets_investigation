"""
dataset/formats/segmentation_formats.py
-----------------------------------------
Dataset loaders for the three supported segmentation input formats.

Format A — "image_mask"  : parallel image + mask folders
Format B — "coco_json"   : COCO polygon annotations → rasterised masks
Format C — "cityscapes"  : Cityscapes directory structure

All three return a Dataset producing:
  (image_tensor, mask_tensor)

mask_tensor is a LongTensor of shape (H, W) where each value is a
class index in [0, num_classes-1]. This is exactly what the segmentation
loss functions (DiceLoss, BCEDiceLoss) expect.
"""

import logging
import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


# ---------------------------------------------------------------------------
# Format A: Image + Mask pairs
# ---------------------------------------------------------------------------

class ImageMaskDataset(Dataset):
    """
    Segmentation dataset from parallel image and mask directories.

    Expected structure
    ------------------
    images/
        img001.jpg
        img002.jpg
    masks/
        img001.png    ← grayscale PNG, pixel values = class indices
        img002.png

    Image and mask filenames must match (same stem, any extension).

    Parameters
    ----------
    images_dir  : str   Directory containing input images.
    masks_dir   : str   Directory containing mask images.
    transform   : transforms.Compose  Applied to images only.
    mask_size   : int   Target H=W for mask (resized with nearest-neighbour).
    """

    def __init__(
        self,
        images_dir : str,
        masks_dir  : str,
        transform  : transforms.Compose = None,
        mask_size  : int = 256,
    ):
        self.transform = transform
        self.mask_size = mask_size
        self.samples: List[Tuple[str, str]] = []
        self._load_pairs(images_dir, masks_dir)

    def _load_pairs(self, images_dir: str, masks_dir: str) -> None:
        img_dir  = Path(images_dir)
        mask_dir = Path(masks_dir)

        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            # Find matching mask (same stem, any extension)
            for ext in (".png", ".jpg", ".jpeg"):
                mask_path = mask_dir / (img_path.stem + ext)
                if mask_path.exists():
                    self.samples.append((str(img_path), str(mask_path)))
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")   # grayscale: values = class ids

        if self.transform:
            image = self.transform(image)

        # Resize mask with nearest-neighbour to preserve class indices
        mask = mask.resize((self.mask_size, self.mask_size), Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask


def load_image_mask_dataset(
    images_dir : str,
    masks_dir  : str,
    transform  : transforms.Compose,
    mask_size  : int = 256,
    logger     : logging.Logger = None,
) -> Dataset:
    """Load an image+mask segmentation dataset."""
    log     = logger.info if logger else print
    dataset = ImageMaskDataset(images_dir, masks_dir, transform, mask_size)
    log(f"Image+Mask dataset loaded: {len(dataset):,} pairs")
    return dataset


# ---------------------------------------------------------------------------
# Format B: COCO JSON with polygon annotations
# ---------------------------------------------------------------------------

class COCOSegmentationDataset(Dataset):
    """
    COCO polygon annotation format for segmentation.

    Converts COCO polygon annotations (list of x,y vertex coords) into
    rasterised pixel masks using PIL's ImageDraw.

    Parameters
    ----------
    images_dir  : str   Path to images directory.
    annotation  : str   Path to COCO JSON annotations file.
    mask_size   : int   Target H=W for output masks.
    transform   : transforms.Compose  Applied to images.
    """

    def __init__(
        self,
        images_dir : str,
        annotation : str,
        mask_size  : int = 256,
        transform  : transforms.Compose = None,
    ):
        from collections import defaultdict
        self.transform  = transform
        self.mask_size  = mask_size
        self.samples    = []

        with open(annotation) as f:
            coco = json.load(f)

        img_to_anns = defaultdict(list)
        for ann in coco.get("annotations", []):
            img_to_anns[ann["image_id"]].append(ann)

        for img_info in coco.get("images", []):
            img_id   = img_info["id"]
            img_path = Path(images_dir) / img_info["file_name"]
            if not img_path.exists():
                continue
            anns = img_to_anns.get(img_id, [])
            if anns:
                self.samples.append((
                    str(img_path),
                    anns,
                    img_info["width"],
                    img_info["height"],
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import ImageDraw
        img_path, anns, orig_w, orig_h = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Rasterise polygons into a mask
        mask = Image.new("L", (orig_w, orig_h), 0)
        draw = ImageDraw.Draw(mask)
        for ann in anns:
            cat_id = ann["category_id"]
            for seg in ann.get("segmentation", []):
                if len(seg) >= 6:
                    poly = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                    draw.polygon(poly, fill=cat_id)

        mask = mask.resize((self.mask_size, self.mask_size), Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask


def load_coco_segmentation_dataset(
    images_dir : str,
    annotation : str,
    transform  : transforms.Compose,
    mask_size  : int = 256,
    logger     : logging.Logger = None,
) -> Dataset:
    """Load a COCO polygon segmentation dataset."""
    log     = logger.info if logger else print
    dataset = COCOSegmentationDataset(images_dir, annotation, mask_size, transform)
    log(f"COCO segmentation dataset loaded: {len(dataset):,} images")
    return dataset


# ---------------------------------------------------------------------------
# Format C: Cityscapes
# ---------------------------------------------------------------------------

class CityscapesDataset(Dataset):
    """
    Cityscapes-format segmentation dataset.

    Standard Cityscapes structure
    -----------------------------
    leftImg8bit/
        train/
            aachen/
                aachen_000000_000019_leftImg8bit.png
    gtFine/
        train/
            aachen/
                aachen_000000_000019_gtFine_labelIds.png

    Maps the 34 Cityscapes label IDs to 19 training class IDs
    (ignores void/unlabelled classes by mapping them to 255).

    Parameters
    ----------
    root_dir  : str   Cityscapes dataset root (parent of leftImg8bit/).
    split     : str   "train" | "val" | "test"
    transform : transforms.Compose  Applied to images.
    mask_size : int   Target mask size.
    """

    # Cityscapes: label_id → training_id mapping (255 = ignore)
    LABEL_TO_TRAIN = {
        0:255,1:255,2:255,3:255,4:255,5:255,6:255,
        7:0,8:1,9:255,10:255,11:2,12:3,13:4,14:255,
        15:255,16:255,17:5,18:255,19:6,20:7,21:8,22:9,
        23:10,24:11,25:12,26:13,27:14,28:15,29:255,
        30:255,31:16,32:17,33:18,-1:255,
    }

    def __init__(
        self,
        root_dir  : str,
        split     : str = "train",
        transform : transforms.Compose = None,
        mask_size : int = 512,
    ):
        self.transform = transform
        self.mask_size = mask_size
        self.samples   = []
        self._load_pairs(root_dir, split)

    def _load_pairs(self, root_dir: str, split: str) -> None:
        img_root  = Path(root_dir) / "leftImg8bit" / split
        mask_root = Path(root_dir) / "gtFine"      / split

        for city_dir in sorted(img_root.glob("*")):
            if not city_dir.is_dir():
                continue
            for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
                stem      = img_path.name.replace("_leftImg8bit.png", "")
                mask_path = mask_root / city_dir.name / f"{stem}_gtFine_labelIds.png"
                if mask_path.exists():
                    self.samples.append((str(img_path), str(mask_path)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path)  # uint8 label IDs

        if self.transform:
            image = self.transform(image)

        mask = mask.resize((self.mask_size, self.mask_size), Image.NEAREST)
        mask_np = np.array(mask, dtype=np.int64)

        # Remap label IDs to training IDs
        remapped = np.full_like(mask_np, 255)
        for label_id, train_id in self.LABEL_TO_TRAIN.items():
            remapped[mask_np == label_id] = train_id

        return image, torch.from_numpy(remapped).long()


def load_cityscapes_dataset(
    root_dir  : str,
    split     : str,
    transform : transforms.Compose,
    mask_size : int = 512,
    logger    : logging.Logger = None,
) -> Dataset:
    """Load a Cityscapes segmentation dataset."""
    log     = logger.info if logger else print
    dataset = CityscapesDataset(root_dir, split, transform, mask_size)
    log(f"Cityscapes ({split}) loaded: {len(dataset):,} images")
    return dataset