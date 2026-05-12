"""
evaluation/inference.py
-----------------------
Runs predictions on a single image or a batch of images using the best
trained checkpoint from the training phase.

Two modes
---------
  predict_single(image_path)  → returns (predicted_class, confidence)
  predict_batch(loader)       → returns lists of predictions and true labels

Both modes load the best checkpoint automatically from the training
manifest written by training/train.py.

Usage
-----
>>> from evaluation.inference import Predictor
>>> predictor = Predictor(logger=logger)
>>> label, conf = predictor.predict_single("path/to/image.png")
>>> print(f"Predicted: {label}  Confidence: {conf:.2%}")
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config              import CHECKPOINT_DIR, DATASET
from architectures.base  import CNNModel
from training.model_factory import get_model
from utils               import Timer


class Predictor:
    """
    Loads the best trained model and runs predictions.

    The model and checkpoint are loaded once at construction time, so
    repeated calls to predict_single() or predict_batch() are fast.

    Parameters
    ----------
    logger : logging.Logger, optional

    Raises
    ------
    FileNotFoundError  If training_manifest.json does not exist.
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger
        self.log    = logger.info if logger else print
        self.device = self._get_device()
        self.model  = self._load_model()
        self.transform = self._build_transform()

    def predict_single(self, image_path: str) -> Tuple[int, float]:
        """
        Predict the class of a single image file.

        Parameters
        ----------
        image_path : str  Path to any image file (JPEG, PNG, etc.)

        Returns
        -------
        (predicted_class_index, confidence)
            predicted_class_index : int    The argmax class index.
            confidence            : float  Softmax probability for that class (0–1).

        Example
        -------
        >>> label, conf = predictor.predict_single("test_image.png")
        >>> print(f"Class {label} with {conf:.2%} confidence")
        """
        img    = Image.open(image_path).convert("RGB" if DATASET["in_channels"] == 3 else "L")
        tensor = self.transform(img).unsqueeze(0).to(self.device)  # add batch dim

        with torch.no_grad():
            logits      = self.model(tensor)
            probs       = F.softmax(logits, dim=1)
            confidence, pred = probs.max(dim=1)

        return pred.item(), confidence.item()

    def predict_batch(
        self,
        loader: torch.utils.data.DataLoader,
    ) -> Tuple[List[int], List[int]]:
        """
        Run predictions over an entire DataLoader.

        Parameters
        ----------
        loader : DataLoader  Any DataLoader yielding (images, labels) pairs.

        Returns
        -------
        (all_preds, all_labels)
            all_preds  : list of int  Predicted class indices.
            all_labels : list of int  True class indices (from the DataLoader).

        Example
        -------
        >>> preds, labels = predictor.predict_batch(test_loader)
        >>> accuracy = sum(p == l for p, l in zip(preds, labels)) / len(labels)
        """
        from tqdm import tqdm

        self.model.eval()
        all_preds  : List[int] = []
        all_labels : List[int] = []

        with torch.no_grad():
            for images, labels in tqdm(
                loader,
                desc         = "  [Inference]",
                leave        = False,
                dynamic_ncols= True,
            ):
                images = images.to(self.device)
                logits = self.model(images)
                preds  = logits.argmax(dim=1).cpu().tolist()
                all_preds  .extend(preds)
                all_labels .extend(labels.tolist())

        return all_preds, all_labels

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _load_model(self) -> CNNModel:
        """
        Load the best checkpoint identified by training_manifest.json.

        Returns
        -------
        CNNModel  Loaded model in eval mode, on self.device.
        """
        manifest_path = CHECKPOINT_DIR / "training_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Training manifest not found: {manifest_path}\n"
                f"Run the training phase first: python main.py --stage training"
            )

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        ckpt_path = Path(manifest["best_checkpoint"])
        self.log(f"Loading checkpoint: {ckpt_path.name}")
        self.log(f"Best val accuracy : {manifest['best_val_acc']:.4f}")

        with Timer("Loading model weights", logger=self.logger):
            model = get_model(logger=self.logger).to(self.device)
            ckpt  = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()

        return model

    def _build_transform(self) -> transforms.Compose:
        """Build the same preprocessing pipeline used during training."""
        return transforms.Compose([
            transforms.Resize((DATASET["image_size"], DATASET["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=DATASET["mean"], std=DATASET["std"]),
        ])

    @staticmethod
    def _get_device() -> torch.device:
        from config import TRAIN
        pref = TRAIN["device"]
        if pref == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(pref)