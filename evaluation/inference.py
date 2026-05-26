"""
evaluation/inference.py (v2)
-----------------------------
Task-aware predictor. Loads the best checkpoint and runs inference
on the test set, returning predictions in the format each metric
module expects.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from config              import CHECKPOINT_DIR, DATASET, TRAIN, EXPERIMENT, EVAL
from training.model_factory import get_model
from utils               import Timer


class Predictor:
    """
    Loads the best checkpoint and runs task-aware inference.

    Parameters
    ----------
    logger : logging.Logger, optional
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger
        self.log    = logger.info if logger else print
        self.task   = EXPERIMENT["task"].lower()
        self.device = self._get_device()
        self.model  = self._load_model()

    def predict_batch(self, loader) -> Tuple[list, list]:
        """
        Run inference over a DataLoader and return (predictions, labels)
        in the format expected by the metrics module for the current task.

        Returns
        -------
        (all_preds, all_labels)
          Classification : lists of int
          Detection      : lists of dicts {"boxes":arr,"scores":arr,"labels":arr}
          Segmentation   : lists of np.ndarray (H,W) class maps
        """
        self.model.eval()
        all_preds, all_labels = [], []

        conf_thr = EVAL.get("detection", {}).get("conf_threshold", 0.25)
        nms_thr  = EVAL.get("detection", {}).get("nms_iou_threshold", 0.45)

        with torch.no_grad():
            for batch in tqdm(loader, desc="  [Inference]",
                              leave=False, dynamic_ncols=True):
                images = batch[0].to(self.device)
                targets = batch[1]

                output = self.model(images)

                if self.task == "classification":
                    preds = output.argmax(dim=1).cpu().tolist()
                    labs  = targets.tolist() if hasattr(targets, "tolist") else list(targets)
                    all_preds.extend(preds)
                    all_labels.extend(labs)

                elif self.task == "detection":
                    B = images.shape[0]
                    for b in range(B):
                        boxes  = output["boxes"][b].cpu().numpy()
                        scores_logits = output["scores"][b].cpu()
                        obj    = torch.sigmoid(output["objectness"][b]).cpu().numpy().squeeze(1)

                        scores_prob = torch.softmax(scores_logits, dim=1).numpy()
                        cls_scores  = scores_prob.max(axis=1)
                        cls_labels  = scores_prob.argmax(axis=1)

                        # Combined score = objectness × class confidence
                        combined = obj * cls_scores
                        keep     = combined > conf_thr

                        all_preds.append({
                            "boxes" : boxes[keep],
                            "scores": combined[keep],
                            "labels": cls_labels[keep],
                        })
                        tgt = targets[b] if isinstance(targets, list) else {}
                        all_labels.append({
                            "boxes" : tgt.get("boxes", torch.zeros(0,4)).cpu().numpy(),
                            "labels": tgt.get("labels", torch.zeros(0)).cpu().numpy(),
                        })

                elif self.task == "segmentation":
                    if output.shape[2:] != images.shape[2:]:
                        output = F.interpolate(
                            output, size=images.shape[2:],
                            mode="bilinear", align_corners=False,
                        )
                    preds = output.argmax(dim=1).cpu().numpy()
                    for b in range(preds.shape[0]):
                        all_preds.append(preds[b])
                        lbl = targets[b] if isinstance(targets, (list, tuple)) else targets[b]
                        all_labels.append(
                            lbl.cpu().numpy() if hasattr(lbl, "cpu") else np.array(lbl)
                        )

        return all_preds, all_labels

    def _load_model(self):
        manifest_path = CHECKPOINT_DIR / "training_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Training manifest not found: {manifest_path}\n"
                f"Run training phase first: python main.py --stage training"
            )
        with open(manifest_path) as f:
            manifest = json.load(f)

        ckpt_path = Path(manifest["best_checkpoint"])
        self.log(f"Loading checkpoint: {ckpt_path.name}")
        self.log(f"Best val metric   : {manifest.get('best_val_metric', 'N/A')}")

        with Timer("Loading model weights", logger=self.logger):
            model = get_model(logger=self.logger).to(self.device)
            ckpt  = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
        return model

    @staticmethod
    def _get_device():
        pref = TRAIN["device"]
        if pref == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(pref)