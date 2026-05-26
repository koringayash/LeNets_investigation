# CV Framework v2

A modular, configurable, and resumable deep learning pipeline for three Computer Vision tasks — **Classification, Detection, and Segmentation** — all driven by a single `config.py` file.

---

## Features

- **Three tasks** — image classification, object detection, semantic segmentation
- **Config-driven** — change task, model, dataset, hyperparameters in `config.py` only
- **Resumable** — crash mid-training? `--resume` picks up from the exact epoch
- **9 dataset formats** — 3 per task (see Dataset Formats below)
- **12 architectures** — LeNet, AlexNet, VGG-11/16, ResNet-18/34, YOLOv1, SSD-MobileNet, Faster R-CNN, U-Net, FCN, DeepLab v3
- **Task-aware losses** — CrossEntropy/Focal/LabelSmoothing | CIoU+Focal+BCE | BCE+Dice
- **Task-aware metrics** — Accuracy/P/R/F1/CM | mAP@0.5:0.95+AR | mIoU+Dice+PixelAcc
- **Full logging** — CSV + JSON metrics, per-phase log files, live tqdm bars

---

## Project Structure

```
cv_framework_v2/
│
├── main.py                        ← master entry point
├── config.py                      ← edit this to configure everything
├── pipeline_state.py              ← crash-safe resume state manager
├── requirements.txt
├── Run.sh
│
├── utils/                         ← Timer, Logger, MetricWriter, seed, SystemInfo
├── losses/                        ← classification_loss, detection_loss, segmentation_loss
│
├── architectures/
│   ├── base.py                    ← shared layer builders (all new v2 layers)
│   ├── classifier.py              ← CNNModel (sequential classification)
│   ├── segmentor.py               ← EncoderDecoderModel (U-Net style)
│   ├── detector.py                ← DetectionModel (backbone+neck+head)
│   ├── classification_configs.py  ← LeNet, AlexNet, VGG, ResNet builders
│   ├── detection_configs.py       ← YOLOv1, SSD-MobileNet, Faster R-CNN builders
│   └── segmentation_configs.py    ← U-Net, FCN, DeepLab v3 builders
│
├── dataset/
│   ├── download.py                ← 4 source strategies (torchvision/local/url/github)
│   ├── preprocess.py              ← task-aware loading + normalisation + split
│   ├── save_dataset.py            ← .pt files + task-aware collate functions
│   ├── info.py                    ← dataset summary printer
│   ├── main.py                    ← dataset phase orchestrator
│   └── formats/
│       ├── classification_formats.py  ← ImageFolder | CSV | torchvision
│       ├── detection_formats.py       ← YOLO txt | COCO JSON | Pascal VOC XML
│       └── segmentation_formats.py    ← Image+Mask | COCO polygons | Cityscapes
│
├── training/
│   ├── model_factory.py           ← task-aware model builder
│   ├── task_loop.py               ← per-task forward pass + loss functions
│   ├── train.py                   ← training loop with resume + checkpointing
│   └── main.py                    ← training phase orchestrator
│
├── evaluation/
│   ├── inference.py               ← task-aware predictor
│   ├── evaluate.py                ← metric computation + JSON save
│   ├── results.py                 ← training curves + task-specific plots
│   ├── main.py                    ← evaluation phase orchestrator
│   └── metrics/
│       ├── classification_metrics.py  ← Accuracy, P, R, F1, Confusion Matrix
│       ├── detection_metrics.py       ← mAP@0.5:0.95 + AR (COCO, from scratch)
│       └── segmentation_metrics.py    ← mIoU + Dice + Pixel Accuracy
│
├── Data/          ← raw + processed datasets
├── Checkpoint/    ← model weights + training_manifest.json
├── logs/          ← log files, results.csv, results.json, eval_metrics.json
└── plots/         ← generated PNGs
```

---

## Quick Start

```bash
git clone <repo-url> && cd cv_framework_v2
pip install -r requirements.txt
bash Run.sh
```

---

## Common Commands

```bash
# Full pipeline from scratch
python main.py

# Single stage only
python main.py --stage dataset
python main.py --stage training
python main.py --stage evaluation

# Resume after a crash
python main.py --resume
python main.py --stage training --resume

# Override epochs without editing config
python main.py --epochs 5

# VM background run
nohup bash Run.sh > run.log 2>&1 &
tail -f logs/main.log
```

---

## Switching Tasks

Open `config.py` — change these two fields:

```python
EXPERIMENT = {
    "task": "classification",   # ← "classification" | "detection" | "segmentation"
    "name": "my_experiment",
    ...
}

MODEL = {
    "type": "predefined",
    "name": "lenet5",           # ← see supported models below
    ...
}
```

The entire pipeline (loss, metrics, dataset format, model class, collate function, inference interpretation) adapts automatically.

---

## Supported Models

| Task | Model name | Notes |
|---|---|---|
| classification | `lenet5` | image_size=32, grayscale |
| classification | `alexnet` | image_size≥64 |
| classification | `vgg11` | image_size≥32 |
| classification | `vgg16` | image_size≥32 |
| classification | `resnet18` | image_size≥32 |
| classification | `resnet34` | image_size≥32 |
| detection | `yolov1` | image_size=448 |
| detection | `ssd_mobilenet` | image_size=300 |
| detection | `faster_rcnn` | image_size≥600 |
| segmentation | `unet` | image_size≥256 |
| segmentation | `fcn` | image_size≥256 |
| segmentation | `deeplabv3` | image_size≥512 |

---

## Dataset Formats

Set `DATASET["format"]` in config.py to match your data:

### Classification
| Format | Value | Structure |
|---|---|---|
| torchvision built-in | `"torchvision"` | Automatic (MNIST, CIFAR-10 etc.) |
| ImageFolder | `"imagefolder"` | `class_name/image.jpg` |
| CSV | `"csv"` | CSV with `image_path, label` columns |

### Detection
| Format | Value | Structure |
|---|---|---|
| YOLO txt | `"yolo"` | `images/` + `labels/*.txt` (cx cy w h class, normalised) |
| COCO JSON | `"coco_json"` | `images/` + single JSON annotation file |
| Pascal VOC | `"pascal_voc"` | `images/` + `annotations/*.xml` |

### Segmentation
| Format | Value | Structure |
|---|---|---|
| Image + Mask pairs | `"image_mask"` | `images/` + `masks/` (grayscale PNGs, values=class idx) |
| COCO JSON polygons | `"coco_json"` | `images/` + COCO polygon annotation JSON |
| Cityscapes | `"cityscapes"` | Standard Cityscapes `leftImg8bit/` + `gtFine/` tree |

---

## Supported Layer Types (Custom Architectures)

Set `MODEL["type"] = "custom"` and define `MODEL["layer_configs"]` as a list of dicts:

| Type | Required keys | Notes |
|---|---|---|
| `conv` | `out_channels`, `kernel_size` | `stride`, `padding` optional |
| `pool` | `name` (max/avg), `kernel_size` | `stride` optional |
| `activation` | `name` (relu/sigmoid/tanh/leakyrelu) | |
| `linear` | `out_features` | After flatten |
| `batchnorm` | — | Auto 1D/2D |
| `dropout` | `p` | |
| `flatten` | — | |
| `residual_block` | `out_channels` | `stride`, `activation` optional |
| `depthwise_conv` | `out_channels`, `kernel_size` | MobileNet-style |
| `dilated_conv` | `out_channels`, `kernel_size`, `dilation` | Atrous conv |
| `transposed_conv` | `out_channels`, `kernel_size` | Learnable upsample |
| `bilinear_upsample` | `scale_factor` | Non-learnable upsample |
| `aspp_block` | `out_channels` | `dilations` optional, default [6,12,18] |

---

## Loss Functions

| Task | Default | Options |
|---|---|---|
| Classification | `cross_entropy` | `focal`, `label_smoothing` |
| Detection | CIoU + Focal + BCE (combined) | `box_loss`: `ciou` \| `smooth_l1` |
| Segmentation | BCE + Dice (combined) | `bce_dice`, `dice`, `cross_entropy` |

Configure via `LOSS` section in `config.py`.

---

## Evaluation Metrics

| Task | Metrics |
|---|---|
| Classification | Accuracy, Precision (macro), Recall (macro), F1 (macro), Confusion Matrix |
| Detection | mAP@0.5, mAP@0.5:0.95, AR (full COCO suite, from scratch) |
| Segmentation | mIoU, Dice Coefficient, Pixel Accuracy (per-class breakdowns in JSON) |

---

## Resume System

`pipeline_state.json` is written atomically after every stage and every training epoch:

```json
{
  "experiment": "cifar10_resnet18",
  "stages": {
    "dataset":    "done",
    "training":   {"status": "in_progress", "last_epoch": 12, "total_epochs": 50},
    "evaluation": "pending"
  }
}
```

`python main.py --resume` reads this and skips done stages, resumes training from epoch 13.
Without `--resume` the state resets and all stages run fresh.

---

## Output Files

| File | Contents |
|---|---|
| `logs/results.csv` | Per-epoch metrics (Excel-friendly) |
| `logs/results.json` | Same data in JSON (used by plots) |
| `logs/eval_metrics.json` | Final test metrics + confusion matrix / per-class AP / per-class IoU |
| `logs/*.log` | Per-phase log files with timestamps |
| `Checkpoint/*_best.pth` | Best model weights |
| `Checkpoint/*_latest.pth` | Most recent epoch (for resume) |
| `Checkpoint/training_manifest.json` | Handoff: training → evaluation |
| `plots/val_metric_curves.png` | Validation metric over epochs |
| `plots/train_loss_curves.png` | Training loss over epochs |
| `plots/confusion_matrix.png` | Classification: confusion matrix heatmap |
| `plots/per_class_ap.png` | Detection: per-class AP bar chart |
| `plots/per_class_iou.png` | Segmentation: per-class IoU bar chart |

---

## Adding a New Architecture

1. Write `build_mymodel_config(num_classes, ...)` in the appropriate config file
2. Add one entry to `REGISTRY` in `architectures/__init__.py`
3. Set `MODEL["name"] = "mymodel"` in `config.py`

Nothing else changes.