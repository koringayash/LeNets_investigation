# CV Framework

A modular, configurable, and resumable deep learning pipeline for Computer Vision research.

Train any standard or custom CNN architecture on any dataset — by editing **one file**: `config.py`.

---

## Features

- **Config-driven** — change model, dataset, hyperparameters in `config.py` only
- **Resumable** — crash mid-training? `--resume` picks up from the exact epoch
- **4 dataset sources** — torchvision, local folder, direct URL, GitHub
- **6 standard architectures** — LeNet-5, AlexNet, VGG-11, VGG-16, ResNet-18, ResNet-34
- **Custom architectures** — define any sequential or residual CNN as a list of dicts
- **Full logging** — CSV + JSON metrics, per-phase log files, live tqdm bars
- **Evaluation** — accuracy, precision, recall, F1 (macro), confusion matrix + plots

---

## Project Structure

```
cv_framework/
├── main.py                  ← master entry point
├── config.py                ← edit this to configure your experiment
├── pipeline_state.py        ← crash-safe resume state manager
├── requirements.txt
├── Run.sh
│
├── utils/                   ← Timer, Logger, MetricWriter, SystemInfo, seed
├── architectures/           ← CNNModel builder + LeNet/AlexNet/VGG/ResNet configs
├── dataset/                 ← download, preprocess, save, info
├── training/                ← model factory, training loop, checkpointing
├── evaluation/              ← inference, metrics, plots
│
├── Data/                    ← raw + processed datasets (auto-created)
├── Checkpoint/              ← model weights (auto-created)
├── logs/                    ← log files + results.csv/json (auto-created)
└── plots/                   ← generated PNGs (auto-created)
```

---

## Quick Start

```bash
git clone <your-repo-url> && cd cv_framework
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

# VM background run (safe to disconnect)
nohup bash Run.sh > run.log 2>&1 &
tail -f logs/main.log
```

---

## Configuring an Experiment

Open `config.py` — it is the only file you need to edit.

### Switch dataset

```python
# Built-in torchvision dataset
DATASET = {"source": "torchvision", "name": "CIFAR10", "in_channels": 3,
           "num_classes": 10, "image_size": 32, ...}

# Local folder (ImageFolder format)
DATASET = {"source": "local", "local_path": "/data/my_dataset", ...}

# Direct URL to a zip file
DATASET = {"source": "url", "url": "https://example.com/dataset.zip", ...}

# GitHub folder or repo
DATASET = {"source": "github", "url": "https://github.com/user/repo/tree/main/data", ...}
```

### Switch model

```python
# Standard architecture
MODEL = {"type": "predefined", "name": "resnet18"}
# Options: "lenet5", "alexnet", "vgg11", "vgg16", "resnet18", "resnet34"

# Custom architecture (your own layer list)
MODEL = {
    "type": "custom",
    "layer_configs": [
        {"type": "conv",       "out_channels": 32, "kernel_size": 3, "padding": 1},
        {"type": "batchnorm"},
        {"type": "activation", "name": "relu"},
        {"type": "pool",       "name": "max", "kernel_size": 2, "stride": 2},
        {"type": "flatten"},
        {"type": "linear",     "out_features": 10},
    ]
}
```

### Supported layer types for custom architectures

| Type | Required keys | Optional keys |
|---|---|---|
| `conv` | `out_channels`, `kernel_size` | `stride` (1), `padding` (0) |
| `pool` | `name` (max/avg), `kernel_size` | `stride` (=kernel_size) |
| `activation` | `name` (relu/sigmoid/tanh/leakyrelu) | `negative_slope` |
| `linear` | `out_features` | — |
| `batchnorm` | — | — |
| `dropout` | `p` | `spatial` (False) |
| `flatten` | — | — |
| `residual_block` | `out_channels` | `stride` (1), `activation` (relu) |

---

## Resume System

A `pipeline_state.json` file is written atomically after every completed stage and after every training epoch.

```json
{
  "experiment": "mnist_lenet5",
  "stages": {
    "dataset":    "done",
    "training":   {"status": "in_progress", "last_epoch": 12, "total_epochs": 50},
    "evaluation": "pending"
  }
}
```

Running `python main.py --resume` reads this file and:
- Skips stages marked `done`
- Resumes training from epoch 13 (loads `Checkpoint/latest.pth`)
- Runs evaluation fresh when training completes

Without `--resume`, the state file is always reset and all stages run fresh.

---

## Adding a New Architecture

1. Create `architectures/mynet.py` with a `build_mynet_config(num_classes)` function
2. Add one line to `architectures/__init__.py`:
   ```python
   from architectures.mynet import build_mynet_config
   REGISTRY["mynet"] = build_mynet_config
   ```
3. Set `MODEL = {"type": "predefined", "name": "mynet"}` in `config.py`

Nothing else needs to change.

---

## Output Files

| File | Contents |
|---|---|
| `logs/results.csv` | Per-epoch metrics for all runs (Excel-friendly) |
| `logs/results.json` | Same data in JSON (used by plots) |
| `logs/eval_metrics.json` | Final test accuracy + confusion matrix |
| `logs/*.log` | Per-phase log files with timestamps |
| `Checkpoint/*_best.pth` | Best model weights (highest val accuracy) |
| `Checkpoint/*_latest.pth` | Most recent epoch weights (for resume) |
| `Checkpoint/training_manifest.json` | Handoff from training → evaluation |
| `plots/val_accuracy_curves.png` | Val accuracy over epochs |
| `plots/train_loss_curves.png` | Training loss over epochs |
| `plots/confusion_matrix.png` | Test set confusion matrix heatmap |