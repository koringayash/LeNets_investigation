# LeNet Empirical Study

A clean, beginner-friendly PyTorch project for training LeNet-5 variants on MNIST and empirically comparing **activation functions** (ReLU, Sigmoid, Tanh, LeakyReLU) and **pooling strategies** (MaxPool vs AvgPool).

---

## What This Project Does

1. Builds a **config-driven CNN class** (`CNNModel`) — describe any architecture as a list of dicts, no boilerplate required.
2. Trains **8 LeNet-5 variants** (4 activations × 2 pooling types), all with Adam optimiser.
3. Logs every epoch's metrics to a **master CSV** for easy analysis.
4. Saves the **best checkpoint** per experiment (by validation accuracy).
5. Times every phase (download, preprocessing, each epoch) so bottlenecks are visible.

---

## Project Structure

```
lenet_study/
├── config.py          ← All hyperparameters and experiment definitions (edit me!)
├── network.py         ← CNNModel class — the generic CNN builder
├── dataset.py         ← Download, preprocess, and split MNIST
├── train.py           ← Training loop, evaluation, CLI interface
├── utils.py           ← Timer, Logger, SystemInfo, CSV helper
├── Run.sh             ← One command to run the whole study
├── requirements.txt   ← Python dependencies
│
├── Data/              ← MNIST downloaded here automatically
├── Checkpoint/        ← Best model weights saved here (one .pth per experiment)
└── logs/
    ├── main.log       ← Master log (shared across all experiments)
    ├── lenet_relu_maxpool.log     ← One log file per experiment
    └── results.csv    ← Master results table (all experiments, all epochs)
```

---

## Quick Start

### 1 — Clone and install

```bash
git clone <your-repo-url>
cd lenet_study
pip install -r requirements.txt
```

### 2 — Run everything

```bash
bash Run.sh
```

This will:
- Install dependencies
- Do a 1-batch dry-run to verify the pipeline works
- Train all 8 experiments and save results to `logs/results.csv`

### 3 — View results

```bash
# Pretty-print the CSV in the terminal
column -t -s, logs/results.csv

# Or open it in Excel / LibreOffice / pandas
```

---

## Running Individual Experiments

```bash
# Single experiment (by name)
python train.py --experiment lenet_relu_maxpool

# Override epoch count
python train.py --epochs 5

# Sanity check — 1 batch, no checkpoint saved
python train.py --dry-run

# All options combined
python train.py --experiment lenet_tanh_avgpool --epochs 20
```

---

## Adding a New Experiment

Open `config.py` and append a dict to `EXPERIMENTS`:

```python
EXPERIMENTS = [
    ...
    {"activation": "leakyrelu", "pooling": "avg"},   # already there
    {"activation": "sigmoid",   "pooling": "max"},   # new variant
]
```

That's it. The next `python train.py` run will include your new experiment automatically.

---

## Using the CNN Builder Class (network.py)

`CNNModel` can build **any** sequential CNN from a list of layer-config dicts — not just LeNet. Here's how to use it directly:

```python
from network import CNNModel

layer_configs = [
    {"type": "conv",       "out_channels": 6,  "kernel_size": 5, "stride": 1, "padding": 0},
    {"type": "activation", "name": "relu"},
    {"type": "pool",       "name": "max",  "kernel_size": 2, "stride": 2},
    {"type": "conv",       "out_channels": 16, "kernel_size": 5},
    {"type": "activation", "name": "relu"},
    {"type": "pool",       "name": "max",  "kernel_size": 2, "stride": 2},
    {"type": "flatten"},
    {"type": "linear",     "out_features": 120},
    {"type": "activation", "name": "relu"},
    {"type": "linear",     "out_features": 84},
    {"type": "activation", "name": "relu"},
    {"type": "linear",     "out_features": 10},
]

model = CNNModel(layer_configs, input_shape=(1, 32, 32))

# Useful inspection methods:
model.summary()                 # prints a table of layers + output shapes
model.count_parameters()        # → 61706
model.get_input_shape()         # → (1, 32, 32)
model.get_output_shape()        # → (10,)
model.get_layer_output_shapes() # → [(6,28,28), (6,14,14), ..., (10,)]
model.model_size_mb()           # → 0.235 MB
```

### Supported Layer Types

| Type         | Required keys                          | Optional keys                    |
|--------------|----------------------------------------|----------------------------------|
| `conv`       | `out_channels`, `kernel_size`          | `stride` (1), `padding` (0)      |
| `pool`       | `name` (max/avg), `kernel_size`        | `stride` (=kernel_size)          |
| `activation` | `name` (relu/sigmoid/tanh/leakyrelu)   | `negative_slope` (LeakyReLU)     |
| `linear`     | `out_features`                         | —                                |
| `dropout`    | `p`                                    | `spatial` (False)                |
| `flatten`    | —                                      | —                                |
| `batchnorm`  | —                                      | —                                |

---

## Experiment Results

Training setup: **15 epochs · batch size 64 · Adam (lr=1e-3) · MNIST 32×32**

| Rank | Experiment               | Best Val Acc | Test Acc | Val Acc @ Ep.1 |
|:----:|--------------------------|:------------:|:--------:|:--------------:|
|  1   | lenet_**leakyrelu**_maxpool | 99.13%    | **99.08%** | 97.2%       |
|  2   | lenet_**leakyrelu**_avgpool | 99.10%    | 99.06%   | 96.5%          |
|  3   | lenet_**tanh**_maxpool      | 99.02%    | 98.90%   | **97.8%**      |
|  4   | lenet_**relu**_avgpool      | 98.92%    | 98.80%   | 96.5%          |
|  5   | lenet_**relu**_maxpool      | 98.95%    | 98.76%   | 97.0%          |
|  6   | lenet_**tanh**_avgpool      | 98.87%    | 98.70%   | 96.7%          |
|  7   | lenet_**sigmoid**_maxpool   | 98.68%    | 98.69%   | 92.7%          |
|  8   | lenet_**sigmoid**_avgpool   | 98.53%    | 98.52%   | 90.2%          |

**Gap between best and worst: 0.56 percentage points.**

---

## Key Findings

### 1. Activation Function matters more than Pooling type

The top 2 and bottom 2 spots are decided entirely by the activation function, not pooling. LeakyReLU wins both its MaxPool and AvgPool variants; Sigmoid loses both of its variants.

### 2. LeakyReLU is the best activation on this task

LeakyReLU (MaxPool) achieves **99.08% test accuracy** — the highest of all 8 experiments. Its small negative slope (0.01) for negative inputs prevents the "dying ReLU" problem, which matters even for a shallow network like LeNet-5.

### 3. Sigmoid is the worst activation — and the slowest to converge

Sigmoid reaches only **90.2% val accuracy at epoch 1** (avgpool variant), compared to 97.8% for Tanh and 97.2% for LeakyReLU. The reason: sigmoid saturates for large/small inputs, causing gradients to near-zero ("vanishing gradient"), which slows learning especially in early epochs.

### 4. Tanh converges fastest

Tanh (MaxPool) achieves **97.8% val accuracy in epoch 1** — the highest convergence speed across all experiments. It reaches near-peak performance very early because its output is zero-centred (unlike Sigmoid which is biased toward positive values), leading to better gradient flow.

### 5. MaxPool vs AvgPool: negligible difference (<0.15 pp)

The pooling type accounts for at most 0.15 percentage points of difference within any activation group. For MNIST — a simple, clean dataset — both strategies extract sufficient spatial information. MaxPool has a slight edge for LeakyReLU and Sigmoid; AvgPool has a slight edge for ReLU.

### 6. All experiments converge. No training instability observed.

Training loss decreases smoothly for all 8 variants. Val loss stays close to training loss throughout (no severe overfitting), which is expected given MNIST's simplicity relative to LeNet's capacity.

---

## Visualisations

Generate all plots from your results CSV with one command:

```bash
python plot_results.py                         # reads logs/results.csv by default
python plot_results.py --csv logs/results.csv --out plots/
```

This produces 5 PNG files in `plots/`:

| File | What it shows |
|------|--------------|
| `val_accuracy_curves.png` | Val accuracy over all 15 epochs for all 8 experiments |
| `train_loss_curves.png` | Training loss over all 15 epochs — shows Sigmoid's slow start |
| `final_test_accuracy.png` | Ranked bar chart of final test accuracy |
| `activation_vs_pooling.png` | Grouped bars: MaxPool vs AvgPool per activation |
| `convergence_speed.png` | Val accuracy at epoch 1 — shows which activation learns fastest |

---

## Monitoring on a VM

All output is mirrored to log files, so you can disconnect and reconnect freely.

```bash
# Follow the master log live
tail -f logs/main.log

# Follow a specific experiment
tail -f logs/lenet_relu_maxpool.log

# Check which experiments have finished
ls Checkpoint/
```

---

## Hyperparameter Reference

All defaults live in `config.py`. Key values:

| Parameter     | Default | Location              |
|---------------|---------|-----------------------|
| Epochs        | 15      | `TRAIN["epochs"]`     |
| Batch size    | 64      | `TRAIN["batch_size"]` |
| Learning rate | 1e-3    | `TRAIN["learning_rate"]` |
| Image size    | 32×32   | `DATASET["image_size"]`  |
| Train split   | 80%     | `DATASET["train_fraction"]` |
| Val split     | 10%     | `DATASET["val_fraction"]`   |

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- torchvision ≥ 0.15
- tqdm ≥ 4.65
- numpy ≥ 1.24

GPU is optional — the project runs on CPU, just slower (~5× per epoch on MNIST).