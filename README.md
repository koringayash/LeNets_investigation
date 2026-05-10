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

*(Fill this table after running the study)*

| Rank | Experiment              | Best Val Acc | Test Acc |
|------|-------------------------|:------------:|:--------:|
| 1    | lenet_relu_maxpool      | —            | —        |
| 2    | lenet_leakyrelu_maxpool | —            | —        |
| 3    | lenet_tanh_maxpool      | —            | —        |
| 4    | lenet_relu_avgpool      | —            | —        |
| 5    | lenet_leakyrelu_avgpool | —            | —        |
| 6    | lenet_tanh_avgpool      | —            | —        |
| 7    | lenet_sigmoid_maxpool   | —            | —        |
| 8    | lenet_sigmoid_avgpool   | —            | —        |

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
