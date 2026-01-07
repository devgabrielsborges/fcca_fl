git clone <repository-url>
dvc init
dvc repro prepare_data
dvc repro train_mnist
dvc repro train_fashion_mnist
dvc metrics show
dvc metrics diff
dvc dag
# FCCA: Federated CINN Clustering for Accurate Clustered Federated Learning

This repository provides a complete implementation of the FCCA (Federated cINN Clustering Algorithm) and several clustered federated learning baselines from the paper "Federated CINN Clustering for Accurate Clustered Federated Learning" (ICASSP 2024).

## Overview

FCCA addresses the challenge of data heterogeneity in Federated Learning by clustering clients with similar data distributions while preserving privacy. The algorithm uses:

- **Global Encoder**: Maps input data to a shared latent space across all clients
- **Conditional Invertible Neural Network (cINN)**: Learns local data distributions without mode collapse
- **Similarity Assessment**: Clusters clients based on learned distributions
- **Cluster-wise Classifiers**: Train specialized models for each cluster

### FCCA Algorithm (4-Step Process)

**Step 1: Train cINNs on Clients**  
Each client trains a conditional Invertible Neural Network (cINN) to learn the distribution of their local encoded features, with the global encoder frozen.

**Step 2: Train Encoders and Classifiers**  
Clients train both the global encoder and cluster-wise classifier using their local data with cross-entropy loss.

**Step 3: Clustering on Server**  
The server performs similarity assessment by:
1. Generating synthetic samples from each client's cINN
2. Computing similarity matrix based on distribution statistics
3. Applying K-Means clustering

**Step 4: Aggregation**  
- **Global Encoder**: Aggregated across all clients using federated averaging
- **Cluster-wise Classifiers**: Aggregated within each cluster separately

## What's in this repository

- Single notebook/script with the full experimental suite (datasets, models, baselines, FCCA, and an experiment runner) in [src/fcca.py](src/fcca.py)
- A Jupyter version of the same workflow in [src/notebooks/FCCA.ipynb](src/notebooks/FCCA.ipynb)
- Baseline config templates in [configs/](configs) (reference only; not auto-loaded by the notebook)
- Paper PDF in [docs/Federated_CINN_Clustering_for_Accurate_Clustered_Federated_Learning.pdf](docs/Federated_CINN_Clustering_for_Accurate_Clustered_Federated_Learning.pdf)
- MLflow SQLite backend tracked by DVC in [mlflow.db.dvc](mlflow.db.dvc) (optional)

### Implemented Features

**Datasets:**
- MNIST
- Fashion-MNIST
- CIFAR-10
- CIFAR-100
- Synthetic (following FedProx paper)

**Algorithms:**
- FedAvg (baseline)
- IFCA (Iterative Federated Clustering Algorithm)
- CFL (Clustered Federated Learning)
- FL-HC (Federated Learning with Hierarchical Clustering)
- FeSEM (Federated Expectation Maximization)
- FCCA (Federated cINN Clustering Algorithm)
- Commented code for personalized variants: FCCA+FedPer, FCCA+FedProx, FCCA+PerFedAvg

**Model Architectures:**
- 11-layer MLP (for MNIST, Fashion-MNIST, and Synthetic)
- 18-layer CNN (for CIFAR-10 and CIFAR-100)

**Data Partitioning:**
- Dirichlet distribution-based non-IID partitioning with configurable concentration parameter α
- Cluster-aware label exchange for simulating clustered FL settings

## Install dependencies

Using uv (recommended):

```bash
uv sync
source .venv/bin/activate
```

Using pip:

```bash
pip install -e .
```

> The code relies on PyTorch, torchvision, scikit-learn, mlflow, FrEIA, and Jupyter. Installing via the commands above pulls all required packages from [pyproject.toml](pyproject.toml).

## Running experiments

### 1) Notebook workflow (recommended)

- Open [src/notebooks/FCCA.ipynb](src/notebooks/FCCA.ipynb) (or [src/fcca.py](src/fcca.py) in VS Code/Jupyter) and run cells interactively.
- Adjust the `DATASETS`, `ALGORITHMS`, `MODEL_TYPES`, `NUM_CLIENTS`, `NUM_CLUSTERS`, `NUM_ROUNDS`, and `LOCAL_EPOCHS` variables in the experiment cell before running the sweep.
- Start small (e.g., 5–10 clients, 2–3 rounds) to verify everything before scaling up.

### 2) Quick smoke test from the CLI

```bash
PYTHONPATH=src uv run python - <<'PY'
from fcca import ExperimentRunner
import torch

runner = ExperimentRunner(device="cuda" if torch.cuda.is_available() else "cpu")

    dataset_name="mnist",
    algorithm="fcca",
    model_type="mlp",
    num_clients=5,
    num_clusters=2,
    num_rounds=2,
    local_epochs=1,
    batch_size=64,
    lr=1e-2,
    alpha=1.0,
)
print({k: v[-1] for k, v in history.items() if len(v)})
PY
```

This imports the notebook code as a module (by adding `src` to `PYTHONPATH`) and runs a tiny FCCA experiment to confirm the stack works.

### 3) Full sweep (slow)

- The bottom of [src/fcca.py](src/fcca.py) defines the experiment grid. Increase `NUM_ROUNDS`, `NUM_CLIENTS`, and `LOCAL_EPOCHS` cautiously—the full paper settings (N=100, E=100, K=20) take hours to days.
- CNN runs on CIFAR datasets are the most compute- and memory-intensive; reduce batch size or clients if you hit OOM.

## Configuration

While the notebook doesn't auto-load config files, the templates in [configs/](configs) show the paper's recommended hyperparameters:

```yaml
# Example from configs/mnist_baseline.yaml
model:
  latent_dim: 64
  encoder_hidden_dims: [128, 256, 512]
  classifier_hidden_dims: [256, 128]
  cinn_num_blocks: 4
  cinn_hidden_dim: 128

training:
  num_rounds: 100
  num_clusters: 5
  batch_size: 32
  local_epochs: 5
  cinn_lr: 0.0001
  encoder_classifier_lr: 0.001
  clustering_interval: 10
  warmup_rounds: 10

data:
  num_clients: 10
  partition_method: clustered
  dirichlet_alpha: 0.5
```

In the notebook, adjust these values directly in the experiment runner call or in the global variables (`NUM_CLIENTS`, `NUM_ROUNDS`, etc.).

## Experiment Tracking & Results

### MLflow

The notebook sets the experiment name to `FCCA-Colab`. To view tracked runs:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

The MLflow database (`mlflow.db`) is tracked by DVC, so you can restore previous experiment states:

```bash
dvc checkout mlflow.db.dvc
```

### Tracked Metrics

Each experiment history dictionary contains:

- `test_acc`: Global test accuracy per round
- `train_loss`: Average training loss per round
- `cluster_assignments`: Client-to-cluster mapping over time (for clustered algorithms)
- `silhouette_scores`: Clustering quality metrics (where applicable)

Use `ExperimentRunner.get_results_dataframe()` to convert results into a Pandas DataFrame for analysis.

### Reproducing Paper Results

Paper settings (⚠️ **very slow**, days on CPU):
- **Clients:** N = 100
- **Clusters:** M = 5
- **Rounds:** E = 100
- **Local epochs:** K = 20
- **Batch size:** 64
- **Learning rate:** η = 0.01
- **Dirichlet α:** 1.0

Recommended testing settings (completes in ~1 hour):
- **Clients:** N = 20
- **Clusters:** M = 5
- **Rounds:** E = 20
- **Local epochs:** K = 5
- **Batch size:** 64
- **Learning rate:** η = 0.01
- **Dirichlet α:** 1.0

## Project Structure

```
fcca/
├── src/
│   ├── fcca.py                    # Main notebook-style script
│   └── notebooks/
│       └── FCCA.ipynb             # Jupyter notebook version
├── configs/                        # Config templates (reference only)
│   ├── mnist_baseline.yaml
│   └── fashion_mnist_baseline.yaml
├── docs/
│   └── Federated_CINN_..._.pdf    # Original paper
├── mlflow.db.dvc                  # DVC-tracked MLflow database
├── pyproject.toml                 # Dependencies
└── README.md
```

## Tips and Troubleshooting

### Performance & Memory

- **Out of Memory (OOM):** Reduce `batch_size` (64 → 32 → 16) or `num_clients`
- **Too slow:** Reduce `num_rounds`, `local_epochs`, or use MLP instead of CNN
- **GPU acceleration:** The code auto-detects CUDA; ensure PyTorch is installed with GPU support

### Logging & Analysis

- The script sets the MLflow experiment name to `FCCA-Colab`
- Use `ExperimentRunner.get_results_dataframe()` to convert collected histories into a Pandas table for analysis/plotting
- For reproducibility, seeds are fixed via `set_seed(42)` at import time

### Common Issues

1. **FrEIA import errors:** Ensure `FrEIA` is installed: `pip install FrEIA`
2. **Dataset download failures:** Check internet connection; datasets auto-download to `./data/`
3. **CUDA version mismatch:** Verify PyTorch CUDA version matches your driver

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{zhou2024fcca,
  title={Federated CINN Clustering for Accurate Clustered Federated Learning},
  author={Zhou, Yuhao and Shi, Minjia and Tian, Yuxin and Li, Yuanxi and Ye, Qing and Lv, Jiancheng},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5590--5594},
  year={2024},
  organization={IEEE}
}
```

## References

- Original Paper: [Federated CINN Clustering for Accurate Clustered Federated Learning](https://ieeexplore.ieee.org/document/10447282)
- Conditional INNs: [Guided Image Generation with Conditional Invertible Neural Networks](https://arxiv.org/abs/1907.02392)
- Federated Learning: [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)

## License

This project is for research and educational purposes.
## Algorithm Overview
