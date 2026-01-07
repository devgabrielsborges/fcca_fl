# FCCA: Federated CINN Clustering for Accurate Clustered Federated Learning

This repository implements the FCCA (Federated cINN Clustering Algorithm) from the paper "Federated CINN Clustering for Accurate Clustered Federated Learning" (ICASSP 2024).

## Overview

FCCA addresses the challenge of data heterogeneity in Federated Learning by clustering clients with similar data distributions while preserving privacy. The algorithm uses:

- **Global Encoder**: Maps input data to a shared latent space across all clients
- **Conditional Invertible Neural Network (cINN)**: Learns local data distributions without mode collapse
- **Similarity Assessment**: Clusters clients based on learned distributions
- **Cluster-wise Classifiers**: Train specialized models for each cluster

## Features

- ✅ PyTorch implementation of FCCA
- ✅ Support for MNIST and Fashion-MNIST datasets
- ✅ Non-IID data partitioning strategies (clustered, Dirichlet, label-based)
- ✅ MLflow integration for experiment tracking
- ✅ DVC for data versioning and pipeline management
- ✅ Reproducible experiments with configuration files

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone <repository-url>
cd fcca

# Install dependencies with uv
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

### Using pip

```bash
pip install -e .
```

## Quick Start

### Basic Training

Train FCCA on MNIST with default settings:

```bash
python train.py --dataset mnist --num-clients 10 --num-clusters 5 --num-rounds 100
```

Train on Fashion-MNIST:

```bash
python train.py --dataset fashion_mnist --num-clients 10 --num-clusters 5 --num-rounds 100
```

### Using Configuration Files

```bash
python train.py --config configs/mnist_baseline.yaml
```

### With DVC Pipeline

Initialize DVC (first time only):

```bash
dvc init
```

Run the complete pipeline:

```bash
# Prepare data
dvc repro prepare_data

# Train MNIST
dvc repro train_mnist

# Train Fashion-MNIST
dvc repro train_fashion_mnist
```

## Project Structure

```
fcca/
├── src/fcca/                 # Main package
│   ├── models/              # Neural network models
│   │   ├── encoder.py      # Global encoder
│   │   ├── classifier.py   # Cluster-wise classifier
│   │   └── cinn.py         # Conditional INN
│   ├── federated/           # Federated learning components
│   │   ├── client.py       # Client implementation
│   │   ├── server.py       # Server implementation
│   │   └── fcca.py         # Main FCCA algorithm
│   ├── data/                # Data utilities
│   │   ├── loaders.py      # Dataset loaders
│   │   └── partition.py    # Data partitioning strategies
│   └── utils/               # Utilities
│       ├── config.py       # Configuration management
│       └── mlflow_logger.py # MLflow integration
├── configs/                 # Configuration files
│   ├── mnist_baseline.yaml
│   └── fashion_mnist_baseline.yaml
├── train.py                 # Main training script
├── dvc.yaml                 # DVC pipeline definition
└── pyproject.toml          # Project dependencies

## Algorithm Overview

FCCA operates in 4 main steps:

### Step 1: Train cINNs on Clients
Each client trains a conditional Invertible Neural Network (cINN) to learn the distribution of their local encoded features, with the global encoder frozen.

### Step 2: Train Encoders and Classifiers
Clients train both the global encoder and cluster-wise classifier using their local data with cross-entropy loss.

### Step 3: Clustering on Server
The server performs similarity assessment by:
1. Generating synthetic samples from each client's cINN
2. Computing similarity matrix based on distribution statistics
3. Applying K-Means clustering

### Step 4: Aggregation
- **Global Encoder**: Aggregated across all clients using federated averaging
- **Cluster-wise Classifiers**: Aggregated within each cluster separately

## Configuration

Key configuration parameters:

```yaml
training:
  num_rounds: 100          # Number of federated rounds
  num_clusters: 5          # Number of client clusters
  batch_size: 32           # Batch size for local training
  local_epochs: 5          # Local training epochs per round
  cinn_lr: 0.001          # Learning rate for cINN
  encoder_classifier_lr: 0.001  # Learning rate for encoder/classifier
  clustering_interval: 10  # Perform clustering every N rounds

model:
  latent_dim: 64           # Dimension of latent space
  encoder_hidden_dims: [128, 256, 512]
  classifier_hidden_dims: [256, 128]
  cinn_num_blocks: 4       # Number of coupling blocks in cINN
  cinn_hidden_dim: 128

data:
  num_clients: 10
  partition_method: clustered  # iid, non_iid, dirichlet, clustered
```

## Experiment Tracking

### MLflow

View experiments in MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open http://localhost:5000 in your browser.

### DVC

Track experiments with DVC:

```bash
# Show metrics
dvc metrics show

# Compare experiments
dvc metrics diff

# Show pipeline DAG
dvc dag
```

## Results

The implementation tracks:

- Global accuracy (averaged across all clients)
- Personalized accuracy (cluster-specific performance)
- Cluster assignments over time
- Silhouette scores for clustering quality

Results are saved to:

- `results/` directory (JSON files)
- MLflow tracking server
- DVC metrics

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
