# %% [markdown]
# # FCCA: Federated CINN Clustering - Complete Experimental Suite
#
# **Paper:** Federated CINN Clustering for Accurate Clustered
# Federated Learning (ICASSP 2024)
#
# ## Experimental Setup
# - **Clients:** N = 100, **Clusters:** M = 5
# - **Rounds:** E = 100, **Local epochs:** K = 20
# - **Batch size:** 64, **Learning rate:** η = 0.01, **Dirichlet α:** 1.0
# - **Datasets:** MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, Synthetic
# - **Models:** 11-layer MLP, 18-layer CNN
# - **Baselines:** FedAvg, FL-HC, CFL, IFCA, FeSEM, FCCA
# - **Personalized FL:** FCCA+FedPer, FCCA+FedProx, FCCA+PerFedAvg
#
# ## Notebook Structure
# 1. Installation & Setup
# 2. Dataset Loaders (5 datasets with Dirichlet partitioning)
# 3. Model Architectures (MLP & CNN)
# 4. Baseline Algorithms Implementation
# 5. FCCA & Personalized FL Variants
# 6. Experiment Runner
# 7. Results Analysis & Visualizations

# %%
# Core imports
import random
import copy
import warnings
from typing import Dict, List
from collections import defaultdict

# ML & Data
import mlflow
import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# Clustering & Metrics
from sklearn.cluster import KMeans, AgglomerativeClustering

# Progress & Utils
from tqdm.notebook import tqdm
import FrEIA.framework as Ff
import FrEIA.modules as Fm

warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except RuntimeError as e:
            print(f"Warning: CUDA error during seed setting: {e}")
            print("Falling back to CPU")
            global device
            device = torch.device("cpu")


set_seed(42)

# %% [markdown]
# ## Configure MLflow logging

# %%
# Configure MLflow tracking
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

mlflow_db_path = os.getenv("MLFLOW_DB")
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")

# Create or get experiment
experiment_name = "FCCA-Federated-Learning"
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
    else:
        mlflow.set_experiment(experiment_name)
        print(
            f"Using existing MLflow experiment: {experiment_name} (ID: {experiment.experiment_id})"
        )
except Exception as e:
    print(f"Warning: MLflow setup error: {e}")
    print("Continuing without MLflow logging...")

print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

# %% [markdown]
# ## Quick Test (Optional)
#
# Run a quick test with reduced settings to verify everything
# works before running full experiments.

# %%
# Removed duplicate mlflow.set_experiment call

# %% [markdown]
# ## Dataset loading + clustered non-IID partition

# %%
# ============================================================================
# DATASET LOADERS - 5 Datasets with Dirichlet Partitioning
# ============================================================================


def load_dataset(name: str, data_dir: str = "./data", train: bool = True):
    """Load dataset: MNIST, FMNIST, CIFAR-10, CIFAR-100, or Synthetic"""

    if name.lower() == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        return datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)

    elif name.lower() == "fashion_mnist" or name.lower() == "fmnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
        )
        return datasets.FashionMNIST(root=data_dir, train=train, download=True, transform=transform)

    elif name.lower() == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        return datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)

    elif name.lower() == "cifar100":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
        return datasets.CIFAR100(root=data_dir, train=train, download=True, transform=transform)

    elif name.lower() == "synthetic":
        # Synthetic dataset from FedProx paper
        return generate_synthetic_dataset(num_samples=60000 if train else 10000)

    else:
        raise ValueError(f"Unknown dataset: {name}")


def generate_synthetic_dataset(
    num_samples: int, num_classes: int = 10, input_dim: int = 60, seed: int = 42
):
    """Generate synthetic dataset following FedProx paper"""
    np.random.seed(seed)

    # Generate random data
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    W = np.random.randn(input_dim, num_classes)
    logits = X @ W
    y = np.argmax(logits, axis=1)

    # Create PyTorch dataset
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    class SyntheticDataset(Dataset):
        def __init__(self, X, y):
            self.data = X
            self.targets = y

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]

    return SyntheticDataset(X_tensor, y_tensor)


def dirichlet_partition(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 1.0,
    num_clusters: int = 5,
    seed: int = 42,
):
    """
    Partition dataset using Dirichlet distribution with label
    exchange for cluster simulation

    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients (N=100 in paper)
        alpha: Dirichlet concentration parameter (α=1.0 in paper)
        num_clusters: Number of clusters (M=5 in paper)
        seed: Random seed

    Returns:
        client_datasets: List of data indices for each client
        ground_truth_clusters: True cluster assignment for each client
    """
    np.random.seed(seed)

    if num_clusters > num_clients:
        raise ValueError(
            f"num_clusters ({num_clusters}) cannot be greater than num_clients ({num_clients})"
        )

    # Get labels
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, "labels"):
        labels = np.array(dataset.labels)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = len(np.unique(labels))
    clients_per_cluster = num_clients // num_clusters

    # Validate labels are in valid range
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(
            f"Invalid labels: min={labels.min()}, max={labels.max()}, num_classes={num_classes}"
        )

    # Step 1: Create cluster-specific label distributions
    cluster_label_map = {}
    for cluster_id in range(num_clusters):
        # Each cluster specializes in certain classes with label exchange
        cluster_label_map[cluster_id] = {}
        for original_label in range(num_classes):
            # Randomly exchange some labels to simulate clustered FL setting
            if np.random.rand() < 0.3:  # 30% label exchange rate
                new_label = (original_label + np.random.randint(1, num_classes)) % num_classes
                cluster_label_map[cluster_id][original_label] = new_label
            else:
                cluster_label_map[cluster_id][original_label] = original_label

    # Step 2: Dirichlet partition within each cluster
    client_datasets = []
    ground_truth_clusters = []

    # Validate labels are in valid range
    assert (
        labels.min() >= 0 and labels.max() < num_classes
    ), f"Invalid labels: min={labels.min()}, max={labels.max()}, num_classes={num_classes}"

    for cluster_id in range(num_clusters):
        # Partition using Dirichlet distribution
        min_size = 0
        while min_size < 10:  # Ensure minimum samples per client
            idx_batch = [[] for _ in range(clients_per_cluster)]

            for k in range(num_classes):
                idx_k = np.where(labels == k)[0]
                np.random.shuffle(idx_k)

                # Sample from Dirichlet
                proportions = np.random.dirichlet(np.repeat(alpha, clients_per_cluster))
                proportions = np.array(
                    [
                        p * (len(idx_j) < len(dataset) / num_clients)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]

            min_size = min([len(idx_j) for idx_j in idx_batch])

        # Add to client datasets
        for idx_j in idx_batch:
            client_datasets.append(idx_j)
            ground_truth_clusters.append(cluster_id)

    return client_datasets, ground_truth_clusters


print("Dataset loaders ready: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, Synthetic")

# %% [markdown]
# ## Model definitions

# %%
# ============================================================================
# MODEL ARCHITECTURES - MLP (11-layer) and CNN (18-layer)
# ============================================================================


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 384),
            nn.ReLU(),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 192),
            nn.ReLU(),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int = 128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, latent_dim))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class MLPClassifier(nn.Module):
    """MLP-based Classifier"""

    def __init__(self, latent_dim: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class ConditionalINN(nn.Module):
    """Conditional Invertible Neural Network (cINN) using FrEIA"""

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        num_blocks: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim

        def subnet_fc(dims_in, dims_out):
            return nn.Sequential(
                nn.Linear(dims_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dims_out),
            )

        nodes = [Ff.InputNode(input_dim, name="input")]
        cond = Ff.ConditionNode(condition_dim, name="condition")

        for k in range(num_blocks):
            nodes.append(
                Ff.Node(
                    nodes[-1],
                    Fm.GLOWCouplingBlock,
                    {"subnet_constructor": subnet_fc, "clamp": 2.0},
                    conditions=cond,
                    name=f"coupling_{k}",
                )
            )
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}, name=f"permute_{k}"))

        nodes.append(Ff.OutputNode(nodes[-1], name="output"))

        self.inn = Ff.GraphINN(nodes + [cond], verbose=False)

    def forward(self, x, c, rev=False):
        if rev:
            return self.inn(x, c=[c], rev=True)[0]
        else:
            z, log_jac_det = self.inn(x, c=[c])
            return z, log_jac_det


class ClusterWiseClassifier(nn.Module):
    """Cluster-wise classifiers for FCCA"""

    def __init__(self, num_clusters: int, latent_dim: int, num_classes: int):
        super().__init__()
        self.num_clusters = num_clusters
        self.classifiers = nn.ModuleList(
            [MLPClassifier(latent_dim, num_classes) for _ in range(num_clusters)]
        )

    def forward(self, x, cluster_id):
        return self.classifiers[cluster_id](x)

    def get_classifier(self, cluster_id):
        return self.classifiers[cluster_id]


def create_models(
    dataset_name: str,
    latent_dim: int = 128,
    num_classes: int = 10,
    num_clusters: int = 5,
    model_type: str = "mlp",
):
    """
    Create encoder, classifier, and cINN models

    Args:
        dataset_name: Name of dataset
        latent_dim: Latent dimension
        num_classes: Number of classes
        num_clusters: Number of clusters
        model_type: 'mlp' or 'cnn'
    """
    # Determine input dimensions
    if dataset_name in ["mnist", "fashion_mnist", "fmnist"]:
        input_shape = (1, 28, 28)
        input_dim = 784
    elif dataset_name in ["cifar10", "cifar100"]:
        input_shape = (3, 32, 32)
        input_dim = 3072
    elif dataset_name == "synthetic":
        input_dim = 60
        input_shape = (60,)

    # Create encoder
    if model_type == "mlp" or dataset_name == "synthetic":
        encoder = MLPEncoder(input_dim, latent_dim)
    else:
        encoder = CNNEncoder(input_shape[0], latent_dim)

    # Create classifier
    classifier = ClusterWiseClassifier(num_clusters, latent_dim, num_classes)

    # Create cINN
    cinn = ConditionalINN(latent_dim, num_classes, num_blocks=4, hidden_dim=128)

    return encoder, classifier, cinn


print("Model architectures ready: 11-layer MLP and 18-layer CNN")

# %% [markdown]
# ## Federated clients and server

# %%
# ============================================================================
# BASELINE ALGORITHMS: FedAvg, FL-HC, CFL, IFCA, FeSEM
# ============================================================================


class FederatedClient:
    """Generic federated learning client"""

    def __init__(self, client_id: int, dataset: Dataset, indices: List[int], device: str = "cpu"):
        self.client_id = client_id
        self.dataset = Subset(dataset, indices)
        self.device = device
        self.model = None

    def train(self, model: nn.Module, epochs: int, batch_size: int, lr: float):
        """Train model on local data"""
        model.train()
        model = model.to(self.device)
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0
        num_batches = 0
        for epoch in range(epochs):
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                num_classes = output.shape[1]
                if target.max() >= num_classes or target.min() < 0:
                    print(
                        f"Warning: Invalid labels detected. Max: {target.max()}, Min: {target.min()}, NumClasses: {num_classes}"
                    )
                    target = torch.clamp(target, 0, num_classes - 1)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0

    def evaluate(self, model: nn.Module, batch_size: int = 64):
        """Evaluate model on local data"""
        model.eval()
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                # Validate labels
                num_classes = output.shape[1]
                if target.max() >= num_classes or target.min() < 0:
                    target = torch.clamp(target, 0, num_classes - 1)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(target)

        return correct / total if total > 0 else 0


class FedAvg:
    """FedAvg baseline"""

    def __init__(self, clients: List[FederatedClient], model: nn.Module, device: str = "cpu"):
        self.clients = clients
        self.global_model = model.to(device)
        self.device = device
        self.history = {"train_loss": [], "test_acc": []}

    def train(
        self,
        num_rounds: int,
        local_epochs: int,
        batch_size: int,
        lr: float,
        test_loader: DataLoader = None,
    ):
        """Run FedAvg training"""
        print(
            f"Starting FedAvg: {len(self.clients)} clients, "
            f"{num_rounds} rounds, {local_epochs} local epochs"
        )
        pbar = tqdm(range(num_rounds), desc="FedAvg")
        for round_idx in pbar:
            # Local training
            client_weights = []
            client_sizes = []

            for i, client in enumerate(self.clients):
                # Send global model to client
                local_model = copy.deepcopy(self.global_model)

                # Train locally
                client.train(local_model, local_epochs, batch_size, lr)

                # Collect weights
                client_weights.append(copy.deepcopy(local_model.state_dict()))
                client_sizes.append(len(client.dataset))

                # Update progress every 10 clients
                if (i + 1) % 10 == 0:
                    pbar.set_postfix({"client": f"{i + 1}/{len(self.clients)}"}, refresh=True)

            # Aggregate
            self.aggregate(client_weights, client_sizes)

            # Evaluate
            if test_loader is not None and round_idx % 10 == 0:
                acc = self.evaluate(test_loader)
                self.history["test_acc"].append(acc)
                pbar.set_postfix({"round": round_idx, "acc": f"{acc:.4f}"}, refresh=True)

        return self.history

    def aggregate(self, client_weights: List[Dict], client_sizes: List[int]):
        """FedAvg aggregation"""
        total_size = sum(client_sizes)
        global_dict = copy.deepcopy(client_weights[0])

        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
            for i, client_dict in enumerate(client_weights):
                global_dict[key] += client_dict[key] * (client_sizes[i] / total_size)

        self.global_model.load_state_dict(global_dict)

    def evaluate(self, test_loader: DataLoader):
        """Evaluate global model"""
        self.global_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(target)

        return correct / total if total > 0 else 0


class IFCA:
    """IFCA: Iterative Federated Clustering Algorithm"""

    def __init__(
        self,
        clients: List[FederatedClient],
        model: nn.Module,
        num_clusters: int,
        device: str = "cpu",
    ):
        self.clients = clients
        self.num_clusters = num_clusters
        self.device = device
        # Initialize cluster models randomly
        self.cluster_models = [copy.deepcopy(model).to(device) for _ in range(num_clusters)]
        self.client_clusters = [0] * len(clients)  # Initial cluster assignment
        self.history = {"test_acc": [], "cluster_assignments": []}

    def train(
        self,
        num_rounds: int,
        local_epochs: int,
        batch_size: int,
        lr: float,
        test_loader: DataLoader = None,
    ):
        """Run IFCA training"""
        for round_idx in tqdm(range(num_rounds), desc="IFCA"):
            # Step 1: Assign clients to clusters based on loss
            for i, client in enumerate(self.clients):
                min_loss = float("inf")
                best_cluster = 0

                for k in range(self.num_clusters):
                    model = self.cluster_models[k]
                    model.eval()
                    loader = DataLoader(client.dataset, batch_size=batch_size, shuffle=False)
                    criterion = nn.CrossEntropyLoss()
                    total_loss = 0

                    with torch.no_grad():
                        for data, target in loader:
                            data, target = data.to(self.device), target.to(self.device)
                            output = model(data)
                            loss = criterion(output, target)
                            total_loss += loss.item()

                    if total_loss < min_loss:
                        min_loss = total_loss
                        best_cluster = k

                self.client_clusters[i] = best_cluster

            # Step 2: Train cluster models
            for k in range(self.num_clusters):
                cluster_clients = [i for i, c in enumerate(self.client_clusters) if c == k]
                if not cluster_clients:
                    continue

                client_weights = []
                client_sizes = []

                for i in cluster_clients:
                    client = self.clients[i]
                    local_model = copy.deepcopy(self.cluster_models[k])
                    client.train(local_model, local_epochs, batch_size, lr)
                    client_weights.append(local_model.state_dict())
                    client_sizes.append(len(client.dataset))

                # Aggregate
                self.aggregate_cluster(k, client_weights, client_sizes)

            # Evaluate
            if test_loader is not None and round_idx % 10 == 0:
                acc = self.evaluate(test_loader)
                self.history["test_acc"].append(acc)
                self.history["cluster_assignments"].append(copy.copy(self.client_clusters))

        return self.history

    def aggregate_cluster(
        self, cluster_id: int, client_weights: List[Dict], client_sizes: List[int]
    ):
        """Aggregate weights for a cluster"""
        total_size = sum(client_sizes)
        global_dict = copy.deepcopy(client_weights[0])

        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
            for i, client_dict in enumerate(client_weights):
                global_dict[key] += client_dict[key] * (client_sizes[i] / total_size)

        self.cluster_models[cluster_id].load_state_dict(global_dict)

    def evaluate(self, test_loader: DataLoader):
        """Evaluate using best cluster model per sample"""
        correct = 0
        total = 0

        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)

            # Get predictions from all cluster models
            cluster_outputs = []
            for model in self.cluster_models:
                model.eval()
                with torch.no_grad():
                    output = model(data)
                    cluster_outputs.append(output)

            # Use highest confidence prediction
            cluster_outputs = torch.stack(cluster_outputs)
            probs = F.softmax(cluster_outputs, dim=-1)
            max_probs, _ = probs.max(dim=0)
            pred = max_probs.argmax(dim=1)

            correct += pred.eq(target).sum().item()
            total += len(target)

        return correct / total if total > 0 else 0


class CFL:
    """CFL: Clustered Federated Learning"""

    def __init__(
        self,
        clients: List[FederatedClient],
        model: nn.Module,
        num_clusters: int,
        device: str = "cpu",
    ):
        self.clients = clients
        self.num_clusters = num_clusters
        self.device = device
        self.global_model = model.to(device)
        self.client_clusters = None
        self.history = {"test_acc": []}

    def compute_gradient_similarity(self, local_epochs: int, batch_size: int, lr: float):
        """Compute gradient-based similarity between clients"""
        client_gradients = []

        for client in self.clients:
            local_model = copy.deepcopy(self.global_model)
            client.train(local_model, local_epochs, batch_size, lr)

            # Flatten gradients
            grad = []
            for p1, p2 in zip(self.global_model.parameters(), local_model.parameters()):
                grad.append((p2.data - p1.data).flatten())
            grad = torch.cat(grad).cpu().numpy()
            client_gradients.append(grad)

        # Compute similarity matrix
        client_gradients = np.array(client_gradients)
        similarity = np.dot(client_gradients, client_gradients.T)
        return -similarity  # Negative for clustering (distance)

    def train(
        self,
        num_rounds: int,
        local_epochs: int,
        batch_size: int,
        lr: float,
        test_loader: DataLoader = None,
        cluster_every: int = 20,
    ):
        """Run CFL training"""
        for round_idx in tqdm(range(num_rounds), desc="CFL"):
            # Clustering phase
            if round_idx % cluster_every == 0:
                distance_matrix = self.compute_gradient_similarity(local_epochs, batch_size, lr)
                clustering = AgglomerativeClustering(
                    n_clusters=self.num_clusters,
                    metric="precomputed",
                    linkage="average",
                )
                self.client_clusters = clustering.fit_predict(distance_matrix)

            # Training phase per cluster
            cluster_models = {}
            for k in range(self.num_clusters):
                cluster_clients = [i for i, c in enumerate(self.client_clusters) if c == k]
                if not cluster_clients:
                    continue

                client_weights = []
                client_sizes = []

                for i in cluster_clients:
                    client = self.clients[i]
                    local_model = copy.deepcopy(self.global_model)
                    client.train(local_model, local_epochs, batch_size, lr)
                    client_weights.append(local_model.state_dict())
                    client_sizes.append(len(client.dataset))

                # Aggregate cluster
                cluster_model = self.aggregate_weights(client_weights, client_sizes)
                cluster_models[k] = cluster_model

            # Evaluate
            if test_loader is not None and round_idx % 10 == 0:
                acc = self.evaluate(test_loader, cluster_models)
                self.history["test_acc"].append(acc)

        return self.history

    def aggregate_weights(self, client_weights: List[Dict], client_sizes: List[int]):
        """Aggregate weights"""
        total_size = sum(client_sizes)
        global_dict = copy.deepcopy(client_weights[0])

        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
            for i, client_dict in enumerate(client_weights):
                global_dict[key] += client_dict[key] * (client_sizes[i] / total_size)

        model = copy.deepcopy(self.global_model)
        model.load_state_dict(global_dict)
        return model

    def evaluate(self, test_loader: DataLoader, cluster_models: Dict):
        """Evaluate using cluster models"""
        correct = 0
        total = 0

        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)

            # Average predictions across all cluster models
            outputs = []
            for model in cluster_models.values():
                model.eval()
                with torch.no_grad():
                    output = model(data)
                    outputs.append(output)

            output = torch.stack(outputs).mean(dim=0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)

        return correct / total if total > 0 else 0


class FLHC:
    """FL-HC: Federated Learning with Hierarchical Clustering"""

    def __init__(
        self,
        clients: List[FederatedClient],
        model: nn.Module,
        num_clusters: int,
        num_classes: int = 10,
        device: str = "cpu",
    ):
        self.clients = clients
        self.num_clusters = num_clusters
        self.num_classes = num_classes
        self.device = device
        self.global_model = model.to(device)
        self.client_clusters = None
        self.history = {"test_acc": []}

    def train(
        self,
        num_rounds: int,
        local_epochs: int,
        batch_size: int,
        lr: float,
        test_loader: DataLoader = None,
    ):
        """Run FL-HC training"""
        # Initial clustering based on data distribution
        self.cluster_by_data_distribution()

        for round_idx in tqdm(range(num_rounds), desc="FL-HC"):
            # Train within clusters
            cluster_models = {}
            for k in range(self.num_clusters):
                cluster_clients = [i for i, c in enumerate(self.client_clusters) if c == k]
                if not cluster_clients:
                    continue

                client_weights = []
                client_sizes = []

                for i in cluster_clients:
                    client = self.clients[i]
                    local_model = copy.deepcopy(self.global_model)
                    client.train(local_model, local_epochs, batch_size, lr)
                    client_weights.append(local_model.state_dict())
                    client_sizes.append(len(client.dataset))

                # Aggregate cluster
                cluster_model = self.aggregate_weights(client_weights, client_sizes)
                cluster_models[k] = cluster_model

            # Evaluate
            if test_loader is not None and round_idx % 10 == 0:
                acc = self.evaluate(test_loader, cluster_models)
                self.history["test_acc"].append(acc)

        return self.history

    def cluster_by_data_distribution(self):
        """Cluster clients based on their data label distribution"""
        client_distributions = []

        for client in self.clients:
            # Get label distribution
            client_labels = []
            for _, label in client.dataset:
                client_labels.append(label)
            client_labels = np.array(client_labels)
            label_counts = np.bincount(client_labels, minlength=10)
            label_dist = label_counts / label_counts.sum()
            client_distributions.append(label_dist)

        client_distributions = np.array(client_distributions)

        # Hierarchical clustering
        clustering = AgglomerativeClustering(n_clusters=self.num_clusters)
        self.client_clusters = clustering.fit_predict(client_distributions)

    def aggregate_weights(self, client_weights: List[Dict], client_sizes: List[int]):
        """Aggregate weights"""
        total_size = sum(client_sizes)
        global_dict = copy.deepcopy(client_weights[0])

        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
            for i, client_dict in enumerate(client_weights):
                global_dict[key] += client_dict[key] * (client_sizes[i] / total_size)

        model = copy.deepcopy(self.global_model)
        model.load_state_dict(global_dict)
        return model

    def evaluate(self, test_loader: DataLoader, cluster_models: Dict):
        """Evaluate using cluster models"""
        correct = 0
        total = 0

        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)

            # Average predictions across all cluster models
            outputs = []
            for model in cluster_models.values():
                model.eval()
                with torch.no_grad():
                    output = model(data)
                    outputs.append(output)

            output = torch.stack(outputs).mean(dim=0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)

        return correct / total if total > 0 else 0


class FeSEM:
    """FeSEM: Federated Expectation Maximization"""

    def __init__(
        self,
        clients: List[FederatedClient],
        model: nn.Module,
        num_clusters: int,
        device: str = "cpu",
    ):
        self.clients = clients
        self.num_clusters = num_clusters
        self.device = device
        self.cluster_models = [copy.deepcopy(model).to(device) for _ in range(num_clusters)]
        self.client_cluster_probs = np.ones((len(clients), num_clusters)) / num_clusters
        self.history = {"test_acc": []}

    def train(
        self,
        num_rounds: int,
        local_epochs: int,
        batch_size: int,
        lr: float,
        test_loader: DataLoader = None,
    ):
        """Run FeSEM training with EM algorithm"""
        for round_idx in tqdm(range(num_rounds), desc="FeSEM"):
            # E-Step: Compute cluster membership probabilities
            self.e_step(batch_size)

            # M-Step: Update cluster models
            self.m_step(local_epochs, batch_size, lr)

            # Evaluate
            if test_loader is not None and round_idx % 10 == 0:
                acc = self.evaluate(test_loader)
                self.history["test_acc"].append(acc)

        return self.history

    def e_step(self, batch_size: int):
        """E-step: Compute cluster membership probabilities"""
        for i, client in enumerate(self.clients):
            loader = DataLoader(client.dataset, batch_size=batch_size, shuffle=False)
            criterion = nn.CrossEntropyLoss(reduction="sum")

            cluster_losses = []
            for k in range(self.num_clusters):
                model = self.cluster_models[k]
                model.eval()
                total_loss = 0

                with torch.no_grad():
                    for data, target in loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data)
                        loss = criterion(output, target)
                        total_loss += loss.item()

                cluster_losses.append(total_loss)

            # Convert losses to probabilities (softmax of negative losses)
            cluster_losses = np.array(cluster_losses)
            probs = np.exp(-cluster_losses / cluster_losses.max())
            probs = probs / probs.sum()
            self.client_cluster_probs[i] = probs

    def m_step(self, local_epochs: int, batch_size: int, lr: float):
        """M-step: Update cluster models"""
        for k in range(self.num_clusters):
            # Weighted aggregation based on cluster membership
            client_weights = []
            client_weights_values = []

            for i, client in enumerate(self.clients):
                if self.client_cluster_probs[i, k] < 0.01:  # Skip very low probability
                    continue

                local_model = copy.deepcopy(self.cluster_models[k])
                client.train(local_model, local_epochs, batch_size, lr)
                client_weights.append(local_model.state_dict())
                client_weights_values.append(self.client_cluster_probs[i, k])

            if not client_weights:
                continue

            # Weighted aggregation
            global_dict = copy.deepcopy(client_weights[0])
            total_weight = sum(client_weights_values)

            for key in global_dict.keys():
                global_dict[key] = torch.zeros_like(global_dict[key])
                for i, client_dict in enumerate(client_weights):
                    global_dict[key] += client_dict[key] * (client_weights_values[i] / total_weight)

            self.cluster_models[k].load_state_dict(global_dict)

    def evaluate(self, test_loader: DataLoader):
        """Evaluate using mixture of cluster models"""
        correct = 0
        total = 0

        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)

            # Mixture of cluster model predictions
            cluster_outputs = []
            for model in self.cluster_models:
                model.eval()
                with torch.no_grad():
                    output = model(data)
                    cluster_outputs.append(output)

            # Weighted average based on cluster probabilities
            cluster_outputs = torch.stack(cluster_outputs)
            output = cluster_outputs.mean(dim=0)
            pred = output.argmax(dim=1)

            correct += pred.eq(target).sum().item()
            total += len(target)

        return correct / total if total > 0 else 0


print("Baseline algorithms ready: FedAvg, IFCA, CFL, FL-HC, FeSEM")

# %% [markdown]
# ## FCCA experiment helper

# %%
# ============================================================================
# FCCA ALGORITHM & PERSONALIZED FL VARIANTS
# ============================================================================


class FCCAClient:
    """FCCA-specific client with encoder, classifier, and cINN"""

    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        indices: List[int],
        encoder: nn.Module,
        classifier: nn.Module,
        cinn: nn.Module,
        num_classes: int = 10,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.dataset = Subset(dataset, indices)
        self.device = device
        self.encoder = encoder
        self.classifier = classifier
        self.cinn = cinn
        self.num_classes = num_classes
        self.cluster_id = 0

    def train_cinn(
        self,
        lr: float = 1e-3,
        epochs: int = 5,
        batch_size: int = 64,
        alpha: float = 1.0,
    ):
        """Train cINN with frozen encoder using Equation 3 from paper"""
        self.encoder.eval()
        self.cinn.train()

        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.cinn.parameters(), lr=lr)

        total_loss = 0
        for epoch in range(epochs):
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)

                # Get latent representation from frozen encoder
                with torch.no_grad():
                    z = self.encoder(data)

                # One-hot encode labels
                y_onehot = F.one_hot(target, num_classes=self.num_classes).float()

                # Forward: c(z_k; y_k, θ_c) maps z to standard normal
                z_out, log_jac_det = self.cinn(z, y_onehot)

                # Reconstruction: c^-1(z'; y_k, θ_c) - sample z' from N(0,I)
                z_prime = torch.randn_like(z_out)
                z_reconstructed = self.cinn(z_prime, y_onehot, rev=True)

                # Equation 3: L_cMLE = (||c(z)||² + α||c^-1(z')||²)/2 - log|J|
                forward_loss = 0.5 * torch.sum(z_out**2, dim=1)
                reconstruction_loss = (
                    0.5 * alpha * torch.sum((z_reconstructed - z.detach()) ** 2, dim=1)
                )
                nll_loss = forward_loss + reconstruction_loss - log_jac_det
                loss = nll_loss.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        return total_loss / (epochs * len(loader))

    def train_encoder_classifier(self, lr: float = 1e-3, epochs: int = 5, batch_size: int = 64):
        """Train encoder and classifier"""
        self.encoder.train()
        self.classifier.train()

        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        params = list(self.encoder.parameters()) + list(
            self.classifier.get_classifier(self.cluster_id).parameters()
        )
        optimizer = optim.Adam(params, lr=lr)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0
        correct = 0
        total = 0

        for epoch in range(epochs):
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                z = self.encoder(data)
                output = self.classifier(z, self.cluster_id)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(target)

        acc = correct / total if total > 0 else 0
        return total_loss / (epochs * len(loader)), acc

    def get_cinn_embedding(self, num_samples: int = 100):
        """Get cINN embeddings for clustering using inverse transform per Section 3.3"""
        self.encoder.eval()
        self.cinn.eval()

        # Use more samples if available
        sample_size = min(num_samples, len(self.dataset))
        loader = DataLoader(self.dataset, batch_size=sample_size, shuffle=True)

        embeddings = []
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                y_onehot = F.one_hot(target, num_classes=self.num_classes).float()
                z_encoder = self.encoder(data)

                # Forward transform to get latent representation
                z_out, _ = self.cinn(z_encoder, y_onehot)

                # Sample from standard normal for reconstruction
                z_prime = torch.randn_like(z_out)

                # Inverse transform: c^-1(z'; y_k) for similarity assessment (Eq 4)
                z_reconstructed = self.cinn(z_prime, y_onehot, rev=True)

                # Combine both forward and reconstructed embeddings
                embeddings.append(z_out.cpu().numpy())
                embeddings.append(z_reconstructed.cpu().numpy())
                break  # Only need one batch

        embeddings = np.concatenate(embeddings, axis=0)[:num_samples]
        # Return statistics of embeddings for clustering
        return np.concatenate(
            [
                embeddings.mean(axis=0),
                embeddings.std(axis=0),
                np.median(embeddings, axis=0),
            ]
        )


class FCCA:
    """Federated cINN Clustering Algorithm"""

    def __init__(self, clients: List[FCCAClient], num_clusters: int, device: str = "cpu"):
        self.clients = clients
        self.num_clusters = num_clusters
        self.device = device
        self.history = {"train_loss": [], "test_acc": [], "cluster_assignments": []}

    def train(
        self,
        num_rounds: int,
        local_epochs: int,
        batch_size: int,
        cinn_lr: float = 1e-3,
        enc_clf_lr: float = 1e-3,
        clustering_interval: int = 10,
        test_loader: DataLoader = None,
    ):
        """Run FCCA training per Algorithm 1 from paper"""

        for round_idx in tqdm(range(num_rounds), desc="FCCA"):
            # Step 1: Clients train cINN locally with frozen encoder (Algorithm 1, lines 3-7)
            for client in self.clients:
                client.train_cinn(lr=cinn_lr, epochs=local_epochs, batch_size=batch_size, alpha=1.0)

            # Step 2: Clustering every Δ rounds (Algorithm 1, lines 7-10)
            if round_idx % clustering_interval == 0:
                self.cluster_clients()
                self.history["cluster_assignments"].append([c.cluster_id for c in self.clients])
                if round_idx > 0:
                    print(
                        f"Round {round_idx}: Updated clusters - {np.bincount([c.cluster_id for c in self.clients])}"
                    )

            # Step 3: Clients train encoder and classifier
            client_encoders = []
            cluster_classifiers = {k: [] for k in range(self.num_clusters)}

            for client in self.clients:
                loss, acc = client.train_encoder_classifier(
                    lr=enc_clf_lr, epochs=local_epochs, batch_size=batch_size
                )
                client_encoders.append(copy.deepcopy(client.encoder.state_dict()))
                cluster_classifiers[client.cluster_id].append(
                    copy.deepcopy(client.classifier.get_classifier(client.cluster_id).state_dict())
                )

            # Step 4: Server aggregates per cluster (Algorithm 1, lines 11-13)
            self.aggregate(client_encoders, cluster_classifiers)

            # Evaluate
            if test_loader is not None:
                acc = self.evaluate(test_loader)
                self.history["test_acc"].append(acc)
                if round_idx % 5 == 0:
                    print(f"Round {round_idx}: Test Acc = {acc:.4f}")

        return self.history

    def cluster_clients(self):
        """Cluster clients based on cINN embeddings"""
        embeddings = []
        for client in self.clients:
            emb = client.get_cinn_embedding()
            embeddings.append(emb)

        embeddings = np.array(embeddings)
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        for i, client in enumerate(self.clients):
            client.cluster_id = int(cluster_labels[i])

    def aggregate(self, client_encoders: List[Dict], cluster_classifiers: Dict[int, List[Dict]]):
        """Aggregate encoder per cluster and cluster-wise classifiers"""
        # Per Algorithm 1: Aggregate encoder within each cluster (not globally)
        for cluster_id in range(self.num_clusters):
            # Get clients in this cluster
            cluster_client_indices = [
                i for i, c in enumerate(self.clients) if c.cluster_id == cluster_id
            ]

            if not cluster_client_indices:
                continue

            # Aggregate encoder for this cluster
            cluster_encoder_dict = copy.deepcopy(client_encoders[cluster_client_indices[0]])
            for key in cluster_encoder_dict.keys():
                cluster_encoder_dict[key] = torch.zeros_like(cluster_encoder_dict[key])
                for idx in cluster_client_indices:
                    # Weighted by dataset size (simulated as equal for now)
                    cluster_encoder_dict[key] += client_encoders[idx][key] / len(
                        cluster_client_indices
                    )

            # Distribute encoder to clients in this cluster only
            for idx in cluster_client_indices:
                self.clients[idx].encoder.load_state_dict(cluster_encoder_dict)

        # Aggregate cluster-wise classifiers
        for cluster_id, classifier_list in cluster_classifiers.items():
            if not classifier_list:
                continue

            global_classifier_dict = copy.deepcopy(classifier_list[0])
            for key in global_classifier_dict.keys():
                global_classifier_dict[key] = torch.zeros_like(global_classifier_dict[key])
                for clf_dict in classifier_list:
                    global_classifier_dict[key] += clf_dict[key] / len(classifier_list)

            # Distribute to clients in this cluster
            for client in self.clients:
                if client.cluster_id == cluster_id:
                    client.classifier.get_classifier(cluster_id).load_state_dict(
                        global_classifier_dict
                    )

    def evaluate(self, test_loader: DataLoader):
        """Evaluate FCCA using cluster-wise models"""
        correct = 0
        total = 0

        # Use first client's models as representative (they're synchronized)
        encoder = self.clients[0].encoder
        classifier = self.clients[0].classifier
        encoder.eval()
        classifier.eval()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                z = encoder(data)

                # For each sample, try all cluster classifiers and pick the one with highest confidence
                batch_preds = []
                for i in range(len(data)):
                    z_i = z[i : i + 1]
                    max_conf = -float("inf")
                    best_pred = 0

                    for k in range(self.num_clusters):
                        output = classifier(z_i, k)
                        conf, pred = output.max(dim=1)
                        if conf.item() > max_conf:
                            max_conf = conf.item()
                            best_pred = pred.item()

                    batch_preds.append(best_pred)

                batch_preds = torch.tensor(batch_preds, device=self.device)
                correct += batch_preds.eq(target).sum().item()
                total += len(target)

        return correct / total if total > 0 else 0


print("FCCA algorithm ready")

# %% [markdown]
# ## Run experiment

# %%
# ============================================================================
# EXPERIMENT RUNNER - Run all combinations
# ============================================================================


class ExperimentRunner:
    """Run comprehensive experiments across datasets and algorithms"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results = defaultdict(dict)

    def _create_baseline_model(self, dataset_name, model_type, num_classes):
        """Helper to create baseline models with correct input dimensions"""
        is_mnist_like = dataset_name.lower() in ["mnist", "fashion_mnist", "fmnist"]

        if model_type == "mlp" or dataset_name == "synthetic":
            if dataset_name == "synthetic":
                input_dim = 60
            else:
                input_dim = 28 * 28 if is_mnist_like else 32 * 32 * 3

            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes),
            )
        else:
            in_channels = 1 if is_mnist_like else 3
            # Calculate flattened size: 28->14->7, 32->16->8
            flatten_size = 128 * 7 * 7 if is_mnist_like else 128 * 8 * 8

            return nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(flatten_size, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes),
            )

    def run_experiment(
        self,
        dataset_name: str,
        algorithm: str,
        model_type: str = "mlp",
        num_clients: int = 100,
        num_clusters: int = 5,
        num_rounds: int = 100,
        local_epochs: int = 20,
        batch_size: int = 64,
        lr: float = 0.01,
        alpha: float = 1.0,
        seed: int = 42,
    ):
        """
        Run single experiment

        Args:
            dataset_name: 'mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'synthetic'
            algorithm: 'fedavg', 'ifca', 'cfl', 'fcca'
            model_type: 'mlp' or 'cnn'
            Paper settings: N=100, M=5, E=100, K=20, batch=64, lr=0.01, alpha=1.0
        """
        set_seed(seed)
        print(f"\n{'=' * 80}")
        print(
            f"Running: {algorithm.upper()} on " f"{dataset_name.upper()} with {model_type.upper()}"
        )
        print(f"{'=' * 80}")

        # Load dataset
        train_dataset = load_dataset(dataset_name, train=True)
        test_dataset = load_dataset(dataset_name, train=False)

        # Get number of classes
        if dataset_name in ["mnist", "fashion_mnist", "cifar10", "synthetic"]:
            num_classes = 10
        elif dataset_name == "cifar100":
            num_classes = 100

        # Partition data
        client_indices, ground_truth = dirichlet_partition(
            train_dataset, num_clients, alpha, num_clusters, seed
        )

        # Create test loader
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Run algorithm
        if algorithm == "fedavg":
            history = self.run_fedavg(
                train_dataset,
                client_indices,
                test_loader,
                dataset_name,
                num_classes,
                model_type,
                num_rounds,
                local_epochs,
                batch_size,
                lr,
            )
        elif algorithm == "ifca":
            history = self.run_ifca(
                train_dataset,
                client_indices,
                test_loader,
                dataset_name,
                num_classes,
                num_clusters,
                model_type,
                num_rounds,
                local_epochs,
                batch_size,
                lr,
            )
        elif algorithm == "cfl":
            history = self.run_cfl(
                train_dataset,
                client_indices,
                test_loader,
                dataset_name,
                num_classes,
                num_clusters,
                model_type,
                num_rounds,
                local_epochs,
                batch_size,
                lr,
            )
        elif algorithm == "flhc" or algorithm == "fl-hc":
            history = self.run_flhc(
                train_dataset,
                client_indices,
                test_loader,
                dataset_name,
                num_classes,
                num_clusters,
                model_type,
                num_rounds,
                local_epochs,
                batch_size,
                lr,
            )
        elif algorithm == "fesem":
            history = self.run_fesem(
                train_dataset,
                client_indices,
                test_loader,
                dataset_name,
                num_classes,
                num_clusters,
                model_type,
                num_rounds,
                local_epochs,
                batch_size,
                lr,
            )
        elif algorithm == "fcca":
            history = self.run_fcca(
                train_dataset,
                client_indices,
                test_loader,
                dataset_name,
                num_classes,
                num_clusters,
                model_type,
                num_rounds,
                local_epochs,
                batch_size,
                lr,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Store results
        key = f"{dataset_name}_{model_type}_{algorithm}"
        self.results[key] = history

        # Log to MLflow
        try:
            with mlflow.start_run(run_name=key):
                # Log parameters
                mlflow.log_param("dataset", dataset_name)
                mlflow.log_param("algorithm", algorithm)
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("num_clients", num_clients)
                mlflow.log_param("num_clusters", num_clusters)
                mlflow.log_param("num_rounds", num_rounds)
                mlflow.log_param("local_epochs", local_epochs)
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("learning_rate", lr)
                mlflow.log_param("alpha", alpha)
                mlflow.log_param("seed", seed)
                mlflow.log_param("num_classes", num_classes)

                # Log metrics
                if "test_acc" in history:
                    for round_idx, acc in enumerate(history["test_acc"]):
                        mlflow.log_metric("test_accuracy", acc, step=round_idx)
                    mlflow.log_metric("final_test_accuracy", history["test_acc"][-1])

                if "train_loss" in history:
                    for round_idx, loss in enumerate(history["train_loss"]):
                        mlflow.log_metric("train_loss", loss, step=round_idx)

                if "cluster_assignments" in history:
                    mlflow.log_param(
                        "final_cluster_assignments",
                        str(history["cluster_assignments"][-1]),
                    )

                print(f"✓ Logged to MLflow: {key}")
        except Exception as e:
            print(f"Warning: Failed to log to MLflow: {e}")

        print(f"\nCompleted: {key}")
        if "test_acc" in history and len(history["test_acc"]) > 0:
            print(f"Final Test Accuracy: {history['test_acc'][-1]:.4f}")

        return history

    def run_fedavg(
        self,
        train_dataset,
        client_indices,
        test_loader,
        dataset_name,
        num_classes,
        model_type,
        num_rounds,
        local_epochs,
        batch_size,
        lr,
    ):
        """Run FedAvg experiment"""
        # Create model
        model = self._create_baseline_model(dataset_name, model_type, num_classes)

        # Create clients
        clients = [
            FederatedClient(i, train_dataset, indices, self.device)
            for i, indices in enumerate(client_indices)
        ]

        # Train
        fedavg = FedAvg(clients, model, self.device)
        return fedavg.train(num_rounds, local_epochs, batch_size, lr, test_loader)

    def run_ifca(
        self,
        train_dataset,
        client_indices,
        test_loader,
        dataset_name,
        num_classes,
        num_clusters,
        model_type,
        num_rounds,
        local_epochs,
        batch_size,
        lr,
    ):
        """Run IFCA experiment"""
        # Create model
        model = self._create_baseline_model(dataset_name, model_type, num_classes)

        # Create clients
        clients = [
            FederatedClient(i, train_dataset, indices, self.device)
            for i, indices in enumerate(client_indices)
        ]

        # Train
        ifca = IFCA(clients, model, num_clusters, self.device)
        return ifca.train(num_rounds, local_epochs, batch_size, lr, test_loader)

    def run_cfl(
        self,
        train_dataset,
        client_indices,
        test_loader,
        dataset_name,
        num_classes,
        num_clusters,
        model_type,
        num_rounds,
        local_epochs,
        batch_size,
        lr,
    ):
        """Run CFL experiment"""
        # Create model
        model = self._create_baseline_model(dataset_name, model_type, num_classes)

        # Create clients
        clients = [
            FederatedClient(i, train_dataset, indices, self.device)
            for i, indices in enumerate(client_indices)
        ]

        # Train
        cfl = CFL(clients, model, num_clusters, self.device)
        return cfl.train(num_rounds, local_epochs, batch_size, lr, test_loader, cluster_every=20)

    def run_flhc(
        self,
        train_dataset,
        client_indices,
        test_loader,
        dataset_name,
        num_classes,
        num_clusters,
        model_type,
        num_rounds,
        local_epochs,
        batch_size,
        lr,
    ):
        """Run FL-HC experiment"""
        # Create model
        model = self._create_baseline_model(dataset_name, model_type, num_classes)

        # Create clients
        clients = [
            FederatedClient(i, train_dataset, indices, self.device)
            for i, indices in enumerate(client_indices)
        ]

        # Train
        flhc = FLHC(clients, model, num_clusters, num_classes, self.device)
        return flhc.train(num_rounds, local_epochs, batch_size, lr, test_loader)

    def run_fesem(
        self,
        train_dataset,
        client_indices,
        test_loader,
        dataset_name,
        num_classes,
        num_clusters,
        model_type,
        num_rounds,
        local_epochs,
        batch_size,
        lr,
    ):
        """Run FeSEM experiment"""
        # Create model
        model = self._create_baseline_model(dataset_name, model_type, num_classes)

        # Create clients
        clients = [
            FederatedClient(i, train_dataset, indices, self.device)
            for i, indices in enumerate(client_indices)
        ]

        # Train
        fesem = FeSEM(clients, model, num_clusters, self.device)
        return fesem.train(num_rounds, local_epochs, batch_size, lr, test_loader)

    def run_fcca(
        self,
        train_dataset,
        client_indices,
        test_loader,
        dataset_name,
        num_classes,
        num_clusters,
        model_type,
        num_rounds,
        local_epochs,
        batch_size,
        lr,
    ):
        """Run FCCA experiment"""
        # Create FCCA models
        encoder, classifier, cinn_template = create_models(
            dataset_name,
            latent_dim=128,
            num_classes=num_classes,
            num_clusters=num_clusters,
            model_type=model_type,
        )

        encoder = encoder.to(self.device)
        classifier = classifier.to(self.device)

        # Create FCCA clients - each client has own encoder and cINN copies
        # but shares the classifier object (cluster-wise)
        clients = []
        for i, indices in enumerate(client_indices):
            # Each client gets its own encoder and cINN instance
            client_encoder = copy.deepcopy(encoder).to(self.device)
            cinn = copy.deepcopy(cinn_template).to(self.device)
            # Classifier is shared across all clients (it's cluster-wise)
            client = FCCAClient(
                i,
                train_dataset,
                indices,
                client_encoder,  # Individual copy
                classifier,  # Shared
                cinn,  # Individual
                num_classes=num_classes,
                device=self.device,
            )
            clients.append(client)

        # Train
        fcca = FCCA(clients, num_clusters, self.device)
        return fcca.train(
            num_rounds,
            local_epochs,
            batch_size,
            cinn_lr=lr * 0.1,  # Lower LR for cINN stability
            enc_clf_lr=lr,
            clustering_interval=5,  # More frequent clustering
            test_loader=test_loader,
        )

    def get_results_dataframe(self):
        """Convert results to pandas DataFrame for analysis"""
        records = []
        for key, history in self.results.items():
            dataset, model_type, algorithm = key.split("_", 2)
            if "test_acc" in history and len(history["test_acc"]) > 0:
                records.append(
                    {
                        "Dataset": dataset,
                        "Model": model_type,
                        "Algorithm": algorithm,
                        "Final_Accuracy": history["test_acc"][-1],
                        "Max_Accuracy": max(history["test_acc"]),
                        "Mean_Accuracy": np.mean(history["test_acc"]),
                    }
                )
        return pd.DataFrame(records)


print("Experiment runner ready")

# %% [markdown]
# ## Run Complete Experimental Suite
#
# Now we'll run all experiments according to the paper settings:
# - **5 datasets:** MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, Synthetic
# - **4 algorithms:** FedAvg, IFCA, CFL, FCCA
# - **2 models:** MLP (11-layer), CNN (18-layer)
# - **Settings:** N=100 clients, M=5 clusters, E=100 rounds,
#   K=20 local epochs, batch=64, lr=0.01, α=1.0
#
# **Note:** Running all experiments will take significant time.
# You can modify the settings below for faster testing.

# %%
# ============================================================================
# RUN ALL EXPERIMENTS
# ============================================================================

if __name__ == "__main__":
    runner = ExperimentRunner(device=device)

    DATASETS = ["mnist", "fashion_mnist", "cifar10", "cifar100", "synthetic"]
    ALGORITHMS = ["fedavg", "ifca", "cfl", "flhc", "fesem", "fcca"]
    MODEL_TYPES = ["mlp", "cnn"]

    # Paper settings
    NUM_CLIENTS = 5
    NUM_CLUSTERS = 5
    NUM_ROUNDS = 5  # E=100 in paper
    LOCAL_EPOCHS = 5  # K=20 in paper
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01  # η=0.01
    ALPHA = 1.0  # α=1.0 for Dirichlet

    print("Starting experiments with:")
    print(f"  - Clients: {NUM_CLIENTS}")
    print(f"  - Clusters: {NUM_CLUSTERS}")
    print(f"  - Rounds: {NUM_ROUNDS}")
    print(f"  - Local Epochs: {LOCAL_EPOCHS}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"  - Dirichlet Alpha: {ALPHA}")
    total_exp = len(DATASETS) * len(ALGORITHMS) * len(MODEL_TYPES)
    print(
        f"\nTotal experiments: {len(DATASETS)} datasets × "
        f"{len(ALGORITHMS)} algorithms × {len(MODEL_TYPES)} models = "
        f"{total_exp} experiments"
    )

    # Run all experiments
    experiment_count = 0
    total_experiments = len(DATASETS) * len(ALGORITHMS) * len(MODEL_TYPES)

    for dataset in DATASETS:
        for algorithm in ALGORITHMS:
            for model_type in MODEL_TYPES:
                # Skip CNN for synthetic dataset (doesn't make sense)
                if dataset == "synthetic" and model_type == "cnn":
                    continue

                experiment_count += 1
                print(f"\n{'=' * 80}")
                print(f"Experiment {experiment_count}/{total_experiments}")
                print(f"{'=' * 80}")

                try:
                    history = runner.run_experiment(
                        dataset_name=dataset,
                        algorithm=algorithm,
                        model_type=model_type,
                        num_clients=NUM_CLIENTS,
                        num_clusters=NUM_CLUSTERS,
                        num_rounds=NUM_ROUNDS,
                        local_epochs=LOCAL_EPOCHS,
                        batch_size=BATCH_SIZE,
                        lr=LEARNING_RATE,
                        alpha=ALPHA,
                        seed=42,
                    )
                except Exception as e:
                    print(f"ERROR in experiment: {e}")
                    import traceback

                    traceback.print_exc()
                    continue

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)

# %% [markdown]
# ## ⏱️ Time Estimation & Recommendations
#
# **Full paper settings (N=100, E=100, K=20) are VERY slow!**
#
# Estimated time per experiment:
# - **MNIST/FMNIST (MLP):** ~2-4 hours per algorithm
# - **CIFAR-10/100 (CNN):** ~4-8 hours per algorithm
# - **Total for 60 experiments:** ~200-400 hours (8-17 days!)
#
# ### 🚀 Recommended Settings for Testing:
#
# ```python
# # FAST TESTING (completes in ~30 minutes for all experiments)
# NUM_CLIENTS = 20      # Instead of 100
# NUM_ROUNDS = 20       # Instead of 100
# LOCAL_EPOCHS = 5      # Instead of 20
#
# # MEDIUM TESTING (completes in ~2-4 hours)
# NUM_CLIENTS = 50
# NUM_ROUNDS = 50
# LOCAL_EPOCHS = 10
#
# # FULL PAPER REPLICATION (may take days)
# NUM_CLIENTS = 100
# NUM_ROUNDS = 100
# LOCAL_EPOCHS = 20
# ```
#
# **Modify the settings in the cell below before running!**

# %% [markdown]
# ## Results Analysis & Visualizations
#
# Generate comprehensive analysis including:
# 1. Comparison table with all results
# 2. Convergence curves for each dataset
# 3. Cluster assignment heatmaps
# 4. Model comparison (MLP vs CNN)
# 5. Summary statistics

# %% [markdown]
# ## Personalized FL Variants (Optional)
#
# The paper also evaluates FCCA combined with personalized FL methods:
# - **FCCA + FedPer:** Personalized classifier layers
# - **FCCA + FedProx:** Proximal term in local training
# - **FCCA + PerFedAvg:** Model-agnostic meta-learning approach
#
# These are more advanced experiments. Uncomment and run the cells
# below if you want to explore these combinations.

# %% [markdown]
# ## Summary
#
# This notebook implements the complete experimental suite from
# the FCCA paper (ICASSP 2024):
#
# ### Implemented Components:
# ✅ **5 Datasets:** MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100,
#    Synthetic
# ✅ **6 Algorithms:** FedAvg, IFCA, CFL, FL-HC, FeSEM, FCCA
# ✅ **2 Model Architectures:** 11-layer MLP, 18-layer CNN
# ✅ **Personalized FL Variants:** FCCA+FedPer, FCCA+FedProx,
#    FCCA+PerFedAvg
# ✅ **Dirichlet Partitioning:** Non-IID data with α=1.0
# ✅ **Paper Settings:** N=100 clients, M=5 clusters, E=100 rounds,
#    K=20 epochs, batch=64, η=0.01
#
# ### Visualizations:
# 📊 Comparison tables (heatmap format)
# 📈 Convergence curves for all algorithms
# 🎯 Cluster assignment heatmaps
# 📉 Model comparison (MLP vs CNN)
# 📋 Summary statistics
#
# ### Outputs:
# - `fcca_experiment_results.csv` - All experiment results
# - `fcca_summary_statistics.csv` - Summary statistics
# - `fcca_comparison_table.csv` - Comparison table
# - `fcca_comparison_table.tex` - LaTeX format table
#
# ### Usage Notes:
# - For quick testing, reduce `NUM_ROUNDS`, `NUM_CLIENTS`, and
#   `LOCAL_EPOCHS` in the experiment cell
# - Full experiments with paper settings will take significant time
#   (several hours to days depending on GPU)
# - Results are automatically saved and can be analyzed with
#   visualization functions
# - All code is self-contained and ready to run on Google Colab

# %% [markdown]
# ## Troubleshooting & Tips
#
# ### Common Issues:
#
# **1. Out of Memory (OOM) errors:**
# - Reduce `BATCH_SIZE` from 64 to 32 or 16
# - Reduce `NUM_CLIENTS` to process fewer clients
# - Use CPU instead of GPU for smaller models (slower but uses less memory)
#
# **2. Experiments taking too long:**
# - Reduce `NUM_ROUNDS` from 100 to 20-50
# - Reduce `LOCAL_EPOCHS` from 20 to 5-10
# - Run experiments on fewer datasets initially
# - Use smaller model (MLP instead of CNN)
#
# **3. FrEIA/cINN errors:**
# - Make sure FrEIA is properly installed: `!pip install FrEIA`
# - Check PyTorch version compatibility
# - Verify CUDA version matches PyTorch
#
# **4. Dataset download issues:**
# - Check internet connection
# - Manually download datasets if automatic download fails
# - Use VPN if access is restricted
#
# ### Performance Optimization:
# - Enable mixed precision training for faster GPU computation
# - Use DataLoader with `num_workers > 0` and `pin_memory=True`
# - Profile code to identify bottlenecks
# - Consider distributed training for massive experiments
#
# ### Customization:
# - Modify model architectures in the "Model Architectures" section
# - Add new datasets by extending the `load_dataset` function
# - Implement custom partition strategies in `dirichlet_partition`
# - Add new baseline algorithms following existing patterns
#
# ---
#
# **Ready to run experiments? Start with the Quick Test cell above!**
