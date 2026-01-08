#!/usr/bin/env python3
"""
FCCA-Only Experiments Runner
Runs ONLY FCCA algorithm across all available datasets and model architectures.

Datasets: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, Synthetic
Models: MLP, CNN (CNN not used for Synthetic)
"""

import sys
import warnings
from pathlib import Path

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))

# Import necessary components from fcca.py
from fcca import (
    device,
    ExperimentRunner,
    mlflow,
)

warnings.filterwarnings("ignore")

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Datasets and models
DATASETS = ["mnist", "fashion_mnist", "cifar10", "cifar100", "synthetic"]
MODEL_TYPES = ["mlp", "cnn"]

# Experiment parameters
NUM_CLIENTS = 5
NUM_CLUSTERS = 5
NUM_ROUNDS = 5  # Increase to 100 for paper results
LOCAL_EPOCHS = 5  # Increase to 20 for paper results
BATCH_SIZE = 64
LEARNING_RATE = 0.01
ALPHA = 1.0  # Dirichlet alpha for non-IID data
SEED = 42

# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    print("=" * 80)
    print("FCCA-ONLY FEDERATED LEARNING EXPERIMENTS")
    print("=" * 80)
    print(f"\nUsing device: {device}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    print("\nExperiment Configuration:")
    print(f"  - Algorithm: FCCA ONLY")
    print(f"  - Clients: {NUM_CLIENTS}")
    print(f"  - Clusters: {NUM_CLUSTERS}")
    print(f"  - Rounds: {NUM_ROUNDS}")
    print(f"  - Local Epochs: {LOCAL_EPOCHS}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"  - Dirichlet Alpha: {ALPHA}")
    print(f"  - Random Seed: {SEED}")

    # Calculate total experiments
    total_experiments = 0
    for dataset in DATASETS:
        for model_type in MODEL_TYPES:
            if dataset == "synthetic" and model_type == "cnn":
                continue
            total_experiments += 1

    print(f"\nTotal FCCA experiments: {total_experiments}")
    print(
        f"(5 datasets × 2 models - 1 synthetic/CNN = {total_experiments} experiments)"
    )
    print("=" * 80)

    # Initialize experiment runner
    runner = ExperimentRunner(device=device)

    # Run experiments
    experiment_count = 0
    failed_experiments = []
    successful_experiments = []

    for dataset in DATASETS:
        for model_type in MODEL_TYPES:
            # Skip CNN for synthetic dataset
            if dataset == "synthetic" and model_type == "cnn":
                continue

            experiment_count += 1
            experiment_name = f"{dataset}_{model_type}_fcca"

            print(f"\n{'=' * 80}")
            print(f"Experiment {experiment_count}/{total_experiments}")
            print(f"{'=' * 80}")
            print(f"\nRunning: FCCA on {dataset.upper()} with {model_type.upper()}")
            print(f"{'=' * 80}")

            try:
                history = runner.run_experiment(
                    dataset_name=dataset,
                    algorithm="fcca",
                    model_type=model_type,
                    num_clients=NUM_CLIENTS,
                    num_clusters=NUM_CLUSTERS,
                    num_rounds=NUM_ROUNDS,
                    local_epochs=LOCAL_EPOCHS,
                    batch_size=BATCH_SIZE,
                    lr=LEARNING_RATE,
                    alpha=ALPHA,
                    seed=SEED,
                )

                successful_experiments.append(experiment_name)
                print(f"\n✓ Successfully completed: {experiment_name}")

                if history:
                    final_acc = history.get("test_accuracy", [0])[-1]
                    print(f"  Final Test Accuracy: {final_acc:.4f}")

            except Exception as e:
                failed_experiments.append((experiment_name, str(e)))
                print(f"\n✗ ERROR in {experiment_name}:")
                print(f"  {e}")
                import traceback

                traceback.print_exc()
                continue

    # Print summary
    print("\n" + "=" * 80)
    print("FCCA-ONLY EXPERIMENTS SUMMARY")
    print("=" * 80)
    print(f"\nTotal experiments run: {experiment_count}")
    print(f"Successful: {len(successful_experiments)}")
    print(f"Failed: {len(failed_experiments)}")

    if successful_experiments:
        print("\n✓ Successful experiments:")
        for exp in successful_experiments:
            print(f"  - {exp}")

    if failed_experiments:
        print("\n✗ Failed experiments:")
        for exp, error in failed_experiments:
            print(f"  - {exp}")
            print(f"    Error: {error[:100]}...")

    print("\n" + "=" * 80)
    print("ALL FCCA-ONLY EXPERIMENTS COMPLETED!")
    print("=" * 80)
    print(f"\nResults logged to MLflow: {mlflow.get_tracking_uri()}")
    print("View results with: mlflow ui")


if __name__ == "__main__":
    main()
