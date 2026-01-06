#!/usr/bin/env python
"""
train_GB1_oracle.py - Train fitness predictor (oracle) for GB1 dataset

This script trains a CNN-based fitness predictor on the GB1 dataset,
compatible with LatProtRL's oracle architecture.

Usage:
    python train_GB1_oracle.py --device cuda:0 --epochs 100
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from net.rew import BaseCNN
from utils.constants import ALPHABET, AATOIDX


class GB1Dataset(Dataset):
    """Dataset for GB1 sequences and fitness values."""

    def __init__(self, sequences: list, fitness: np.ndarray):
        self.sequences = sequences
        self.fitness = fitness

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Convert to one-hot encoding
        one_hot = torch.zeros(len(seq), 20)
        for i, aa in enumerate(seq):
            if aa in AATOIDX:
                one_hot[i, AATOIDX[aa]] = 1.0
        return one_hot, torch.tensor(self.fitness[idx], dtype=torch.float32)


def load_gb1_data(data_dir: str):
    """Load and preprocess GB1 data."""
    data_path = os.path.join(data_dir, 'GB1', 'data.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"GB1 data not found at {data_path}")

    df = pd.read_csv(data_path)
    sequences = df['seq'].tolist()
    fitness = df['fitness'].values

    # Normalize fitness to [0, 1]
    fitness_norm = fitness / np.max(fitness)

    return sequences, fitness_norm


def train_oracle(
    device: str = "cuda:0",
    data_dir: str = "/home/xux/Desktop/AlphaVariant/Benchmark/data",
    output_dir: str = "ckpt/GB1",
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    val_split: float = 0.1,
    test_split: float = 0.1,
):
    """Train the oracle model for GB1."""

    print("Loading GB1 data...")
    sequences, fitness = load_gb1_data(data_dir)
    print(f"  Total samples: {len(sequences)}")
    print(f"  Sequence length: {len(sequences[0])}")
    print(f"  Fitness range: [{fitness.min():.4f}, {fitness.max():.4f}]")

    # Create dataset
    dataset = GB1Dataset(sequences, fitness)

    # Split data
    n_val = int(len(dataset) * val_split)
    n_test = int(len(dataset) * test_split)
    n_train = len(dataset) - n_val - n_test

    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"  Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Create model
    model = BaseCNN(make_one_hot=False)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Training loop
    best_val_loss = float('inf')
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred.squeeze(), y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(x)

        train_loss /= len(train_set)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred.squeeze(), y)
                val_loss += loss.item() * len(x)
                val_preds.extend(y_pred.squeeze().cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        val_loss /= len(val_set)
        val_r2 = r2_score(val_targets, val_preds)
        val_spearman, _ = spearmanr(val_targets, val_preds)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"Val R2: {val_r2:.4f}, "
              f"Val Spearman: {val_spearman:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'oracle.ckpt'))
            print(f"  Saved best model (val_loss: {val_loss:.6f})")

    # Test evaluation
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'oracle.ckpt')))
    model.eval()

    test_preds = []
    test_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            test_preds.extend(y_pred.squeeze().cpu().numpy())
            test_targets.extend(y.cpu().numpy())

    test_r2 = r2_score(test_targets, test_preds)
    test_spearman, _ = spearmanr(test_targets, test_preds)
    test_mse = np.mean((np.array(test_preds) - np.array(test_targets))**2)

    print(f"Test Results:")
    print(f"  MSE: {test_mse:.6f}")
    print(f"  R2: {test_r2:.4f}")
    print(f"  Spearman: {test_spearman:.4f}")

    # Also save level-specific checkpoints (same model for now)
    torch.save(model.state_dict(), os.path.join(output_dir, 'hard.ckpt'))
    torch.save(model.state_dict(), os.path.join(output_dir, 'medium.ckpt'))
    print(f"\nSaved oracle checkpoints to {output_dir}/")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train GB1 fitness predictor (oracle)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--data_dir", type=str,
                        default="/home/xux/Desktop/AlphaVariant/Benchmark/data",
                        help="Data directory")
    parser.add_argument("--output_dir", type=str, default="ckpt/GB1", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()

    train_oracle(
        device=args.device,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
