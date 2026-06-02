#!/usr/bin/env python
"""
train_oracle.py - Train a learned fitness oracle (GGS/LatProtRL BaseCNN) for any
multi-site dataset.

Mirrors the GGS oracle protocol (configs/train_predictor.yaml + AAV/GFP-oracle.yaml):
    - BaseCNN, one-hot, alphabet ARNDCQEGHILKMFPSTWYV
    - MSE loss, Adam(lr=1e-4, wd=1e-4), batch 1024, <=100 epochs
    - trained on ALL variants (no fitness/distance filtering)
    - weighted sampling: w ~ 1/(target - min + 1)  (upweights low-fitness variants)

Differences from GGS (intentional, for a quality gate): we hold out a test split
and report test Spearman/R2 as a go/no-go gate before using the oracle downstream.

Fitness is min-max normalized to [0,1] for training stability (CreiLOV is ~1e4
scale); the scaler (fit_min, fit_max) is stored in the checkpoint so the landscape
can map oracle outputs back.

Usage:
    # cheap pipeline check (subsample + few epochs)
    python scripts/train_oracle.py --dataset ms_CreiLOV --smoke
    # full training
    python scripts/train_oracle.py --dataset ms_GFP --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.oracle_model import BaseCNN, encode_int, N_TOKENS


def load_data(dataset: str, data_dir: str):
    path = os.path.join(data_dir, dataset, "data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    seq_col = "seq" if "seq" in df.columns else "sequence"
    seqs = df[seq_col].astype(str).tolist()
    fitness = df["fitness"].values.astype(np.float64)
    return seqs, fitness


def train_oracle(
    dataset: str,
    data_dir: str,
    output_dir: str,
    device: str = "cuda:0",
    epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    test_split: float = 0.1,
    seed: int = 420,
    smoke: bool = False,
    max_variants: int = None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    seqs, fitness = load_data(dataset, data_dir)
    seq_len = len(seqs[0])
    if not all(len(s) == seq_len for s in seqs):
        raise ValueError(f"{dataset}: non-uniform sequence length; oracle assumes fixed L.")

    if smoke:
        epochs = min(epochs, 3)
        max_variants = max_variants or 4000
    if max_variants and len(seqs) > max_variants:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(seqs), size=max_variants, replace=False)
        seqs = [seqs[i] for i in idx]
        fitness = fitness[idx]

    n = len(seqs)
    print(f"[{dataset}] N={n}  seq_len={seq_len}  "
          f"fit[min={fitness.min():.4g} max={fitness.max():.4g}]  epochs={epochs}")

    # min-max normalize fitness to [0,1]; store scaler
    fit_min, fit_max = float(fitness.min()), float(fitness.max())
    y = (fitness - fit_min) / (fit_max - fit_min + 1e-12)

    X = encode_int(seqs, seq_len)
    y_t = torch.tensor(y, dtype=torch.float32)

    # train/test split
    perm = np.random.permutation(n)
    n_test = int(n * test_split)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    X_tr, y_tr = X[train_idx], y_t[train_idx]
    X_te, y_te = X[test_idx], y_t[test_idx]

    # GGS weighted sampling: w ~ 1/(target - min + 1)
    w = 1.0 / (y_tr.numpy() - y_tr.numpy().min() + 1.0)
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double),
                                    num_samples=len(w), replacement=True)
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size,
                              sampler=sampler)

    model = BaseCNN(n_tokens=N_TOKENS, make_one_hot=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    def evaluate(Xe, ye):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xe), 4096):
                preds.append(model(Xe[i:i + 4096].to(device)).cpu().numpy())
        p = np.concatenate(preds)
        t = ye.numpy()
        rho = spearmanr(t, p).correlation
        return rho, r2_score(t, p)

    start = datetime.now()
    best_rho = -1.0
    os.makedirs(os.path.join(output_dir, dataset), exist_ok=True)
    ckpt_path = os.path.join(output_dir, dataset, "oracle.pt")

    for epoch in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tot += loss.item() * len(xb)
        rho, r2 = evaluate(X_te, y_te)
        print(f"  epoch {epoch + 1:3d}/{epochs}  train_mse={tot / len(X_tr):.5f}  "
              f"test_spearman={rho:.4f}  test_r2={r2:.4f}")
        if rho > best_rho:
            best_rho = rho
            torch.save({
                "state_dict": model.state_dict(),
                "dataset": dataset, "seq_len": seq_len,
                "fit_min": fit_min, "fit_max": fit_max,
                "test_spearman": float(rho), "test_r2": float(r2),
                "n_train": len(X_tr), "n_test": len(X_te),
                "arch": {"n_tokens": N_TOKENS, "kernel_size": 5,
                         "input_size": 256, "make_one_hot": True},
            }, ckpt_path)

    runtime = (datetime.now() - start).total_seconds()
    gate = "PASS" if best_rho > 0.6 else "FLAG (rho<=0.6)"
    print(f"\n[{dataset}] best test Spearman={best_rho:.4f}  [{gate}]  "
          f"runtime={runtime:.1f}s  -> {ckpt_path}")

    with open(os.path.join(output_dir, dataset, "oracle_meta.json"), "w") as f:
        json.dump({"dataset": dataset, "best_test_spearman": best_rho,
                   "seq_len": seq_len, "fit_min": fit_min, "fit_max": fit_max,
                   "n": n, "epochs": epochs, "runtime_s": runtime,
                   "smoke": smoke}, f, indent=2)
    return best_rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. ms_AAV, ms_CreiLOV, ms_GFP, ms_PAB1")
    ap.add_argument("--data_dir", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data")))
    ap.add_argument("--output_dir", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "oracles")))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--smoke", action="store_true", help="subsample + few epochs to verify pipeline")
    ap.add_argument("--max_variants", type=int, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        args.device = "cpu"
    train_oracle(
        dataset=args.dataset, data_dir=args.data_dir, output_dir=args.output_dir,
        device=args.device, epochs=args.epochs, batch_size=args.batch_size,
        smoke=args.smoke, max_variants=args.max_variants,
    )


if __name__ == "__main__":
    main()
