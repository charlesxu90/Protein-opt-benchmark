#!/usr/bin/env python
"""
plot_oracle_diagnostics.py - Supplementary oracle-support figure for the ms_* oracles.

For each multi-site dataset, produces two diagnostics:
  (row 1) Calibration: oracle prediction vs true fitness on the held-out TEST split
          (the same split train_oracle.py held out: np.random.seed(420)), annotated
          with RMSE (raw fitness units), Spearman rho, and R2.
  (row 2) Generalization: oracle score distribution on measured TEST variants vs an
          EQUAL number of NOVEL variants that are NOT in the wet-lab data. Novel
          variants are drawn to match each dataset's empirical mutation-count
          distribution and per-position observed alphabet, so the comparison stays
          inside the realistic design space (not wild extrapolation).

Output: figures/ms_oracles/oracle_diagnostics.{png,pdf,svg}

Usage:
    python scripts/plot_oracle_diagnostics.py --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.oracle_model import BaseCNN, encode_int, N_TOKENS
from utils.plot_style_utils import (
    BASE_FONTSIZE, DEFAULT_FIGURE_RCPARAMS, TITLE_FONTSIZE, XLABEL_FONTSIZE,
    apply_nature_rcparams, prettify_ax, save_figure,
)

MEASURED_COLOR = "#2c7fb8"
NOVEL_COLOR = "#de2d26"
MM_TO_IN = 1 / 25.4

DATASETS = ["ms_AAV", "ms_CreiLOV", "ms_PAB1"]
TEST_SPLIT = 0.1
TRAIN_SEED = 420          # must match train_oracle.py
MAX_CMP = 10000           # cap per group for the distribution panel
SAMPLE_SEED = 7


def load_dataset(dataset, data_dir):
    df = pd.read_csv(os.path.join(data_dir, dataset, "data.csv"))
    seqs = df["seq"].astype(str).tolist()
    fitness = df["fitness"].values.astype(np.float64)
    nmuts = df["n_muts"].values.astype(int) if "n_muts" in df.columns else None
    wt_path = os.path.join(data_dir, dataset, "wt.fasta")
    wt = "".join(l.strip() for l in open(wt_path)
                 if l.strip() and not l.startswith(">"))
    return seqs, fitness, nmuts, wt


def reproduce_test_idx(n):
    """Recreate the held-out test indices exactly as train_oracle.py did."""
    np.random.seed(TRAIN_SEED)
    perm = np.random.permutation(n)
    return perm[:int(n * TEST_SPLIT)]


def load_oracle(dataset, oracle_dir, device):
    ckpt = torch.load(os.path.join(oracle_dir, dataset, "oracle.pt"),
                      map_location=device)
    model = BaseCNN(n_tokens=N_TOKENS, make_one_hot=True).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["fit_min"], ckpt["fit_max"], ckpt["seq_len"]


def predict(model, seqs, seq_len, device, bs=4096):
    X = encode_int(seqs, seq_len)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(X[i:i + bs].to(device)).cpu().numpy())
    return np.concatenate(out)


def per_position_alphabet(seqs, seq_len):
    cols = [set() for _ in range(seq_len)]
    for s in seqs:
        for j, a in enumerate(s):
            cols[j].add(a)
    return cols


def sample_novel(seqs, nmuts, wt, seq_len, n_sample, rng):
    """Sample novel (unmeasured) variants matching the empirical mutation-count
    distribution and per-position observed alphabet."""
    measured = set(seqs)
    cols = per_position_alphabet(seqs, seq_len)
    varying = [j for j in range(seq_len) if len(cols[j]) > 1]
    # per-position mutation choices (observed AAs other than WT)
    choices = {j: [a for a in cols[j] if a != wt[j]] for j in varying}
    choices = {j: v for j, v in choices.items() if v}
    var_positions = list(choices.keys())
    if nmuts is not None:
        k_pool = nmuts[nmuts >= 1]
    else:
        k_pool = np.array([len(varying)])
    novel, seen, attempts = [], set(), 0
    max_attempts = n_sample * 50
    while len(novel) < n_sample and attempts < max_attempts:
        attempts += 1
        k = int(rng.choice(k_pool))
        k = max(1, min(k, len(var_positions)))
        pos = rng.choice(var_positions, size=k, replace=False)
        s = list(wt)
        for p in pos:
            s[p] = rng.choice(choices[p])
        cand = "".join(s)
        if cand in measured or cand in seen or cand == wt:
            continue
        seen.add(cand)
        novel.append(cand)
    return novel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data")))
    ap.add_argument("--oracle_dir", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "oracles")))
    ap.add_argument("--out_dir", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "figures", "ms_oracles")))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        args.device = "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    apply_nature_rcparams(DEFAULT_FIGURE_RCPARAMS)
    fig, axes = plt.subplots(2, len(DATASETS),
                             figsize=(170 * MM_TO_IN, 90 * MM_TO_IN))
    summary = []

    for c, dataset in enumerate(DATASETS):
        seqs, fitness, nmuts, wt = load_dataset(dataset, args.data_dir)
        n = len(seqs)
        model, fmin, fmax, seq_len = load_oracle(dataset, args.oracle_dir, args.device)
        scale = (fmax - fmin)

        # ---- held-out test calibration ----
        test_idx = reproduce_test_idx(n)
        test_seqs = [seqs[i] for i in test_idx]
        true_raw = fitness[test_idx]
        pred_norm = predict(model, test_seqs, seq_len, args.device)
        pred_raw = pred_norm * scale + fmin

        rmse_raw = float(np.sqrt(np.mean((pred_raw - true_raw) ** 2)))
        rmse_norm = rmse_raw / (scale + 1e-12)
        rho = spearmanr(true_raw, pred_raw).correlation
        r2 = r2_score(true_raw, pred_raw)
        summary.append((dataset, rmse_raw, rmse_norm, rho, r2, len(test_idx)))

        # Normalize to [0,1] using the original dataset's fitness range so every
        # panel shares a 0-1 scale (this is the same min-max the oracle was fit
        # with; see train_oracle.py).
        data_min, data_max = float(fitness.min()), float(fitness.max())
        data_range = data_max - data_min + 1e-12
        true_n = (true_raw - data_min) / data_range
        pred_n = (pred_raw - data_min) / data_range

        ax = axes[0, c]
        ax.hexbin(true_n, pred_n, gridsize=45, cmap="viridis", bins="log", mincnt=1)
        lo = min(true_n.min(), pred_n.min())
        hi = max(true_n.max(), pred_n.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=0.8, alpha=0.8)
        ax.set_xlabel("True fitness", fontsize=XLABEL_FONTSIZE)
        if c == 0:
            ax.set_ylabel("Oracle prediction", fontsize=XLABEL_FONTSIZE)
        ax.set_title(f"{dataset.replace('ms_', '')}  (n={len(test_idx)})",
                     fontsize=TITLE_FONTSIZE)
        ax.text(0.04, 0.96,
                f"RMSE={rmse_raw:.3g}\n(norm {rmse_norm:.3f})\n$\\rho$={rho:.3f}  $R^2$={r2:.3f}",
                transform=ax.transAxes, va="top", ha="left", ma="left",
                fontsize=BASE_FONTSIZE)
        prettify_ax(ax)

        # ---- measured vs novel (equal N) oracle-score distribution ----
        rng = np.random.RandomState(SAMPLE_SEED)
        n_cmp = min(len(test_idx), MAX_CMP)
        meas_n = pred_n[rng.choice(len(pred_n), n_cmp, replace=False)] \
            if len(pred_n) > n_cmp else pred_n
        novel = sample_novel(seqs, nmuts, wt, seq_len, n_cmp, rng)
        novel_raw = predict(model, novel, seq_len, args.device) * scale + fmin
        novel_n = (novel_raw - data_min) / data_range

        ax2 = axes[1, c]
        bins = np.linspace(min(meas_n.min(), novel_n.min()),
                           max(meas_n.max(), novel_n.max()), 50)
        ax2.hist(meas_n, bins=bins, alpha=0.6, density=True,
                 label=f"Measured test (n={len(meas_n)})", color=MEASURED_COLOR)
        ax2.hist(novel_n, bins=bins, alpha=0.6, density=True,
                 label=f"Novel unmeasured (n={len(novel_n)})", color=NOVEL_COLOR)
        ax2.set_xlabel("Oracle score (normalized fitness)", fontsize=XLABEL_FONTSIZE)
        if c == 0:
            ax2.set_ylabel("Density", fontsize=XLABEL_FONTSIZE)
        ax2.set_title(f"Measured vs novel  (med {np.median(meas_n):.3g} / "
                      f"{np.median(novel_n):.3g})", fontsize=TITLE_FONTSIZE)
        prettify_ax(ax2)

    # One shared legend for the measured/novel histograms, placed above the
    # figure (up and outside the panels) so it never overlaps the bars.
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=BASE_FONTSIZE)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, args.out_dir, "oracle_diagnostics")

    print("\n=== oracle test-set summary ===")
    print(f"{'dataset':12s} {'RMSE(raw)':>10s} {'RMSE(norm)':>11s} {'spearman':>9s} {'R2':>7s} {'n_test':>7s}")
    sm = pd.DataFrame(summary, columns=["dataset", "rmse_raw", "rmse_norm",
                                        "spearman", "r2", "n_test"])
    for _, r in sm.iterrows():
        print(f"{r['dataset']:12s} {r['rmse_raw']:10.4g} {r['rmse_norm']:11.4f} "
              f"{r['spearman']:9.4f} {r['r2']:7.4f} {int(r['n_test']):7d}")
    sm.to_csv(os.path.join(args.out_dir, "oracle_test_metrics.csv"), index=False)
    print(f"\nsaved {os.path.join(args.out_dir, 'oracle_test_metrics.csv')}")


if __name__ == "__main__":
    main()
