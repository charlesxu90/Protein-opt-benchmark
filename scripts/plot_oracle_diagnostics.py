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

Output: figures/ms_oracles/oracle_diagnostics.pdf

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

# Print size (mm). Fonts and line weights come from plot_style_utils:
# 6 pt tick labels, 7 pt axis labels and panel titles, 0.3 pt axes and ticks.
FIG_WIDTH_MM = 180
FIG_HEIGHT_MM = 85
# Vertical bands in mm, top to bottom: row-1 titles, row-1 panels, row-1 x tick
# labels + x label, the shared legend, row-2 titles, row-2 panels, row-2 x tick
# labels + x label. Explicit axes placement (rather than tight_layout) is what
# lets the legend sit in its own band directly above the histogram row.
TOP_BAND_MM = 4.2
ROW1_LABEL_BAND_MM = 7.4
LEGEND_BAND_MM = 5.0
ROW2_TITLE_BAND_MM = 4.2
BOTTOM_BAND_MM = 7.4
LEFT_MM, RIGHT_PAD_MM, COL_GAP_MM = 11.0, 2.0, 9.0

# Both rows plot normalised fitness on x (oracle prediction / oracle score), so
# the two panels of a column share one fixed x range, set per dataset. The y
# axis of the calibration row is true fitness, which reaches higher than the
# predictions do, so it is fitted to the data instead of reusing the x range.
AXIS_MIN = 0.0
AXIS_STEP = 0.05
XLIM_BY_DATASET = {
    "ms_AAV": (0.0, 0.8),
    "ms_CreiLOV": (0.0, 1.0),
    "ms_PAB1": (0.0, 0.5),
}


def axis_max(*arrays):
    """Upper axis limit covering every array, rounded up to AXIS_STEP."""
    return float(np.ceil(max(a.max() for a in arrays) / AXIS_STEP) * AXIS_STEP)

MEASURED_COLOR = "#5BA4CF"  # measured test histogram
NOVEL_COLOR = "#E8746A"     # novel unmeasured histogram
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
    ap.add_argument("--hist_alpha", type=float, default=0.9,
                    help="opacity of the measured/novel histograms (default: 0.9)")
    args = ap.parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        args.device = "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    apply_nature_rcparams(DEFAULT_FIGURE_RCPARAMS)
    n_cols = len(DATASETS)
    fig = plt.figure(figsize=(FIG_WIDTH_MM * MM_TO_IN, FIG_HEIGHT_MM * MM_TO_IN))
    panel_h = (FIG_HEIGHT_MM - TOP_BAND_MM - ROW1_LABEL_BAND_MM - LEGEND_BAND_MM
               - ROW2_TITLE_BAND_MM - BOTTOM_BAND_MM) / 2
    panel_w = (FIG_WIDTH_MM - LEFT_MM - RIGHT_PAD_MM
               - (n_cols - 1) * COL_GAP_MM) / n_cols
    row1_bottom_mm = (BOTTOM_BAND_MM + panel_h + ROW2_TITLE_BAND_MM
                      + LEGEND_BAND_MM + ROW1_LABEL_BAND_MM)
    axes = np.empty((2, n_cols), dtype=object)
    for _c in range(n_cols):
        _x0 = (LEFT_MM + _c * (panel_w + COL_GAP_MM)) / FIG_WIDTH_MM
        for _r, _y0 in ((0, row1_bottom_mm), (1, BOTTOM_BAND_MM)):
            axes[_r, _c] = fig.add_axes([_x0, _y0 / FIG_HEIGHT_MM,
                                         panel_w / FIG_WIDTH_MM,
                                         panel_h / FIG_HEIGHT_MM])
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

        # ---- measured vs novel (equal N) oracle-score distribution ----
        # Computed before drawing so both rows of this column can share one
        # x range (0 -> this dataset's max, rounded up).
        rng = np.random.RandomState(SAMPLE_SEED)
        n_cmp = min(len(test_idx), MAX_CMP)
        meas_n = pred_n[rng.choice(len(pred_n), n_cmp, replace=False)] \
            if len(pred_n) > n_cmp else pred_n
        novel = sample_novel(seqs, nmuts, wt, seq_len, n_cmp, rng)
        novel_raw = predict(model, novel, seq_len, args.device) * scale + fmin
        novel_n = (novel_raw - data_min) / data_range

        xlim = XLIM_BY_DATASET.get(dataset, (AXIS_MIN, axis_max(pred_n, meas_n, novel_n)))
        ylim = (AXIS_MIN, axis_max(true_n))
        for name, vals, rng_ in (("prediction", pred_n, xlim),
                                 ("measured", meas_n, xlim),
                                 ("novel", novel_n, xlim),
                                 ("true", true_n, ylim)):
            outside = int(np.sum((vals < rng_[0]) | (vals > rng_[1])))
            if outside:
                print(f"  WARNING: {dataset} {name}: {outside}/{len(vals)} "
                      f"({outside / len(vals) * 100:.1f}%) outside {rng_}; "
                      f"data spans {vals.min():.3f} to {vals.max():.3f}")

        # Oracle prediction on x, matching the histogram row below; true fitness
        # on y, fitted to its own range.
        ax = axes[0, c]
        ax.hexbin(pred_n, true_n, gridsize=45, cmap="viridis", bins="log", mincnt=1)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        # Freeze the limits so the reference line cannot rescale the panel.
        ax.autoscale(False)
        diag = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        ax.plot(diag, diag, "r--", lw=0.8, alpha=0.8)
        ax.set_xlabel("Oracle prediction", fontsize=XLABEL_FONTSIZE)
        if c == 0:
            ax.set_ylabel("True fitness", fontsize=XLABEL_FONTSIZE)
        ax.set_title(f"{dataset.replace('ms_', '')}  (n={len(test_idx)})",
                     fontsize=TITLE_FONTSIZE)
        ax.text(0.04, 0.96,
                f"RMSE={rmse_raw:.3g}\n(norm {rmse_norm:.3f})\n$\\rho$={rho:.3f}  $R^2$={r2:.3f}",
                transform=ax.transAxes, va="top", ha="left", ma="left",
                fontsize=BASE_FONTSIZE)
        prettify_ax(ax)

        ax2 = axes[1, c]
        bins = np.linspace(min(meas_n.min(), novel_n.min()),
                           max(meas_n.max(), novel_n.max()), 50)
        ax2.hist(meas_n, bins=bins, alpha=args.hist_alpha, density=True,
                 label=f"Measured test (n={len(meas_n)})", color=MEASURED_COLOR)
        ax2.hist(novel_n, bins=bins, alpha=args.hist_alpha, density=True,
                 label=f"Novel unmeasured (n={len(novel_n)})", color=NOVEL_COLOR)
        ax2.set_xlim(*xlim)
        ax2.set_xlabel("Oracle score (normalized fitness)", fontsize=XLABEL_FONTSIZE)
        if c == 0:
            ax2.set_ylabel("Density", fontsize=XLABEL_FONTSIZE)
        ax2.set_title(f"Measured vs novel  (med {np.median(meas_n):.3g} / "
                      f"{np.median(novel_n):.3g})", fontsize=TITLE_FONTSIZE)
        prettify_ax(ax2)

    # One shared legend for the measured/novel histograms, centred in its own
    # band directly above the histogram row.
    legend_y = (BOTTOM_BAND_MM + panel_h + ROW2_TITLE_BAND_MM
                + LEGEND_BAND_MM / 2) / FIG_HEIGHT_MM
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, legend_y),
               ncol=2, frameon=False, fontsize=BASE_FONTSIZE,
               handlelength=1.4, handletextpad=0.5, columnspacing=1.6)
    # bbox_inches=None keeps the page exactly FIG_WIDTH_MM x FIG_HEIGHT_MM.
    save_figure(fig, args.out_dir, "oracle_diagnostics", bbox_inches=None)

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
