#!/usr/bin/env python
"""
aggregate_oracle_results.py - Aggregate + plot the ms_* oracle benchmark.

Reads results_oracle/<dataset>/<method>/seed*.json, aggregates median/IQR across
seeds for the key metrics, writes a tidy CSV, and renders a summary figure
(max-fitness bars + fitness trajectories) to figures/ms_oracles/.

Usage:
    python scripts/aggregate_oracle_results.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = ["ms_AAV", "ms_CreiLOV", "ms_GFP", "ms_PAB1"]
METHODS = ["Random", "GreedyWalk", "ftMLDE", "CLADE", "ALDE"]
COLORS = {"Random": "#9e9e9e", "GreedyWalk": "#fdae61", "ftMLDE": "#2c7fb8",
          "CLADE": "#7fbc41", "ALDE": "#d7191c"}


def load(results_dir, DATASETS):
    rows, trajs = [], {}
    for d in DATASETS:
        for m in METHODS:
            files = glob.glob(os.path.join(results_dir, d, m, "seed*.json"))
            for fp in files:
                r = json.load(open(fp))
                mt = r["metrics"]
                rows.append({"dataset": d, "method": m, "seed": r["seed"],
                             "max_fitness_norm": mt["max_fitness_norm"],
                             "top128_mean_norm": mt["top128_mean_norm"],
                             "diversity_top128": mt["diversity_top128"],
                             "novelty_top128_vs_wt": mt["novelty_top128_vs_wt"],
                             "best_n_muts": mt["best_n_muts"]})
                trajs.setdefault((d, m), []).append(mt["fitness_trajectory"])
    return pd.DataFrame(rows), trajs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results_oracle")))
    ap.add_argument("--out_dir", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "figures", "ms_oracles")))
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--fig_prefix", default="oracle_benchmark")
    ap.add_argument("--title", default="ms_* oracle benchmark")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    DS = args.datasets

    df, trajs = load(args.results_dir, DS)
    if df.empty:
        print("No results found in", args.results_dir); return

    # aggregate median/IQR
    agg = (df.groupby(["dataset", "method"])
             .agg(n_seeds=("seed", "count"),
                  max_med=("max_fitness_norm", "median"),
                  max_q1=("max_fitness_norm", lambda x: x.quantile(0.25)),
                  max_q3=("max_fitness_norm", lambda x: x.quantile(0.75)),
                  top128_med=("top128_mean_norm", "median"),
                  top128_q1=("top128_mean_norm", lambda x: x.quantile(0.25)),
                  top128_q3=("top128_mean_norm", lambda x: x.quantile(0.75)),
                  div_med=("diversity_top128", "median"),
                  best_muts_med=("best_n_muts", "median"))
             .reset_index())
    csv = os.path.join(args.out_dir, f"{args.fig_prefix}_summary.csv")
    agg.to_csv(csv, index=False)
    print("saved", csv)
    print(agg.to_string(index=False))

    # figure: row1 = max-fitness bars; row2 = top-128 mean bars (median +- IQR)
    present = [m for m in METHODS if m in df["method"].unique().tolist()]
    fig, axes = plt.subplots(2, len(DS), figsize=(4.3 * len(DS), 8), squeeze=False)
    panels = [("max", "max_fitness", "max fitness (norm)"),
              ("top128", "top128_mean", "top-128 mean fitness (norm)")]
    for row, (key, _, ylabel) in enumerate(panels):
        for c, d in enumerate(DS):
            sub = agg[agg["dataset"] == d].set_index("method")
            xs = [m for m in present if m in sub.index]
            med = [sub.loc[m, f"{key}_med"] for m in xs]
            lo = [sub.loc[m, f"{key}_med"] - sub.loc[m, f"{key}_q1"] for m in xs]
            hi = [sub.loc[m, f"{key}_q3"] - sub.loc[m, f"{key}_med"] for m in xs]
            ax = axes[row, c]
            ax.bar(xs, med, yerr=[lo, hi], capsize=3,
                   color=[COLORS[m] for m in xs], alpha=0.85)
            if row == 0:
                ax.set_title(d, fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1.0)
            ax.tick_params(axis="x", rotation=30)

    fig.suptitle(f"{args.title} — max fitness (top) and top-128 mean fitness (bottom); "
                 "median ± IQR over seeds", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        p = os.path.join(args.out_dir, f"{args.fig_prefix}.{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print("saved", p)


if __name__ == "__main__":
    main()
