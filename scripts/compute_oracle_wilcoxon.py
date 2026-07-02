#!/usr/bin/env python
"""
Paired Wilcoxon signed-rank significance tests for the multi-site oracle benchmark.

All 9 methods share the same 30 seeds per dataset, so a *paired* signed-rank test is
appropriate. For each dataset x metric (max fitness, top-128 mean) we compute:

  1. Full pairwise two-sided p-value matrix (9x9), Bonferroni-corrected over the
     k(k-1)/2 = 36 unordered pairs.
  2. A "vs-best" table: the top method (by median) vs every other, one-sided
     (best > other), Bonferroni over k-1 = 8 comparisons, with median difference.

Outputs (figures/ms_oracles/):
  wilcoxon_pairwise_<metric>.csv        long-form pairwise results
  wilcoxon_vs_best.csv                  vs-best ranking (both metrics)
  wilcoxon_heatmap_<metric>.{png,pdf}   per-dataset dominance heatmaps
  wilcoxon_summary.md                   human-readable summary

Usage: python scripts/compute_oracle_wilcoxon.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

BENCH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BENCH)

from utils.plot_style_utils import (
    BASE_FONTSIZE, DEFAULT_FIGURE_RCPARAMS, TITLE_FONTSIZE, XLABEL_FONTSIZE,
    apply_nature_rcparams, save_figure,
)

OUT = os.path.join(BENCH, "figures", "ms_oracles")
DATASETS = ["ms_AAV", "ms_CreiLOV", "ms_PAB1"]
LABELS = {"ms_AAV": "AAV", "ms_CreiLOV": "CreiLOV", "ms_PAB1": "PAB1"}
METHODS = ["Random", "GreedyWalk", "ALDE", "CLADE", "ftMLDE",
           "AdaLead", "MULTIevolve", "AiCE", "EVOLVEpro", "AlphaVariant"]
METRICS = [("max_fitness_norm", "max fitness"),
           ("top128_mean_norm", "top-128 mean")]
# Only this metric's dominance heatmap is drawn; the other metric still feeds
# the pairwise/vs-best CSVs and the markdown summary.
HEATMAP_METRIC = "top128_mean_norm"
ALPHA = 0.05


def load(dataset, metric):
    """Return {method: {seed: value}} for one dataset/metric."""
    out = {}
    for m in METHODS:
        d = {}
        for fp in glob.glob(os.path.join(BENCH, "results_oracle", dataset, m, "seed*.json")):
            seed = int(os.path.basename(fp)[4:-5])
            d[seed] = json.load(open(fp))["metrics"][metric]
        if d:
            out[m] = d
    return out


def paired(a, b):
    """Aligned paired vectors over common seeds."""
    seeds = sorted(set(a) & set(b))
    return np.array([a[s] for s in seeds]), np.array([b[s] for s in seeds])


def wtest(x, y, alternative):
    """Wilcoxon signed-rank; robust to all-zero differences."""
    if np.allclose(x, y):
        return 1.0
    try:
        return wilcoxon(x, y, alternative=alternative, zero_method="wilcox").pvalue
    except ValueError:
        return 1.0


def global_heatmap_order(datasets, metric):
    """Single method order shared by every heatmap panel, so the y-tick labels
    are identical across panels and only need to be drawn once (on the left).

    Methods are ranked by their mean per-dataset median (best first), matching
    the "best at top" convention of the other figures.
    """
    medians = {}
    for ds in datasets:
        data = load(ds, metric)
        for m, d in data.items():
            medians.setdefault(m, []).append(float(np.median(list(d.values()))))
    avg = {m: float(np.mean(v)) for m, v in medians.items() if v}
    return sorted(avg, key=lambda m: avg[m], reverse=True)


def main():
    global DATASETS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=DATASETS,
                    help="subset of datasets to test (default: all four)")
    DATASETS = ap.parse_args().datasets
    os.makedirs(OUT, exist_ok=True)
    apply_nature_rcparams(DEFAULT_FIGURE_RCPARAMS)
    MM_TO_IN = 1 / 25.4
    n_pairs = len(METHODS) * (len(METHODS) - 1) // 2
    alpha_pair = ALPHA / n_pairs            # Bonferroni, full pairwise
    alpha_vsbest = ALPHA / (len(METHODS) - 1)  # Bonferroni, vs-best
    md = ["# Multi-site oracle benchmark — paired Wilcoxon (n=30 seeds)\n",
          f"Bonferroni: pairwise α={ALPHA}/{n_pairs}={alpha_pair:.2e}; "
          f"vs-best α={ALPHA}/{len(METHODS)-1}={alpha_vsbest:.2e}\n"]

    vsbest_rows = []
    for key, label in METRICS:
        pair_rows = []
        draw = (key == HEATMAP_METRIC)
        if draw:
            heat_order = global_heatmap_order(DATASETS, key)
            fig, axes = plt.subplots(
                1, len(DATASETS),
                figsize=(170 * MM_TO_IN, 45 * MM_TO_IN))
            axes = np.atleast_1d(axes)
        for c, ds in enumerate(DATASETS):
            data = load(ds, key)
            order = sorted(data, key=lambda m: np.median(list(data[m].values())), reverse=True)
            k = len(order)
            P = np.ones((k, k))      # one-sided p: row > col
            for i in range(k):
                for j in range(k):
                    if i == j:
                        continue
                    x, y = paired(data[order[i]], data[order[j]])
                    p_gt = wtest(x, y, "greater")
                    P[i, j] = p_gt
                    if i < j:
                        x2, y2 = paired(data[order[i]], data[order[j]])
                        pair_rows.append({
                            "dataset": ds, "metric": key,
                            "method_a": order[i], "method_b": order[j],
                            "median_a": float(np.median(x2)), "median_b": float(np.median(y2)),
                            "median_diff": float(np.median(x2) - np.median(y2)),
                            "p_two_sided": wtest(x2, y2, "two-sided"),
                            "sig_bonferroni": wtest(x2, y2, "two-sided") < alpha_pair,
                        })
            # vs-best (best = order[0])
            best = order[0]
            for other in order[1:]:
                x, y = paired(data[best], data[other])
                p = wtest(x, y, "greater")
                vsbest_rows.append({
                    "dataset": ds, "metric": key, "best": best, "other": other,
                    "median_best": float(np.median(x)), "median_other": float(np.median(y)),
                    "median_diff": float(np.median(x) - np.median(y)),
                    "p_greater": p, "sig_bonferroni": p < alpha_vsbest})

            if not draw:
                continue
            # heatmap: -log10(one-sided p, row>col), star if Bonferroni-sig.
            # Remap the per-dataset-ordered P matrix into the shared global order
            # so every panel uses identical row/column method positions.
            present = [m for m in heat_order if m in data]
            pos = {m: i for i, m in enumerate(order)}
            perm = [pos[m] for m in present]
            Pg = P[np.ix_(perm, perm)]
            kk = len(present)

            ax = axes[c]
            with np.errstate(divide="ignore"):
                M = -np.log10(np.clip(Pg, 1e-300, 1))
            np.fill_diagonal(M, np.nan)
            im = ax.imshow(M, cmap="viridis", vmin=0, vmax=6, aspect="auto")
            ax.set_xticks(range(kk))
            ax.set_yticks(range(kk))
            ax.set_xticklabels(present, rotation=60, ha="right", fontsize=BASE_FONTSIZE)
            # Unified y-tick labels: draw method names only on the leftmost panel.
            ax.set_yticklabels(present if c == 0 else [], fontsize=BASE_FONTSIZE)
            ax.tick_params(axis="both", labelsize=BASE_FONTSIZE)
            ax.set_title(LABELS.get(ds, ds), fontsize=TITLE_FONTSIZE)
            for i in range(kk):
                for j in range(kk):
                    if i != j and Pg[i, j] < alpha_pair:
                        ax.text(j, i, "*", ha="center", va="center", color="w",
                                fontsize=BASE_FONTSIZE)
        if draw:
            # Shared y-tick labels let the panels sit close together.
            fig.subplots_adjust(wspace=0.12)
            cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
            cbar.set_label("-log10 p", fontsize=XLABEL_FONTSIZE)
            cbar.ax.tick_params(labelsize=BASE_FONTSIZE)
            save_figure(fig, OUT, f"wilcoxon_heatmap_{key}")
            plt.close(fig)
        pd.DataFrame(pair_rows).to_csv(
            os.path.join(OUT, f"wilcoxon_pairwise_{key}.csv"), index=False)

    vb = pd.DataFrame(vsbest_rows)
    vb.to_csv(os.path.join(OUT, "wilcoxon_vs_best.csv"), index=False)

    # markdown vs-best summary
    for key, label in METRICS:
        md.append(f"\n## {label} — best method vs rest (one-sided, Bonferroni)\n")
        for ds in DATASETS:
            sub = vb[(vb.dataset == ds) & (vb.metric == key)]
            best = sub.iloc[0]["best"]
            n_sig = int(sub["sig_bonferroni"].sum())
            md.append(f"\n**{ds}** — best = **{best}** (median "
                      f"{sub.iloc[0]['median_best']:.3f}); significantly beats "
                      f"{n_sig}/{len(sub)} other methods:\n")
            md.append("| vs | Δmedian | p(>) | sig |\n|---|---|---|---|")
            for _, r in sub.iterrows():
                md.append(f"| {r['other']} | {r['median_diff']:+.3f} | "
                          f"{r['p_greater']:.1e} | {'**yes**' if r['sig_bonferroni'] else 'ns'} |")
    with open(os.path.join(OUT, "wilcoxon_summary.md"), "w") as f:
        f.write("\n".join(md))
    print("wrote wilcoxon_vs_best.csv, wilcoxon_pairwise_*.csv, "
          "wilcoxon_heatmap_*.{png,pdf}, wilcoxon_summary.md to", OUT)
    # quick console view: vs-best for top-128
    print("\n=== vs-best (top-128 mean) ===")
    t = vb[vb.metric == "top128_mean_norm"]
    for ds in DATASETS:
        sub = t[t.dataset == ds]
        sig = sub[sub.sig_bonferroni]["other"].tolist()
        ns = sub[~sub.sig_bonferroni]["other"].tolist()
        print(f"{ds}: best={sub.iloc[0]['best']} | NOT-sig-different: {ns or '(none)'}")


if __name__ == "__main__":
    main()
