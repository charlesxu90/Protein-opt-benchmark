#!/usr/bin/env python
"""
Paired Wilcoxon signed-rank significance tests for the benchmark methods.

  --task multisite : multi-site oracle benchmark (AAV, CreiLOV, PAB1)  [default]
  --task 4site     : four-site benchmark (GB1, PhoQ, TrpB)

All 9 methods share the same 30 seeds per dataset, so a *paired* signed-rank test is
appropriate. For each dataset x metric (max fitness, top-128 mean) we compute:

  1. Full pairwise two-sided p-value matrix (9x9), Bonferroni-corrected over the
     k(k-1)/2 = 36 unordered pairs.
  2. A "vs-best" table: the top method (by median) vs every other, one-sided
     (best > other), Bonferroni over k-1 = 8 comparisons, with median difference.

Outputs (the task's figure directory):
  wilcoxon_pairwise_<metric>.csv        long-form pairwise results
  wilcoxon_vs_best.csv                  vs-best ranking (both metrics)
  wilcoxon_heatmap_<metric>.pdf         per-dataset dominance heatmaps (both metrics)
  wilcoxon_summary.md                   human-readable summary

Usage: python scripts/compute_oracle_wilcoxon.py [--task 4site]
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
from utils.seed_values import load_seeds

_FIGDIR = os.path.join(BENCH, "figures")

# metrics: (generic key for utils.seed_values, file/label key, human label,
#           label drawn under the panels)
TASKS = {
    "multisite": {
        "outdir": os.path.join(_FIGDIR, "ms_oracles"),
        "datasets": ["ms_AAV", "ms_CreiLOV", "ms_PAB1"],
        "labels": {"ms_AAV": "AAV", "ms_CreiLOV": "CreiLOV", "ms_PAB1": "PAB1"},
        "methods": ["Random", "GreedyWalk", "ALDE", "CLADE", "ftMLDE",
                    "AdaLead", "MULTIevolve", "AiCE", "EVOLVEpro", "AlphaVariant"],
        "metrics": [("max_fitness", "max_fitness_norm", "max fitness",
                     "Max fitness"),
                    ("top128", "top128_mean_norm", "top-128 mean",
                     "Top-128 fitness")],
        "console_metric": "top128_mean_norm",  # metric echoed to stdout
        "title": "Multi-site oracle benchmark",
    },
    "4site": {
        "outdir": _FIGDIR,
        "datasets": ["4site_GB1", "4site_PhoQ", "4site_TRPB"],
        "labels": {"4site_GB1": "GB1", "4site_PhoQ": "PhoQ", "4site_TRPB": "TrpB"},
        # FLEXS is the same AdaLead algorithm; displayed under that name.
        "methods": ["Random", "GreedyWalk", "ALDE", "CLADE", "ftMLDE",
                    "FLEXS", "MULTIevolve", "AiCE", "EVOLVEpro", "AlphaVariant"],
        "metrics": [("max_fitness", "max_fitness", "max fitness",
                     "Max fitness"),
                    ("top128", "top128", "top-128 median",
                     "Top-128 fitness")],
        "console_metric": "top128",  # metric echoed to stdout
        "title": "Four-site benchmark",
    },
}
DISPLAY_NAMES = {"FLEXS": "AdaLead"}
ALPHA = 0.05

# Print size, matching the other main figures.
FIG_WIDTH_MM = 89
FIG_HEIGHT_MM = 35
HEAT_FONTSIZE = 6.0        # method labels, column numbers, significance stars
METRIC_LABEL_PT = 7.0      # the metric name under the panels
TOP_BAND_MM = 3.4          # dataset titles
BOTTOM_BAND_MM = 7.2       # column numbers + the metric label
METRIC_LABEL_Y_MM = 1.9    # metric label centre, mm from the page bottom


def load(dataset, metric, cfg, task):
    """Return {method: {seed: value}} for one dataset/metric."""
    out = {}
    for m in cfg["methods"]:
        d = load_seeds(m, dataset, task, metric=metric)
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


def global_heatmap_order(datasets, metric, cfg, task):
    """Single method order shared by every heatmap panel, so the y-tick labels
    are identical across panels and only need to be drawn once (on the left).

    Methods are ranked by their mean per-dataset median (best first), matching
    the "best at top" convention of the other figures.
    """
    medians = {}
    for ds in datasets:
        data = load(ds, metric, cfg, task)
        for m, d in data.items():
            medians.setdefault(m, []).append(float(np.median(list(d.values()))))
    avg = {m: float(np.mean(v)) for m, v in medians.items() if v}
    return sorted(avg, key=lambda m: avg[m], reverse=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=list(TASKS), default="multisite",
                    help="which benchmark to test (default: multisite)")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="subset of datasets to test (default: the task's three)")
    args = ap.parse_args()
    task = args.task
    cfg = TASKS[task]
    DATASETS = args.datasets or cfg["datasets"]
    METHODS = cfg["methods"]
    OUT = cfg["outdir"]
    LABELS = cfg["labels"]
    CONSOLE_METRIC = cfg["console_metric"]
    os.makedirs(OUT, exist_ok=True)
    apply_nature_rcparams(DEFAULT_FIGURE_RCPARAMS)
    MM_TO_IN = 1 / 25.4
    n_pairs = len(METHODS) * (len(METHODS) - 1) // 2
    alpha_pair = ALPHA / n_pairs            # Bonferroni, full pairwise
    alpha_vsbest = ALPHA / (len(METHODS) - 1)  # Bonferroni, vs-best
    md = [f"# {cfg['title']} — paired Wilcoxon (n=30 seeds)\n",
          f"Bonferroni: pairwise α={ALPHA}/{n_pairs}={alpha_pair:.2e}; "
          f"vs-best α={ALPHA}/{len(METHODS)-1}={alpha_vsbest:.2e}\n"]

    vsbest_rows = []
    for metric, key, label, fig_label in cfg["metrics"]:
        pair_rows = []
        heat_order = global_heatmap_order(DATASETS, metric, cfg, task)
        fig, axes = plt.subplots(
            1, len(DATASETS),
            figsize=(FIG_WIDTH_MM * MM_TO_IN, FIG_HEIGHT_MM * MM_TO_IN))
        axes = np.atleast_1d(axes)
        for c, ds in enumerate(DATASETS):
            data = load(ds, metric, cfg, task)
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
            # Rows and columns share one method order, so the names are written
            # once on the y axis with an index, and the columns carry just the
            # index. That drops three repeats of the method list and lets the
            # cells grow, since no vertical label band is needed.
            shown = [f"{i + 1} {DISPLAY_NAMES.get(m, m)}"
                     for i, m in enumerate(present)]
            ax.set_xticklabels([str(i + 1) for i in range(kk)],
                               fontsize=HEAT_FONTSIZE)
            ax.set_yticklabels(shown if c == 0 else [], fontsize=HEAT_FONTSIZE)
            # No tick marks; the cell grid already delimits rows and columns.
            ax.tick_params(axis="both", labelsize=HEAT_FONTSIZE, length=0,
                           width=0, pad=1.5)
            ax.set_title(LABELS.get(ds, ds), fontsize=TITLE_FONTSIZE, pad=2)
            for i in range(kk):
                for j in range(kk):
                    if i != j and Pg[i, j] < alpha_pair:
                        ax.text(j, i, "*", ha="center", va="center", color="w",
                                fontsize=HEAT_FONTSIZE)
        # Margins in absolute mm; only the left column carries method names,
        # so the panels themselves sit close together.
        left_mm, right_mm, gap_mm = 14.9, 8.4, 1.2
        left = left_mm / FIG_WIDTH_MM
        right = 1 - right_mm / FIG_WIDTH_MM
        axis_w = ((right - left) * FIG_WIDTH_MM
                  - (len(DATASETS) - 1) * gap_mm) / len(DATASETS)
        bottom = BOTTOM_BAND_MM / FIG_HEIGHT_MM
        top = 1 - TOP_BAND_MM / FIG_HEIGHT_MM
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                            wspace=gap_mm / axis_w)
        # Explicit colourbar axes: fig.colorbar(ax=...) would steal space
        # from the panels and undo the mm layout above.
        cax = fig.add_axes([(FIG_WIDTH_MM - right_mm + 1.2) / FIG_WIDTH_MM,
                            bottom, 1.6 / FIG_WIDTH_MM, top - bottom])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("$-$log$_{10}$ p", fontsize=BASE_FONTSIZE, labelpad=1)
        cbar.ax.tick_params(labelsize=HEAT_FONTSIZE, length=1.5, width=0.3,
                            pad=1)
        cbar.outline.set_linewidth(0.3)
        # Metric name once per figure, centred on the panel block.
        fig.text((left + right) / 2, METRIC_LABEL_Y_MM / FIG_HEIGHT_MM,
                 fig_label, fontsize=METRIC_LABEL_PT, ha="center", va="center")
        save_figure(fig, OUT, f"wilcoxon_heatmap_{key}", bbox_inches=None)
        plt.close(fig)
        pd.DataFrame(pair_rows).to_csv(
            os.path.join(OUT, f"wilcoxon_pairwise_{key}.csv"), index=False)

    vb = pd.DataFrame(vsbest_rows)
    vb.to_csv(os.path.join(OUT, "wilcoxon_vs_best.csv"), index=False)

    # markdown vs-best summary
    for _metric, key, label, _fig in cfg["metrics"]:
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
          "wilcoxon_heatmap_*.pdf, wilcoxon_summary.md to", OUT)
    # quick console view: vs-best for the heatmap metric
    print(f"\n=== vs-best ({CONSOLE_METRIC}) ===")
    t = vb[vb.metric == CONSOLE_METRIC]
    for ds in DATASETS:
        sub = t[t.dataset == ds]
        sig = sub[sub.sig_bonferroni]["other"].tolist()
        ns = sub[~sub.sig_bonferroni]["other"].tolist()
        print(f"{ds}: best={sub.iloc[0]['best']} | NOT-sig-different: {ns or '(none)'}")


if __name__ == "__main__":
    main()
