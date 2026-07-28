#!/usr/bin/env python
"""
plot_4site_density.py - Landscape-difficulty diagnostic for the 4-site
benchmarks. For each dataset:
    (row 1) fitness density histogram (log-count y) of normalized fitness, with
            median / p99 / p99.9 / max markers and the fraction above 0.5.
    (row 2) survival curve P(fitness > x) on log-y = the probability a uniformly
            random draw exceeds threshold x. This is the single best predictor of
            how hard "find a high-fitness variant in 480 queries" is.

Saves figures/4site_benchmarks/density_4site.pdf + a stats CSV.

Usage: python scripts/plot_4site_density.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_BENCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _BENCH_ROOT)

from utils.plot_style_utils import (
    BASE_FONTSIZE, DEFAULT_FIGURE_RCPARAMS, TITLE_FONTSIZE, XLABEL_FONTSIZE,
    apply_nature_rcparams, prettify_ax, save_figure,
)

BENCH = _BENCH_ROOT
DATASETS = ["4site_GB1", "4site_PhoQ", "4site_TRPB"]
LABELS = {"4site_GB1": "GB1", "4site_PhoQ": "PhoQ",
          "4site_TRPB": "TrpB"}
COLOR = {"4site_GB1": "#2c7fb8", "4site_PhoQ": "#d7191c",
         "4site_TRPB": "#fdae61"}


def main():
    apply_nature_rcparams(DEFAULT_FIGURE_RCPARAMS)

    outdir = os.path.join(BENCH, "figures", "4site_benchmarks")
    os.makedirs(outdir, exist_ok=True)

    MM_TO_IN = 1 / 25.4
    fig, axes = plt.subplots(2, 3, figsize=(170 * MM_TO_IN, 90 * MM_TO_IN))
    rows = []
    for c, d in enumerate(DATASETS):
        df = pd.read_csv(os.path.join(BENCH, "data", d, "data.csv"))
        f = df["fitness"].values.astype(float)
        fn = f / f.max()
        col = COLOR[d]
        med, p99, p999 = np.quantile(fn, [0.5, 0.99, 0.999])
        frac50 = float(np.mean(fn > 0.5))
        rows.append({"dataset": d, "N": len(fn), "median": med, "p99": p99,
                     "p99.9": p999, "frac>0.5": frac50, "frac>0.8": float(np.mean(fn > 0.8))})

        # row 1: density (log count)
        ax = axes[0, c]
        ax.hist(fn, bins=80, color=col, alpha=0.85, log=True)
        for x, lab, ls in [(med, "median", ":"), (p99, "p99", "--"),
                           (1.0, "max", "-")]:
            ax.axvline(x, color="#333", ls=ls, lw=0.5)
        ax.set_title(LABELS[d], fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("Normalized fitness", fontsize=XLABEL_FONTSIZE)
        ax.set_xlim(min(0, fn.min()), 1.02)
        if c == 0:
            ax.set_ylabel("Variant count (log)", fontsize=XLABEL_FONTSIZE)
        ax.text(0.96, 0.94, f"N={len(fn):,}\nmedian={med:.3g}\np99={p99:.2f}\n"
                f">0.5: {frac50*100:.2f}%", transform=ax.transAxes, va="top", ha="right",
                ma="left", fontsize=BASE_FONTSIZE)
        prettify_ax(ax)

        # row 2: survival P(F > x), log y
        ax2 = axes[1, c]
        xs = np.linspace(0, 1, 200)
        surv = np.array([np.mean(fn > x) for x in xs])
        surv = np.clip(surv, 1.0 / len(fn), 1.0)
        ax2.semilogy(xs, surv, color=col, lw=1.2)
        ax2.axhline(480 / len(fn), color="#999", ls="--", lw=0.5)
        ax2.text(0.98, 480 / len(fn) * 1.3, "1 expected hit\nin 480 random draws",
                 ha="right", va="bottom", fontsize=BASE_FONTSIZE, color="#666")
        ax2.set_xlabel("Fitness threshold x", fontsize=XLABEL_FONTSIZE)
        ax2.set_xlim(0, 1)
        if c == 0:
            ax2.set_ylabel("P(fitness > x)", fontsize=XLABEL_FONTSIZE)
        prettify_ax(ax2)

    fig.tight_layout()

    save_figure(fig, outdir, "density_4site")
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "density_4site_stats.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
