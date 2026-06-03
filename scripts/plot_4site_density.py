#!/usr/bin/env python
"""
plot_4site_density.py - Landscape-difficulty diagnostic for the four 4-site
benchmarks. For each dataset:
    (row 1) fitness density histogram (log-count y) of normalized fitness, with
            median / p99 / p99.9 / max markers and the fraction above 0.5.
    (row 2) survival curve P(fitness > x) on log-y = the probability a uniformly
            random draw exceeds threshold x. This is the single best predictor of
            how hard "find a high-fitness variant in 480 queries" is.

Saves figures/4site_diagnostics/density_4site.{png,pdf} + a stats CSV.

Usage: python scripts/plot_4site_density.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

BENCH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS = ["4site_GB1", "4site_PhoQ", "4site_TEV", "4site_TRPB"]
LABELS = {"4site_GB1": "GB1 4-site", "4site_PhoQ": "PhoQ 4-site",
          "4site_TEV": "TEV 4-site", "4site_TRPB": "TrpB 4-site"}
COLOR = {"4site_GB1": "#2c7fb8", "4site_PhoQ": "#d7191c",
         "4site_TEV": "#7fbc41", "4site_TRPB": "#fdae61"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.65, "axes.labelsize": 8.4, "axes.titlesize": 9.2,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    "figure.dpi": 180, "savefig.dpi": 600,
})


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#333333"); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(length=2.4, width=0.55, color="#333333")


def main():
    outdir = os.path.join(BENCH, "figures", "4site_benchmarks")
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.4))
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
            ax.axvline(x, color="#333", ls=ls, lw=0.7)
        ax.set_title(LABELS[d], fontweight="bold")
        ax.set_xlabel("normalized fitness"); ax.set_xlim(min(0, fn.min()), 1.02)
        if c == 0:
            ax.set_ylabel("variant count (log)")
        ax.text(0.96, 0.94, f"N={len(fn):,}\nmedian={med:.3g}\np99={p99:.2f}\n"
                f">0.5: {frac50*100:.2f}%", transform=ax.transAxes, va="top", ha="right",
                fontsize=6.4, bbox=dict(boxstyle="round", fc="white", alpha=0.8, lw=0.4))
        style(ax)

        # row 2: survival P(F > x), log y
        ax2 = axes[1, c]
        xs = np.linspace(0, 1, 200)
        surv = np.array([np.mean(fn > x) for x in xs])
        surv = np.clip(surv, 1.0 / len(fn), 1.0)
        ax2.semilogy(xs, surv, color=col, lw=1.6)
        ax2.axhline(480 / len(fn), color="#999", ls="--", lw=0.7)
        ax2.text(0.98, 480 / len(fn) * 1.3, "1 expected hit\nin 480 random draws",
                 ha="right", va="bottom", fontsize=5.6, color="#666")
        ax2.set_xlabel("fitness threshold x"); ax2.set_xlim(0, 1)
        if c == 0:
            ax2.set_ylabel("P(fitness > x)")
        style(ax2)

    fig.text(0.012, 0.985, "a", fontsize=11, fontweight="bold", va="top")
    fig.suptitle("4-site landscape difficulty: fitness density (top) and survival "
                 "P(fitness > x) (bottom)", x=0.5, y=0.995, fontsize=11, fontweight="bold")
    fig.text(0.5, 0.005,
             "All four landscapes are extreme needle-in-haystack: >90% of variants are near-dead; "
             "high-fitness mass is a thin tail. The dashed line marks the fitness a uniformly random "
             "480-query budget can expect to reach.",
             ha="center", va="bottom", fontsize=6.2, color="#4D4D4D")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"density_4site.{ext}"), bbox_inches="tight")
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "density_4site_stats.csv"), index=False)
    print("saved", os.path.join(outdir, "density_4site.png"))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
