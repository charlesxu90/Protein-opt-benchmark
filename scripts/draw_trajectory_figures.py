#!/usr/bin/env python
"""
Draw AlphaVariant per-round fitness improvement trajectories on multi-site benchmarks.

Shows per-seed thin lines, Q1-Q3 IQR band, and median trajectory for
AAV, CreiLOV, and PAB1.

Data source: results_oracle/{dataset}/AlphaVariant/seed*.json
  -> metrics.fitness_trajectory (5-element list, normalised to [0, 1])

Outputs: {outdir}/alphavariant_trajectory_multisite.{png,pdf,svg}
"""
import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BENCH_ROOT)
os.chdir(_BENCH_ROOT)

from utils.plot_style_utils import (
    VERMILION,
    apply_nature_rcparams, save_figure, style_axis_vbar,
)

_DEFAULT_OUTDIR = os.path.join(_BENCH_ROOT, "figures", "ms_oracles")

DATASETS = {
    "ms_AAV":     "AAV",
    "ms_CreiLOV": "CreiLOV",
    "ms_PAB1":    "PAB1",
}
N_ROUNDS = 5
SALMON = "#F4A07A"
IQR_BG = "#F8CBB0"


def load_trajectories(dataset_key, cap=30):
    """Return array of shape (n_seeds, N_ROUNDS) from fitness_trajectory in seed JSONs."""
    pattern = f"results_oracle/{dataset_key}/AlphaVariant/seed*.json"
    trajs = []
    for fp in sorted(glob.glob(pattern))[:cap]:
        try:
            d = json.load(open(fp))
            traj = d.get("metrics", {}).get("fitness_trajectory")
            if traj is None:
                traj = d.get("fitness_trajectory")
            if traj is not None and len(traj) == N_ROUNDS:
                trajs.append([float(v) for v in traj])
        except Exception:
            pass
    return np.array(trajs) if trajs else None


def plot_trajectory_figure(data, outdir):
    rounds = np.arange(1, N_ROUNDS + 1)
    n_panels = len(data)
    n_seeds_legend = list(data.values())[0].shape[0]

    fig, axes = plt.subplots(1, n_panels,
                             figsize=(2.6 * n_panels + 0.7, 2.8),
                             sharey=False)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    for ax, (key, arr) in zip(axes, data.items()):
        label = DATASETS[key]

        for seed_row in arr:
            ax.plot(rounds, seed_row, color=SALMON, lw=0.7, alpha=0.28, zorder=2)

        q1 = np.percentile(arr, 25, axis=0)
        q3 = np.percentile(arr, 75, axis=0)
        med = np.median(arr, axis=0)
        ax.fill_between(rounds, q1, q3, color=IQR_BG, alpha=0.65, zorder=3, linewidth=0)
        ax.plot(rounds, q1, color=SALMON, lw=0.4, alpha=0.55, zorder=3)
        ax.plot(rounds, q3, color=SALMON, lw=0.4, alpha=0.55, zorder=3)
        ax.plot(rounds, med, color=VERMILION, lw=2.0, zorder=5, solid_capstyle="round")
        ax.scatter(rounds, med, s=28, color=VERMILION, zorder=6,
                   edgecolors="white", linewidths=0.6)

        style_axis_vbar(ax)
        ax.set_title(label, fontweight="bold", pad=5, fontsize=8.0)
        ax.set_xlabel("Round", labelpad=3)
        ax.set_xlim(0.6, N_ROUNDS + 0.4)
        ax.set_xticks(rounds)
        # Tight per-panel y-limits (each dataset has a different fitness range)
        y_lo = max(0.0, arr.min() - 0.04)
        y_hi = min(1.01, arr.max() + 0.04)
        ax.set_ylim(y_lo, y_hi)

    axes[0].set_ylabel("Max fitness", labelpad=3)

    legend_handles = [
        Line2D([0], [0], color=SALMON, lw=1.2, alpha=0.55,
               label=f"Individual run (n = {n_seeds_legend})"),
        Patch(facecolor=IQR_BG, edgecolor=SALMON, lw=0.5, label="Q1–Q3 (IQR)"),
        Line2D([0], [0], color=VERMILION, lw=2.0, marker="o", markersize=5.5,
               markerfacecolor=VERMILION, markeredgecolor="white",
               markeredgewidth=0.6, label="Median"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False,
               fontsize=6.2, handletextpad=0.5, columnspacing=1.2)

    fig.text(0.015, 0.975, "e", fontsize=11.5, fontweight="bold", va="top", ha="left")
    fig.suptitle(
        f"AlphaVariant optimization trajectories across multi-site datasets "
        f"(n = {n_seeds_legend} seeds)",
        x=0.055, y=0.975, ha="left", va="top",
        fontsize=8.5, fontweight="bold")
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.22, top=0.85, wspace=0.35)

    save_figure(fig, outdir, "alphavariant_trajectory_multisite")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=_DEFAULT_OUTDIR,
                    help=f"output directory (default: {_DEFAULT_OUTDIR})")
    ap.add_argument("--cap", type=int, default=30,
                    help="max seeds to load per dataset (default: 30)")
    args = ap.parse_args()

    apply_nature_rcparams()

    data = {}
    for key, label in DATASETS.items():
        arr = load_trajectories(key, cap=args.cap)
        if arr is not None:
            data[key] = arr
            print(f"  {label}: {arr.shape[0]} seeds loaded")
        else:
            print(f"  {label}: no data at results_oracle/{key}/AlphaVariant/seed*.json")

    if not data:
        sys.exit("No trajectory data found. Check results_oracle/ paths.")

    plot_trajectory_figure(data, args.outdir)
    print("Trajectory figure complete.")


if __name__ == "__main__":
    main()
