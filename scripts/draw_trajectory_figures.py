#!/usr/bin/env python
"""
Draw AlphaVariant per-round fitness improvement trajectories on multi-site benchmarks.

Style: Scheme B (alpha=0.18) — per-seed thin lines, 5th-95th and Q1-Q3 bands,
and a median trajectory for AAV, CreiLOV, and PAB1.

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
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

_BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BENCH_ROOT)
os.chdir(_BENCH_ROOT)

from utils.plot_style_utils import apply_nature_rcparams, save_figure

_DEFAULT_OUTDIR = os.path.join(_BENCH_ROOT, "figures", "ms_oracles")

DATASETS = {
    "ms_AAV":     "AAV",
    "ms_CreiLOV": "CreiLOV",
    "ms_PAB1":    "PAB1",
}
N_ROUNDS = 5

YLIMS = {
    "ms_AAV":     (0.60, 0.75),
    "ms_CreiLOV": (0.70, 1.02),
    "ms_PAB1":    (0.30, 0.65),
}

# Scheme B (alpha=0.18) palette — individual runs share the median's color,
# only the two percentile bands get their own pastel shades.
RUN_COLOR = "#C0392B"
BAND_90 = "#FAE3DE"   # 5th-95th percentile, outer/lighter
BAND_IQR = "#F5C6BC"  # Q1-Q3 (IQR), inner/darker


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

    MM_TO_IN = 1 / 25.4
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(170 * MM_TO_IN, 45 * MM_TO_IN),
                             sharey=False)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    for ax, (key, arr) in zip(axes, data.items()):
        label = DATASETS[key]

        p05 = np.percentile(arr, 5, axis=0)
        p95 = np.percentile(arr, 95, axis=0)
        q1 = np.percentile(arr, 25, axis=0)
        q3 = np.percentile(arr, 75, axis=0)
        med = np.median(arr, axis=0)

        ax.fill_between(rounds, p05, p95, color=BAND_90, alpha=0.90,
                        zorder=1, linewidth=0)
        ax.fill_between(rounds, q1, q3, color=BAND_IQR, alpha=0.90,
                        zorder=2, linewidth=0)
        for seed_row in arr:
            ax.plot(rounds, seed_row, color=RUN_COLOR, lw=0.65, alpha=0.18, zorder=3)
        ax.plot(rounds, med, color=RUN_COLOR, lw=1.6, zorder=4, solid_capstyle="round")
        ax.scatter(rounds, med, s=26, color=RUN_COLOR, zorder=5)

        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="both", labelsize=6)
        ax.set_title(label, fontweight="bold", fontsize=7)
        ax.set_xlabel("Round", fontsize=7, fontweight="normal")
        ax.set_xlim(0.6, N_ROUNDS + 0.4)
        ax.set_xticks(rounds)
        ax.set_ylim(*YLIMS[key])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    axes[0].set_ylabel("Max fitness", fontsize=7, fontweight="normal")

    legend_handles = [
        Line2D([0], [0], color=RUN_COLOR, lw=1.2, alpha=0.5,
               label=f"Individual run (n={n_seeds_legend})"),
        Patch(facecolor=BAND_90, label="5th–95th percentile"),
        Patch(facecolor=BAND_IQR, label="Q1–Q3 (IQR)"),
        Line2D([0], [0], color=RUN_COLOR, lw=1.6, marker="o", markersize=5,
               label="Median"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.10), ncol=4, frameon=False, fontsize=7.5)

    fig.tight_layout()

    save_figure(fig, outdir, "alphavariant_trajectory_multisite", dpi=180)
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
