#!/usr/bin/env python
"""
Draw AlphaVariant per-round fitness improvement trajectories.

Style: 5th-95th and Q1-Q3 percentile bands with a median trajectory per dataset.
One x-axis label centred under the middle panel.

  --task multisite : oracle benchmarks (AAV, CreiLOV, PAB1)  [default]
  --task 4site     : four-site benchmarks (GB1, PhoQ, TrpB)

Data sources (both -> fitness_trajectory, a 5-element per-round list):
  multisite: results_oracle/{dataset}/AlphaVariant/seed*.json, already in [0, 1]
  4site:     alphavariant/results/_archive_tier1B_canonical/{arch}/seed_*/
             metrics.json, raw fitness divided by the dataset's global max
Exported per-seed copy of the multisite values: figures/ms_oracles/
alphavariant_trajectory_per_seed.csv (see build_trajectory_dataset.py)

Outputs: {outdir}/{prefix}.pdf
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

from utils.plot_style_utils import (
    BASE_FONTSIZE, DEFAULT_FIGURE_RCPARAMS, DOT_DIAMETER_PT, DOT_SIZE,
    TITLE_FONTSIZE, XLABEL_FONTSIZE, apply_nature_rcparams, save_figure,
)

_FIGDIR = os.path.join(_BENCH_ROOT, "figures")
N_ROUNDS = 5

# Global maxima used to normalise the raw 4-site trajectories to [0, 1].
GMAX_4SITE = {
    "4site_GB1":  8.761966,
    "4site_PhoQ": 133.5943,
    "TRPB":       1.0,
}

TASKS = {
    "multisite": {
        "datasets": {"ms_AAV": "AAV", "ms_CreiLOV": "CreiLOV", "ms_PAB1": "PAB1"},
        # Each oracle sits in its own narrow band, so panels get their own y axis.
        "ylims": {"ms_AAV": (0.60, 0.75), "ms_CreiLOV": (0.70, 1.02),
                  "ms_PAB1": (0.30, 0.65)},
        "shared_y": False,
        "gap_mm": 8.5,   # holds the next panel's y tick labels
        "outdir": os.path.join(_FIGDIR, "ms_oracles"),
        "prefix": "alphavariant_trajectory_multisite",
    },
    "4site": {
        "datasets": {"4site_GB1": "GB1", "4site_PhoQ": "PhoQ", "TRPB": "TrpB"},
        # All three span the same normalised 0-1 range (5th-95th bands reach 1.0),
        # but each panel carries its own y axis to match the multisite layout.
        "ylims": {"4site_GB1": (0.0, 1.02), "4site_PhoQ": (0.0, 1.02),
                  "TRPB": (0.0, 1.02)},
        "shared_y": False,
        "gap_mm": 8.5,   # holds the next panel's y tick labels
        "outdir": _FIGDIR,
        "prefix": "alphavariant_trajectory_4site",
    },
}

RUN_COLOR = "#C0392B"  # median line + dots
BAND_90 = "#F9D4CE"    # 5th-95th percentile, outer/lighter
BAND_IQR = "#F2B5AC"   # Q1-Q3 (IQR), inner/darker

# Print size, matching the main dot + whisker figures. The median dot uses the
# shared DOT_SIZE so both figure families draw the same marker.
FIG_WIDTH_MM = 89
FIG_HEIGHT_MM = 45
MED_LINE_LW = 0.75
LEGEND_FONTSIZE = 5.5  # one point under the tick labels; the band key is secondary
# Text bands in mm, so they stay constant when the figure height changes and the
# panels absorb the difference. Bottom stacks the x tick labels, the single
# x-axis label and the legend row; top holds the panel titles.
BOTTOM_BAND_MM = 12.8
TOP_BAND_MM = 4.8


def _read_trajectories(pattern, cap, scale=1.0):
    """Return array (n_seeds, N_ROUNDS) of fitness_trajectory values, scaled."""
    trajs = []
    for fp in sorted(glob.glob(pattern))[:cap]:
        try:
            d = json.load(open(fp))
            traj = d.get("metrics", {}).get("fitness_trajectory")
            if traj is None:
                traj = d.get("fitness_trajectory")
            if traj is not None and len(traj) == N_ROUNDS:
                trajs.append([float(v) / scale for v in traj])
        except Exception:
            pass
    return np.array(trajs) if trajs else None


def load_trajectories(dataset_key, task="multisite", cap=30):
    """Load one dataset's per-seed trajectories for the given task."""
    if task == "4site":
        return _read_trajectories(
            f"alphavariant/results/_archive_tier1B_canonical/{dataset_key}/"
            "seed_*/metrics.json",
            cap, scale=GMAX_4SITE[dataset_key])
    return _read_trajectories(
        f"results_oracle/{dataset_key}/AlphaVariant/seed*.json", cap)


def plot_trajectory_figure(data, cfg, outdir):
    rounds = np.arange(1, N_ROUNDS + 1)
    n_panels = len(data)
    n_seeds_legend = list(data.values())[0].shape[0]

    MM_TO_IN = 1 / 25.4
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(FIG_WIDTH_MM * MM_TO_IN,
                                      FIG_HEIGHT_MM * MM_TO_IN),
                             sharey=cfg["shared_y"])
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    for col, (ax, (key, arr)) in enumerate(zip(axes, data.items())):
        p05 = np.percentile(arr, 5, axis=0)
        p95 = np.percentile(arr, 95, axis=0)
        q1 = np.percentile(arr, 25, axis=0)
        q3 = np.percentile(arr, 75, axis=0)
        med = np.median(arr, axis=0)

        ax.fill_between(rounds, p05, p95, color=BAND_90, alpha=0.90,
                        zorder=1, linewidth=0)
        ax.fill_between(rounds, q1, q3, color=BAND_IQR, alpha=0.90,
                        zorder=2, linewidth=0)
        ax.plot(rounds, med, color=RUN_COLOR, lw=MED_LINE_LW, zorder=4,
                solid_capstyle="round")
        ax.scatter(rounds, med, s=DOT_SIZE, color=RUN_COLOR,
                   edgecolors="none", linewidths=0, zorder=5)

        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="both", labelsize=BASE_FONTSIZE)
        ax.set_title(cfg["datasets"][key], pad=4, fontsize=TITLE_FONTSIZE)
        if col == n_panels // 2:  # one x-axis label, centred under the middle panel
            ax.set_xlabel("Round", labelpad=3, fontsize=XLABEL_FONTSIZE)
        ax.set_xlim(0.6, N_ROUNDS + 0.4)
        ax.set_xticks(rounds)
        ax.set_ylim(*cfg["ylims"][key])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        if cfg["shared_y"] and col > 0:  # one y-axis serves every panel
            ax.tick_params(axis="y", labelleft=False)

    axes[0].set_ylabel("Max fitness", labelpad=2, fontsize=BASE_FONTSIZE)

    legend_handles = [
        Patch(facecolor=BAND_90, label="5th–95th percentile"),
        Patch(facecolor=BAND_IQR, label="Q1–Q3 (IQR)"),
        Line2D([0], [0], color=RUN_COLOR, lw=MED_LINE_LW, marker="o",
               markersize=DOT_DIAMETER_PT, markeredgewidth=0,
               label=f"Median (n={n_seeds_legend})"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
               fontsize=LEGEND_FONTSIZE, handlelength=1.6, columnspacing=1.4,
               handletextpad=0.5)

    # Margins in absolute mm: the left column holds the y-axis label plus its
    # tick labels. cfg["gap_mm"] must additionally hold the next panel's y tick
    # labels (~4.2 mm) plus its tick marks where each panel has its own y axis.
    left_mm, right_pad_mm = 9.8, 1.0
    gap_mm = cfg["gap_mm"]
    left = left_mm / FIG_WIDTH_MM
    right = 1 - right_pad_mm / FIG_WIDTH_MM
    axis_width_mm = ((right - left) * FIG_WIDTH_MM - (n_panels - 1) * gap_mm) / n_panels
    fig.subplots_adjust(left=left, right=right,
                        bottom=BOTTOM_BAND_MM / FIG_HEIGHT_MM,
                        top=1 - TOP_BAND_MM / FIG_HEIGHT_MM,
                        wspace=gap_mm / axis_width_mm)

    # bbox_inches=None keeps the page exactly FIG_WIDTH_MM x FIG_HEIGHT_MM.
    save_figure(fig, outdir, cfg["prefix"], bbox_inches=None)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(TASKS), default="multisite",
                    help="which benchmark to draw (default: multisite)")
    ap.add_argument("--outdir", default=None,
                    help="output directory (default: task-specific figures/ subdir)")
    ap.add_argument("--cap", type=int, default=30,
                    help="max seeds to load per dataset (default: 30)")
    args = ap.parse_args()

    apply_nature_rcparams(DEFAULT_FIGURE_RCPARAMS)
    cfg = TASKS[args.task]

    data = {}
    for key, label in cfg["datasets"].items():
        arr = load_trajectories(key, task=args.task, cap=args.cap)
        if arr is not None:
            data[key] = arr
            print(f"  {label}: {arr.shape[0]} seeds loaded")
        else:
            print(f"  {label}: no trajectory data found for {key}")

    if not data:
        sys.exit(f"No trajectory data found for task '{args.task}'.")

    plot_trajectory_figure(data, cfg, args.outdir or cfg["outdir"])
    print(f"Trajectory figure ({args.task}) complete.")


if __name__ == "__main__":
    main()
