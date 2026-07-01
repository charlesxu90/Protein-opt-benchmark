#!/usr/bin/env python
"""
Draw horizontal raincloud plots for the protein optimization benchmark.

Each method row shows three layers:
  - Half-violin (KDE of per-seed distribution) above the row centre
  - Jittered scatter of individual seed values below
  - IQR box with median dot at centre

  --task 4site     : 4-site benchmark (GB1, PhoQ, TrpB)  [default]
  --task multisite : multi-site oracle benchmark (AAV, CreiLOV, PAB1)

Outputs: {outdir}/{prefix}.{png,pdf,svg}
"""
import argparse
import glob
import json
import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde

_BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BENCH_ROOT)
os.chdir(_BENCH_ROOT)

from utils.plot_style_utils import (
    VERMILION, method_color,
    apply_nature_rcparams, save_figure, style_axis_hbar,
)

_DEFAULT_FIGDIR = os.path.join(_BENCH_ROOT, "figures")

DISPLAY_NAMES = {"FLEXS": "AdaLead"}

# ------------------------------------------------------------------
# 4-site: per-seed value loading
# ------------------------------------------------------------------
GMAX_4SITE = {
    "4site_GB1": 8.761966,
    "4site_PhoQ": 133.5943,
    "4site_TEV": 1.0,
    "4site_TRPB": 1.0,
}

COMPETITOR_PATTERNS_4SITE = {
    "Random":       "Random/results/{a}_Random/{a}/random/metrics_seed*.json",
    "GreedyWalk":   "GreedyWalk/results/{a}_GreedyWalk/{a}/greedy/metrics_seed*.json",
    "ALDE":         "ALDE/results/{a}_ALDE/{a}/onehot/metrics_seed*.json",
    "FLEXS":        "FLEXS/results/{a}_AdaLead/{a}/metrics_seed*.json",
    "AiCE":         "AiCE/results/{a}_AiCE/{a}/aice/metrics_seed*.json",
    "ftMLDE":       "ftMLDE/results/{a}_ftMLDE/{a}/ftmlde/metrics_seed*.json",
    "CLADE":        "CLADE/results/{a}_CLADE/{a}/clade/metrics_seed*.json",
    "EVOLVEpro":    "EVOLVEpro/results/{a}_EVOLVEpro/{a}/*/metrics_seed*.json",
    "MULTIevolve":  "MULTIevolve/results/{a}_MULTIevolve/{a}/*/metrics_seed*.json",
    "AlphaVariant": "alphavariant/results/_archive_tier1B_canonical/{arch}/seed_*/metrics.json",
}

DS_ARCH = {
    "4site_GB1":  "4site_GB1",
    "4site_PhoQ": "4site_PhoQ",
    "4site_TEV":  "4site_TEV",
    "4site_TRPB": "TRPB",
}


def _load_4site_seeds(method, dataset, cap=30):
    arch = DS_ARCH.get(dataset, dataset)
    pat = COMPETITOR_PATTERNS_4SITE.get(method)
    if pat is None:
        return []
    pattern = pat.format(a=arch, arch=arch)
    gmax = GMAX_4SITE.get(dataset, 1.0)
    vals = []
    for fp in sorted(glob.glob(pattern))[:cap]:
        try:
            d = json.load(open(fp))
            m = d.get("metrics") or d.get("final_metrics") or d
            if isinstance(m, list):
                m = m[-1]
            v = m.get("max_fitness")
            if v is None:
                continue
            if v > 1.5 and gmax != 1.0:
                v = v / gmax
            if 0.0 <= v <= 1.5:
                vals.append(float(v))
        except Exception:
            pass
    return vals


def _load_oracle_seeds(method, dataset, cap=30):
    pattern = f"results_oracle/{dataset}/{method}/seed*.json"
    vals = []
    for fp in sorted(glob.glob(pattern))[:cap]:
        try:
            d = json.load(open(fp))
            v = d.get("metrics", {}).get("max_fitness_norm")
            if v is not None and 0.0 <= v <= 1.5:
                vals.append(float(v))
        except Exception:
            pass
    return vals


def load_seed_values(method, dataset, task):
    if task == "4site":
        return _load_4site_seeds(method, dataset)
    return _load_oracle_seeds(method, dataset)


# ------------------------------------------------------------------
# Task configs
# ------------------------------------------------------------------
TASKS = {
    "4site": {
        "csv": os.path.join(_DEFAULT_FIGDIR, "alphavariant_comparison_median_iqr.csv"),
        "outdir": _DEFAULT_FIGDIR,
        "dataset_order": ["4site_GB1", "4site_PhoQ", "4site_TRPB"],
        "dataset_labels": {
            "4site_GB1": "GB1",
            "4site_PhoQ": "PhoQ",
            "4site_TRPB": "TrpB",
        },
        "main_methods": {
            "Random", "GreedyWalk", "ALDE", "FLEXS", "AiCE",
            "ftMLDE", "CLADE", "AlphaVariant", "MULTIevolve", "EVOLVEpro",
        },
        "highlight": {"AlphaVariant"},
        "xlim": (0.2, 1.05),
        "xticks": np.arange(0.2, 1.01, 0.2),
        "prefix": "main_figure_max_fitness_raincloud",
        "panel_letter": "c",
        "label_kind": "four-site datasets",
    },
    "multisite": {
        "csv": os.path.join(_DEFAULT_FIGDIR, "ms_oracles", "multisite_oracle_median_iqr.csv"),
        "outdir": os.path.join(_DEFAULT_FIGDIR, "ms_oracles"),
        "dataset_order": ["ms_AAV", "ms_CreiLOV", "ms_PAB1"],
        "dataset_labels": {
            "ms_AAV": "AAV",
            "ms_CreiLOV": "CreiLOV",
            "ms_PAB1": "PAB1",
        },
        "main_methods": {
            "Random", "GreedyWalk", "ALDE", "CLADE", "ftMLDE",
            "AdaLead", "MULTIevolve", "EVOLVEpro", "AiCE", "AlphaVariant",
        },
        "highlight": {"AlphaVariant"},
        "xlim": (0.2, 1.05),
        "xticks": np.arange(0.2, 1.01, 0.2),
        "prefix": "main_figure_multisite_max_fitness_raincloud",
        "panel_letter": "d",
        "label_kind": "multi-site datasets",
    },
}


# ------------------------------------------------------------------
# Raincloud row drawing
# ------------------------------------------------------------------
def draw_raincloud_row(ax, values, y_center, color, highlight=False,
                       violin_height=0.26, jitter_range=0.17):
    """Draw one horizontal raincloud row: half-violin + scatter + IQR box."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return

    dot_s = 9 if highlight else 5
    dot_alpha = 0.65 if highlight else 0.45
    box_h = 0.08 if highlight else 0.055
    vl_alpha = 0.50 if highlight else 0.30
    vl_lw = 0.9 if highlight else 0.5
    med_s = 42 if highlight else 20
    ec = "white" if highlight else "none"
    ec_lw = 0.9 if highlight else 0

    if len(values) >= 4:
        try:
            kde = gaussian_kde(values)
            x_lo = max(values.min() - 0.06, 0.0)
            x_hi = min(values.max() + 0.06, 1.05)
            x_grid = np.linspace(x_lo, x_hi, 200)
            density = kde(x_grid)
            density_scaled = density / density.max() * violin_height
            ax.fill_between(x_grid, y_center, y_center + density_scaled,
                            color=color, alpha=vl_alpha, linewidth=0, zorder=2)
            ax.plot(x_grid, y_center + density_scaled,
                    color=color, lw=vl_lw, alpha=0.85, zorder=3)
        except Exception:
            pass

    jitter = np.random.uniform(-jitter_range, -0.02, len(values))
    ax.scatter(values, y_center + jitter, s=dot_s, color=color,
               alpha=dot_alpha, linewidths=0, zorder=3)

    q1, med, q3 = np.percentile(values, [25, 50, 75])
    rect = Rectangle((q1, y_center - box_h / 2), q3 - q1, box_h,
                      facecolor="white", edgecolor=color,
                      linewidth=1.1 if highlight else 0.75, zorder=4)
    ax.add_patch(rect)
    ax.vlines(med, y_center - box_h / 2, y_center + box_h / 2,
              color=color, linewidth=1.6 if highlight else 1.0, zorder=5)
    ax.scatter([med], [y_center], s=med_s, color=color,
               edgecolors=ec, linewidths=ec_lw, zorder=6)


def global_method_order(df, cfg):
    """Fixed method order (best overall mean rank → worst) from summary CSV."""
    datasets = cfg["dataset_order"]
    main_methods = cfg["main_methods"]
    sub = df[(df["dataset"].isin(datasets)) & (df["method"].isin(main_methods))].copy()
    sub["rank"] = sub.groupby("dataset")["max_fitness_median"].rank(
        ascending=False, method="average")
    return sub.groupby("method")["rank"].mean().sort_values().index.tolist()


def plot_raincloud_figure(task, outdir=None):
    cfg = TASKS[task]
    df = pd.read_csv(cfg["csv"])
    seed_note = int(df["n"].max()) if "n" in df.columns else 30
    effective_outdir = outdir or cfg["outdir"]

    # Best method (AlphaVariant) at y=0 → bottom row; worst at y=n-1 → top row.
    methods = global_method_order(df, cfg)  # best first (rank 1 = index 0)
    n_methods = len(methods)
    ROW_STEP = 0.75
    y_centers = np.arange(n_methods, dtype=float) * ROW_STEP

    dataset_order = cfg["dataset_order"]
    n_panels = len(dataset_order)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(2.35 * n_panels + 1.25, 0.31 * n_methods + 0.9),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    raw_from_display = {DISPLAY_NAMES.get(m, m): m for m in methods}

    for col, (ax, dataset) in enumerate(zip(axes, dataset_order)):
        np.random.seed(col)
        for row_i, method in enumerate(methods):
            vals = load_seed_values(method, dataset, task)
            if not vals:
                continue
            draw_raincloud_row(ax, vals, y_centers[row_i],
                               method_color(method),
                               highlight=(method in cfg["highlight"]))

        style_axis_hbar(ax, cfg["xlim"], cfg["xticks"])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-0.30, (n_methods - 1) * ROW_STEP + 0.36)
        ax.set_yticks(y_centers)
        ax.set_title(cfg["dataset_labels"][dataset], pad=5, fontweight="bold", fontsize=8.0)
        ax.set_xlabel("Median max fitness", labelpad=3)

        if col == 0:
            ax.tick_params(axis="y", length=0)
            ax.set_yticklabels(
                [DISPLAY_NAMES.get(m, m) for m in methods], fontsize=7.0)
            for tick in ax.get_yticklabels():
                raw = raw_from_display.get(tick.get_text(), tick.get_text())
                if raw in cfg["highlight"]:
                    tick.set_fontweight("bold")
                    tick.set_color(mcolors.to_hex(VERMILION))
        else:
            ax.tick_params(axis="y", length=0, labelleft=False)

    fig.text(0.015, 0.975, cfg["panel_letter"],
             fontsize=11.5, fontweight="bold", va="top", ha="left")
    fig.suptitle(
        f"Max fitness across {cfg['label_kind']}\n"
        f"(n = {seed_note} independent runs; median with Q1–Q3 IQR)",
        x=0.055, y=0.975, ha="left", va="top",
        fontsize=8.5, fontweight="bold")
    fig.subplots_adjust(left=0.155, right=0.988, bottom=0.16, top=0.83, wspace=0.12)

    save_figure(fig, effective_outdir, cfg["prefix"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["4site", "multisite"], default="4site",
                    help="which benchmark to draw (default: 4site)")
    ap.add_argument("--outdir", default=None,
                    help="output directory (default: task-specific figures/ subdir)")
    args = ap.parse_args()

    apply_nature_rcparams({"xtick.labelsize": 6.8})
    np.random.seed(0)

    csv_path = TASKS[args.task]["csv"]
    if not os.path.exists(csv_path):
        sys.exit(f"Source CSV not found: {csv_path}")

    plot_raincloud_figure(args.task, outdir=args.outdir)
    print(f"Raincloud figure ({args.task}) complete.")


if __name__ == "__main__":
    main()
