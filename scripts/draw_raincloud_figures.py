#!/usr/bin/env python
"""
Draw horizontal raincloud plots for the protein optimization benchmark.

Each method row shows three layers:
  - Half-violin (KDE of per-seed distribution) above the row centre
  - Jittered scatter of individual seed values below
  - IQR box with median dot at centre

  --task 4site     : 4-site benchmark (GB1, PhoQ, TrpB)  [default]
  --task multisite : multi-site oracle benchmark (AAV, CreiLOV, PAB1)

  --metric max_fitness : per-seed best fitness found                [default]
  --metric top128       : per-seed top-128 mean fitness

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
    VERMILION, SECONDARY_GRAY,
    BASE_FONTSIZE, DEFAULT_FIGURE_RCPARAMS, TITLE_FONTSIZE, XLABEL_FONTSIZE,
    apply_nature_rcparams, save_figure, style_axis_hbar,
)

BASELINE_GRAY = SECONDARY_GRAY  # single flat color for all non-highlighted methods


def _row_color(method, cfg):
    return VERMILION if method in cfg["highlight"] else BASELINE_GRAY


def _darken(color, factor=0.55):
    """Darken a color for higher-contrast error-bar strokes."""
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)

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

# EVOLVEpro/MULTIevolve stored their TRPB run under the full "4site_TRPB"
# archive name instead of the bare "TRPB" alias every other method uses.
ARCH_OVERRIDE = {
    ("EVOLVEpro", "4site_TRPB"): "4site_TRPB",
    ("MULTIevolve", "4site_TRPB"): "4site_TRPB",
}


# Per-seed metric key to read from the raw metrics.json, per task/metric.
# 4-site "top128" is already normalized to [0,1] (no gmax division needed);
# multisite oracle "top128" uses its own pre-normalized field.
METRIC_KEY_4SITE = {
    "max_fitness": "max_fitness",
    "top128": "normalized_fitness_median_top128",
}
METRIC_KEY_ORACLE = {
    "max_fitness": "max_fitness_norm",
    "top128": "top128_mean_norm",
}


def _load_4site_seeds(method, dataset, metric="max_fitness", cap=30):
    arch = ARCH_OVERRIDE.get((method, dataset), DS_ARCH.get(dataset, dataset))
    pat = COMPETITOR_PATTERNS_4SITE.get(method)
    if pat is None:
        return []
    pattern = pat.format(a=arch, arch=arch)
    gmax = GMAX_4SITE.get(dataset, 1.0)
    key = METRIC_KEY_4SITE[metric]
    vals = []
    for fp in sorted(glob.glob(pattern))[:cap]:
        try:
            d = json.load(open(fp))
            m = d.get("metrics") or d.get("final_metrics") or d
            if isinstance(m, list):
                m = m[-1]
            v = m.get(key)
            if v is None:
                continue
            if metric == "max_fitness" and v > 1.5 and gmax != 1.0:
                v = v / gmax
            if 0.0 <= v <= 1.5:
                vals.append(float(v))
        except Exception:
            pass
    return vals


def _load_oracle_seeds(method, dataset, metric="max_fitness", cap=30):
    pattern = f"results_oracle/{dataset}/{method}/seed*.json"
    key = METRIC_KEY_ORACLE[metric]
    vals = []
    for fp in sorted(glob.glob(pattern))[:cap]:
        try:
            d = json.load(open(fp))
            v = d.get("metrics", {}).get(key)
            if v is not None and 0.0 <= v <= 1.5:
                vals.append(float(v))
        except Exception:
            pass
    return vals


def load_seed_values(method, dataset, task, metric="max_fitness"):
    if task == "4site":
        return _load_4site_seeds(method, dataset, metric=metric)
    return _load_oracle_seeds(method, dataset, metric=metric)


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
        "xlim": (0.0, 1.05),
        "xticks": np.arange(0.0, 1.01, 0.2),
        "prefix": {
            "max_fitness": "main_figure_max_fitness_raincloud",
            "top128": "supplementary_figure_top128_mean_fitness_raincloud",
        },
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
        "xlim": (0.0, 1.05),
        "xticks": np.arange(0.0, 1.01, 0.2),
        "prefix": {
            "max_fitness": "main_figure_multisite_max_fitness_raincloud",
            "top128": "supplementary_figure_multisite_top128_mean_fitness_raincloud",
        },
        "panel_letter": "d",
        "label_kind": "multi-site datasets",
    },
}

# Metric to rank/plot by. rank_col must exist in each task's source CSV.
METRIC_CONFIG = {
    "max_fitness": {
        "rank_col": "max_fitness_median",
        "xlabel": "Median max fitness",
    },
    "top128": {
        "rank_col": "top128_median",
        "xlabel": "Top-128 fitness",
    },
}


# ------------------------------------------------------------------
# Raincloud row drawing
# ------------------------------------------------------------------
def draw_raincloud_row(ax, values, y_center, color, bar_box_h, highlight=False,
                       violin_height=0.26, jitter_range=0.17):
    """Draw one horizontal raincloud row: half-violin + scatter + IQR box."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return

    dot_s = 2.5
    dot_alpha = 0.65 if highlight else 0.45
    vl_alpha = 0.50 if highlight else 0.30
    vl_lw = 0.9 if highlight else 0.5
    med_s = 12
    bar_color = _darken(color)  # darker, higher-contrast Q1-Q3 bar + median line
    med_color = bar_color if highlight else "black"
    gap = bar_box_h / 2 + 0.035  # clearance between the bar and the violin/jitter

    if len(values) >= 4:
        try:
            kde = gaussian_kde(values)
            x_lo = max(values.min() - 0.06, 0.0)
            x_hi = min(values.max() + 0.06, 1.0)
            x_grid = np.linspace(x_lo, x_hi, 200)
            density = kde(x_grid)
            density_scaled = density / density.max() * violin_height
            ax.fill_between(x_grid, y_center + gap, y_center + gap + density_scaled,
                            color=color, alpha=vl_alpha, linewidth=0, zorder=2)
            ax.plot(x_grid, y_center + gap + density_scaled,
                    color=color, lw=vl_lw, alpha=0.85, zorder=3)
        except Exception:
            pass

    jitter = np.random.uniform(-jitter_range - gap, -gap, len(values))
    ax.scatter(values, y_center + jitter, s=dot_s, color=color,
               alpha=dot_alpha, linewidths=0, zorder=3)

    q1, med, q3 = np.percentile(values, [25, 50, 75])
    rect = Rectangle((q1, y_center - bar_box_h / 2), q3 - q1, bar_box_h,
                      facecolor=bar_color, edgecolor=bar_color,
                      linewidth=0.3, zorder=4)
    ax.add_patch(rect)
    ax.vlines(med, y_center - bar_box_h / 2, y_center + bar_box_h / 2,
              color="white", linewidth=1.1 if highlight else 0.7, zorder=5)
    ax.scatter([med], [y_center], s=med_s, color=med_color,
               edgecolors="none", linewidths=0, zorder=6)


def global_method_order(df, cfg, rank_col="max_fitness_median"):
    """Fixed method order: best mean rank first, mean score breaks ties."""
    datasets = cfg["dataset_order"]
    main_methods = cfg["main_methods"]
    sub = df[(df["dataset"].isin(datasets)) & (df["method"].isin(main_methods))].copy()
    sub["rank"] = sub.groupby("dataset")[rank_col].rank(
        ascending=False, method="average")
    agg = sub.groupby("method").agg(
        mean_rank=("rank", "mean"),
        mean_score=(rank_col, "mean"),
    )
    agg = agg.sort_values(["mean_rank", "mean_score"], ascending=[True, False])
    return agg.index.tolist()


def plot_raincloud_figure(task, metric="max_fitness", outdir=None):
    cfg = TASKS[task]
    metric_cfg = METRIC_CONFIG[metric]
    df = pd.read_csv(cfg["csv"])
    effective_outdir = outdir or cfg["outdir"]

    # Best method (AlphaVariant) at y=n-1 → top row; worst at y=0 → bottom row.
    methods = global_method_order(df, cfg, rank_col=metric_cfg["rank_col"])[::-1]  # worst first (rank 1 = last index)
    n_methods = len(methods)
    ROW_STEP = 0.75
    y_centers = np.arange(n_methods, dtype=float) * ROW_STEP

    dataset_order = cfg["dataset_order"]
    n_panels = len(dataset_order)
    MM_TO_IN = 1 / 25.4
    FIG_HEIGHT_MM = 40
    AXIS_BOTTOM, AXIS_TOP = 0.16, 0.90  # must match the subplots_adjust call below
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(80 * MM_TO_IN, FIG_HEIGHT_MM * MM_TO_IN),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    # Q1-Q3 bar height fixed at 0.3 mm in print, converted to data units from
    # the actual axis geometry so it stays 0.3 mm regardless of n_methods.
    data_y_range = (n_methods - 1) * ROW_STEP + 0.36 + 0.30
    axis_height_mm = FIG_HEIGHT_MM * (AXIS_TOP - AXIS_BOTTOM)
    mm_per_data_unit = axis_height_mm / data_y_range
    bar_box_h = 0.3 / mm_per_data_unit

    raw_from_display = {DISPLAY_NAMES.get(m, m): m for m in methods}

    for col, (ax, dataset) in enumerate(zip(axes, dataset_order)):
        np.random.seed(col)
        for row_i, method in enumerate(methods):
            vals = load_seed_values(method, dataset, task, metric=metric)
            if not vals:
                continue
            draw_raincloud_row(ax, vals, y_centers[row_i],
                               _row_color(method, cfg), bar_box_h,
                               highlight=(method in cfg["highlight"]))

        style_axis_hbar(ax, cfg["xlim"], cfg["xticks"])
        ax.set_ylim(-0.30, (n_methods - 1) * ROW_STEP + 0.36)
        ax.set_yticks(y_centers)
        ax.set_title(cfg["dataset_labels"][dataset], pad=4, fontsize=TITLE_FONTSIZE)
        ax.set_xlabel(metric_cfg["xlabel"], labelpad=3, fontsize=XLABEL_FONTSIZE)

        if col == 0:
            ax.set_yticklabels(
                [DISPLAY_NAMES.get(m, m) for m in methods], fontsize=BASE_FONTSIZE)
            for tick in ax.get_yticklabels():
                raw = raw_from_display.get(tick.get_text(), tick.get_text())
                if raw in cfg["highlight"]:
                    tick.set_color(mcolors.to_hex(VERMILION))
        else:
            ax.tick_params(axis="y", labelleft=False)

    # Margins are set in absolute mm (not a fixed fraction) so the label column
    # and inter-panel gaps stay constant in size as the figure widens.
    fig_width_mm = fig.get_size_inches()[0] * 25.4
    left_mm, right_pad_mm, gap_mm = 12.4, 0.96, 2.66
    left = left_mm / fig_width_mm
    right = 1 - right_pad_mm / fig_width_mm
    axis_width_mm = ((right - left) * fig_width_mm - (n_panels - 1) * gap_mm) / n_panels
    wspace = gap_mm / axis_width_mm
    fig.subplots_adjust(left=left, right=right, bottom=AXIS_BOTTOM, top=AXIS_TOP, wspace=wspace)

    save_figure(fig, effective_outdir, cfg["prefix"][metric])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["4site", "multisite"], default="4site",
                    help="which benchmark to draw (default: 4site)")
    ap.add_argument("--metric", choices=list(METRIC_CONFIG), default="max_fitness",
                    help="which metric to plot (default: max_fitness)")
    ap.add_argument("--outdir", default=None,
                    help="output directory (default: task-specific figures/ subdir)")
    args = ap.parse_args()

    apply_nature_rcparams(DEFAULT_FIGURE_RCPARAMS)
    np.random.seed(0)

    csv_path = TASKS[args.task]["csv"]
    if not os.path.exists(csv_path):
        sys.exit(f"Source CSV not found: {csv_path}")

    plot_raincloud_figure(args.task, metric=args.metric, outdir=args.outdir)
    print(f"Raincloud figure ({args.task}, {args.metric}) complete.")


if __name__ == "__main__":
    main()
