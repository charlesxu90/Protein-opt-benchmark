#!/usr/bin/env python
"""
Draw horizontal dot + IQR whisker plots for the protein optimization benchmark.

Each method row shows the per-seed distribution as:
  - Whisker spanning Q1-Q3 with end caps
  - Median dot at the row centre

  --task 4site     : 4-site benchmark (GB1, PhoQ, TrpB)  [default]
  --task multisite : multi-site oracle benchmark (AAV, CreiLOV, PAB1)

  --metric max_fitness : per-seed best fitness found                [default]
  --metric top128       : per-seed top-128 mean fitness

Outputs: {outdir}/{prefix}.pdf
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

_BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BENCH_ROOT)
os.chdir(_BENCH_ROOT)

from utils.plot_style_utils import (
    BASE_FONTSIZE, DEFAULT_FIGURE_RCPARAMS, DOT_DIAMETER_MM, DOT_SIZE,
    TITLE_FONTSIZE, XLABEL_FONTSIZE, apply_nature_rcparams, save_figure,
    style_axis_hbar,
)
from utils.seed_values import load_seeds

AV_DOT = "#C0392B"        # AlphaVariant median dot + y-tick label
AV_WHISKER = "#C0392B"    # AlphaVariant IQR whisker
BASE_DOT = "#2C3E50"      # all other methods, median dot
BASE_WHISKER = "#95A5A6"  # all other methods, IQR whisker


def _row_colors(method, cfg):
    """(median dot, IQR whisker) colors for one method row."""
    if method in cfg["highlight"]:
        return AV_DOT, AV_WHISKER
    return BASE_DOT, BASE_WHISKER

_DEFAULT_FIGDIR = os.path.join(_BENCH_ROOT, "figures")

DISPLAY_NAMES = {"FLEXS": "AdaLead"}

def load_seed_values(method, dataset, task, metric="max_fitness"):
    """Per-seed values for one method/dataset (see utils.seed_values)."""
    return list(load_seeds(method, dataset, task, metric=metric).values())


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
        # max_fitness: one shared tick list for all three panels, whose spreads
        # are wide enough (3.5-4.3 mm) to read on a common 0-1 axis.
        # top128: per-panel, since the panels top out at very different values
        # (PhoQ max Q3 = 0.14 vs TrpB 0.66) and a shared axis leaves PhoQ's IQR
        # at 0.93 mm — under the dot diameter.
        "xrange": {
            "max_fitness": [0.0, 0.5, 1.0],
            "top128": {
                "4site_GB1":  [0.0, 0.2, 0.4, 0.6],
                "4site_PhoQ": [0.0, 0.05, 0.10, 0.15],
                "4site_TRPB": [0.0, 0.4, 0.8],  # 3 ticks: 21 mm panels are narrow
            },
        },
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
        # max_fitness: one shared range across the three panels (panel d of the
        # combined figure). top128: None = fit each panel to its own data, since
        # a pinned 0.4-1.0 there would clip four rows, two of them median dots.
        # Per-panel fixed ranges: every method sits in a narrow band, so a
        # shared axis compresses the IQR whiskers below the dot diameter. Ticks
        # are round, fixed values bracketing the panel's Q1-Q3 extent; the axis
        # limits are derived from them (see fixed_xlim_from_ticks).
        "xrange": {
            "max_fitness": {
                "ms_AAV":     [0.50, 0.60, 0.70, 0.80],
                "ms_CreiLOV": [0.85, 0.90, 0.95, 1.00],
                "ms_PAB1":    [0.40, 0.50, 0.60],
            },
            "top128": {
                "ms_AAV":     [0.30, 0.50, 0.70],  # min Q1 = 0.396, max Q3 = 0.664
                "ms_CreiLOV": [0.75, 0.85, 0.95],
                "ms_PAB1":    [0.25, 0.35, 0.45, 0.55],
            },
        },
        "prefix": {
            "max_fitness": "main_figure_multisite_max_fitness_raincloud",
            "top128": "supplementary_figure_multisite_top128_mean_fitness_raincloud",
        },
        "panel_letter": "d",
        "label_kind": "multi-site datasets",
    },
}

# Metric to rank/plot by. rank_col must exist in each task's source CSV.
# fig_w_mm / fig_h_mm set the exact print size (max fitness is the single-column
# main figure; top-128 is drawn wider for the supplementary layout).
METRIC_CONFIG = {
    "max_fitness": {
        "rank_col": "max_fitness_median",
        "xlabel": "Median max fitness",
        "fig_w_mm": 89,
        "fig_h_mm": 40,
    },
    "top128": {
        "rank_col": "top128_median",
        "xlabel": "Top-128 fitness",
        "fig_w_mm": 89,
        "fig_h_mm": 40,
    },
}


# ------------------------------------------------------------------
# Dot + IQR whisker row drawing
# ------------------------------------------------------------------
# Uniform for every method: only colour marks the highlighted method.
# DOT_SIZE is shared with the trajectory figures via plot_style_utils.
WHISKER_LW = 0.5
CAP_SIZE = 1.5   # points, so it stays fixed in print
DOT_RADIUS_MM = DOT_DIAMETER_MM / 2  # sets the right-edge clearance


def draw_dot_whisker_row(ax, values, y_center, dot_color, whisker_color):
    """Draw one horizontal row: Q1-Q3 whisker with caps + median dot."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return

    q1, med, q3 = np.percentile(values, [25, 50, 75])
    ax.errorbar(med, y_center, xerr=[[med - q1], [q3 - med]], fmt="none",
                ecolor=whisker_color, elinewidth=WHISKER_LW,
                capsize=CAP_SIZE, capthick=WHISKER_LW, zorder=4)
    ax.scatter([med], [y_center], s=DOT_SIZE, color=dot_color,
               edgecolors="none", linewidths=0, zorder=5)


def fixed_xlim_from_ticks(ticks, axis_width_mm):
    """Axis limits for a fixed tick list: starts exactly on the first tick.

    The right bound clears the last tick by half a dot so a median landing on it
    (e.g. 4-site methods that reach 1.0) is not sliced by the panel edge.
    """
    lo, hi = ticks[0], ticks[-1]
    margin = DOT_RADIUS_MM * (hi - lo) / axis_width_mm
    return (lo, hi + margin)


def tick_decimals(tick_lists):
    """1 decimal if every tick lands on a 0.1 grid, else 2 — one format per figure."""
    flat = [t for ticks in tick_lists for t in ticks]
    return 1 if all(abs(t * 10 - round(t * 10)) < 1e-9 for t in flat) else 2


def warn_if_clipped(panel_values, methods, dataset, xlim):
    """Report any drawn element (Q1, median, Q3) outside a pinned x range."""
    for method, values in zip(methods, panel_values):
        if len(values) < 4:
            continue
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        outside = [f"{name}={v:.3f}" for name, v in
                   (("Q1", q1), ("median", med), ("Q3", q3))
                   if v < xlim[0] or v > xlim[1]]
        if outside:
            print(f"  WARNING: {dataset} {method} outside xlim {xlim}: "
                  f"{', '.join(outside)}")


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
    FIG_WIDTH_MM = metric_cfg["fig_w_mm"]
    FIG_HEIGHT_MM = metric_cfg["fig_h_mm"]
    # Bottom band holds the x tick labels + the single x-axis label; sized to
    # keep them inside the page now that it is no longer cropped to content.
    AXIS_BOTTOM, AXIS_TOP = 0.175, 0.90  # must match the subplots_adjust call below
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(FIG_WIDTH_MM * MM_TO_IN, FIG_HEIGHT_MM * MM_TO_IN),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    # Margins are set in absolute mm (not a fixed fraction) so the label column
    # and inter-panel gaps stay constant in size as the figure widens. Resolved
    # before drawing because the panel width sets the x-limit margin below.
    # left_mm fits the longest method label ("MULTIevolve") inside the page.
    # right_pad_mm holds the half of the last tick label that overhangs the panel
    # edge, now that the axis stops just past the final tick.
    # gap_mm must fit the facing halves of two 4-character tick labels (~4.2 mm
    # wide each) now that ticks sit at the panel edges.
    left_mm, right_pad_mm, gap_mm = 14.0, 1.8, 4.8
    left = left_mm / FIG_WIDTH_MM
    right = 1 - right_pad_mm / FIG_WIDTH_MM
    axis_width_mm = ((right - left) * FIG_WIDTH_MM - (n_panels - 1) * gap_mm) / n_panels

    raw_from_display = {DISPLAY_NAMES.get(m, m): m for m in methods}

    # A tick list per panel; one shared list is broadcast to every panel.
    spec = cfg["xrange"][metric]
    panel_ticks = (dict(spec) if isinstance(spec, dict)
                   else {ds: spec for ds in dataset_order})
    tick_format = FormatStrFormatter(f"%.{tick_decimals(panel_ticks.values())}f")

    for col, (ax, dataset) in enumerate(zip(axes, dataset_order)):
        panel_values = [load_seed_values(method, dataset, task, metric=metric)
                        for method in methods]
        for row_i, method in enumerate(methods):
            vals = panel_values[row_i]
            if not vals:
                continue
            dot_color, whisker_color = _row_colors(method, cfg)
            draw_dot_whisker_row(ax, vals, y_centers[row_i],
                                 dot_color, whisker_color)

        xticks = panel_ticks[dataset]
        xlim = fixed_xlim_from_ticks(xticks, axis_width_mm)
        warn_if_clipped(panel_values, methods, dataset, xlim)
        ax.xaxis.set_major_formatter(tick_format)
        style_axis_hbar(ax, xlim, xticks)
        ax.set_ylim(-0.30, (n_methods - 1) * ROW_STEP + 0.36)
        ax.set_yticks(y_centers)
        ax.set_title(cfg["dataset_labels"][dataset], pad=4, fontsize=TITLE_FONTSIZE)
        if col == n_panels // 2:  # one x-axis label, centred under the middle panel
            ax.set_xlabel(metric_cfg["xlabel"], labelpad=3, fontsize=XLABEL_FONTSIZE)

        if col == 0:
            ax.set_yticklabels(
                [DISPLAY_NAMES.get(m, m) for m in methods], fontsize=BASE_FONTSIZE)
            for tick in ax.get_yticklabels():
                raw = raw_from_display.get(tick.get_text(), tick.get_text())
                if raw in cfg["highlight"]:
                    tick.set_color(AV_DOT)
        else:
            ax.tick_params(axis="y", labelleft=False)

    fig.subplots_adjust(left=left, right=right, bottom=AXIS_BOTTOM, top=AXIS_TOP,
                        wspace=gap_mm / axis_width_mm)

    # bbox_inches=None: the layout is authored in absolute mm, so the page must
    # stay exactly fig_w_mm x fig_h_mm rather than being cropped to content.
    save_figure(fig, effective_outdir, cfg["prefix"][metric], bbox_inches=None)
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

    csv_path = TASKS[args.task]["csv"]
    if not os.path.exists(csv_path):
        sys.exit(f"Source CSV not found: {csv_path}")

    plot_raincloud_figure(args.task, metric=args.metric, outdir=args.outdir)
    print(f"Raincloud figure ({args.task}, {args.metric}) complete.")


if __name__ == "__main__":
    main()
