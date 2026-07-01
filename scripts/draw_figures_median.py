#!/usr/bin/env python
"""
Render the main-figure max-fitness dot plot (and the supplementary
top-128 mean-fitness dot plot) as horizontal dots with Q1/Q3 IQR whiskers.

Nature-style presentation:
  * horizontal layout (method names as y-tick labels, no rotation),
  * dot = median, whiskers = Q1-Q3 (IQR),
  * a single fixed method order per group (by mean rank across the group's
    datasets, best at top) shared by every panel,
  * vermilion AlphaVariant + two-tier gray baselines (no rainbow).

Supports two tasks (select with --task):
    4site     : the 4-site combinatorial benchmark (GB1/PhoQ/TrpB)  [default]
    multisite : the multi-site learned-oracle benchmark (AAV/CreiLOV/PAB1)

Input: a CSV with columns
    dataset, method,
    max_fitness_median, max_fitness_q1, max_fitness_q3,
    top128_median, top128_q1, top128_q3, n
(4site: produced by scripts/build_median_iqr_csv.py;
 multisite: produced by scripts/build_oracle_median_iqr_csv.py)

Outputs (per --outdir):
    main_figure[_multisite]_max_fitness_median_iqr.{png,pdf}
    supplementary_figure[_multisite]_top128_mean_fitness_median_iqr.{png,pdf}
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.plot_style_utils import (
    PRIMARY_COMPARISON, VERMILION, method_color,
    apply_nature_rcparams, save_figure, style_axis_hbar,
)

_DEFAULT_BASE = "/home/xux/Desktop/AlphaVariant/Benchmark/figures"

apply_nature_rcparams({"axes.labelsize": 8.2, "xtick.labelsize": 6.8})

# Display-name mapping: FLEXS algorithm is AdaLead (canonical name in the
# directed-evolution literature); user requested label rename in figures.
DISPLAY_NAMES = {
    "FLEXS": "AdaLead",
}

# ---------------------------------------------------------------------------
# Per-task configuration. 4site preserves the original behavior verbatim.
# ---------------------------------------------------------------------------
TASKS = {
    "4site": {
        "default_csv": os.path.join(_DEFAULT_BASE, "alphavariant_comparison_median_iqr.csv"),
        "default_outdir": _DEFAULT_BASE,
        "dataset_order": ["4site_GB1", "4site_PhoQ", "4site_TRPB"],
        "dataset_labels": {
            "4site_GB1": "GB1 4-site", "4site_PhoQ": "PhoQ 4-site",
            "4site_TRPB": "TrpB 4-site",
        },
        # Main-figure method allow-list (drops AlphaVariant ablation rows;
        # delta_cs excluded per user request).
        "main_methods": {"Random", "GreedyWalk", "ALDE", "FLEXS", "AiCE",
                         "ftMLDE", "CLADE", "AlphaVariant",
                         "MULTIevolve", "EVOLVEpro"},
        "highlight": {"AlphaVariant"},
        "max_prefix": "main_figure_max_fitness_median_iqr",
        "top_prefix": "supplementary_figure_top128_mean_fitness_median_iqr",
        "max_ylim": (0.0, 1.1), "max_yticks": np.arange(0.0, 1.01, 0.2),
        "top_ylim": (0.0, 0.82), "top_yticks": np.arange(0.0, 0.71, 0.1),
    },
    "multisite": {
        "default_csv": os.path.join(_DEFAULT_BASE, "ms_oracles", "multisite_oracle_median_iqr.csv"),
        "default_outdir": os.path.join(_DEFAULT_BASE, "ms_oracles"),
        "dataset_order": ["ms_AAV", "ms_CreiLOV", "ms_PAB1"],
        "dataset_labels": {
            "ms_AAV": "AAV multi-site", "ms_CreiLOV": "CreiLOV multi-site",
            "ms_PAB1": "PAB1 multi-site",
        },
        "main_methods": {"Random", "GreedyWalk", "ALDE", "CLADE", "ftMLDE",
                         "AdaLead", "MULTIevolve", "EVOLVEpro", "AiCE", "AlphaVariant"},
        "highlight": {"AlphaVariant"},
        "max_prefix": "main_figure_multisite_max_fitness_median_iqr",
        "top_prefix": "supplementary_figure_multisite_top128_mean_fitness_median_iqr",
        "max_ylim": (0.0, 1.1), "max_yticks": np.arange(0.0, 1.01, 0.2),
        "top_ylim": (0.0, 1.05), "top_yticks": np.arange(0.0, 1.01, 0.2),
    },
}


def display_name(method: str) -> str:
    """Return the figure-display label for a method (e.g. FLEXS → AdaLead)."""
    return DISPLAY_NAMES.get(method, method)


def global_method_order(df, cfg, metric_med):
    """Return one fixed method order for the group, best → worst.

    Order = ascending mean rank across the group's datasets, where per dataset
    rank 1 = best median. This single order is reused for every panel so the eye
    can track a method across datasets (reviewer: 'consistent ordering').
    """
    datasets = cfg["dataset_order"]
    main_methods = cfg["main_methods"]
    sub = df[(df["dataset"].isin(datasets)) & (df["method"].isin(main_methods))].copy()
    sub["rank"] = sub.groupby("dataset")[metric_med].rank(ascending=False, method="average")
    mean_rank = sub.groupby("method")["rank"].mean().sort_values()
    return mean_rank.index.tolist()


def plot_metric_figure(df, cfg, outdir, metric_med, metric_q1, metric_q3,
                       ylabel, title, output_prefix, panel_letter, ylim, yticks,
                       seed_note):
    # `ylim`/`yticks` describe the fitness axis; in the horizontal layout the
    # fitness axis is x, so they are applied as xlim/xticks.
    xlim, xticks = ylim, yticks
    dataset_order = cfg["dataset_order"]
    dataset_labels = cfg["dataset_labels"]
    main_methods = cfg["main_methods"]
    highlight = cfg["highlight"]

    # Fixed order (best→worst); plotted bottom→top so the best sits at the top.
    order_best_first = global_method_order(df, cfg, metric_med)
    methods_btt = order_best_first[::-1]
    y = np.arange(len(methods_btt))

    n_panels = len(dataset_order)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.15 * n_panels + 1.35, 3.35),
                             sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    for col, (ax, dataset) in enumerate(zip(axes, dataset_order)):
        sub = df[(df["dataset"] == dataset) & (df["method"].isin(main_methods))]
        sub = sub.set_index("method")
        med = np.array([sub.loc[m, metric_med] if m in sub.index else np.nan
                        for m in methods_btt], dtype=float)
        q1 = np.array([sub.loc[m, metric_q1] if m in sub.index else np.nan
                       for m in methods_btt], dtype=float)
        q3 = np.array([sub.loc[m, metric_q3] if m in sub.index else np.nan
                       for m in methods_btt], dtype=float)
        err_lo = np.clip(med - q1, 0.0, None)
        err_hi = np.clip(q3 - med, 0.0, None)

        pt_colors = [method_color(m) for m in methods_btt]
        pt_sizes = [40 if m in highlight else 26 for m in methods_btt]
        pt_edges = ["#111111" if m in highlight else "white" for m in methods_btt]
        pt_edge_w = [0.9 if m in highlight else 0.45 for m in methods_btt]

        # IQR whiskers first (behind the dots).
        ax.errorbar(med, y, xerr=np.vstack([err_lo, err_hi]), fmt="none",
                    ecolor="#5A5A5A", elinewidth=0.7, capsize=2.0,
                    capthick=0.6, zorder=3)
        ax.scatter(med, y, s=pt_sizes, c=pt_colors, edgecolors=pt_edges,
                   linewidths=pt_edge_w, zorder=4, clip_on=False)

        ax.set_title(dataset_labels[dataset], pad=6, fontweight="bold")
        ax.set_ylim(-0.6, len(methods_btt) - 0.4)
        ax.set_yticks(y)
        style_axis_hbar(ax, xlim, xticks)
        ax.set_xlabel(ylabel, labelpad=3)
        if col == 0:
            ax.set_yticklabels([display_name(m) for m in methods_btt])
            highlight_color = mcolors.to_hex(VERMILION)
            highlight_display = {display_name(m) for m in highlight}
            for tick in ax.get_yticklabels():
                if tick.get_text() in highlight_display:
                    tick.set_fontweight("bold")
                    tick.set_color(highlight_color)
        else:
            ax.tick_params(axis="y", length=0)

    fig.text(0.008, 0.985, panel_letter, ha="left", va="top",
             fontsize=10.5, fontweight="bold")
    fig.suptitle(title, x=0.052, y=0.985, ha="left", va="top",
                 fontsize=10.0, fontweight="bold")
    fig.text(
        0.052, 0.015,
        f"Dots indicate the median across {seed_note} independent seeds; "
        "whiskers indicate Q1–Q3 (interquartile range). Methods share a fixed "
        "order (mean rank across this group's datasets, best at top) in every "
        "panel. Pairwise comparisons: Bonferroni-corrected Wilcoxon signed-rank test.",
        ha="left", va="bottom", fontsize=5.6, color="#4D4D4D", wrap=True,
    )
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.20, top=0.86, wspace=0.18)

    save_figure(fig, outdir, output_prefix)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TASKS.keys()), default="4site")
    parser.add_argument("--csv", default=None, help="defaults to the task's CSV")
    parser.add_argument("--outdir", default=None, help="defaults to the task's outdir")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="subset/reorder of datasets to plot (default: task's full order)")
    args = parser.parse_args()
    cfg = TASKS[args.task]
    if args.datasets:
        cfg = {**cfg, "dataset_order": [d for d in cfg["dataset_order"]
                                        if d in args.datasets]}

    source_csv = args.csv or cfg["default_csv"]
    outdir = args.outdir or cfg["default_outdir"]
    os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(source_csv):
        raise FileNotFoundError(f"Source CSV not found at {source_csv}.")
    df = pd.read_csv(source_csv)
    seed_note = int(df["n"].max()) if "n" in df.columns else 30

    label_kind = "multi-site oracles" if args.task == "multisite" else "datasets"
    plot_metric_figure(
        df, cfg, outdir,
        metric_med="max_fitness_median", metric_q1="max_fitness_q1",
        metric_q3="max_fitness_q3",
        ylabel="Median max fitness",
        title=f"Maximum fitness across {label_kind} (median, Q1–Q3 whiskers)",
        output_prefix=cfg["max_prefix"], panel_letter="a",
        ylim=cfg["max_ylim"], yticks=cfg["max_yticks"], seed_note=seed_note,
    )
    plot_metric_figure(
        df, cfg, outdir,
        metric_med="top128_median", metric_q1="top128_q1",
        metric_q3="top128_q3",
        ylabel="Median of top-128 mean fitness",
        title=f"Top-128 mean fitness across {label_kind} (median, Q1–Q3 whiskers)",
        output_prefix=cfg["top_prefix"], panel_letter="a",
        ylim=cfg["top_ylim"], yticks=cfg["top_yticks"], seed_note=seed_note,
    )
    print(f"Wrote median+IQR figures ({args.task}) to {outdir}")


if __name__ == "__main__":
    main()
