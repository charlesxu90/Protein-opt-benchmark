#!/usr/bin/env python
"""
Render the main-figure max-fitness bar chart (and the supplementary
top-128 mean-fitness bar chart) using median + Q1/Q3 IQR error bars
instead of the mean ± std presentation in `draw_figures.py`.

Input: a CSV with columns
    dataset, method,
    max_fitness_median, max_fitness_q1, max_fitness_q3,
    top128_median, top128_q1, top128_q3, n
(produced by `scripts/build_median_iqr_csv.py`).

Outputs (per --outdir):
    main_figure_max_fitness_median_iqr.{png,pdf}
    supplementary_figure_top128_mean_fitness_median_iqr.{png,pdf}
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

_DEFAULT_BASE = "/home/xux/Desktop/AlphaVariant/Benchmark/figures"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--csv", default=os.path.join(_DEFAULT_BASE, "alphavariant_comparison_median_iqr.csv"))
parser.add_argument("--outdir", default=_DEFAULT_BASE)
_args = parser.parse_args()

OUTDIR = _args.outdir
os.makedirs(OUTDIR, exist_ok=True)
source_csv = _args.csv
if not os.path.exists(source_csv):
    raise FileNotFoundError(f"Source CSV not found at {source_csv}.")
df = pd.read_csv(source_csv)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.65,
    "axes.labelsize": 8.2,
    "axes.titlesize": 8.8,
    "xtick.labelsize": 6.1,
    "ytick.labelsize": 6.8,
    "figure.dpi": 180,
    "savefig.dpi": 600,
})

colors = {
    "AlphaVariant": "#8B1A1A",
    "ALDE": "#0072B2",
    "FLEXS": "#009E73",
    "ftMLDE": "#E69F00",
    "CLADE": "#CC79A7",
    "GreedyWalk": "#56B4E9",
    "AiCE": "#D55E00",
    "Random": "#7F7F7F",
    "delta_cs": "#6A3D9A",
    "MULTIevolve": "#1F9E89",
    "EVOLVEpro": "#B26000",
}
HIGHLIGHT_METHODS = {"AlphaVariant"}

# Main-figure method allow-list (drops AlphaVariant_base/_SHAP/_PLM/_Hybrid ablation rows).
MAIN_METHODS = {"Random", "GreedyWalk", "ALDE", "FLEXS", "AiCE",
                "ftMLDE", "CLADE", "delta_cs", "AlphaVariant",
                "MULTIevolve", "EVOLVEpro"}

dataset_order = ["4site_GB1", "4site_PhoQ", "4site_TEV", "4site_TRPB"]
dataset_labels = {
    "4site_GB1":  "GB1 4-site",
    "4site_PhoQ": "PhoQ 4-site",
    "4site_TEV":  "TEV 4-site",
    "4site_TRPB": "TrpB 4-site",
}


def lighten(hex_color, amount=0.12):
    h = hex_color.lstrip("#")
    rgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0
    return tuple(rgb + (1 - rgb) * amount)


def style_axis(ax, ylim, yticks):
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.55, zorder=0)
    ax.xaxis.grid(False)
    ax.tick_params(axis="both", length=2.4, width=0.55, color="#333333", pad=2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#333333")
        ax.spines[spine].set_linewidth(0.6)


def plot_metric_figure(metric_med, metric_q1, metric_q3, ylabel, title,
                        output_prefix, panel_letter, ylim, yticks):
    fig, axes = plt.subplots(1, 4, figsize=(7.25, 2.72), sharey=True)
    fig.patch.set_facecolor("white")

    for col, (ax, dataset) in enumerate(zip(axes, dataset_order)):
        sub = df[(df["dataset"] == dataset) & (df["method"].isin(MAIN_METHODS))].copy()
        # Primary sort: median ascending. Secondary sort: Q1 ascending — breaks
        # ceiling ties (e.g. GB1 where multiple methods all have median = 1.0)
        # by putting the method with the tightest lower quartile to the right.
        sub = sub.sort_values([metric_med, metric_q1], ascending=[True, True])
        methods = sub["method"].tolist()
        med = sub[metric_med].to_numpy(dtype=float)
        q1 = sub[metric_q1].to_numpy(dtype=float)
        q3 = sub[metric_q3].to_numpy(dtype=float)
        # Asymmetric error bars: (lower, upper) distances from the median.
        err_lo = np.clip(med - q1, 0.0, None)
        err_hi = np.clip(q3 - med, 0.0, None)
        yerr = np.vstack([err_lo, err_hi])
        x = np.arange(len(methods))

        bar_colors = [lighten(colors[m], 0.05 if m in HIGHLIGHT_METHODS else 0.16) for m in methods]
        edge_colors = ["#111111" if m in HIGHLIGHT_METHODS else "white" for m in methods]
        line_widths = [0.80 if m in HIGHLIGHT_METHODS else 0.30 for m in methods]

        bars = ax.bar(x, med, width=0.76, color=bar_colors, edgecolor=edge_colors,
                      linewidth=line_widths, zorder=3)
        ax.errorbar(x, med, yerr=yerr, fmt="none", ecolor="#333333",
                    elinewidth=0.6, capsize=1.8, capthick=0.55, zorder=4)

        offset = (ylim[1] - ylim[0]) * 0.018
        for bar, m_val, hi in zip(bars, med, err_hi):
            ax.text(bar.get_x() + bar.get_width() / 2, m_val + hi + offset,
                    f"{m_val:.3f}",
                    ha="center", va="bottom", fontsize=5.0, rotation=90,
                    color="#303030", clip_on=False)

        ax.set_title(dataset_labels[dataset], pad=7, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=60, ha="right", rotation_mode="anchor")
        for tick in ax.get_xticklabels():
            if tick.get_text() in HIGHLIGHT_METHODS:
                tick.set_fontweight("bold")
                tick.set_color("#8B1A1A")
        style_axis(ax, ylim, yticks)
        if col == 0:
            ax.set_ylabel(ylabel, labelpad=4)
        else:
            ax.tick_params(axis="y", length=0)

    fig.text(0.012, 0.985, panel_letter, ha="left", va="top",
              fontsize=10.5, fontweight="bold")
    fig.suptitle(title, x=0.066, y=0.985, ha="left", va="top",
                  fontsize=10.5, fontweight="bold")
    fig.text(
        0.081, 0.012,
        "Bars show median across 30 seeds per method per dataset; error bars span Q1–Q3 (IQR). "
        "Within each panel methods are ordered by median ascending; ties broken by Q1 ascending.",
        ha="left", va="bottom", fontsize=5.5, color="#4D4D4D",
    )
    fig.subplots_adjust(left=0.08, right=0.996, bottom=0.34, top=0.82, wspace=0.28)

    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUTDIR, f"{output_prefix}.{ext}"),
                     bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


plot_metric_figure(
    metric_med="max_fitness_median",
    metric_q1="max_fitness_q1",
    metric_q3="max_fitness_q3",
    ylabel="Median max fitness",
    title="Max fitness across datasets — median ± Q1–Q3 IQR",
    output_prefix="main_figure_max_fitness_median_iqr",
    panel_letter="a",
    ylim=(0.0, 1.1),
    yticks=np.arange(0.0, 1.01, 0.2),
)

plot_metric_figure(
    metric_med="top128_median",
    metric_q1="top128_q1",
    metric_q3="top128_q3",
    ylabel="Median of top-128 mean fitness",
    title="Top-128 mean fitness across datasets — median ± Q1–Q3 IQR",
    output_prefix="supplementary_figure_top128_mean_fitness_median_iqr",
    panel_letter="a",
    ylim=(0.0, 0.82),
    yticks=np.arange(0.0, 0.71, 0.1),
)

print(f"Wrote median+IQR figures to {OUTDIR}")
