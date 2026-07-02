import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BENCH_ROOT)

from utils.plot_style_utils import (
    COMPACT_6PT_RCPARAMS, apply_nature_rcparams, save_figure, style_axis_vbar,
)

_DEFAULT_BASE = os.path.join(_BENCH_ROOT, "figures")

parser = argparse.ArgumentParser(description="Render main + supplementary AlphaVariant comparison figures.")
parser.add_argument("--csv", default=os.path.join(_DEFAULT_BASE, "alphavariant_comparison_values.csv"),
                    help="Source comparison-values CSV.")
parser.add_argument("--outdir", default=_DEFAULT_BASE,
                    help="Directory where the rendered figures (PNG+PDF+SVG) will be written.")
_args = parser.parse_args()

OUTDIR = _args.outdir
os.makedirs(OUTDIR, exist_ok=True)

source_csv = _args.csv
if not os.path.exists(source_csv):
    raise FileNotFoundError(
        f"Source CSV not found at {source_csv}. "
        "Run the upstream aggregation step to regenerate it."
    )
df = pd.read_csv(source_csv)

# Mirror the source CSV into the output directory for provenance.
_mirror_path = os.path.join(OUTDIR, "alphavariant_comparison_values.csv")
if os.path.abspath(_mirror_path) != os.path.abspath(source_csv):
    df.to_csv(_mirror_path, index=False)

apply_nature_rcparams(COMPACT_6PT_RCPARAMS)

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
}
HIGHLIGHT_METHODS = {"AlphaVariant"}

# Main-figure method allow-list. AlphaVariant_base / _plm / _hybrid are
# ablation-only and excluded from the main figure (they appear in the
# supplementary ablation figure produced by draw_supplementary_ablation.py).
MAIN_METHODS = {"Random", "GreedyWalk", "ALDE", "FLEXS", "AiCE",
                "ftMLDE", "CLADE", "delta_cs", "AlphaVariant"}

dataset_order = ["4site_GB1", "4site_PhoQ", "4site_TEV", "4site_TRPB"]
dataset_labels = {
    "4site_GB1": "GB1 4-site",
    "4site_PhoQ": "PhoQ 4-site",
    "4site_TEV": "TEV 4-site",
    "4site_TRPB": "TrpB 4-site",
}


def lighten(hex_color, amount=0.12):
    hex_color = hex_color.lstrip("#")
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0
    return tuple(rgb + (1 - rgb) * amount)


def style_axis(ax, ylim, yticks):
    style_axis_vbar(ax)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)


def plot_metric_figure(metric, ylabel, output_prefix, ylim, yticks, std_col):
    MM_TO_IN = 1 / 25.4
    fig, axes = plt.subplots(1, 4, figsize=(184 * MM_TO_IN, 69 * MM_TO_IN), sharey=True)
    fig.patch.set_facecolor("white")

    for col, (ax, dataset) in enumerate(zip(axes, dataset_order)):
        # Filter to main-figure methods only (drops ablation-only AlphaVariant
        # variants), then sort within each panel by the metric shown.
        sub = df[(df["dataset"] == dataset) & (df["method"].isin(MAIN_METHODS))].copy()
        sub = sub.sort_values(metric, ascending=True)
        methods = sub["method"].tolist()
        vals = sub[metric].to_numpy(dtype=float)
        stds = sub[std_col].to_numpy(dtype=float) if std_col in sub.columns else np.zeros(len(vals))
        x = np.arange(len(methods))

        bar_colors = [lighten(colors[m], 0.05 if m in HIGHLIGHT_METHODS else 0.16) for m in methods]
        edge_colors = ["#111111" if m in HIGHLIGHT_METHODS else "white" for m in methods]
        line_widths = [0.80 if m in HIGHLIGHT_METHODS else 0.30 for m in methods]

        bars = ax.bar(x, vals, width=0.76, color=bar_colors, edgecolor=edge_colors, linewidth=line_widths, zorder=3)
        ax.errorbar(x, vals, yerr=stds, fmt="none", ecolor="#333333", elinewidth=0.6,
                    capsize=1.8, capthick=0.55, zorder=4)
        best_i = int(np.argmax(vals))
        ax.scatter(x[best_i], vals[best_i], s=40, facecolors="none", edgecolors="#111111", linewidths=0.75, zorder=5)

        offset = (ylim[1] - ylim[0]) * 0.018
        for bar, val, sd in zip(bars, vals, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, val + sd + offset, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=6.0, rotation=90,
                    color="#303030", clip_on=False)

        ax.set_title(dataset_labels[dataset], pad=7, fontweight="bold", fontsize=6.0)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=60, ha="right", rotation_mode="anchor", fontsize=6.0)
        for tick in ax.get_xticklabels():
            if tick.get_text() in HIGHLIGHT_METHODS:
                tick.set_fontweight("bold")
                tick.set_color("#8B1A1A")
        style_axis(ax, ylim, yticks)
        if col == 0:
            ax.set_ylabel(ylabel, labelpad=4, fontsize=6.0)
        else:
            ax.tick_params(axis="y", length=0)

    fig.text(
        0.081,
        0.012,
        "Within each dataset, methods are ordered from lowest to highest mean value; open circle marks the best method.",
        ha="left",
        va="bottom",
        fontsize=6.0,
        color="#4D4D4D",
    )
    fig.subplots_adjust(left=0.08, right=0.996, bottom=0.34, top=0.92, wspace=0.28)

    save_figure(fig, OUTDIR, output_prefix)
    plt.close(fig)


plot_metric_figure(
    metric="max_fitness",
    ylabel="Max fitness",
    output_prefix="main_figure_max_fitness_four_datasets",
    ylim=(0.0, 1.1),
    yticks=np.arange(0.0, 1.01, 0.2),
    std_col="max_fitness_std",
)

plot_metric_figure(
    metric="top128_mean_fitness",
    ylabel="Mean of top 128 fitness",
    output_prefix="supplementary_figure_top128_mean_fitness_four_datasets",
    ylim=(0.0, 0.82),
    yticks=np.arange(0.0, 0.71, 0.1),
    std_col="top128_std",
)

print(f"Generated final separated-metric figures in {OUTDIR}")
