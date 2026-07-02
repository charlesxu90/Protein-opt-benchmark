#!/usr/bin/env python
"""
Supplementary ablation figure for AlphaVariant.

4 panels (one per dataset). Each panel shows 4 bars:
    1. AlphaVariant         — Tier 1B base (shipped default)
    2. AlphaVariant + PLM-reward
    3. AlphaVariant + Hybrid (weighted-allocation selection)
    4. AlphaVariant + SHAP  — alphabet pruning

Each bar represents the base method augmented with one optional
extension. None of the extensions universally improves over the base at
n=30 (paired Wilcoxon p > 0.10 on every dataset / extension combination);
the figure documents the per-landscape pattern. Combos that were not
evaluated on a given dataset are drawn as empty hatched bars with "NR".
Source data: `figures/alphavariant_comparison_values.csv`.
"""
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

parser = argparse.ArgumentParser(description="Render the supplementary AlphaVariant ablation figure.")
parser.add_argument("--csv", default=os.path.join(_DEFAULT_BASE, "alphavariant_comparison_values.csv"))
parser.add_argument("--outdir", default=_DEFAULT_BASE)
parser.add_argument("--shipped-default", default="AlphaVariant",
                    help="Which CSV method name represents the shipped method (red bold x-tick).")
parser.add_argument("--bar-methods", default="AlphaVariant,AlphaVariant_PLM,AlphaVariant_Hybrid,AlphaVariant_SHAP",
                    help="Comma-separated CSV method names for the 4 ablation bars, left → right. "
                         "Plan A default. For Plan B pass: "
                         "'AlphaVariant_base,AlphaVariant_PLM,AlphaVariant_Hybrid,AlphaVariant'.")
parser.add_argument("--bar-labels", default="base,+PLM-reward,+Hybrid,+SHAP",
                    help="Comma-separated display labels for the 4 ablation bars.")
_args = parser.parse_args()

OUTDIR = _args.outdir
os.makedirs(OUTDIR, exist_ok=True)
SRC = _args.csv
SHIPPED_DEFAULT = _args.shipped_default

df = pd.read_csv(SRC)

apply_nature_rcparams(COMPACT_6PT_RCPARAMS)

# Ablation bar order (left → right) — populated from CLI.
_methods = [m.strip() for m in _args.bar_methods.split(",")]
_labels = [l.strip() for l in _args.bar_labels.split(",")]
if len(_methods) != len(_labels):
    raise SystemExit("--bar-methods and --bar-labels must have the same count.")
ABLATION_ORDER = list(zip(_methods, _labels))

# Colours: shipped-default gets the headline red; other slots get muted reds /
# greys distinguishable from each other.
_PALETTE = {
    "shipped": "#8B1A1A",
    "base":    "#B3B3B3",
    "plm":     "#FF7F50",
    "hybrid":  "#9B2D30",
    "shap":    "#D62728",
}
def _bar_color(method_name):
    if method_name == SHIPPED_DEFAULT:
        return _PALETTE["shipped"]
    if method_name == "AlphaVariant_base":
        return _PALETTE["base"]
    if "PLM" in method_name:
        return _PALETTE["plm"]
    if "Hybrid" in method_name:
        return _PALETTE["hybrid"]
    if "SHAP" in method_name:
        return _PALETTE["shap"]
    return "#888888"
BAR_COLORS = [_bar_color(m) for m, _ in ABLATION_ORDER]

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


def plot_ablation(metric, ylabel, output_prefix, ylim, yticks, std_col):
    MM_TO_IN = 1 / 25.4
    fig, axes = plt.subplots(1, 4, figsize=(184 * MM_TO_IN, 66 * MM_TO_IN), sharey=True)
    fig.patch.set_facecolor("white")

    for col, (ax, dataset) in enumerate(zip(axes, dataset_order)):
        sub = df[df["dataset"] == dataset]
        method_to_val = dict(zip(sub["method"], sub[metric]))
        method_to_std = dict(zip(sub["method"], sub[std_col]))

        vals, stds, present = [], [], []
        for m, _ in ABLATION_ORDER:
            v = method_to_val.get(m)
            s = method_to_std.get(m)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                vals.append(0.0); stds.append(0.0); present.append(False)
            else:
                vals.append(float(v)); stds.append(float(s) if s is not None else 0.0); present.append(True)

        x = np.arange(len(ABLATION_ORDER))
        bar_colors_pn = [lighten(c, 0.05) for c in BAR_COLORS]
        bars = ax.bar(x, vals, width=0.74,
                       color=[bc if p else "white" for bc, p in zip(bar_colors_pn, present)],
                       edgecolor=["#111111" if p else "#BBBBBB" for p in present],
                       linewidth=[0.7 if p else 0.5 for p in present],
                       hatch=[None if p else "//" for p in present],
                       zorder=3)
        ax.errorbar(x[np.array(present)], np.array(vals)[np.array(present)],
                     yerr=np.array(stds)[np.array(present)], fmt="none",
                     ecolor="#333333", elinewidth=0.6, capsize=1.8, capthick=0.55, zorder=4)

        # Best-of-present open circle
        present_idx = np.where(np.array(present))[0]
        if len(present_idx) > 0:
            best_local = present_idx[np.argmax(np.array(vals)[present_idx])]
            ax.scatter(x[best_local], vals[best_local], s=40, facecolors="none",
                        edgecolors="#111111", linewidths=0.75, zorder=5)

        offset = (ylim[1] - ylim[0]) * 0.018
        for bar, v, s, p in zip(bars, vals, stds, present):
            if p:
                ax.text(bar.get_x() + bar.get_width() / 2, v + s + offset, f"{v:.3f}",
                         ha="center", va="bottom", fontsize=6.0, rotation=90,
                         color="#303030", clip_on=False)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, ylim[0] + offset, "NR",
                         ha="center", va="bottom", fontsize=6.0, fontweight="bold",
                         color="#888888", clip_on=False)

        ax.set_title(dataset_labels[dataset], pad=7, fontweight="bold", fontsize=6.0)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in ABLATION_ORDER], rotation=35, ha="right",
                            rotation_mode="anchor", fontsize=6.0)
        # Highlight the shipped-default bar on the x-axis (red, bold).
        for tick, (m, _) in zip(ax.get_xticklabels(), ABLATION_ORDER):
            if m == SHIPPED_DEFAULT:
                tick.set_fontweight("bold")
                tick.set_color("#8B1A1A")
        style_axis(ax, ylim, yticks)
        if col == 0:
            ax.set_ylabel(ylabel, labelpad=4, fontsize=6.0)
        else:
            ax.tick_params(axis="y", length=0)

    fig.text(
        0.081, 0.012,
        "AlphaVariant extension ablation. Leftmost (red) bar is the shipped AlphaVariant base; "
        "remaining bars add one optional extension. NR = not evaluated. None of the extensions "
        "universally improves over the base at n=30 (paired Wilcoxon p > 0.10).",
        ha="left", va="bottom", fontsize=6.0, color="#4D4D4D",
    )
    fig.subplots_adjust(left=0.08, right=0.996, bottom=0.34, top=0.92, wspace=0.28)

    save_figure(fig, OUTDIR, output_prefix)
    plt.close(fig)


plot_ablation(
    metric="max_fitness",
    ylabel="Max fitness",
    output_prefix="supp_figure_ablation_max_fitness",
    ylim=(0.0, 1.1),
    yticks=np.arange(0.0, 1.01, 0.2),
    std_col="max_fitness_std",
)

plot_ablation(
    metric="top128_mean_fitness",
    ylabel="Mean of top 128 fitness",
    output_prefix="supp_figure_ablation_top128_mean_fitness",
    ylim=(0.0, 0.82),
    yticks=np.arange(0.0, 0.71, 0.1),
    std_col="top128_std",
)

print(f"Generated supplementary ablation figures in {OUTDIR}")
