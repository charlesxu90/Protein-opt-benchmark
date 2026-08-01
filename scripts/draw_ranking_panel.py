#!/usr/bin/env python3
"""Average-method-ranking panel for the AlphaVariant benchmark.

One single-panel lollipop figure per benchmark x metric (four-site / multi-site
crossed with max fitness / top-128), showing each method's mean rank across that
group's datasets. Visual encoding follows
``figures/alphavariant_ranking_panel_gray_preferred.py``:

- light-gray dots: a method's rank in each individual dataset,
- open black circles: the mean rank across datasets,
- dark-red filled circle + red label: AlphaVariant's mean rank.

Unlike the prototype, ranks are computed directly from the benchmark median
CSVs (``figures/alphavariant_comparison_median_iqr.csv`` and
``figures/ms_oracles/multisite_oracle_median_iqr.csv``) for the chosen metric,
so the figure stays in sync with the data. Lower rank = better (rank 1 = best
median on that dataset; ties share the average rank).

Outputs ``ranking_panel_<group>_<metric>.pdf`` into ``--outdir``
(default: figures/), plus a matching ``_values.csv`` of the rank table.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FIGDIR = os.path.join(ROOT, "figures")
MM_TO_IN = 1 / 25.4

# --- palette -----------------------------------------------------------------
sys.path.insert(0, ROOT)
from utils.plot_style_utils import (  # noqa: E402
    AXIS_LW, GRID_COLOR, GRID_DASH, apply_nature_rcparams, save_figure,
)

ALPHA_RED = "#C0392B"   # AlphaVariant mean rank + its tick label
DOT_GRAY = "#AAAAAA"    # individual dataset ranks
MEAN_EDGE = "#333333"   # mean rank across datasets (open circle)
LINE_GRAY = "#D4D4D4"

# Marker diameters in points; scatter() takes the area, hence the squares.
DATASET_DOT_PT = 2.0
MEAN_DOT_PT = 3.0
DATASET_DOT_SIZE = DATASET_DOT_PT ** 2
MEAN_DOT_SIZE = MEAN_DOT_PT ** 2

# Canonical method label set (10 methods). FLEXS (4-site) and AdaLead
# (multi-site) are the same AdaLead algorithm and are unified under "AdaLead".
DISPLAY_NAMES = {"FLEXS": "AdaLead"}
METHOD_ORDER = [
    "Random", "GreedyWalk", "ftMLDE", "ALDE", "CLADE",
    "EVOLVEpro", "MULTIevolve", "AiCE", "AdaLead", "AlphaVariant",
]

# Per-group main-figure method allow-lists (mirror draw_figures_median.py:
# AlphaVariant_* ablations and delta_cs are excluded from the headline figure).
MAIN_METHODS_4SITE = {
    "Random", "GreedyWalk", "ALDE", "FLEXS", "AiCE",
    "ftMLDE", "CLADE", "AlphaVariant", "MULTIevolve", "EVOLVEpro",
}
MAIN_METHODS_MULTISITE = {
    "Random", "GreedyWalk", "ALDE", "CLADE", "ftMLDE",
    "AdaLead", "MULTIevolve", "EVOLVEpro", "AiCE", "AlphaVariant",
}

GROUPS = {
    "Four-site": {
        "slug": "4site",
        "csv": os.path.join(DEFAULT_FIGDIR, "alphavariant_comparison_median_iqr.csv"),
        "allow": MAIN_METHODS_4SITE,
        "dataset_labels": {
            "4site_GB1": "GB1", "4site_PhoQ": "PhoQ",
            "4site_TRPB": "TrpB",
        },
    },
    "Multisite": {
        "slug": "multisite",
        "csv": os.path.join(DEFAULT_FIGDIR, "ms_oracles", "multisite_oracle_median_iqr.csv"),
        "allow": MAIN_METHODS_MULTISITE,
        "dataset_labels": {
            "ms_AAV": "AAV", "ms_CreiLOV": "CreiLOV",
            "ms_PAB1": "PAB1",
        },
    },
}

# Print size. The text bands are absolute mm so the panel absorbs any height
# change: the top holds the "<benchmark>, <metric>" caption, the bottom the x
# tick labels, the x-axis label and the legend.
FIG_WIDTH_MM = 57
FIG_HEIGHT_MM = 40
TOP_BAND_MM = 4.4  # benchmark + metric caption
# The bottom band stacks, from the axes down: x tick labels, the x-axis label
# and a two-row legend. At 57 mm the legend cannot run in one row, so it wraps
# to two columns; the band is sized to that.
BOTTOM_BAND_MM = 12.4

apply_nature_rcparams()


def build_rank_table(group_cfg: dict, metric_col: str) -> pd.DataFrame:
    """Rank methods within each dataset of a group by ``metric_col`` (desc).

    Returns a method x dataset table of ranks plus 'Mean rank' and 'Mean score'
    columns, sorted best-to-worst by mean rank, with mean score (higher = better)
    breaking ties. Rank 1 = best median; ties share the average rank.
    """
    df = pd.read_csv(group_cfg["csv"])
    df = df[df["method"].isin(group_cfg["allow"])].copy()
    df["method"] = df["method"].replace(DISPLAY_NAMES)

    datasets = list(group_cfg["dataset_labels"].keys())
    df = df[df["dataset"].isin(datasets)]

    # rank within each dataset: higher metric -> better -> lower (=1) rank.
    df["rank"] = df.groupby("dataset")[metric_col].rank(ascending=False, method="average")

    table = df.pivot(index="method", columns="dataset", values="rank")
    table = table.reindex(METHOD_ORDER).dropna(how="all")
    # keep dataset columns in declared order
    table = table[[d for d in datasets if d in table.columns]]
    table["Mean rank"] = table.mean(axis=1, skipna=True)
    # mean metric score across datasets (higher = better) — the tiebreaker.
    table["Mean score"] = df.groupby("method")[metric_col].mean().reindex(table.index)
    # primary key: mean rank ascending (lower = better);
    # tiebreaker: mean score descending (higher = better).
    table = table.sort_values(["Mean rank", "Mean score"], ascending=[True, False],
                              na_position="last")
    return table


def _draw_group(ax, table: pd.DataFrame, n_methods: int) -> None:
    # reverse so the best mean rank sits at the top
    methods = table.index.tolist()[::-1]
    tp = table.loc[methods]
    dataset_cols = [c for c in tp.columns if c not in ("Mean rank", "Mean score")]
    y = np.arange(len(tp))
    mean_rank = tp["Mean rank"].to_numpy(float)

    worst = n_methods  # stem origin = worst possible rank
    ax.hlines(y, worst, mean_rank, color=LINE_GRAY, lw=0.75, zorder=1)

    # individual dataset ranks: deliberately gray, jittered vertically
    offsets = np.linspace(-0.20, 0.20, len(dataset_cols))
    for di, ds in enumerate(dataset_cols):
        vals = tp[ds].to_numpy(float)
        mask = ~np.isnan(vals)
        ax.scatter(vals[mask], y[mask] + offsets[di], s=DATASET_DOT_SIZE,
                   facecolor=DOT_GRAY, edgecolors="none", linewidth=0,
                   alpha=0.85, zorder=2)

    # mean rank: open circles, AlphaVariant as red filled
    is_alpha = np.array([m == "AlphaVariant" for m in tp.index])
    ax.scatter(mean_rank[~is_alpha], y[~is_alpha], s=MEAN_DOT_SIZE,
               facecolor="white", edgecolor=MEAN_EDGE, linewidth=0.5, zorder=4)
    ax.scatter(mean_rank[is_alpha], y[is_alpha], s=MEAN_DOT_SIZE,
               facecolor=ALPHA_RED, edgecolors="none", linewidth=0, zorder=5)

    # Single panel per figure, so the axis label centres itself on the plot area
    # and the metric name moves up into the caption. "(1 = best)" carries the
    # direction that the shared "better" arrow used to.
    ax.set_xlabel("Mean rank (1 = best)", fontsize=7, fontweight="normal",
                  labelpad=2)
    ax.set_yticks(y)
    ax.set_yticklabels(tp.index, fontsize=6)
    for lab in ax.get_yticklabels():
        if lab.get_text() == "AlphaVariant":
            lab.set_color(ALPHA_RED)
            lab.set_fontweight("bold")

    ax.set_xlim(0.55, worst + 0.4)  # Rank 1 (best) on the left
    ticks = sorted({1, 2, 4, 6, 8, worst})
    ax.set_xticks([t for t in ticks if t <= worst])
    # x-axis label and the "better" arrow are drawn once per figure, in
    # draw_panel, centred on the plot area.
    ax.grid(axis="x", color=GRID_COLOR, lw=0.55, linestyle=GRID_DASH)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_linewidth(AXIS_LW)
    ax.spines["bottom"].set_linewidth(AXIS_LW)
    ax.tick_params(axis="y", labelsize=6, length=2.4, width=AXIS_LW, pad=2)
    ax.tick_params(axis="x", labelsize=6, length=2.4, width=AXIS_LW, pad=2)



def draw_panel(table: pd.DataFrame, outdir: str, prefix: str,
               caption: str = "") -> List[str]:
    """One single-panel figure for a given benchmark x metric rank table."""
    import matplotlib.lines as mlines

    fig = plt.figure(figsize=(FIG_WIDTH_MM * MM_TO_IN, FIG_HEIGHT_MM * MM_TO_IN),
                     dpi=600)
    # Margins in absolute mm; the left column holds the method names
    # ("MULTIevolve" is the longest).
    left_mm, right_pad_mm = 15.0, 1.2
    left = left_mm / FIG_WIDTH_MM
    right = 1 - right_pad_mm / FIG_WIDTH_MM
    gs = fig.add_gridspec(1, 1, left=left, right=right,
                          bottom=BOTTOM_BAND_MM / FIG_HEIGHT_MM,
                          top=1 - TOP_BAND_MM / FIG_HEIGHT_MM)
    ax = fig.add_subplot(gs[0, 0])
    _draw_group(ax, table, len(table))

    # Name the benchmark and metric centred at the top: the four files are
    # otherwise identical in layout and could not be told apart once exported.
    if caption:
        fig.text(0.5, 1 - 0.9 / FIG_HEIGHT_MM, caption,
                 fontsize=7, ha="center", va="top")

    # Keyed legend with explicit symbol descriptions
    legend_handles = [
        mlines.Line2D([0], [0], marker="o", color="none",
                      markersize=DATASET_DOT_PT, markerfacecolor=DOT_GRAY,
                      markeredgewidth=0, label="Rank in one dataset"),
        mlines.Line2D([0], [0], marker="o", color="none",
                      markersize=MEAN_DOT_PT, markerfacecolor="white",
                      markeredgecolor=MEAN_EDGE, markeredgewidth=0.5,
                      label="Mean rank across datasets"),
        mlines.Line2D([0], [0], marker="o", color="none",
                      markersize=MEAN_DOT_PT, markerfacecolor=ALPHA_RED,
                      markeredgewidth=0, label="AlphaVariant mean rank"),
    ]
    # ncol=2 wraps the three keys to two rows; one row would need ~70 mm.
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
               fontsize=5.5, handletextpad=0.4, columnspacing=1.0,
               labelspacing=0.3, borderaxespad=0.2)

    # bbox_inches=None keeps the page exactly FIG_WIDTH_MM x FIG_HEIGHT_MM.
    paths = save_figure(fig, outdir, prefix, bbox_inches=None)
    plt.close(fig)
    return paths


# metric key -> (source column, panel title)
METRIC_COLS = {
    "max_fitness": ("max_fitness_median", "Max fitness"),
    "top128": ("top128_median", "Top-128 fitness"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=[c["slug"] for c in GROUPS.values()],
                    default=None,
                    help="Benchmark to draw (default: both).")
    ap.add_argument("--metric", choices=list(METRIC_COLS), default=None,
                    help="Metric to draw (default: both, one figure each).")
    ap.add_argument("--outdir", default=DEFAULT_FIGDIR,
                    help="Output directory (default: figures/).")
    args = ap.parse_args()

    for group, cfg in GROUPS.items():
        if args.group and cfg["slug"] != args.group:
            continue
        if not os.path.exists(cfg["csv"]):
            sys.exit(f"Missing source CSV for {group}: {cfg['csv']}")

        # One figure per metric, each covering a single benchmark.
        for metric, (col, title) in METRIC_COLS.items():
            if args.metric and metric != args.metric:
                continue
            table = build_rank_table(cfg, col)
            prefix = f"ranking_panel_{cfg['slug']}_{metric}"
            paths = draw_panel(table, args.outdir, prefix,
                               caption=f"{group}, {title.lower()}")

            csv_path = os.path.join(args.outdir, f"{prefix}_values.csv")
            table.to_csv(csv_path)
            print(f"{group} / {title}: wrote")
            for p in paths + [csv_path]:
                print(f"  {p}")


if __name__ == "__main__":
    main()
