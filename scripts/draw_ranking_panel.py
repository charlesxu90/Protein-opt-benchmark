#!/usr/bin/env python3
"""Average-method-ranking panel for the AlphaVariant benchmark.

Two side-by-side lollipop panels (four-site, multi-site) showing each method's
mean rank across that group's datasets. Visual encoding follows
``figures/alphavariant_ranking_panel_gray_preferred.py``:

- light-gray dots: a method's rank in each individual dataset,
- open black circles: the mean rank across datasets,
- dark-red filled circle + red label: AlphaVariant's mean rank.

Unlike the prototype, ranks are computed directly from the benchmark median
CSVs (``figures/alphavariant_comparison_median_iqr.csv`` and
``figures/ms_oracles/multisite_oracle_median_iqr.csv``) for the chosen metric,
so the figure stays in sync with the data. Lower rank = better (rank 1 = best
median on that dataset; ties share the average rank).

Outputs ``<prefix>.{pdf,png}`` into ``--outdir`` (default: figures/).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FIGDIR = os.path.join(ROOT, "figures")
MM_TO_IN = 1 / 25.4

# --- palette -----------------------------------------------------------------
# AlphaVariant colour unified with the performance figures (vermilion). Per-
# dataset dots darkened from the prototype (#B8B8B8) so they survive printing
# and typesetting.
sys.path.insert(0, ROOT)
from utils.plot_style_utils import VERMILION, apply_nature_rcparams, save_figure  # noqa: E402

ALPHA_RED = VERMILION
DOT_GRAY = "#8A8A8A"
DOT_GRAY_EDGE = "#5F5F5F"
LINE_GRAY = "#D4D4D4"
GRID_GRAY = "#E6E6E6"
TEXT_GRAY = "#444444"

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
        "csv": os.path.join(DEFAULT_FIGDIR, "alphavariant_comparison_median_iqr.csv"),
        "allow": MAIN_METHODS_4SITE,
        "dataset_labels": {
            "4site_GB1": "GB1", "4site_PhoQ": "PhoQ",
            "4site_TRPB": "TrpB",
        },
    },
    "Multisite": {
        "csv": os.path.join(DEFAULT_FIGDIR, "ms_oracles", "multisite_oracle_median_iqr.csv"),
        "allow": MAIN_METHODS_MULTISITE,
        "dataset_labels": {
            "ms_AAV": "AAV", "ms_CreiLOV": "CreiLOV",
            "ms_PAB1": "PAB1",
        },
    },
}

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


def _draw_group(ax, table: pd.DataFrame, title: str, n_methods: int) -> None:
    # reverse so the best mean rank sits at the top
    methods = table.index.tolist()[::-1]
    tp = table.loc[methods]
    dataset_cols = [c for c in tp.columns if c not in ("Mean rank", "Mean score")]
    y = np.arange(len(tp))
    mean_rank = tp["Mean rank"].to_numpy(float)

    worst = n_methods  # stem origin = worst possible rank
    ax.hlines(y, worst, mean_rank, color=LINE_GRAY, lw=0.8, zorder=1)

    # individual dataset ranks: deliberately gray, jittered vertically
    offsets = np.linspace(-0.20, 0.20, len(dataset_cols))
    for di, ds in enumerate(dataset_cols):
        vals = tp[ds].to_numpy(float)
        mask = ~np.isnan(vals)
        ax.scatter(vals[mask], y[mask] + offsets[di], s=10,
                   facecolor=DOT_GRAY, edgecolor=DOT_GRAY_EDGE,
                   linewidth=0.25, alpha=0.85, zorder=2)

    # mean rank: open circles, AlphaVariant as red filled
    is_alpha = np.array([m == "AlphaVariant" for m in tp.index])
    ax.scatter(mean_rank[~is_alpha], y[~is_alpha], s=30, facecolor="white",
               edgecolor="#202020", linewidth=0.7, zorder=4)
    ax.scatter(mean_rank[is_alpha], y[is_alpha], s=38, facecolor=ALPHA_RED,
               edgecolor="white", linewidth=0.6, zorder=5)

    ax.set_title(title, loc="left", fontsize=7, fontweight="normal", pad=5)
    ax.set_yticks(y)
    ax.set_yticklabels(tp.index, fontsize=6)
    for lab in ax.get_yticklabels():
        if lab.get_text() == "AlphaVariant":
            lab.set_color(ALPHA_RED)
            lab.set_fontweight("bold")

    ax.set_xlim(0.55, worst + 0.4)  # Rank 1 (best) on the left
    ticks = sorted({1, 2, 4, 6, 8, worst})
    ax.set_xticks([t for t in ticks if t <= worst])
    ax.set_xlabel("Mean rank (1 = best)", fontsize=7, fontweight="normal", labelpad=8)
    ax.grid(axis="x", color=GRID_GRAY, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_linewidth(0.25)
    ax.spines["bottom"].set_linewidth(0.25)
    ax.tick_params(axis="y", labelsize=6, length=2.4, width=0.25, pad=2)
    ax.tick_params(axis="x", labelsize=6, length=2.4, width=0.25, pad=2)

    # directional indicator: best rank sits on the left
    ax.annotate("better", xy=(0.0, -0.185), xytext=(0.30, -0.185),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=6, color=TEXT_GRAY, va="center", ha="left",
                arrowprops=dict(arrowstyle="->", color=TEXT_GRAY, lw=0.8))


def draw_panel(tables: Dict[str, pd.DataFrame], metric_label: str,
               outdir: str, prefix: str, n_seeds: int = 30) -> List[str]:
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    fig = plt.figure(figsize=(100 * MM_TO_IN, 45 * MM_TO_IN), dpi=600)
    gs = fig.add_gridspec(1, 2, left=0.14, right=0.985,
                          bottom=0.30, top=0.86, wspace=0.55)
    for gi, (group, table) in enumerate(tables.items()):
        ax = fig.add_subplot(gs[0, gi])
        _draw_group(ax, table, f"{group}", len(table))

    # Keyed legend with explicit symbol descriptions
    legend_handles = [
        mlines.Line2D([0], [0], marker="o", color="none", markersize=4,
                      markerfacecolor=DOT_GRAY, markeredgecolor=DOT_GRAY_EDGE,
                      markeredgewidth=0.4, label="Rank in one dataset"),
        mlines.Line2D([0], [0], marker="o", color="none", markersize=5,
                      markerfacecolor="white", markeredgecolor="#202020",
                      markeredgewidth=0.7, label="Mean rank across datasets"),
        mlines.Line2D([0], [0], marker="o", color="none", markersize=5.5,
                      markerfacecolor=ALPHA_RED, markeredgecolor="white",
                      markeredgewidth=0.6, label="AlphaVariant mean rank"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.56, -0.02), ncol=3, frameon=False,
               fontsize=6, handletextpad=0.4, columnspacing=1.0)

    paths = save_figure(fig, outdir, prefix)
    plt.close(fig)
    return paths


METRIC_COLS = {
    "max_fitness": ("max_fitness_median", "max fitness"),
    "top128": ("top128_median", "top-128 mean fitness"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", choices=list(METRIC_COLS), default="max_fitness",
                    help="Metric to rank by (default: max_fitness).")
    ap.add_argument("--outdir", default=DEFAULT_FIGDIR,
                    help="Output directory (default: figures/).")
    ap.add_argument("--prefix", default=None,
                    help="Output file prefix (default: ranking_panel_<metric>).")
    args = ap.parse_args()

    metric_col, metric_label = METRIC_COLS[args.metric]
    prefix = args.prefix or f"ranking_panel_{args.metric}"

    tables: Dict[str, pd.DataFrame] = {}
    for group, cfg in GROUPS.items():
        if not os.path.exists(cfg["csv"]):
            sys.exit(f"Missing source CSV for {group}: {cfg['csv']}")
        tables[group] = build_rank_table(cfg, metric_col)

    paths = draw_panel(tables, metric_label, args.outdir, prefix)

    # provenance: write the computed rank table next to the figure
    csv_path = os.path.join(args.outdir, f"{prefix}_values.csv")
    pd.concat({g: t for g, t in tables.items()}, names=["group"]).to_csv(csv_path)
    print(f"Ranked by {metric_label}. Wrote:")
    for p in paths + [csv_path]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
