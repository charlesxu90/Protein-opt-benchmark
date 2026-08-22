#!/usr/bin/env python
"""
Supplementary ablation figure for AlphaVariant (leave-one-out).

Reads the raw per-seed ablation runs under ``results_ablation/`` -- the same
files ``scripts/summarize_ablation.py`` aggregates into
``results_ablation/ablation_summary.csv`` -- and draws one panel per dataset,
one bar per configuration:

  four-site  full | -MutCompute reward | -SHAP pruning | -both
  multi-site full | -EV features | -SHAP/constraint | -mutation cap | -homolog prior

``full`` is the shipped default (red, bold x-tick); every other bar removes one
component. Bars are medians over seeds with Q1-Q3 error bars. Configurations
with no runs on a dataset are drawn as empty hatched bars marked "NR".

Outputs (into --outdir, default figures/):
    supp_figure_ablation_max_fitness.pdf
    supp_figure_ablation_top128_mean_fitness.pdf
    supp_figure_ablation_values.csv      <- every plotted number, for traceability

--verify cross-checks every plotted median against ablation_summary.csv and
exits non-zero on any disagreement.
"""
import argparse
import csv
import glob
import json
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

ABL = os.path.join(_BENCH_ROOT, "results_ablation")

# Four-site global maxima; multi-site oracle metrics are already normalized.
# Kept identical to scripts/summarize_ablation.py so the numbers agree.
GMAX = {"4site_GB1": 8.761966, "4site_PhoQ": 133.5943, "4site_TRPB": 1.0}

FOUR_CONFIGS = [("full", "full"), ("no_mcreward", "− MutCompute\nreward"),
                ("no_shap", "− SHAP\npruning"), ("bare", "− both")]
MULTI_CONFIGS = [("full", "full"), ("no_ev", "− EV\nfeatures"),
                 ("no_shap", "− SHAP/\nconstraint"), ("no_cap", "− mutation\ncap"),
                 ("no_prior", "− homolog\nprior")]

# (prefix, dataset_id, regime, panel title). Live headline datasets only:
# 4site_TEV and ms_GFP are out of the panels, so their ablation rows -- which
# still exist in ablation_summary.csv -- are deliberately not drawn.
DATASETS = [
    ("gb1",  "4site_GB1",   "four",  "GB1 4-site"),
    ("phoq", "4site_PhoQ",  "four",  "PhoQ 4-site"),
    ("trpb", "4site_TRPB",  "four",  "TrpB 4-site"),
    ("cre",  "ms_CreiLOV",  "multi", "CreiLOV multi-site"),
    ("aav",  "ms_AAV",      "multi", "AAV multi-site"),
    ("pab1", "ms_PAB1",     "multi", "PAB1 multi-site"),
]

METRICS = {
    "max_fitness":   {"four": "max_fitness",
                      "multi": "max_fitness_norm",
                      "label": "Max fitness",
                      "summary_col": "max_median"},
    "top128":        {"four": "normalized_fitness_median_top128",
                      "multi": "top128_mean_norm",
                      "label": "Mean of top 128 fitness",
                      "summary_col": "top128_median"},
}

_PALETTE = {"full": "#8B1A1A", "ablated": "#C7C7C7"}


def load_seed_values(prefix, suffix, dataset_id, regime, metric):
    """Per-seed values for one (config, dataset, metric). Mirrors summarize_ablation.py."""
    cfg = f"{prefix}_{suffix}"
    key = METRICS[metric][regime]
    out = []
    if regime == "four":
        g = GMAX[dataset_id]
        pattern = os.path.join(ABL, cfg, "seed_*", "metrics.json")
    else:
        pattern = os.path.join(ABL, cfg, dataset_id, "AlphaVariant", "seed*.json")
    for fp in sorted(glob.glob(pattern)):
        try:
            m = json.load(open(fp)).get("metrics", {})
        except (json.JSONDecodeError, OSError):
            continue
        v = m.get(key)
        if v is None:
            continue
        if regime == "four" and metric == "max_fitness" and g != 1.0 and v > 1.5:
            v = v / g
        out.append(float(v))
    return out


def collect(metric):
    """{(dataset_id, suffix): (median, q1, q3, n)} plus the row list for the CSV."""
    stats, rows = {}, []
    for prefix, dsid, regime, _title in DATASETS:
        configs = FOUR_CONFIGS if regime == "four" else MULTI_CONFIGS
        for suffix, _label in configs:
            vals = load_seed_values(prefix, suffix, dsid, regime, metric)
            if not vals:
                stats[(dsid, suffix)] = None
                rows.append([metric, dsid, f"{prefix}_{suffix}", 0, "", "", ""])
                continue
            a = np.asarray(vals, dtype=float)
            st = (float(np.median(a)), float(np.quantile(a, .25)),
                  float(np.quantile(a, .75)), int(a.size))
            stats[(dsid, suffix)] = st
            rows.append([metric, dsid, f"{prefix}_{suffix}", st[3],
                         f"{st[0]:.6f}", f"{st[1]:.6f}", f"{st[2]:.6f}"])
    return stats, rows


def plot(metric, stats, outdir):
    cfg = METRICS[metric]
    MM_TO_IN = 1 / 25.4
    fig, axes = plt.subplots(2, 3, figsize=(184 * MM_TO_IN, 118 * MM_TO_IN))
    fig.patch.set_facecolor("white")

    for ax, (prefix, dsid, regime, title) in zip(axes.ravel(), DATASETS):
        configs = FOUR_CONFIGS if regime == "four" else MULTI_CONFIGS
        # Per-panel y-limit: an ablation is read *within* a dataset, and a shared
        # scale squashes the low-dynamic-range panels (PhoQ, PAB1) into illegibility.
        panel_top = max((stats[(dsid, s)][2] for s, _ in configs
                         if stats.get((dsid, s))), default=1.0)
        ylim = (0.0, panel_top * 1.34)
        x = np.arange(len(configs))
        meds, lo, hi, present = [], [], [], []
        for suffix, _ in configs:
            st = stats.get((dsid, suffix))
            if st is None:
                meds.append(0.0); lo.append(0.0); hi.append(0.0); present.append(False)
            else:
                meds.append(st[0]); lo.append(st[0] - st[1]); hi.append(st[2] - st[0])
                present.append(True)
        present = np.array(present)
        colors = [_PALETTE["full"] if s == "full" else _PALETTE["ablated"] for s, _ in configs]

        ax.bar(x, meds, width=0.72,
               color=[c if p else "white" for c, p in zip(colors, present)],
               edgecolor=["#111111" if p else "#BBBBBB" for p in present],
               linewidth=[0.7 if p else 0.5 for p in present],
               hatch=[None if p else "//" for p in present], zorder=3)
        if present.any():
            ax.errorbar(x[present], np.array(meds)[present],
                        yerr=[np.array(lo)[present], np.array(hi)[present]],
                        fmt="none", ecolor="#333333", elinewidth=0.6,
                        capsize=1.8, capthick=0.55, zorder=4)

        offset = (ylim[1] - ylim[0]) * 0.02
        for xi, (m, h, p) in enumerate(zip(meds, hi, present)):
            if p:
                ax.text(xi, m + h + offset, f"{m:.3f}", ha="center", va="bottom",
                        fontsize=5.4, rotation=90, color="#303030", clip_on=False)
            else:
                ax.text(xi, offset, "NR", ha="center", va="bottom", fontsize=5.6,
                        fontweight="bold", color="#888888", clip_on=False)

        n_seeds = {st[3] for k, st in stats.items() if k[0] == dsid and st}
        n_txt = f"n={min(n_seeds)}" if len(n_seeds) == 1 else (
                f"n={min(n_seeds)}–{max(n_seeds)}" if n_seeds else "no runs")
        ax.set_title(f"{title}  ({n_txt})", pad=6, fontweight="bold", fontsize=6.0)
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in configs], fontsize=5.4)
        for tick, (s, _) in zip(ax.get_xticklabels(), configs):
            if s == "full":
                tick.set_fontweight("bold"); tick.set_color(_PALETTE["full"])
        style_axis_vbar(ax)
        ax.set_ylim(*ylim)
        ax.set_ylabel(cfg["label"], labelpad=4, fontsize=6.0)

    fig.text(0.012, 0.012,
             "AlphaVariant leave-one-out ablation. Red bar is the shipped default configuration; "
             "each other bar removes one component. Bars are medians over seeds, whiskers Q1–Q3. "
             "NR = not run for that dataset. Source: results_ablation/.",
             ha="left", va="bottom", fontsize=5.8, color="#4D4D4D")
    fig.subplots_adjust(left=0.062, right=0.995, bottom=0.13, top=0.945,
                        wspace=0.26, hspace=0.42)
    paths = save_figure(fig, outdir, f"supp_figure_ablation_{'max_fitness' if metric=='max_fitness' else 'top128_mean_fitness'}")
    plt.close(fig)
    return paths


def verify(all_stats):
    """Cross-check every plotted median against results_ablation/ablation_summary.csv."""
    summary_path = os.path.join(ABL, "ablation_summary.csv")
    if not os.path.exists(summary_path):
        print(f"  ! {summary_path} absent -- run scripts/summarize_ablation.py first")
        return False
    summary = pd.read_csv(summary_path)
    ok = bad = 0
    for metric, stats in all_stats.items():
        col = METRICS[metric]["summary_col"]
        for (dsid, suffix), st in stats.items():
            prefix = next(p for p, d, _r, _t in DATASETS if d == dsid)
            row = summary[summary["config"] == f"{prefix}_{suffix}"]
            if st is None:
                if not row.empty and pd.notna(row.iloc[0][col]):
                    print(f"  MISMATCH {metric} {prefix}_{suffix}: figure=NR, table={row.iloc[0][col]}")
                    bad += 1
                continue
            if row.empty:
                print(f"  MISMATCH {metric} {prefix}_{suffix}: plotted {st[0]:.4f}, absent from table")
                bad += 1; continue
            tv = row.iloc[0][col]
            if pd.isna(tv):
                print(f"  MISMATCH {metric} {prefix}_{suffix}: plotted {st[0]:.4f}, table blank")
                bad += 1; continue
            if abs(float(tv) - st[0]) > 1e-3:
                print(f"  MISMATCH {metric} {prefix}_{suffix}: figure {st[0]:.6f} vs table {float(tv):.6f}")
                bad += 1
            else:
                ok += 1
    print(f"  verify: {ok} medians match ablation_summary.csv, {bad} mismatched")
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=os.path.join(_BENCH_ROOT, "figures"))
    ap.add_argument("--verify", action="store_true",
                    help="Cross-check plotted medians against ablation_summary.csv; "
                         "exit 1 on any disagreement.")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    apply_nature_rcparams(COMPACT_6PT_RCPARAMS)

    all_stats, all_rows = {}, []
    for metric in METRICS:
        stats, rows = collect(metric)
        all_stats[metric] = stats
        all_rows += rows
        for p in plot(metric, stats, args.outdir):
            print(f"  wrote {os.path.relpath(p, _BENCH_ROOT)}")

    csv_path = os.path.join(args.outdir, "supp_figure_ablation_values.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "dataset", "config", "n", "median", "q1", "q3"])
        w.writerows(all_rows)
    print(f"  wrote {os.path.relpath(csv_path, _BENCH_ROOT)}")

    if args.verify and not verify(all_stats):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
