#!/usr/bin/env python
"""
Build the median + Q1/Q3 (IQR) CSV for the multi-site ORACLE benchmark, in the
exact column format consumed by scripts/draw_figures_median.py:

    dataset, method,
    max_fitness_median, max_fitness_q1, max_fitness_q3,
    top128_median, top128_q1, top128_q3, n

Reads results_oracle/<dataset>/<method>/seed*.json (already-normalized oracle
fitness: metrics.max_fitness_norm and metrics.top128_mean_norm).

Usage:
    python scripts/build_oracle_median_iqr_csv.py
    -> figures/ms_oracles/multisite_oracle_median_iqr.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

DATASETS = ["ms_AAV", "ms_CreiLOV", "ms_GFP", "ms_PAB1"]
METHODS = ["Random", "GreedyWalk", "ALDE", "CLADE", "ftMLDE",
           "AdaLead", "MULTIevolve", "EVOLVEpro", "AiCE", "AlphaVariant"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results_oracle")))
    ap.add_argument("--out", default=os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "figures", "ms_oracles",
        "multisite_oracle_median_iqr.csv")))
    ap.add_argument("--datasets", nargs="+", default=DATASETS,
                    help="subset of datasets to aggregate (default: all four)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    for d in args.datasets:
        for m in METHODS:
            maxv, top = [], []
            for fp in glob.glob(os.path.join(args.results_dir, d, m, "seed*.json")):
                mt = json.load(open(fp)).get("metrics", {})
                if "max_fitness_norm" in mt:
                    maxv.append(mt["max_fitness_norm"])
                if "top128_mean_norm" in mt:
                    top.append(mt["top128_mean_norm"])
            if not maxv:
                continue
            maxv, top = np.array(maxv), np.array(top)
            rows.append({
                "dataset": d, "method": m,
                "max_fitness_median": float(np.median(maxv)),
                "max_fitness_q1": float(np.quantile(maxv, 0.25)),
                "max_fitness_q3": float(np.quantile(maxv, 0.75)),
                "top128_median": float(np.median(top)),
                "top128_q1": float(np.quantile(top, 0.25)),
                "top128_q3": float(np.quantile(top, 0.75)),
                "n": len(maxv),
            })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}  ({len(rows)} rows, n per row: "
          f"{sorted(set(r['n'] for r in rows))})")


if __name__ == "__main__":
    main()
