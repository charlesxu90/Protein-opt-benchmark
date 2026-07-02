#!/usr/bin/env python
"""
Build a unified per-seed CSV of AlphaVariant's per-round fitness trajectories
on the multi-site oracle benchmarks plotted by scripts/draw_trajectory_figures.py.

Reads results_oracle/<dataset>/AlphaVariant/seed*.json (metrics.fitness_trajectory,
a 5-element list normalized to [0, 1]) and unpacks it into one row per
(dataset, seed, round).

Usage:
    python scripts/build_trajectory_dataset.py
    -> figures/ms_oracles/alphavariant_trajectory_per_seed.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

DATASETS = {
    "ms_AAV":     "AAV",
    "ms_CreiLOV": "CreiLOV",
    "ms_PAB1":    "PAB1",
}
METHOD = "AlphaVariant"
N_ROUNDS = 5


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results_oracle")))
    ap.add_argument("--out", default=os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "figures", "ms_oracles",
        "alphavariant_trajectory_per_seed.csv")))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                    help="subset of dataset keys to include (default: all three)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    for dataset_key in args.datasets:
        label = DATASETS[dataset_key]
        pattern = os.path.join(args.results_dir, dataset_key, METHOD, "seed*.json")
        for fp in sorted(glob.glob(pattern)):
            d = json.load(open(fp))
            seed = d.get("seed")
            traj = d.get("metrics", {}).get("fitness_trajectory")
            if traj is None:
                traj = d.get("fitness_trajectory")
            if traj is None or len(traj) != N_ROUNDS:
                continue
            for round_no, fitness in enumerate(traj, start=1):
                rows.append({
                    "dataset": label,
                    "seed": seed,
                    "round": round_no,
                    "fitness": float(fitness),
                })

    if not rows:
        raise SystemExit("No trajectory data found. Check results_oracle/ paths.")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "seed", "round", "fitness"])
        w.writeheader()
        w.writerows(rows)

    seeds_per_dataset = {
        label: len({r["seed"] for r in rows if r["dataset"] == label})
        for label in (DATASETS[k] for k in args.datasets)
    }
    print(f"wrote {args.out}  ({len(rows)} rows; seeds per dataset: {seeds_per_dataset})")


if __name__ == "__main__":
    main()
