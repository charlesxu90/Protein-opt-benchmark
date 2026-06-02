#!/usr/bin/env python
"""
Build a median + Q1/Q3 (IQR) CSV per plan (A/B/C).

Per-method, per-dataset rows aggregated directly from the per-seed
metrics.json files. Output CSV columns:

    dataset, method,
    max_fitness_median, max_fitness_q1, max_fitness_q3,
    top128_median, top128_q1, top128_q3, n

Writes `figures/plan_{A,B,C}/alphavariant_comparison_median_iqr.csv`.
Reads the same source dirs that built each plan's mean+std CSV — the
method→source-dir mapping is plan-specific because each plan ships a
different AlphaVariant configuration under the canonical "AlphaVariant"
label.
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

GMAX = {
    "4site_GB1": 8.761966,
    "4site_PhoQ": 133.5943,
    "4site_TEV": 1.0,
    "4site_TRPB": 1.0,
}

# (dataset dir name, archive name used for Tier 1B / competitor paths)
DATASETS = [
    ("4site_TEV",  "4site_TEV"),
    ("4site_GB1",  "4site_GB1"),
    ("4site_PhoQ", "4site_PhoQ"),
    ("4site_TRPB", "TRPB"),
]


def get_metric(fp: str, gmax: float, key: str):
    try:
        r = json.load(open(fp))
        m = r.get("metrics") or r.get("final_metrics") or r
        if isinstance(m, list):
            m = m[-1]
        v = m.get(key)
        if v is None:
            return None
        if key == "max_fitness" and v > 1.5 and gmax != 1.0:
            v = v / gmax
        return v
    except Exception:
        return None


def load_seed_values(pattern: str, gmax: float, key: str, cap: int = 30):
    """Return at most `cap` per-seed values for the given metric key."""
    return [
        v for v in (get_metric(fp, gmax, key) for fp in sorted(glob.glob(pattern))[:cap])
        if v is not None
    ]


def stats_block(vals):
    """Return (median, q1, q3, n) clamped to non-negative reals."""
    vals = [v for v in vals if 0 <= v <= 1.5]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    return (
        float(np.median(arr)),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 75)),
        int(arr.size),
    )


# Method → source-dir pattern. Patterns use {a} for the dataset alias used
# in competitor paths (4site_GB1 / 4site_PhoQ / 4site_TEV / TRPB) and {ds}
# for the dataset dir name (4site_GB1 / 4site_PhoQ / 4site_TEV / 4site_TRPB).
COMMON_COMPETITORS = {
    "Random":     "Random/results/{a}_Random/{a}/random/metrics_seed*.json",
    "GreedyWalk": "GreedyWalk/results/{a}_GreedyWalk/{a}/greedy/metrics_seed*.json",
    "ALDE":       "ALDE/results/{a}_ALDE/{a}/onehot/metrics_seed*.json",
    "FLEXS":      "FLEXS/results/{a}_AdaLead/{a}/metrics_seed*.json",
    "AiCE":       "AiCE/results/{a}_AiCE/{a}/aice/metrics_seed*.json",
    "ftMLDE":     "ftMLDE/results/{a}_ftMLDE/{a}/ftmlde/metrics_seed*.json",
    "CLADE":      "CLADE/results/{a}_CLADE/{a}/clade/metrics_seed*.json",
    "delta_cs":   "delta_cs/BioSeq-GFN-AL/results/{a}_delta_cs/{a}/metrics_seed*.json",
}

PLAN_AV_METHODS = {
    "A": {
        # Plan A: AlphaVariant = Tier 1B base
        "AlphaVariant":        "alphavariant/results/_archive_tier1B_canonical/{arch}/seed_*/metrics.json",
        "AlphaVariant_SHAP":   "alphavariant/results/{ds}_AlphaVariant_shap_late/seed_*/metrics.json",
        "AlphaVariant_PLM":    "alphavariant/results/{ds}_AlphaVariant_plm_reward_winner/seed_*/metrics.json",
        "AlphaVariant_Hybrid": "alphavariant/results/{ds}_AlphaVariant_hybrid_w_winner/seed_*/metrics.json",
    },
    "B": {
        # Plan B: AlphaVariant = base + SHAP (shipped)
        "AlphaVariant":        "alphavariant/results/{ds}_AlphaVariant_shap_late/seed_*/metrics.json",
        "AlphaVariant_base":   "alphavariant/results/_archive_tier1B_canonical/{arch}/seed_*/metrics.json",
        "AlphaVariant_PLM":    "alphavariant/results/{ds}_AlphaVariant_plm_reward_winner/seed_*/metrics.json",
        "AlphaVariant_Hybrid": "alphavariant/results/{ds}_AlphaVariant_hybrid_w_winner/seed_*/metrics.json",
    },
    "C": {
        # Plan C: AlphaVariant = base + MutCompute + SHAP (shipped)
        "AlphaVariant":        "alphavariant/results/{ds}_AlphaVariant_mc_shap_winner/seed_*/metrics.json",
        "AlphaVariant_base":   "alphavariant/results/_archive_tier1B_canonical/{arch}/seed_*/metrics.json",
        "AlphaVariant_SHAP":   "alphavariant/results/{ds}_AlphaVariant_shap_late/seed_*/metrics.json",
        "AlphaVariant_PLM":    "alphavariant/results/{ds}_AlphaVariant_plm_reward_winner/seed_*/metrics.json",
        "AlphaVariant_Hybrid": "alphavariant/results/{ds}_AlphaVariant_hybrid_w_winner/seed_*/metrics.json",
    },
}

DS_ALIAS = {
    "4site_TEV": "4site_TEV",
    "4site_GB1": "4site_GB1",
    "4site_PhoQ": "4site_PhoQ",
    "4site_TRPB": "TRPB",
}


def build_csv(plan: str, out_path: Path):
    rows = []
    av_methods = PLAN_AV_METHODS[plan]
    for ds_dir, ds_arch in DATASETS:
        g = GMAX[ds_dir]
        alias = DS_ALIAS[ds_dir]
        method_patterns = {}
        for name, pat in COMMON_COMPETITORS.items():
            method_patterns[name] = str(ROOT / pat.format(a=alias))
        for name, pat in av_methods.items():
            method_patterns[name] = str(ROOT / pat.format(ds=ds_dir, arch=ds_arch))

        for method, pattern in method_patterns.items():
            mx = load_seed_values(pattern, g, "max_fitness")
            t128 = load_seed_values(pattern, g, "normalized_fitness_median_top128")
            sm = stats_block(mx)
            st = stats_block(t128) if t128 else None
            if sm is None or sm[3] < 5:
                continue
            row = {
                "dataset": ds_dir,
                "method": method,
                "max_fitness_median": round(sm[0], 4),
                "max_fitness_q1":     round(sm[1], 4),
                "max_fitness_q3":     round(sm[2], 4),
                "top128_median": round(st[0], 4) if st else 0.0,
                "top128_q1":     round(st[1], 4) if st else 0.0,
                "top128_q3":     round(st[2], 4) if st else 0.0,
                "n": sm[3],
            }
            rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "dataset", "method",
            "max_fitness_median", "max_fitness_q1", "max_fitness_q3",
            "top128_median", "top128_q1", "top128_q3", "n",
        ])
        for r in rows:
            w.writerow([
                r["dataset"], r["method"],
                r["max_fitness_median"], r["max_fitness_q1"], r["max_fitness_q3"],
                r["top128_median"], r["top128_q1"], r["top128_q3"], r["n"],
            ])
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", default="A,B,C",
                        help="Comma-separated plans to build (default A,B,C).")
    args = parser.parse_args()

    for plan in args.plans.split(","):
        plan = plan.strip()
        if plan not in PLAN_AV_METHODS:
            print(f"Unknown plan {plan}; skipping")
            continue
        out_path = ROOT / f"figures/plan_{plan}/alphavariant_comparison_median_iqr.csv"
        n = build_csv(plan, out_path)
        print(f"Plan {plan}: wrote {n} rows to {out_path}")


if __name__ == "__main__":
    main()
