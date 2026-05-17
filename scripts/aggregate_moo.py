#!/usr/bin/env python
"""
aggregate_moo.py — Multi-objective metrics for joint (*_joint) datasets.

For each method's metrics_seedNN.json on a joint dataset, looks up the
per-query (blue, red) tuple by `queried_indices`, then computes:
  - max scalarized fitness (sqrt(blue*red))
  - max blue, max red (per-objective bests)
  - hypervolume vs (0, 0) reference, normalized to the landscape's max HV
  - Pareto front coverage vs the landscape's true Pareto front

Aggregates across seeds with median + IQR.

Usage
-----
    python scripts/aggregate_moo.py --dataset eqFP611_joint
    python scripts/aggregate_moo.py --dataset mTagBFP2_joint --first_n 30
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data import load_joint_objectives
from utils.multi_objective import (
    pareto_front_mask,
    hypervolume,
    pareto_front_coverage,
)

METHOD_PATTERNS = {
    "Random":     "Random/results/{ds}_Random/{ds}/random/metrics_seed*.json",
    "GreedyWalk": "GreedyWalk/results/{ds}_GreedyWalk/{ds}/greedy/metrics_seed*.json",
    "AiCE":       "AiCE/results/{ds}_AiCE/{ds}/aice/metrics_seed*.json",
    "ALDE":       "ALDE/results/{ds}_ALDE/{ds}/onehot/metrics_seed*.json",
    "FLEXS":      "FLEXS/results/{ds}_AdaLead/{ds}/metrics_seed*.json",
    "ftMLDE":     "ftMLDE/results/{ds}_ftMLDE/{ds}/ftmlde/metrics_seed*.json",
    "CLADE":      "CLADE/results/{ds}_CLADE/{ds}/clade/metrics_seed*.json",
}


def per_seed_moo_metrics(qi, blue, red, ref_front, ref_hv):
    """Compute MOO metrics for one seed's queried indices."""
    qi = np.asarray(qi, dtype=int)
    qb = blue[qi]
    qr = red[qi]
    pts = np.column_stack([qb, qr])
    scal = np.sqrt(np.clip(qb, 0, None) * np.clip(qr, 0, None))
    hv = hypervolume(pts, np.array([0.0, 0.0]))
    return {
        "max_scalarized": float(scal.max()),
        "max_blue":       float(qb.max()),
        "max_red":        float(qr.max()),
        "hypervolume":    float(hv),
        "hv_normalized":  float(hv / ref_hv) if ref_hv > 0 else 0.0,
        "pareto_coverage": float(pareto_front_coverage(pts, ref_front)),
    }


def summarize(vals):
    a = np.asarray(vals, dtype=float)
    return {
        "n":      int(a.size),
        "mean":   float(a.mean()),
        "median": float(np.median(a)),
        "q1":     float(np.percentile(a, 25)),
        "q3":     float(np.percentile(a, 75)),
        "min":    float(a.min()),
        "max":    float(a.max()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="Joint dataset name (e.g. eqFP611_joint)")
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--first_n", type=int, default=None,
                   help="Use only the first N seeds (sorted by seed number)")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    methods = args.methods or list(METHOD_PATTERNS)

    # Build the landscape's reference Pareto front + max-HV bound
    _, blue, red = load_joint_objectives(args.dataset)
    landscape_pts = np.column_stack([blue, red])
    ref_mask = pareto_front_mask(landscape_pts)
    ref_front = landscape_pts[ref_mask]
    ref_hv = hypervolume(landscape_pts, np.array([0.0, 0.0]))
    print(f"\nLandscape: {args.dataset}")
    print(f"  total sequences: {len(blue)}")
    print(f"  Pareto-optimal: {ref_mask.sum()}")
    print(f"  reference hypervolume (vs (0,0)): {ref_hv:.4f}")
    print(f"  max blue: {blue.max():.4f}, max red: {red.max():.4f}")
    print()

    results = {}
    for name in methods:
        pat = METHOD_PATTERNS.get(name)
        if pat is None:
            print(f"  {name:>12}  (unknown method)")
            continue
        files = sorted(ROOT.glob(pat.format(ds=args.dataset)))
        if args.first_n:
            # Sort numerically by seed
            def seed_of(p):
                stem = p.stem  # metrics_seedNN
                try: return int(stem.split("seed")[-1])
                except: return 0
            files = sorted(files, key=seed_of)[: args.first_n]
        if not files:
            print(f"  {name:>12}  <no files>")
            continue
        per_seed = []
        skipped = 0
        for f in files:
            try:
                rec = json.load(open(f))
                m = rec.get("metrics") if isinstance(rec, dict) else None
                if isinstance(m, list): m = m[-1] if m else {}
                qi = (m or {}).get("queried_indices", [])
                if not qi:
                    skipped += 1
                    continue
                per_seed.append(per_seed_moo_metrics(qi, blue, red, ref_front, ref_hv))
            except Exception as e:
                skipped += 1
                continue
        if not per_seed:
            print(f"  {name:>12}  <no usable metrics ({skipped} skipped)>")
            continue
        summary = {}
        for k in per_seed[0]:
            summary[k] = summarize([d[k] for d in per_seed])
        summary["_n_files"] = len(files)
        summary["_n_used"] = len(per_seed)
        summary["_n_skipped"] = skipped
        results[name] = summary

    # Print compact table
    hdr = ["method", "n", "max_blue", "max_red", "max_scal", "hv_norm", "pareto_cov"]
    widths = [12, 4, 10, 10, 10, 10, 12]
    print("".join(f"{h:>{w}}" for h, w in zip(hdr, widths)))
    print("-" * sum(widths))
    for name, s in results.items():
        if name.startswith("_"): continue
        row = [
            name,
            s["_n_used"],
            f"{s['max_blue']['median']:.3f}",
            f"{s['max_red']['median']:.3f}",
            f"{s['max_scalarized']['median']:.3f}",
            f"{s['hv_normalized']['median']:.3f}",
            f"{s['pareto_coverage']['median']:.3f}",
        ]
        line = "".join(f"{str(c):>{w}}" for c, w in zip(row, widths))
        print(line)

    out = (Path(args.output) if args.output
           else ROOT / "tables" / args.dataset / "moo_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"dataset": args.dataset,
         "landscape": {
             "n_total": int(len(blue)),
             "n_pareto": int(ref_mask.sum()),
             "reference_hv": float(ref_hv),
             "max_blue": float(blue.max()),
             "max_red": float(red.max()),
         },
         "methods": results}, indent=2, default=str))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    sys.exit(main())
