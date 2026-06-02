#!/usr/bin/env python
"""
aggregate_moo.py — Multi-objective metrics for joint (*_joint) datasets.

For each method's metrics_seedNN.json on a joint dataset, looks up the
per-query (blue, red) tuple by `queried_indices`, then computes:

  Final-budget metrics
  - max_scalarized:    max sqrt(blue*red)
  - max_blue, max_red: per-objective bests
  - hypervolume:       HV vs (0,0), normalized to landscape HV
  - pareto_coverage:   fraction of landscape Pareto front covered
  - hv_regret:         ref_hv - hv
  - product_score:     max(blue*red), raw product
  - max_min_norm:      max(min(B_tilde, R_tilde)) using min-max norm
  - distance_to_ideal: min ||(1,1) - (B_tilde, R_tilde)||
  - n_hits_wt:         # queries dominating wild-type on both channels
  - n_hits_p75:        # queries above 75th-percentile on both channels
  - frac_hits_wt, frac_hits_p75

  Trajectory metrics (per checkpoint in {96, 192, 288, 384, 480})
  - hv_norm, pareto_coverage, product_score
  - hv_auc: trapezoidal integral over the checkpoint grid

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
    "Random":       "Random/results/{ds}_Random/{ds}/random/metrics_seed*.json",
    "GreedyWalk":   "GreedyWalk/results/{ds}_GreedyWalk/{ds}/greedy/metrics_seed*.json",
    "AiCE":         "AiCE/results/{ds}_AiCE/{ds}/aice/metrics_seed*.json",
    "ALDE":         "ALDE/results/{ds}_ALDE/{ds}/onehot/metrics_seed*.json",
    "FLEXS":        "FLEXS/results/{ds}_AdaLead/{ds}/metrics_seed*.json",
    "ftMLDE":       "ftMLDE/results/{ds}_ftMLDE/{ds}/ftmlde/metrics_seed*.json",
    "CLADE":        "CLADE/results/{ds}_CLADE/{ds}/clade/metrics_seed*.json",
    "AlphaVariant": "alphavariant/results/{ds}_AlphaVariant/seed_*/metrics.json",
}

TRAJECTORY_CHECKPOINTS = [96, 192, 288, 384, 480]


def _wt_index(dataset: str, sequences) -> int | None:
    """Find the wild-type row index in the landscape by reading wt.fasta."""
    wt_path = ROOT / "data" / dataset / "wt.fasta"
    if not wt_path.exists():
        return None
    lines = wt_path.read_text().splitlines()
    wt = "".join(l for l in lines if not l.startswith(">")).strip()
    for i, s in enumerate(sequences):
        if s == wt:
            return i
    return None


def per_seed_moo_metrics(
    qi,
    blue,
    red,
    ref_front,
    ref_hv,
    norm,
    thresholds,
):
    """Compute MOO metrics for one seed's queried indices.

    norm: dict with keys 'b_min','b_max','r_min','r_max'.
    thresholds: dict with keys 'wt_b','wt_r','p75_b','p75_r'. wt_b/wt_r may be None.
    """
    qi = np.asarray(qi, dtype=int)
    qb = blue[qi]
    qr = red[qi]
    pts = np.column_stack([qb, qr])
    scal = np.sqrt(np.clip(qb, 0, None) * np.clip(qr, 0, None))
    hv = hypervolume(pts, np.array([0.0, 0.0]))

    # Normalized objectives over the queried set
    b_range = norm["b_max"] - norm["b_min"]
    r_range = norm["r_max"] - norm["r_min"]
    b_t = (qb - norm["b_min"]) / (b_range if b_range > 0 else 1.0)
    r_t = (qr - norm["r_min"]) / (r_range if r_range > 0 else 1.0)

    product_score = float(np.max(qb * qr))
    max_min_norm = float(np.max(np.minimum(b_t, r_t)))
    distance_to_ideal = float(np.min(np.sqrt((1.0 - b_t) ** 2 + (1.0 - r_t) ** 2)))

    # Threshold hits (count + frac)
    n = max(len(qi), 1)
    if thresholds.get("wt_b") is not None and thresholds.get("wt_r") is not None:
        wt_hits = int(np.sum((qb >= thresholds["wt_b"]) & (qr >= thresholds["wt_r"])))
    else:
        wt_hits = -1
    p75_hits = int(np.sum((qb >= thresholds["p75_b"]) & (qr >= thresholds["p75_r"])))

    # Trajectory: replay qi[:k] for each checkpoint
    traj = {}
    for k in TRAJECTORY_CHECKPOINTS:
        if k > len(qi):
            break
        prefix_pts = pts[:k]
        hv_k = hypervolume(prefix_pts, np.array([0.0, 0.0]))
        traj[k] = {
            "hv_norm": float(hv_k / ref_hv) if ref_hv > 0 else 0.0,
            "pareto_coverage": float(pareto_front_coverage(prefix_pts, ref_front)),
            "product_score": float(np.max(qb[:k] * qr[:k])),
        }

    # HV-AUC: trapezoidal integral of hv_norm over checkpoints
    if len(traj) >= 2:
        xs = sorted(traj.keys())
        ys = [traj[k]["hv_norm"] for k in xs]
        hv_auc = float(np.trapz(ys, xs))
    else:
        hv_auc = 0.0

    return {
        "max_scalarized":   float(scal.max()),
        "max_blue":         float(qb.max()),
        "max_red":          float(qr.max()),
        "hypervolume":      float(hv),
        "hv_normalized":    float(hv / ref_hv) if ref_hv > 0 else 0.0,
        "hv_regret":        float(ref_hv - hv),
        "pareto_coverage":  float(pareto_front_coverage(pts, ref_front)),
        "product_score":    product_score,
        "max_min_norm":     max_min_norm,
        "distance_to_ideal": distance_to_ideal,
        "n_hits_wt":        wt_hits,
        "frac_hits_wt":     float(wt_hits) / n if wt_hits >= 0 else -1.0,
        "n_hits_p75":       p75_hits,
        "frac_hits_p75":    float(p75_hits) / n,
        "hv_auc":           hv_auc,
        "_trajectory":      traj,
    }


def summarize(vals):
    a = np.asarray(vals, dtype=float)
    return {
        "n":      int(a.size),
        "mean":   float(a.mean()),
        "std":    float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "q1":     float(np.percentile(a, 25)),
        "q3":     float(np.percentile(a, 75)),
        "min":    float(a.min()),
        "max":    float(a.max()),
    }


def summarize_trajectory(per_seed_records):
    """Collapse per-seed trajectory dicts into per-checkpoint summaries.

    Each seed contributes a dict {k: {hv_norm, pareto_coverage, product_score}}.
    Output: {k: {metric: summary_stats}} for k present in at least one seed.
    """
    by_ckpt = {}
    for rec in per_seed_records:
        traj = rec.get("_trajectory", {})
        for k, fields in traj.items():
            by_ckpt.setdefault(k, {"hv_norm": [], "pareto_coverage": [], "product_score": []})
            for metric, v in fields.items():
                by_ckpt[k][metric].append(v)
    out = {}
    for k in sorted(by_ckpt):
        out[k] = {m: summarize(v) for m, v in by_ckpt[k].items()}
    return out


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
    sequences, blue, red = load_joint_objectives(args.dataset)
    landscape_pts = np.column_stack([blue, red])
    ref_mask = pareto_front_mask(landscape_pts)
    ref_front = landscape_pts[ref_mask]
    ref_hv = hypervolume(landscape_pts, np.array([0.0, 0.0]))

    # Normalization constants from full landscape
    norm = {
        "b_min": float(blue.min()), "b_max": float(blue.max()),
        "r_min": float(red.min()),  "r_max": float(red.max()),
    }
    # Thresholds: wild-type (if in landscape) and 75th-percentile
    wt_idx = _wt_index(args.dataset, sequences)
    thresholds = {
        "wt_b": float(blue[wt_idx]) if wt_idx is not None else None,
        "wt_r": float(red[wt_idx])  if wt_idx is not None else None,
        "p75_b": float(np.percentile(blue, 75)),
        "p75_r": float(np.percentile(red,  75)),
    }

    print(f"\nLandscape: {args.dataset}")
    print(f"  total sequences: {len(blue)}")
    print(f"  Pareto-optimal: {ref_mask.sum()}")
    print(f"  reference hypervolume (vs (0,0)): {ref_hv:.4f}")
    print(f"  max blue: {blue.max():.4f}, max red: {red.max():.4f}")
    print(f"  WT index: {wt_idx}  (blue={thresholds['wt_b']}, red={thresholds['wt_r']})")
    print(f"  P75 thresholds:  blue={thresholds['p75_b']:.4f}, red={thresholds['p75_r']:.4f}")
    print()

    def seed_of(p):
        """Extract seed id from filename (metrics_seedNN.json) or parent dir (seed_NN/metrics.json)."""
        stem = p.stem
        if stem.startswith("metrics_seed"):
            try:
                return int(stem.split("seed")[-1])
            except ValueError:
                pass
        parent = p.parent.name
        if parent.startswith("seed_"):
            try:
                return int(parent.split("_", 1)[1])
            except (ValueError, IndexError):
                pass
        if parent.startswith("seed"):
            try:
                return int(parent[4:])
            except ValueError:
                pass
        return 0

    results = {}
    for name in methods:
        pat = METHOD_PATTERNS.get(name)
        if pat is None:
            print(f"  {name:>12}  (unknown method)")
            continue
        files = sorted(ROOT.glob(pat.format(ds=args.dataset)))
        if args.first_n:
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
                per_seed.append(per_seed_moo_metrics(qi, blue, red, ref_front, ref_hv, norm, thresholds))
            except Exception as e:
                skipped += 1
                continue
        if not per_seed:
            print(f"  {name:>12}  <no usable metrics ({skipped} skipped)>")
            continue
        summary = {}
        # Summarize scalar fields (skip the per-seed _trajectory dicts)
        for k in per_seed[0]:
            if k.startswith("_"):
                continue
            summary[k] = summarize([d[k] for d in per_seed])
        # Summarize trajectory separately
        summary["trajectory"] = summarize_trajectory(per_seed)
        summary["_n_files"] = len(files)
        summary["_n_used"] = len(per_seed)
        summary["_n_skipped"] = skipped
        results[name] = summary

    # Print compact table
    hdr = ["method", "n", "max_blue", "max_red", "max_scal", "hv_norm", "pareto_cov",
           "prod", "max_min", "d_ideal", "p75_hit", "hv_auc"]
    widths = [12, 4, 9, 9, 9, 9, 11, 9, 9, 9, 9, 9]
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
            f"{s['product_score']['median']:.3f}",
            f"{s['max_min_norm']['median']:.3f}",
            f"{s['distance_to_ideal']['median']:.3f}",
            f"{s['n_hits_p75']['median']:.1f}",
            f"{s['hv_auc']['median']:.1f}",
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
             "max_red":  float(red.max()),
             "min_blue": float(blue.min()),
             "min_red":  float(red.min()),
             "wt_idx":   wt_idx,
             "wt_blue":  thresholds["wt_b"],
             "wt_red":   thresholds["wt_r"],
             "p75_blue": thresholds["p75_b"],
             "p75_red":  thresholds["p75_r"],
         },
         "trajectory_checkpoints": TRAJECTORY_CHECKPOINTS,
         "methods": results}, indent=2, default=str))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    sys.exit(main())
