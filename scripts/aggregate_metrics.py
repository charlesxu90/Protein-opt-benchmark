#!/usr/bin/env python
"""
aggregate_metrics.py — Cross-method metric summary on a single dataset.

Different methods write metrics into different schemas:
    Random / GreedyWalk / AiCE / ALDE / FLEXS / EvoPlay / LatProtRL / alphavariant:
        top-level["metrics"] is a dict with keys high_fitness_proximity, ...
    delta_cs:
        top-level["final_metrics"] is the round-N metrics dict (or
        top-level["metrics"] is a list of per-round dicts).

Some methods report raw max_fitness (range ~[0, GLOBAL_MAX]) while others report
it normalized to [0, 1]. We auto-rescale so the comparison is apples-to-apples.

Usage
-----
    python scripts/aggregate_metrics.py --dataset GB1 --seed 621
    python scripts/aggregate_metrics.py --dataset GB1 --seed 621 --output results/GB1_summary.json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Per-dataset global maxima (used to normalize raw max_fitness reports).
# Values are max(fitness) over data/<name>/data.csv.
GLOBAL_MAX = {
    "GB1":       8.76196565571,
    "4site_GB1": 8.761966,         # same landscape, new dataset name in CombinGym
    "CR9114":    9.834508,         # CombinGym bnAbs_CR9114_H1
    "CreiLOV":   15686.304864,     # CombinGym CreiLOV
    "TRPB":      1.0000,           # CombinGym tryptophan synthase β-subunit
    # ProteinGym / internal landscapes that are already in [0,1] use 1.0
    # so normalization is an identity (raw values pass through).
    "AAV_med":  1.0,
    "AAV_hard": 1.0,
    "GFP_med":  1.0,
    "GFP_hard": 1.561,
    # eqFP611 / mTagBFP2 single-property splits (CombinGym dual-channel data)
    "eqFP611_blue":  1.6077,
    "eqFP611_red":   1.6924,
    "mTagBFP2_blue": 1.6077,
    "mTagBFP2_red":  1.6924,
    # 4-site combinatorial ProteinGym/Wittmann-style libraries
    "PAB1":         2.6279,
    "4site_PhoQ":   133.5943,
    "4site_TEV":    1.0000,
}

# (display name, glob pattern under <method>/results/ for the metrics JSON)
METHOD_LOCATIONS = {
    "Random":      ["Random/results/{dataset}_Random/{dataset}/random/metrics_seed{seed}.json"],
    "GreedyWalk":  ["GreedyWalk/results/{dataset}_GreedyWalk/{dataset}/greedy/metrics_seed{seed}.json"],
    "AiCE":        ["AiCE/results/{dataset}_AiCE/{dataset}/aice/metrics_seed{seed}.json",
                    "AiCE/results/{dataset}_AiCE_experiments/{dataset}/aice/metrics_seed{seed}.json"],
    "ALDE":        ["ALDE/results/{dataset}_ALDE/{dataset}/onehot/metrics_seed{seed}.json",
                    "ALDE/results/{dataset}_TS_experiments/{dataset}/onehot/metrics_seed{seed}.json"],
    "FLEXS":       ["FLEXS/results/{dataset}_AdaLead/{dataset}/metrics_seed{seed}.json",
                    "FLEXS/results/{dataset}_AdaLead_experiments/{dataset}/metrics_seed{seed}.json"],
    "ftMLDE":      ["ftMLDE/results/{dataset}_ftMLDE/{dataset}/ftmlde/metrics_seed{seed}.json"],
    "CLADE":       ["CLADE/results/{dataset}_CLADE/{dataset}/clade/metrics_seed{seed}.json"],
    "delta_cs":    ["delta_cs/BioSeq-GFN-AL/results/{dataset}/metrics_seed{seed}.json"],
    "EvoPlay":     ["EvoPlay/results/{dataset}_EvoPlay_experiments/{dataset}/onehot/metrics_seed{seed}.json",
                    "EvoPlay/results/{dataset}_EvoPlay_experiments/{dataset}/evoplay/metrics_seed{seed}.json"],
    "LatProtRL":   ["LatProtRL/results/{dataset}_LatProtRL/{dataset}_medium_seed{seed}_*/metrics.json",
                    "LatProtRL/results/{dataset}_LatProtRL/{dataset}_*_seed{seed}_*/metrics.json",
                    "LatProtRL/results/{dataset}_LatProtRL/{dataset}/metrics_seed{seed}.json",
                    "LatProtRL/results/{dataset}/metrics_seed{seed}.json"],
    "alphavariant":["alphavariant/results/{dataset}_AlphaVariant/seed_{seed}/metrics.json"],
}

KEYS = [
    "max_fitness",
    "simple_regret",
    "normalized_fitness_median_top128",
    "high_fitness_proximity",
    "novelty",
    "batch_diversity",
    "spearman_correlation",
    "recall_high_order",
    "auoc",
    "miscalibration_area",
    "global_max_found",
]


def get_metrics_block(rec):
    """Extract the metric dict from a method's record."""
    if not isinstance(rec, dict):
        return {}
    if "final_metrics" in rec and isinstance(rec["final_metrics"], dict):
        return rec["final_metrics"]
    m = rec.get("metrics")
    if isinstance(m, dict):
        return m
    if isinstance(m, list) and m and isinstance(m[-1], dict):
        return m[-1]
    return {}


def fmt(v, width=8, prec=4):
    if v is None:
        return f"{'—':>{width}}"
    if isinstance(v, bool):
        return f"{'T' if v else 'F':>{width}}"
    if isinstance(v, (int, float)):
        if abs(v) < 1000:
            return f"{v:>{width}.{prec}f}"
        return f"{v:>{width}.2e}"
    return f"{str(v)[:width]:>{width}}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", default=None,
                   help="Write aggregated JSON here (default: results/<dataset>_summary.json)")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Subset of methods (default: all configured)")
    args = p.parse_args()

    methods = args.methods or list(METHOD_LOCATIONS)
    g_max = GLOBAL_MAX.get(args.dataset)

    def resolve_path(pat: str):
        """Resolve a path pattern (supports glob via {dataset}/{seed} expansion).

        Returns the most recently modified match, or None if no match.
        """
        formatted = pat.format(dataset=args.dataset, seed=args.seed)
        # If the pattern has glob characters, expand
        if "*" in formatted:
            matches = sorted(ROOT.glob(formatted), key=lambda p: p.stat().st_mtime)
            return matches[-1] if matches else None
        p = ROOT / formatted
        return p if p.exists() else None

    cols = ["method", "max(N)", "regret(N)", "norm-Top128",
            "high-fit", "novelty", "diversity",
            "spearman", "recall_HO", "auoc",
            "miscalibr", "global_max"]
    widths = [14, 8, 9, 11, 8, 8, 9, 9, 9, 8, 9, 10]
    print("".join(f"{c:>{w}}" for c, w in zip(cols, widths)))
    print("-" * sum(widths))

    results = {}
    for name in methods:
        patterns = METHOD_LOCATIONS.get(name, [])
        path = None
        for pat in patterns:
            cand = resolve_path(pat)
            if cand is not None:
                path = cand
                break
        if path is None:
            print(f"{name:>14}" + " " * (sum(widths[1:]) - 12) + " <not found>")
            continue
        try:
            rec = json.load(open(path))
        except Exception as e:
            print(f"{name:>14}  parse error: {e}")
            continue
        m = get_metrics_block(rec)
        raw_max = m.get("max_fitness")
        raw_regret = m.get("simple_regret")
        # If raw_max > 1.5, treat as raw fitness (rescale by global max).
        rescale = (raw_max is not None and raw_max > 1.5
                   and g_max is not None)
        max_norm = (raw_max / g_max) if rescale else raw_max
        regret_norm = (raw_regret / g_max) if (rescale and raw_regret is not None) else raw_regret

        row_vals = [
            max_norm, regret_norm,
            m.get("normalized_fitness_median_top128"),
            m.get("high_fitness_proximity"),
            m.get("novelty"),
            m.get("batch_diversity"),
            m.get("spearman_correlation"),
            m.get("recall_high_order"),
            m.get("auoc"),
            m.get("miscalibration_area"),
            m.get("global_max_found", m.get("global_max_hit_count")),
        ]
        line = f"{name:>14}"
        for v, w in zip(row_vals, widths[1:]):
            line += fmt(v, width=w)
        print(line)

        results[name] = {
            "_path": str(path),
            "max_fitness_raw": raw_max,
            "max_fitness_normalized": max_norm,
            "simple_regret_raw": raw_regret,
            "simple_regret_normalized": regret_norm,
            **{k: m.get(k) for k in KEYS if k not in ("max_fitness", "simple_regret")},
        }

    print()
    print("Notes:")
    if g_max:
        print(f"  - max(N) and regret(N) are normalized by GLOBAL_MAX[{args.dataset}] = {g_max:.3f}.")
    print("  - Methods reporting raw max_fitness > 1.5 are auto-rescaled.")
    print("  - global_max column: T/F if available; integer count for delta_cs.")

    out = (Path(args.output) if args.output
           else ROOT / "results" / f"{args.dataset}_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved aggregate to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
