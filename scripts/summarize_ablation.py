#!/usr/bin/env python3
"""Summarize the AlphaVariant leave-one-out ablation across all datasets.

Two result schemas are handled: four-site lookup writes seed_<S>/metrics.json with
raw max_fitness (normalized here by the dataset global max); multi-site oracle writes
<dataset>/AlphaVariant/seed<S>.json with already-normalized metrics. Prints median
[n] of normalized max fitness and top-128 mean fitness per config, with delta versus
that dataset's full (Plan C) configuration, and writes a CSV. Configs with no results
are skipped, so it works incrementally while a sweep is still running.
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ABL = ROOT / "results_ablation"

# four-site global maxima for normalization (multi-site oracle metrics are pre-normalized)
GMAX = {"4site_GB1": 8.761966, "4site_PhoQ": 133.5943, "4site_TEV": 1.0, "4site_TRPB": 1.0}

FOUR_CONFIGS = [("full", "— (Plan C)"), ("no_mcreward", "− MutCompute reward"),
                ("no_shap", "− SHAP pruning"), ("bare", "− both")]
MULTI_CONFIGS = [("full", "— (Plan C)"), ("no_ev", "− EV features"),
                 ("no_shap", "− SHAP/constraint"), ("no_cap", "− mutation cap"),
                 ("no_prior", "− homolog prior")]

# (prefix, dataset_id, regime, display)
DATASETS = [
    ("gb1", "4site_GB1", "four", "GB1"),
    ("phoq", "4site_PhoQ", "four", "PhoQ"),
    ("tev", "4site_TEV", "four", "TEV"),
    ("trpb", "4site_TRPB", "four", "TrpB"),
    ("cre", "ms_CreiLOV", "multi", "CreiLOV"),
    ("aav", "ms_AAV", "multi", "AAV"),
    ("pab1", "ms_PAB1", "multi", "PAB1"),
    ("gfp", "ms_GFP", "multi", "GFP"),
]


def load(prefix, suffix, dataset_id, regime):
    """Return (max_norm[], top128[]) for one ablation config, or ([],[]) if absent."""
    cfg = f"{prefix}_{suffix}"
    mx, t = [], []
    if regime == "four":
        g = GMAX[dataset_id]
        for f in glob.glob(str(ABL / cfg / "seed_*" / "metrics.json")):
            m = json.load(open(f)).get("metrics", {})
            v, tt = m.get("max_fitness"), m.get("normalized_fitness_median_top128")
            if v is not None:
                mx.append(v / g if (g != 1.0 and v > 1.5) else v)
            if tt is not None:
                t.append(tt)
    else:
        for f in glob.glob(str(ABL / cfg / dataset_id / "AlphaVariant" / "seed*.json")):
            m = json.load(open(f)).get("metrics", {})
            v, tt = m.get("max_fitness_norm"), m.get("top128_mean_norm")
            if v is not None:
                mx.append(v)
            if tt is not None:
                t.append(tt)
    return mx, t


def med(vals):
    return float(np.median(vals)) if vals else None


def main():
    print(f"\n{'config':16} {'dataset':8} {'ablated':20} {'n':>3} "
          f"{'max(med)':>9} {'Δmax':>7} {'top128':>8} {'Δtop128':>8}")
    print("-" * 92)
    csv = ["config,dataset,ablated,n,max_median,top128_median"]
    for prefix, dsid, regime, disp in DATASETS:
        configs = FOUR_CONFIGS if regime == "four" else MULTI_CONFIGS
        full_mx, full_t = None, None
        block = []
        for suffix, label in configs:
            mx, t = load(prefix, suffix, dsid, regime)
            if not mx:
                continue
            mm, tm = med(mx), med(t)
            if suffix == "full":
                full_mx, full_t = mm, tm
            block.append((f"{prefix}_{suffix}", disp, label, len(mx), mm, tm))
        for cfg, disp, label, n, mm, tm in block:
            dmx = f"{mm-full_mx:+.3f}" if full_mx is not None else "  ref"
            dt = f"{tm-full_t:+.3f}" if (full_t is not None and tm is not None) else "  ref"
            tms = f"{tm:.3f}" if tm is not None else "n/a"
            print(f"{cfg:16} {disp:8} {label:20} {n:>3} {mm:>9.3f} {dmx:>7} {tms:>8} {dt:>8}")
            csv.append(f"{cfg},{disp},{label},{n},{mm:.4f},{tm if tm is not None else ''}")
        if block:
            print()
    out = ABL / "ablation_summary.csv"
    out.write_text("\n".join(csv) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
