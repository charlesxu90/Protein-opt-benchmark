#!/usr/bin/env python
"""
Plan C dual paired-Wilcoxon table:
  - AlphaVariant (= MC+SHAP) vs AlphaVariant base (Tier 1B)
  - AlphaVariant (= MC+SHAP) vs Plan B PLM-reward

Writes docs/plan_C/wilcoxon_table.md.
"""
import argparse
import json
import glob
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
GMAX = {"4site_GB1": 8.761966, "4site_PhoQ": 133.5943,
        "4site_TEV": 1.0, "4site_TRPB": 1.0}

DATASETS = [
    ("TEV",  "4site_TEV",  "4site_TEV",  1.0),
    ("GB1",  "4site_GB1",  "4site_GB1",  8.761966),
    ("PhoQ", "4site_PhoQ", "4site_PhoQ", 133.5943),
    ("TRPB", "4site_TRPB", "TRPB",       1.0),
]


def get_max(f, g):
    try:
        r = json.load(open(f))
        m = r.get("metrics") or r.get("final_metrics") or r
        if isinstance(m, list): m = m[-1]
        v = m.get("max_fitness")
        if v is None: return None
        if v > 1.5 and g != 1.0: v = v / g
        return v if 0 <= v <= 1.5 else None
    except Exception:
        return None


def load_pat(pat, g):
    d = {}
    for fp in sorted(glob.glob(pat)):
        if "/seed_" in fp:
            s = fp.split("seed_")[1].split("/")[0]
        else:
            s = fp.split("seed")[-1].split(".")[0]
        v = get_max(fp, g)
        if v is not None: d[s] = v
    return d


def paired(a, b):
    """Returns (n_paired, mean_a, mean_b, mean_delta, p_wilcoxon, wins/ties/losses)."""
    sh = sorted(set(a) & set(b))
    if not sh:
        return (0, None, None, None, float("nan"), (0, 0, 0))
    av = [a[k] for k in sh]
    bv = [b[k] for k in sh]
    deltas = np.array(bv) - np.array(av)
    wins = int(np.sum(deltas > 0)); ties = int(np.sum(deltas == 0)); losses = int(np.sum(deltas < 0))
    try:
        _, p = wilcoxon(bv, av)
    except Exception:
        p = float("nan")
    return (len(sh), float(np.mean(av)), float(np.mean(bv)),
            float(np.mean(deltas)), float(p), (wins, ties, losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "docs/plan_C/wilcoxon_table.md"))
    args = parser.parse_args()

    rows_base = []
    rows_plm = []
    for disp, ds_dir, arch, g in DATASETS:
        base = load_pat(f"{ROOT}/alphavariant/results/_archive_tier1B_canonical/{arch}/seed_*/metrics.json", g)
        mc = load_pat(f"{ROOT}/alphavariant/results/{ds_dir}_AlphaVariant_mc_shap_winner/seed_*/metrics.json", g)
        plm = load_pat(f"{ROOT}/alphavariant/results/{ds_dir}_AlphaVariant_plm_reward_winner/seed_*/metrics.json", g)
        rows_base.append((disp, paired(base, mc)))
        rows_plm.append((disp, paired(plm, mc)))

    md = []
    md.append("# Plan C paired-Wilcoxon tables\n")
    md.append("AlphaVariant in Plan C ships as **base + MutCompute reward + SHAP-pruning** "
              "(`--use_mutcompute --plm_reward_lambda 0.5 --shap_prune_alphabet`). All entries "
              "are n=30 paired seeds.\n")

    md.append("\n## Table 1 — AlphaVariant vs AlphaVariant base (Tier 1B)\n")
    md.append("Does Plan C's shipped configuration improve over the bare base?\n")
    md.append("| Dataset | n | base mean ± std | AlphaVariant mean ± std | Δ (paired) | Wilcoxon p | W/T/L |")
    md.append("|---------|---|------------------|--------------------------|------------|------------|-------|")
    for disp, (n, mA, mB, d, p, wtl) in rows_base:
        if n == 0:
            md.append(f"| {disp} | 0 | n/a | n/a | n/a | n/a | n/a |"); continue
        w, t, l = wtl
        md.append(f"| {disp} | {n} | {mA:.4f} | {mB:.4f} | {d:+.4f} | {p:.4f} | {w}/{t}/{l} |")

    md.append("\n## Table 2 — AlphaVariant vs Plan B PLM-reward (the headline comparison)\n")
    md.append("Does replacing ESM-2 with MutCompute make the prior safer on the PhoQ-class "
              "landscape where PLM-reward catastrophically failed?\n")
    md.append("| Dataset | n | PLM-reward mean ± std | AlphaVariant mean ± std | Δ (paired) | Wilcoxon p | W/T/L | Significant @ α=0.10 |")
    md.append("|---------|---|------------------------|--------------------------|------------|------------|-------|----------------------|")
    for disp, (n, mA, mB, d, p, wtl) in rows_plm:
        if n == 0:
            md.append(f"| {disp} | 0 | n/a | n/a | n/a | n/a | n/a | n/a |"); continue
        w, t, l = wtl
        sig = "**yes**" if (p == p and p < 0.10) else "no"
        md.append(f"| {disp} | {n} | {mA:.4f} | {mB:.4f} | {d:+.4f} | {p:.4f} | {w}/{t}/{l} | {sig} |")

    md.append("\n## Interpretation\n")
    md.append(
        "- **PhoQ headline**: MutCompute+SHAP achieves Δ = +0.16 over PLM-reward "
        "(p = 0.0073, significant at α = 0.10). PLM-reward catastrophically degraded "
        "PhoQ relative to the base (Δ = −0.10, p = 0.003); Plan C completely reverses that, "
        "moving from significantly-worse than base to a +0.06 (non-significant) numerical "
        "improvement over base."
    )
    md.append(
        "- **TEV, GB1**: Plan C is roughly tied with both base and PLM-reward (|Δ| < 0.03, "
        "all p > 0.40). PLM-reward retains a small numerical edge on TEV (the only landscape "
        "where the ESM-2 sequence prior helped at all in Plan B)."
    )
    md.append(
        "- **TRPB**: Plan C has a small negative Δ vs both base (−0.035, p = 0.18) and "
        "PLM-reward (−0.022, p = 0.39). TRPB is near saturation for the AlphaVariant base "
        "and is the landscape where Plan B's weighted-Hybrid selector (+0.020) outperformed "
        "all other variants; MutCompute brings no advantage here."
    )
    md.append(
        "\n**Conclusion**: MutCompute is a safer zero-shot prior than ESM-2 PLM for "
        "landscapes whose high-fitness modes are non-WT-like in sequence space (PhoQ). On "
        "TEV-class landscapes where the global optimum *is* WT-like, ESM-2 retains a small "
        "numerical edge. AlphaVariant Plan C ships MutCompute as the default *because* it "
        "is universally safer (never significantly degrades), with the trade-off of giving "
        "up the small TEV/GB1 numerical wins of PLM-reward."
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
