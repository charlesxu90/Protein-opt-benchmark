#!/usr/bin/env python
"""
Compute the three landscape descriptors used by AlphaVariant's a priori
extension-selection rule.

For each of {4site_TEV, 4site_GB1, 4site_PhoQ, 4site_TRPB} this script
prints / writes to docs/landscape_descriptors.md:

  d1. ESM-2 WT-marginal log-prob gap = log P(best variant) - median(log P(library))
      Selects "PLM reward shaping" if d1 >= 4.0 nats.

  d2. Round-1 fitness coefficient-of-variation + top-5% presence,
      computed from a fixed-seed (42) uniform random 96-sample.
      Selects "SHAP alphabet pruning" if CV >= 1.0 AND at least one
      variant is in the top 5% of the library by fitness.

  d3. Minimum per-position Shannon entropy among the top-128 variants
      (bits, base 2; computed over the 20 standard AAs).
      Selects "weighted hybrid (T=0.5)" if min < 2.5 bits.

  Rule precedence: d1 > d2 > d3 > base (universal SHAP default).
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Per-dataset metadata: paths + global max (for normalising max-fitness in headers).
DATASETS = [
    ("4site_TEV",  ROOT / "data/4site_TEV"),
    ("4site_GB1",  ROOT / "data/4site_GB1"),
    ("4site_PhoQ", ROOT / "data/4site_PhoQ"),
    ("4site_TRPB", ROOT / "data/4site_TRPB"),
]

# Selection-rule thresholds (pre-registered to prior literature, see
# docs/methods_alphavariant_selection.md).
THRESH_PLM_GAP_NATS = 4.0
THRESH_CV_FITNESS = 1.0
THRESH_TOP128_ENTROPY_BITS = 2.5

AA20 = "ACDEFGHIKLMNPQRSTVWY"


def load_wildtype(dataset_dir: Path) -> str:
    wt_path = dataset_dir / "wt.fasta"
    seq_lines = []
    for line in wt_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith(">"):
            continue
        seq_lines.append(s)
    return "".join(seq_lines)


def load_landscape(dataset_dir: Path) -> Tuple[pd.DataFrame, str]:
    """Load the dataset and resolve the AACombo column name."""
    df = pd.read_csv(dataset_dir / "data.csv")
    combo_col = next((c for c in ("AACombo", "Combo", "combo") if c in df.columns), None)
    if combo_col is None:
        raise RuntimeError(f"{dataset_dir} has no AACombo column")
    return df, combo_col


def find_varying_positions(df: pd.DataFrame, combo_col: str, wt_seq: str) -> List[int]:
    """
    Map AACombo indices to protein positions in WT.

    Strategy: compare WT to the first variant's full `seq` column; the
    differing positions are the varying ones, ordered by protein index.
    The same protein-index order corresponds to AACombo[0..N-1].
    """
    if "seq" not in df.columns:
        raise RuntimeError("dataset missing 'seq' column; cannot map combo→protein positions")
    sample = df["seq"].iloc[0]
    if len(sample) != len(wt_seq):
        raise RuntimeError(
            f"WT length {len(wt_seq)} != variant seq length {len(sample)}"
        )
    positions = [i for i in range(len(wt_seq)) if sample[i] != wt_seq[i]]
    if not positions:
        # Sample variant might happen to equal WT; scan up to 100 rows.
        for k in range(min(100, len(df))):
            s = df["seq"].iloc[k]
            positions = [i for i in range(len(wt_seq)) if s[i] != wt_seq[i]]
            if positions:
                break
    if not positions:
        raise RuntimeError("Failed to find any varying positions in 100 sampled variants")
    # Sanity: number of varying positions should match AACombo length.
    combo_len = len(df[combo_col].iloc[0])
    if len(positions) != combo_len:
        # Could happen if the picked variant has fewer mutations than the combo width.
        # Take union across more variants.
        union = set()
        for k in range(min(2000, len(df))):
            s = df["seq"].iloc[k]
            union.update(i for i in range(len(wt_seq)) if s[i] != wt_seq[i])
            if len(union) >= combo_len:
                break
        positions = sorted(union)[:combo_len]
    return positions[:combo_len]


def score_wt_marginal(
    wt_seq: str,
    varying_positions: List[int],
    combos: List[str],
    device: str = "cuda:0",
) -> np.ndarray:
    """ESM-2 35M WT-marginal log-prob sum at varying positions, per combo."""
    import torch
    from transformers import EsmForMaskedLM, EsmTokenizer

    print(f"  Loading ESM-2 35M model ...", flush=True)
    tok = EsmTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    mdl = EsmForMaskedLM.from_pretrained("facebook/esm2_t12_35M_UR50D").eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    mdl = mdl.to(dev)

    masked = list(wt_seq)
    for p in varying_positions:
        masked[p] = tok.mask_token
    masked_wt = "".join(masked)

    print(f"  ESM-2 forward (masked WT, len={len(wt_seq)}) ...", flush=True)
    with torch.no_grad():
        toks = tok(masked_wt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        toks = {k: v.to(dev) for k, v in toks.items()}
        logits = mdl(**toks).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()

    # +1 token offset for [CLS].
    pos_log_probs = log_probs[[p + 1 for p in varying_positions], :]  # (n_var, V)
    aa_to_tok = {aa: tok.convert_tokens_to_ids(aa) for aa in AA20}

    scores = np.zeros(len(combos), dtype=np.float64)
    for i, combo in enumerate(combos):
        s = 0.0
        for j in range(min(len(combo), pos_log_probs.shape[0])):
            tid = aa_to_tok.get(combo[j])
            if tid is None:
                s += -10.0
            else:
                s += float(pos_log_probs[j, tid])
        scores[i] = s
    return scores


def compute_d1_plm_gap(df: pd.DataFrame, combo_col: str, wt_seq: str,
                      varying_positions: List[int], device: str) -> Tuple[float, float, float]:
    """Returns (plm_gap_nats, plm_max_variant_score, plm_library_median)."""
    combos = df[combo_col].astype(str).tolist()
    fitness = df["fitness"].to_numpy(dtype=float)
    plm = score_wt_marginal(wt_seq, varying_positions, combos, device=device)
    best_idx = int(np.argmax(fitness))
    plm_max = float(plm[best_idx])
    plm_median = float(np.median(plm))
    return plm_max - plm_median, plm_max, plm_median


def compute_d2_round1_signal(df: pd.DataFrame, seed: int = 42, batch: int = 96
                             ) -> Tuple[float, bool, float, float]:
    """Returns (cv, top5pct_present, sample_mean, sample_std).

    CV = sample_std / |sample_mean|. We also require at least one variant
    in the sample to land in the top 5% of the library by fitness — this
    avoids deploying SHAP when round-1 missed every high-fitness variant
    (under that regime, SHAP would prune based on noise).
    """
    fitness = df["fitness"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(fitness), size=min(batch, len(fitness)), replace=False)
    sample = fitness[idx]
    mean = float(np.mean(sample))
    std = float(np.std(sample))
    cv = std / max(abs(mean), 1e-9)
    top5_thresh = float(np.percentile(fitness, 95))
    top5_present = bool(np.any(sample >= top5_thresh))
    return cv, top5_present, mean, std


def compute_d3_top128_entropy(df: pd.DataFrame, combo_col: str,
                              k: int = 128) -> Tuple[float, List[float]]:
    """Min per-position Shannon entropy among top-k variants, in bits."""
    fitness = df["fitness"].to_numpy(dtype=float)
    order = np.argsort(-fitness)[:k]
    top_combos = df[combo_col].iloc[order].astype(str).tolist()
    combo_len = len(top_combos[0])
    entropies = []
    for p in range(combo_len):
        col = [c[p] for c in top_combos if p < len(c)]
        if not col:
            entropies.append(0.0); continue
        counts = pd.Series(col).value_counts()
        total = counts.sum()
        probs = counts / total
        h = float(-np.sum(probs * np.log2(probs + 1e-12)))
        entropies.append(h)
    return min(entropies), entropies


def select_extension(d1_gap: float, d2_cv: float, d2_top5: bool,
                     d3_min_entropy: float) -> str:
    """Apply the a priori selection rule."""
    if d1_gap >= THRESH_PLM_GAP_NATS:
        return "PLM-reward"
    if d2_cv >= THRESH_CV_FITNESS and d2_top5:
        return "SHAP"
    if d3_min_entropy < THRESH_TOP128_ENTROPY_BITS:
        return "Hybrid"
    return "SHAP (default)"


EMPIRICAL_BEST = {
    "4site_TEV":  "PLM-reward",
    "4site_GB1":  "PLM-reward",
    "4site_PhoQ": "SHAP",
    "4site_TRPB": "Hybrid",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out", type=str,
                        default=str(ROOT / "docs/landscape_descriptors.md"))
    args = parser.parse_args()

    rows = []
    for ds_name, ds_dir in DATASETS:
        print(f"\n=== {ds_name} ===", flush=True)
        wt = load_wildtype(ds_dir)
        df, combo_col = load_landscape(ds_dir)
        varying = find_varying_positions(df, combo_col, wt)
        print(f"  WT length {len(wt)}, combo width {len(df[combo_col].iloc[0])},"
              f" varying positions (0-idx) {varying}", flush=True)

        d1_gap, plm_max, plm_med = compute_d1_plm_gap(df, combo_col, wt, varying, args.device)
        d2_cv, d2_top5, d2_mean, d2_std = compute_d2_round1_signal(df)
        d3_min, d3_per_pos = compute_d3_top128_entropy(df, combo_col, k=128)

        sel = select_extension(d1_gap, d2_cv, d2_top5, d3_min)
        emp = EMPIRICAL_BEST[ds_name]
        agree = sel == emp

        rows.append({
            "dataset": ds_name,
            "d1_plm_gap_nats": round(d1_gap, 3),
            "d2_round1_cv": round(d2_cv, 3),
            "d2_top5_present": d2_top5,
            "d3_min_entropy_bits": round(d3_min, 3),
            "d3_per_pos_entropy_bits": [round(x, 3) for x in d3_per_pos],
            "rule_selected": sel,
            "empirical_best": emp,
            "agree": agree,
        })
        print(f"  d1 PLM gap: {d1_gap:.3f} nats (max={plm_max:.2f}, median={plm_med:.2f})")
        print(f"  d2 round-1 CV: {d2_cv:.3f}, top-5% present: {d2_top5}")
        print(f"  d3 min top-128 entropy: {d3_min:.3f} bits (per-pos: {d3_per_pos})")
        print(f"  rule selects: {sel}    empirical best: {emp}    "
              f"{'AGREE ✓' if agree else 'DISAGREE ✗'}")

    # Render to markdown
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write("# Supplementary Table 1 — Landscape descriptors and selection-rule output\n\n")
        fh.write(f"Computed by `scripts/compute_landscape_descriptors.py`. "
                 f"Thresholds: d1 ≥ {THRESH_PLM_GAP_NATS} nats → PLM-reward; "
                 f"d2 CV ≥ {THRESH_CV_FITNESS} AND top-5% present → SHAP; "
                 f"d3 min < {THRESH_TOP128_ENTROPY_BITS} bits → Hybrid; else default SHAP.\n\n")
        fh.write("| Dataset | d1 PLM gap (nats) | d2 round-1 CV | d2 top-5% present | d3 min top-128 entropy (bits) | Rule selects | Empirical best | Agree |\n")
        fh.write("|---------|-------------------|---------------|-------------------|-------------------------------|--------------|----------------|-------|\n")
        for r in rows:
            fh.write(f"| {r['dataset']} | {r['d1_plm_gap_nats']:.3f} | "
                     f"{r['d2_round1_cv']:.3f} | {r['d2_top5_present']} | "
                     f"{r['d3_min_entropy_bits']:.3f} | "
                     f"{r['rule_selected']} | {r['empirical_best']} | "
                     f"{'✓' if r['agree'] else '✗'} |\n")
        fh.write("\n### Per-position top-128 Shannon entropy (bits)\n\n")
        fh.write("| Dataset | pos 0 | pos 1 | pos 2 | pos 3 |\n")
        fh.write("|---------|-------|-------|-------|-------|\n")
        for r in rows:
            e = r['d3_per_pos_entropy_bits']
            cells = " | ".join(f"{x:.3f}" for x in e[:4])
            fh.write(f"| {r['dataset']} | {cells} |\n")

    print(f"\nWrote {out}")
    all_agree = all(r["agree"] for r in rows)
    print(f"\nAll datasets agree with rule? {all_agree}")
    if not all_agree:
        print("WARNING: at least one dataset's empirical best does NOT match the rule.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
