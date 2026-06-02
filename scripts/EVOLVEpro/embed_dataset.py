#!/usr/bin/env python
"""
Pre-compute ESM-2 per-variant embeddings for one of the 4-site landscapes
(`4site_GB1`, `4site_PhoQ`, `4site_TEV`, `4site_TRPB`), saving the result
in a format EVOLVEpro's `directed_evolution_simulation` can consume.

Output:
    {output_dir}/embeddings_esm2_35M.pt   # torch dict {variant -> (D,) tensor}
    {output_dir}/labels.csv                # variant, activity (= normalised fitness)

The embedding step uses HuggingFace's `transformers.EsmModel` so we can
reuse the existing alphavariant-env rather than the upstream `fair-esm`
plm_env.

The embedding is the mean of last-hidden-state token features at the
varying positions of the WT-substituted full-length sequence — matching
the AlphaVariant feature-extractor convention.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_wildtype(dataset_dir: Path) -> str:
    wt_path = dataset_dir / "wt.fasta"
    lines = []
    for line in wt_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith(">"):
            continue
        lines.append(s)
    return "".join(lines)


def detect_varying_positions(df: pd.DataFrame, combo_col: str, wt_seq: str) -> List[int]:
    """0-indexed positions where the AACombo letters substitute into WT."""
    if "seq" not in df.columns:
        raise RuntimeError("dataset missing 'seq' column")
    sample_seq = df["seq"].iloc[0]
    positions = [i for i in range(len(wt_seq)) if sample_seq[i] != wt_seq[i]]
    if not positions:
        # WT itself may be the first row; scan more
        for k in range(min(2000, len(df))):
            s = df["seq"].iloc[k]
            positions = [i for i in range(len(wt_seq)) if s[i] != wt_seq[i]]
            if positions:
                break
    if not positions:
        raise RuntimeError("Could not infer varying positions from data")
    combo_len = len(df[combo_col].iloc[0])
    if len(positions) > combo_len:
        positions = positions[:combo_len]
    return positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        help="4site_GB1 | 4site_PhoQ | 4site_TEV | 4site_TRPB")
    parser.add_argument("--data_dir", default=str(ROOT / "data"))
    parser.add_argument("--output_dir", default=None,
                        help="Default: data/<dataset>/")
    parser.add_argument("--esm_model", default="facebook/esm2_t12_35M_UR50D",
                        help="HuggingFace model name. 35M default (fast); 650M for full reproduction.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from transformers import EsmModel, EsmTokenizer

    dataset_dir = Path(args.data_dir) / args.dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"{dataset_dir} not found")
    if args.output_dir is None:
        args.output_dir = str(dataset_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[embed_dataset] {args.dataset}  →  {out_dir}", flush=True)
    df = pd.read_csv(dataset_dir / "data.csv")
    combo_col = next((c for c in ("AACombo", "Combo", "combo") if c in df.columns), None)
    if combo_col is None:
        raise RuntimeError(f"{dataset_dir} has no AACombo column")
    wt_seq = load_wildtype(dataset_dir)
    positions = detect_varying_positions(df, combo_col, wt_seq)
    print(f"  WT length {len(wt_seq)}; varying positions {positions}", flush=True)

    # Build full-length sequences with each AACombo substituted into WT.
    combos = df[combo_col].astype(str).tolist()
    full_seqs = []
    for combo in combos:
        arr = list(wt_seq)
        for j, p in enumerate(positions[:len(combo)]):
            arr[p] = combo[j]
        full_seqs.append("".join(arr))

    print(f"  Loading ESM-2 model: {args.esm_model}", flush=True)
    tok = EsmTokenizer.from_pretrained(args.esm_model)
    mdl = EsmModel.from_pretrained(args.esm_model).eval()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mdl = mdl.to(dev)

    print(f"  Embedding {len(full_seqs)} variants (batch {args.batch_size}) ...", flush=True)
    emb_dict = {}
    pos_for_pool = [p + 1 for p in positions]  # +1 for [CLS]
    with torch.no_grad():
        for b in tqdm(range(0, len(full_seqs), args.batch_size), ncols=80):
            batch_seqs = full_seqs[b:b + args.batch_size]
            batch_combos = combos[b:b + args.batch_size]
            tokens = tok(batch_seqs, return_tensors="pt", padding=True,
                         truncation=True, max_length=1024)
            tokens = {k: v.to(dev) for k, v in tokens.items()}
            out = mdl(**tokens)
            # Mean pool over varying positions only.
            h = out.last_hidden_state[:, pos_for_pool, :].mean(dim=1)  # (B, D)
            h = h.cpu().numpy().astype(np.float32)
            for combo, vec in zip(batch_combos, h):
                emb_dict[combo] = vec

    # Save embeddings as a pt-style dict for EVOLVEpro compatibility.
    emb_path = out_dir / "embeddings_evolvepro.pt"
    torch.save(emb_dict, emb_path)
    print(f"  Wrote embeddings: {emb_path} ({len(emb_dict)} variants)", flush=True)

    # Save labels CSV — EVOLVEpro expects: variant + measured_var column.
    fitness = df["fitness"].astype(float).to_numpy()
    # Normalise fitness to [0, 1] within the dataset for stability (EVOLVEpro's regressors).
    gmax = float(np.nanmax(fitness))
    if gmax > 1.5:
        fitness_norm = fitness / gmax
    else:
        # Already-normalised (TEV/TRPB)
        fitness_norm = fitness
    labels_df = pd.DataFrame({
        "variant": combos,
        "activity": fitness_norm,
    })
    labels_path = out_dir / "labels_evolvepro.csv"
    labels_df.to_csv(labels_path, index=False)
    print(f"  Wrote labels: {labels_path}", flush=True)

    # Write a small metadata blob for the run_generic adapter.
    meta = {
        "dataset": args.dataset,
        "esm_model": args.esm_model,
        "wt_length": len(wt_seq),
        "varying_positions": positions,
        "n_variants": len(combos),
        "fitness_normaliser": gmax if gmax > 1.5 else 1.0,
    }
    (out_dir / "embeddings_evolvepro.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  Wrote meta: {out_dir / 'embeddings_evolvepro.meta.json'}", flush=True)


if __name__ == "__main__":
    main()
