#!/usr/bin/env python
"""
align_homologs.py - Align variable-length homologs to a dataset's target sequence
and emit target-coordinate aligned sequences for AlphaVariant prior training.

target_seqs.fasta / alignment.sto hold FULL-LENGTH homologs that are NOT aligned to
the target design region. We locally align each homolog to the WT (BLOSUM62, Biopython),
read off the homolog residue aligned to each WT position (WT residue where the homolog
gaps), and keep homologs with adequate coverage. Output is a CSV of length-seq_len
sequences in target coordinates -> data/<dataset>/prior_aligned.csv.

Then: train_ms_prior.py --dataset <d> --aligned_csv data/<d>/prior_aligned.csv

Usage:
    python scripts/alphavariant/align_homologs.py --dataset ms_AAV
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices

BENCH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def read_fasta(path):
    seqs, cur = [], None
    for line in open(path):
        if line.startswith(">"):
            if cur is not None:
                seqs.append(cur)
            cur = ""
        else:
            cur = (cur or "") + line.strip()
    if cur:
        seqs.append(cur)
    return seqs


def read_stockholm(path):
    d = {}
    for line in open(path):
        line = line.rstrip("\n")
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        p = line.split()
        if len(p) == 2:
            d.setdefault(p[0], "")
            d[p[0]] += p[1]
    return [s.replace("-", "").replace(".", "").upper() for s in d.values()]


VALID = set("ACDEFGHIKLMNPQRSTVWY")


def make_aligner():
    a = PairwiseAligner()
    a.substitution_matrix = substitution_matrices.load("BLOSUM62")
    a.open_gap_score = -11
    a.extend_gap_score = -1
    a.mode = "local"
    return a


def align_one(aligner, wt, homolog):
    """Return a length-len(wt) string: homolog residue at each WT position
    (WT residue where unaligned), and the covered fraction."""
    homolog = "".join(c for c in homolog if c in VALID)
    if len(homolog) < 5:
        return None, 0.0
    try:
        aln = aligner.align(wt, homolog)[0]
    except Exception:
        return None, 0.0
    out = list(wt)
    covered = 0
    # aln.aligned: tuple of (wt_blocks, homolog_blocks); each block is (start,end)
    wt_blocks, h_blocks = aln.aligned
    for (ws, we), (hs, he) in zip(wt_blocks, h_blocks):
        for k in range(we - ws):
            out[ws + k] = homolog[hs + k]
            covered += 1
    return "".join(out), covered / len(wt)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_dir", default=os.path.join(BENCH, "data"))
    ap.add_argument("--max_seqs", type=int, default=20000)
    ap.add_argument("--min_coverage", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dd = os.path.join(args.data_dir, args.dataset)
    wt = "".join(l.strip() for l in open(os.path.join(dd, "wt.fasta"))
                 if l.strip() and not l.startswith(">"))
    import glob as _glob
    _fa = os.path.join(dd, "target_seqs.fasta")
    # Stockholm candidates: alignment.sto, else the jackhmmer iter-*.sto (largest = most homologs)
    _stos = [p for p in ([os.path.join(dd, "alignment.sto")]
                         + sorted(_glob.glob(os.path.join(dd, "iter-*.sto"))))
             if os.path.exists(p) and os.path.getsize(p) > 0]
    if os.path.exists(_fa) and os.path.getsize(_fa) > 0:
        raw = read_fasta(_fa)
    elif _stos:
        src = max(_stos, key=os.path.getsize)
        print(f"  using Stockholm homology: {src}")
        raw = read_stockholm(src)
    else:
        raise FileNotFoundError(f"no non-empty homology (target_seqs.fasta / *.sto) in {dd}")

    rng = np.random.RandomState(args.seed)
    if len(raw) > args.max_seqs:
        raw = [raw[i] for i in rng.choice(len(raw), args.max_seqs, replace=False)]

    aligner = make_aligner()
    aligned, kept, lowcov = [], 0, 0
    for i, h in enumerate(raw):
        s, cov = align_one(aligner, wt, h)
        if s is not None and cov >= args.min_coverage:
            aligned.append(s)
            kept += 1
        else:
            lowcov += 1
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(raw)} aligned ({kept} kept)")
    # always include WT itself
    aligned.append(wt)
    out = os.path.join(dd, "prior_aligned.csv")
    pd.DataFrame({"sequence": aligned}).to_csv(out, index=False)
    print(f"[{args.dataset}] wt_len={len(wt)} homologs={len(raw)} "
          f"kept={kept} dropped_lowcov={lowcov} -> {out} ({len(aligned)} seqs, all len {len(wt)})")


if __name__ == "__main__":
    main()
