#!/usr/bin/env python
"""
compute_mpnn_freqs.py - Precompute ProteinMPNN per-position amino-acid frequencies
for a dataset's structure, aligned to the dataset wild-type sequence. This is the
structure signal AiCE uses (inverse-folding frequencies).

Pipeline (AiCE's inverse_MPNN.sh, replicated):
    parse_multiple_chains.py -> protein_mpnn_run.py (1000 samples, T=0.5)
    -> count per-position AA frequencies over the samples
    -> align the structure's native sequence to the dataset WT (substring offset,
       else best sliding-window identity) -> WT-indexed frequency matrix (L, 20)

Saves data/<dataset>/aice_mpnn_freq.npz with:
    freq      : (L, 20) frequencies in ALPHABET order; uncovered positions = 0
    covered   : (L,) bool mask of WT positions backed by the structure
    alphabet  : the 20-letter order

Run with AiCE/env (has the ProteinMPNN torch deps):
    CUDA_VISIBLE_DEVICES=0 AiCE/env/bin/python scripts/compute_mpnn_freqs.py --dataset ms_CreiLOV
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

BENCH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MPNN = os.path.join(BENCH, "AiCE", "scripts", "ProteinMPNN")
ALPHABET = "ARNDCQEGHILKMFPSTWYV"
AA_IDX = {a: i for i, a in enumerate(ALPHABET)}


def read_wt(dataset):
    p = os.path.join(BENCH, "data", dataset, "wt.fasta")
    return "".join(l.strip() for l in open(p) if l.strip() and not l.startswith(">"))


def run_mpnn(pdb, workdir, num_samples, temp, py):
    pdbs = os.path.join(workdir, "pdbs")
    os.makedirs(pdbs, exist_ok=True)
    shutil.copy(pdb, pdbs)
    jsonl = os.path.join(workdir, "parsed.jsonl")
    subprocess.run([py, os.path.join(MPNN, "helper_scripts", "parse_multiple_chains.py"),
                    "--input_path", pdbs, "--output_path", jsonl], check=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    subprocess.run([py, os.path.join(MPNN, "protein_mpnn_run.py"),
                    "--jsonl_path", jsonl, "--out_folder", workdir,
                    "--num_seq_per_target", str(num_samples),
                    "--sampling_temp", str(temp), "--seed", "37",
                    "--batch_size", "16"], check=True, env=env)
    fa = glob.glob(os.path.join(workdir, "seqs", "*.fa"))[0]
    records = []
    cur = None
    for line in open(fa):
        if line.startswith(">"):
            if cur is not None:
                records.append(cur)
            cur = ""
        else:
            cur += line.strip()
    if cur:
        records.append(cur)
    native, samples = records[0], records[1:]
    return native, samples


def freq_from_samples(samples, struct_len):
    counts = np.zeros((struct_len, 20), dtype=float)
    for s in samples:
        for p in range(min(struct_len, len(s))):
            j = AA_IDX.get(s[p])
            if j is not None:
                counts[p, j] += 1
    row = counts.sum(1, keepdims=True)
    return np.divide(counts, row, out=np.zeros_like(counts), where=row > 0)


def align_offset(native, wt):
    """Return (offset, identity) mapping wt onto native by best contiguous overlap."""
    if wt in native:
        return native.find(wt), 1.0
    best_off, best_id = 0, -1.0
    # slide wt across native (wt may be longer or shorter)
    lo, hi = -len(wt) + 1, len(native)
    for off in range(lo, hi):
        m = tot = 0
        for i in range(len(wt)):
            j = off + i
            if 0 <= j < len(native):
                tot += 1
                m += native[j] == wt[i]
        if tot:
            ident = m / len(wt)
            if ident > best_id:
                best_id, best_off = ident, off
    return best_off, best_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num_samples", type=int, default=1000)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--min_identity", type=float, default=0.4,
                    help="below this, treat structure as non-mapping (no coverage)")
    args = ap.parse_args()

    wt = read_wt(args.dataset)
    pdbs = glob.glob(os.path.join(BENCH, "data", args.dataset, "*.pdb"))
    if not pdbs:
        raise FileNotFoundError(f"no .pdb in data/{args.dataset}")
    pdb = pdbs[0]

    with tempfile.TemporaryDirectory() as wd:
        native, samples = run_mpnn(pdb, wd, args.num_samples, args.temp, sys.executable)
    sf = freq_from_samples(samples, len(native))
    off, ident = align_offset(native, wt)
    print(f"[{args.dataset}] struct_len={len(native)} wt_len={len(wt)} "
          f"offset={off} identity={ident:.3f}")

    L = len(wt)
    freq = np.zeros((L, 20), dtype=float)
    covered = np.zeros(L, dtype=bool)
    if ident >= args.min_identity:
        for i in range(L):
            j = off + i
            if 0 <= j < len(native):
                freq[i] = sf[j]
                covered[i] = True
    else:
        print(f"  WARNING: identity {ident:.3f} < {args.min_identity}; "
              f"structure does not map (e.g. AAV insert). No coverage -> AiCE falls "
              f"back to observed-data frequencies at all positions.")

    out = os.path.join(BENCH, "data", args.dataset, "aice_mpnn_freq.npz")
    np.savez(out, freq=freq, covered=covered, alphabet=ALPHABET,
             offset=off, identity=ident)
    print(f"  saved {out}  (covered {covered.sum()}/{L} positions)")


if __name__ == "__main__":
    main()
