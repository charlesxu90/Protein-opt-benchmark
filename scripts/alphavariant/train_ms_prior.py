#!/usr/bin/env python
"""
train_ms_prior.py - Train an AlphaVariant GPT prior from prepared homology sequences
for a multi-site dataset, producing a checkpoint loadable by run_generic.py
(--prior_model_path).

Input homology:
    data/<dataset>/target_seqs.fasta   (AAV, GFP, PAB1)  -- multi-line FASTA
    data/<dataset>/alignment.sto       (CreiLOV)         -- Stockholm MSA

Sequences are trimmed/padded to the target length (= len(wt.fasta)); the GPT is
built with block_size = seq_len + 7 to match IterativeProteinTrainer's auto-config,
so the agent can deepcopy it directly. Output:
    <out>/<dataset>/prior_model.pt  + prior_model.json   (GPTConfig kwargs)

NOTE on alignment: target_seqs.fasta holds FULL-LENGTH homologs of variable length.
For GFP (homologs ~= target length) trimming is faithful; for AAV/PAB1 (homologs >>
target) trimming takes the N-terminal window and is NOT aligned to the design region.
Use --aligned_csv to pass a pre-aligned MSA CSV instead when available.

Usage:
    python scripts/alphavariant/train_ms_prior.py --dataset ms_GFP --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split, DataLoader

BENCH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(BENCH, "alphavariant"))

from popgen.model.gpt import GPT, GPTConfig
from popgen.model.prior_trainer import Trainer
from popgen.utils.dataset import load_seqs_from_list, get_tensor_dataset, AASeqDictionary
from popgen.utils.utils import set_random_seed


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
    seqs = {}
    for line in open(path):
        line = line.rstrip("\n")
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        parts = line.split()
        if len(parts) == 2:
            seqs.setdefault(parts[0], "")
            seqs[parts[0]] += parts[1]
    # strip gaps/insertions -> ungapped sequences
    return [s.replace("-", "").replace(".", "").upper() for s in seqs.values()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_dir", default=os.path.join(BENCH, "data"))
    ap.add_argument("--out_dir", default=os.path.join(BENCH, "alphavariant", "priors"))
    ap.add_argument("--aligned_csv", default=None,
                    help="optional pre-aligned MSA CSV (column 'sequence') to use instead of fasta")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--max_seqs", type=int, default=40000, help="subsample cap (PAB1 has 522k)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dset_dir = os.path.join(args.data_dir, args.dataset)
    wt = "".join(l.strip() for l in open(os.path.join(dset_dir, "wt.fasta"))
                 if l.strip() and not l.startswith(">"))
    seq_len = len(wt)

    # load homology
    if args.aligned_csv:
        raw = pd.read_csv(args.aligned_csv)["sequence"].astype(str).tolist()
    elif os.path.exists(os.path.join(dset_dir, "target_seqs.fasta")):
        raw = read_fasta(os.path.join(dset_dir, "target_seqs.fasta"))
    elif os.path.exists(os.path.join(dset_dir, "alignment.sto")):
        raw = read_stockholm(os.path.join(dset_dir, "alignment.sto"))
    else:
        raise FileNotFoundError(f"no homology (target_seqs.fasta / alignment.sto) in {dset_dir}")

    rng = np.random.RandomState(args.seed)
    if len(raw) > args.max_seqs:
        raw = [raw[i] for i in rng.choice(len(raw), args.max_seqs, replace=False)]
    print(f"[{args.dataset}] seq_len={seq_len} block_size={seq_len+7} "
          f"n_homologs={len(raw)} (trim/pad to {seq_len})")

    # popgen trims/pads to max_len
    seqs, _ = load_seqs_from_list(raw, max_len=seq_len, rm_duplicates=False)
    ds = get_tensor_dataset(seqs)
    n_val = max(1, int(0.1 * len(ds)))
    train_set, val_set = random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(args.seed))

    sd = AASeqDictionary()
    mconf = GPTConfig(sd.get_char_num(), block_size=seq_len + 7,
                      n_layer=4, n_head=4, n_embd=128)
    model = GPT(mconf)

    bpe = max(1, len(train_set) // args.batch_size)
    train_cfg = Namespace(
        n_epochs=args.epochs, learning_rate=args.lr, lr_decay=True,
        weight_decay=0.1, beta_1=0.9, beta_2=0.95, grad_norm_clip=1.0,
        device="cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu",
        warmup_tokens=int(seq_len * bpe * 0.1 * args.epochs),
        final_tokens=int(seq_len * bpe * args.epochs),
        save_model=True, seed=args.seed, n_devices=1,
        batch_size=args.batch_size, num_workers=4,
    )
    set_random_seed(args.seed)
    out = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out, exist_ok=True)

    # Lightning version compat: prior_trainer calls SingleDeviceStrategy(device="cuda"),
    # but newer Fabric requires an indexed device. Patch without touching the package.
    try:
        import lightning.fabric.strategies as _strat
        _Orig = _strat.SingleDeviceStrategy

        class _PatchedSDS(_Orig):
            def __init__(self, device="cpu", *a, **k):
                if device == "cuda":
                    device = "cuda:0"
                super().__init__(device=device, *a, **k)
        _strat.SingleDeviceStrategy = _PatchedSDS
    except Exception as _e:
        print(f"  (strategy patch skipped: {_e})")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, num_workers=4)
    trainer = Trainer(model, train_cfg, output_dir=out)
    trainer.fit(train_loader, val_loader)

    # save in the layout run_generic.py expects: <stem>.pt + <stem>.json (GPTConfig kwargs)
    ckpt = os.path.join(out, "prior_model.pt")
    torch.save(model.state_dict(), ckpt)
    with open(os.path.join(out, "prior_model.json"), "w") as f:
        json.dump({"vocab_size": sd.get_char_num(), "block_size": seq_len + 7,
                   "n_layer": 4, "n_head": 4, "n_embd": 128}, f, indent=2)
    print(f"  saved prior -> {ckpt}  (+ prior_model.json)")


if __name__ == "__main__":
    main()
