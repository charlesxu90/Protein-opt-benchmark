#!/usr/bin/env python
"""
run_4site_benchmark.py - Benchmark baseline methods on the combinatorially-complete
4-site datasets (true-landscape POOL selection), producing the same two headline
metrics as the ms_* oracle benchmark (max fitness + top-128 mean) for a fair
side-by-side method comparison.

4-site tasks (GB1/PhoQ/TEV/TRPB) enumerate 20^4 ~= 160k variants, all measured. So
the canonical setting is pool selection against the TRUE fitness (no learned oracle):
methods pick 96 sequences/round x 5 rounds = 480 queries from the full library.

Selection rules are IDENTICAL to scripts/run_oracle_benchmark.py (imported), so ALDE/
CLADE/ftMLDE mean the same thing across both benchmark panels:
    Random      : random unqueried picks.
    GreedyWalk  : true-fitness greedy over in-pool single-mutant neighbors of best.
    ftMLDE      : CV-top3 zoo ensemble, top-96 predicted.
    CLADE       : ensemble + MiniBatchKMeans best-per-cluster (diversity).
    ALDE        : RandomForest ensemble + Thompson sampling.

CPU-only (sklearn) so it does not contend with the GPU ms_* sweep.
Output schema matches the oracle runner -> aggregate with scripts/aggregate_oracle_results.py.

Usage:
    python scripts/run_4site_benchmark.py --method ftMLDE --dataset 4site_GB1 --seeds 621 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.compat import load_landscape_data
from utils.oracle_model import ALPHABET
from run_oracle_benchmark import (
    encode_varying, fit_ftmlde, fit_clade, fit_alde, AA_IDX,
)

warnings.filterwarnings("ignore")
METHODS = ["Random", "GreedyWalk", "ftMLDE", "CLADE", "ALDE"]


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def compute_metrics(qseqs, qfit_norm, consensus, batch, n_rounds):
    order = np.argsort(-qfit_norm)
    top = order[:128]
    top_seqs = [qseqs[i] for i in top]
    traj = [float(qfit_norm[:batch * r].max()) for r in range(1, n_rounds + 1)]
    div = 0.0
    if len(top_seqs) > 1:
        ds = [hamming(top_seqs[i], top_seqs[j])
              for i in range(len(top_seqs)) for j in range(i + 1, len(top_seqs))]
        div = float(np.mean(ds))
    nov = float(np.mean([hamming(s, consensus) for s in top_seqs]))
    return {
        "max_fitness_norm": float(qfit_norm.max()),
        "top128_mean_norm": float(qfit_norm[top].mean()),
        "mean_all_norm": float(qfit_norm.mean()),
        "diversity_top128": div,
        "novelty_top128_vs_wt": nov,
        "best_n_muts": hamming(qseqs[int(order[0])], consensus),
        "best_sequence": qseqs[int(order[0])],
        "fitness_trajectory": traj,
    }


def run_one(method, dataset, seed, batch=96, n_rounds=5, out_dir=None,
            cache=None):
    t0 = datetime.now()
    if cache is None:
        cache = {}
    if dataset not in cache:
        seqs, fit_raw = load_landscape_data(dataset)
        seqs = list(seqs)
        fit_raw = np.asarray(fit_raw, dtype=float)
        gmax = float(fit_raw.max())
        ynorm = fit_raw / gmax
        L = len(seqs[0])
        X = encode_varying(seqs, list(range(L)))
        seq_to_idx = {s: i for i, s in enumerate(seqs)}
        consensus = "".join(
            max(set(c), key=[s[p] for s in seqs[:5000]].count)
            for p, c in enumerate(zip(*[s for s in seqs[:5000]])))
        cache[dataset] = (seqs, ynorm, gmax, L, X, seq_to_idx, consensus)
    seqs, ynorm, gmax, L, X, seq_to_idx, consensus = cache[dataset]
    N = len(seqs)
    rng = np.random.RandomState(seed)

    queried = list(rng.choice(N, batch, replace=False))
    for r in range(1, n_rounds):
        qset = set(queried)
        unq = np.array([i for i in range(N) if i not in qset])
        if method == "Random":
            pick = rng.choice(unq, batch, replace=False)
        elif method == "GreedyWalk":
            best = queried[int(np.argmax(ynorm[queried]))]
            bs = seqs[best]
            nbr = []
            for p in range(L):
                for aa in ALPHABET:
                    if aa != bs[p]:
                        j = seq_to_idx.get(bs[:p] + aa + bs[p + 1:])
                        if j is not None and j not in qset:
                            nbr.append(j)
            nbr = np.array(sorted(set(nbr), key=lambda j: -ynorm[j]))
            pick = nbr[:batch]
            if len(pick) < batch:
                fill = rng.choice(np.setdiff1d(unq, pick), batch - len(pick), replace=False)
                pick = np.concatenate([pick, fill])
        else:
            Xtr, ytr = X[queried], ynorm[queried]
            Xpool = X[unq]
            if method == "ftMLDE":
                preds = fit_ftmlde(Xtr, ytr)(Xpool)
                sel = np.argsort(-preds)[:batch]
            elif method == "ALDE":
                mean, std = fit_alde(Xtr, ytr)(Xpool)
                sel = np.argsort(-(mean + std * rng.randn(len(mean))))[:batch]
            elif method == "CLADE":
                from sklearn.cluster import MiniBatchKMeans
                preds = fit_clade(Xtr, ytr)(Xpool)
                k = min(batch, len(Xpool))
                lab = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3).fit(Xpool).labels_
                sel = []
                for c in range(k):
                    idx = np.where(lab == c)[0]
                    if len(idx):
                        sel.append(idx[np.argmax(preds[idx])])
                sel = np.array(sel)
                if len(sel) < batch:
                    rest = np.setdiff1d(np.argsort(-preds), sel)
                    sel = np.concatenate([sel, rest[:batch - len(sel)]])
                sel = sel[:batch]
            pick = unq[sel]
        queried.extend(int(i) for i in pick)

    queried = np.array(queried)
    qseqs = [seqs[i] for i in queried]
    qfit = ynorm[queried]
    metrics = compute_metrics(qseqs, qfit, consensus, batch, n_rounds)
    runtime = (datetime.now() - t0).total_seconds()
    result = {"method": method, "dataset": dataset, "seed": seed,
              "n_queries": len(queried), "global_max_raw": gmax,
              "runtime_seconds": runtime, "metrics": metrics}
    if out_dir:
        sub = os.path.join(out_dir, dataset, method)
        os.makedirs(sub, exist_ok=True)
        with open(os.path.join(sub, f"seed{seed}.json"), "w") as f:
            json.dump(result, f, indent=2)
    print(f"  [{dataset}/{method}/seed{seed}] max={metrics['max_fitness_norm']:.4f} "
          f"top128={metrics['top128_mean_norm']:.4f} div={metrics['diversity_top128']:.2f} "
          f"{runtime:.1f}s")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--dataset", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--seed", type=int)
    g.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--out_dir", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results_4site")))
    args = ap.parse_args()
    seeds = args.seeds or ([args.seed] if args.seed is not None else [42])
    print(f"== {args.method} on {args.dataset}, seeds={seeds} ==")
    cache = {}
    for s in seeds:
        run_one(args.method, args.dataset, s, out_dir=args.out_dir, cache=cache)


if __name__ == "__main__":
    main()
