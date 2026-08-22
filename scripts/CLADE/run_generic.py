#!/usr/bin/env python
"""
run_generic.py — CLADE 2.0 (cluster learning-assisted DE) adapter.

The original `WeilabMSU/CLADE` repo shells out to a Python 3.7 / TF 1.13
MLDE pipeline. This adapter implements the **same algorithm** in modern
Python using sklearn + xgboost:

    Algorithm (per CLADE 2.0 / Qiu Hu Wei 2021):
        1. One-hot encode the entire landscape
        2. KMeans cluster the landscape into K=10 clusters
        3. Round 0: sample uniformly across clusters (96 variants total)
        4. Each round 1..K_round:
            - Train MLDE ensemble on collected (seq, fitness) pairs
            - Predict the unqueried pool
            - For each cluster, pick top variants by predicted fitness
            - Optionally sub-cluster high-fitness regions (CLADE 2.0 hierarchy)
            - Query oracle for the resulting batch
        Total queries = 96 * num_batch (default: 96 * 4 = 384, but our
        benchmark protocol uses 96 * 5 = 480)

We use the simpler "flat" variant (10 clusters, no hierarchy refinement)
to keep wall time tractable. The published CLADE 2.0 uses hierarchy +
zero-shot evolutionary scores; both are optional add-ons.

Usage
-----
    CLADE/env/bin/python scripts/CLADE/run_generic.py --dataset 4site_GB1 --seed 621
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Canonical location is scripts/<Method>/. This file used to be invoked through a
# symlink in <Method>/, where `__file__/..` happened to be the benchmark root; walk
# up to find the root instead, so it runs correctly from either path.
# (Same idiom as scripts/EVOLVEpro/run_generic.py.)
_p = os.path.dirname(os.path.realpath(__file__))
while os.path.dirname(_p) != _p and not os.path.isdir(os.path.join(_p, 'utils')):
    _p = os.path.dirname(_p)
BENCHMARK_ROOT = _p
sys.path.insert(0, BENCHMARK_ROOT)
from utils.compat import (
    compute_all_metrics,
    load_landscape_data,
    MetricsResult,
    hamming_distance,
    high_fitness_proximity,
    novelty,
    batch_diversity,
    normalized_fitness_topk,
    max_fitness,
    simple_regret,
    spearman_correlation,
)

AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_IDX = {a: i for i, a in enumerate(AA)}


def one_hot_encode(sequences: List[str], seq_len: int) -> np.ndarray:
    n = len(sequences)
    out = np.zeros((n, seq_len * 20), dtype=np.float32)
    for i, s in enumerate(sequences):
        for j, c in enumerate(s[:seq_len]):
            idx = AA_IDX.get(c)
            if idx is not None:
                out[i, j * 20 + idx] = 1.0
    return out


def train_ensemble(X_train, y_train, seed):
    """Same ensemble as ftMLDE adapter (XGBoost + sklearn)."""
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.kernel_ridge import KernelRidge
    models = [
        xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                         random_state=seed, n_jobs=4, verbosity=0),
        RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed,
                              n_jobs=4),
        GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed),
        KernelRidge(alpha=1.0),
    ]
    fitted = []
    for m in models:
        try:
            m.fit(X_train, y_train)
            fitted.append(m)
        except Exception:
            pass
    return fitted


def ensemble_predict(models, X):
    preds = np.stack([m.predict(X) for m in models], axis=1)
    return preds.mean(axis=1)


def run_single_experiment(
    dataset: str,
    seed: int,
    batch_size: int = 96,
    n_rounds: int = 5,
    n_clusters: int = 10,
    output_path: str = None,
    data_dir: str = None,
    compute_metrics: bool = True,
) -> Dict[str, Any]:
    from sklearn.cluster import MiniBatchKMeans

    if data_dir is None:
        data_dir = os.path.join(BENCHMARK_ROOT, 'data')
    if output_path is None:
        output_path = os.path.join(BENCHMARK_ROOT, "CLADE", "results", f"{dataset}_CLADE")

    np.random.seed(seed)
    print(f"\n{'='*60}\nCLADE 2.0 (light) on {dataset}")
    print(f"  Seed: {seed}, Batch: {batch_size}, Rounds: {n_rounds}, K={n_clusters}")
    print(f"{'='*60}\n")

    all_sequences, all_fitness_raw = load_landscape_data(dataset, data_dir=data_dir)
    n_total = len(all_sequences)
    global_max_raw = float(np.max(all_fitness_raw))
    all_fitness = all_fitness_raw / global_max_raw
    seq_len = len(all_sequences[0])
    print(f"Loaded {n_total} variants, seq_len={seq_len}")

    print("One-hot encoding landscape...")
    t0 = datetime.now()
    X_all = one_hot_encode(all_sequences, seq_len)
    print(f"  {X_all.shape}, {(datetime.now() - t0).total_seconds():.1f}s")

    print(f"Clustering into {n_clusters} clusters (MiniBatchKMeans)...")
    t0 = datetime.now()
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                         batch_size=4096, n_init=3)
    cluster_labels = km.fit_predict(X_all)
    print(f"  done in {(datetime.now() - t0).total_seconds():.1f}s")

    start_time = datetime.now()

    # Round 0: sample uniformly across clusters
    queried = set()
    init_idx = []
    per_cluster = batch_size // n_clusters
    for k in range(n_clusters):
        members = np.where(cluster_labels == k)[0]
        if len(members) == 0:
            continue
        pick = np.random.choice(members, size=min(per_cluster, len(members)),
                                replace=False)
        init_idx.extend(pick.tolist())
    # Fill to batch_size if rounding lost some
    if len(init_idx) < batch_size:
        remaining = [i for i in range(n_total) if i not in set(init_idx)]
        more = np.random.choice(remaining, size=batch_size - len(init_idx),
                                replace=False)
        init_idx.extend(more.tolist())
    init_idx = init_idx[:batch_size]
    queried.update(init_idx)
    all_indices = list(init_idx)
    initial_indices = list(init_idx)
    print(f"Round 0: cluster-uniform init {len(init_idx)} variants, "
          f"max_fit={float(np.max(all_fitness[init_idx])):.4f}")

    # Iterative rounds: train MLDE on collected, then pick top-per-cluster
    for rnd in range(1, n_rounds):
        y = all_fitness[all_indices].astype(np.float32)
        X = X_all[all_indices]
        print(f"\nRound {rnd}: training ensemble on {len(all_indices)} samples")
        t0 = datetime.now()
        models = train_ensemble(X, y, seed=seed + rnd)
        print(f"  trained {len(models)} models in "
              f"{(datetime.now() - t0).total_seconds():.1f}s")

        pool = np.array([i for i in range(n_total) if i not in queried])
        preds = ensemble_predict(models, X_all[pool])

        # CLADE-style: pick top-per-cluster from the predicted-fitness ranking
        new_picks: List[int] = []
        # Rank clusters by best-predicted fitness; pick more from top clusters
        pool_clusters = cluster_labels[pool]
        per_round = batch_size // n_clusters
        for k in range(n_clusters):
            mask = (pool_clusters == k)
            if not np.any(mask):
                continue
            in_k = pool[mask]
            preds_k = preds[mask]
            top_k_idx = in_k[np.argsort(-preds_k)[:per_round]]
            new_picks.extend(top_k_idx.tolist())
        # Fill any shortage with the global top
        if len(new_picks) < batch_size:
            seen = set(new_picks)
            remaining_order = pool[np.argsort(-preds)]
            for idx in remaining_order:
                if idx not in seen and idx not in queried:
                    new_picks.append(int(idx))
                    if len(new_picks) >= batch_size:
                        break
        new_picks = new_picks[:batch_size]
        queried.update(new_picks)
        all_indices.extend(new_picks)
        print(f"  added {len(new_picks)} variants from {n_clusters} clusters, "
              f"max_fit_so_far={float(np.max(all_fitness[all_indices])):.4f}")

    runtime = (datetime.now() - start_time).total_seconds()
    print(f"\nTotal runtime: {runtime:.1f}s")

    subdir = os.path.join(output_path, dataset, "clade", "")
    os.makedirs(subdir, exist_ok=True)
    import torch
    queried_indices = torch.tensor(all_indices, dtype=torch.long)
    init_t = torch.tensor(initial_indices, dtype=torch.long)
    torch.save(queried_indices, os.path.join(subdir, f"CLADE_seed{seed}_indices.pt"))

    result = {
        'seed': seed,
        'runtime_seconds': runtime,
        'n_queries': len(all_indices),
        'config': {
            'method': 'CLADE2',
            'batch_size': batch_size,
            'n_rounds': n_rounds,
            'n_clusters': n_clusters,
        },
    }

    if compute_metrics:
        print("\nComputing evaluation metrics...")
        wildtype = all_sequences[0]
        metrics_result = compute_all_metrics(
            queried_indices=queried_indices,
            all_sequences=all_sequences,
            all_fitness=all_fitness,
            initial_indices=init_t,
            y_pred=None, y_std=None,
            batch_size=batch_size, wildtype=wildtype,
        )
        m_dict = metrics_result.to_dict()
        m_dict['max_fitness_raw'] = float(np.max(all_fitness_raw[all_indices]))
        m_dict['global_max_raw'] = global_max_raw
        result['metrics'] = m_dict
        result['fitness_trajectory'] = metrics_result.fitness_trajectory
        result['regret_trajectory'] = metrics_result.regret_trajectory
        print(f"  Max Fitness (norm): {metrics_result.max_fitness:.4f}")
        print(f"  Simple Regret:      {metrics_result.simple_regret:.4f}")
        print(f"  Norm-Top128:        {metrics_result.normalized_fitness_median_top128:.4f}")

        out = os.path.join(subdir, f"metrics_seed{seed}.json")
        with open(out, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Saved to: {out}")

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--n_rounds", type=int, default=5)
    p.add_argument("--n_clusters", type=int, default=10)
    p.add_argument("--output_path", type=str, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--skip_metrics", action="store_true")
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    seeds = args.seeds if args.seeds else [args.seed]
    for s in seeds:
        run_single_experiment(
            dataset=args.dataset, seed=s,
            batch_size=args.batch_size, n_rounds=args.n_rounds,
            n_clusters=args.n_clusters,
            output_path=args.output_path,
            data_dir=args.data_dir,
            compute_metrics=not args.skip_metrics,
        )


if __name__ == "__main__":
    main()
