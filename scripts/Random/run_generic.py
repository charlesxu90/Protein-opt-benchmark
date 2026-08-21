#!/usr/bin/env python
"""
run_generic.py - Random sampling baseline for protein optimization benchmark

Uniformly samples sequences from the fitness landscape without any optimization.
Serves as a lower baseline for comparison with optimization methods.

Configuration:
    - Strategy: Uniform random sampling (no model, no optimization)
    - Batch size: 96
    - Rounds: 5 (480 total queries)

Usage:
    python run_generic.py --dataset GB1 --seed 42
    python run_generic.py --dataset AAV_med --seeds 42 123 456
    python run_generic.py --dataset GFP_hard --seed_file ../rand_seeds.txt --num_seeds 10
"""

from __future__ import annotations
import argparse
import json
import numpy as np
import os
import sys
import warnings
from typing import List, Optional, Dict, Any
from datetime import datetime

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
    aggregate_run_metrics,
    load_landscape_data,
    MetricsResult,
    global_max_hit_count,
)


def run_single_experiment(
    dataset: str,
    seed: int,
    batch_size: int = 96,
    n_rounds: int = 5,
    output_path: str = None,
    run_id: Optional[int] = None,
    compute_metrics: bool = True,
    data_dir: str = None,
) -> Dict[str, Any]:
    if data_dir is None:
        data_dir = os.path.join(BENCHMARK_ROOT, 'data')
    if output_path is None:
        output_path = os.path.join(BENCHMARK_ROOT, "Random", "results", f"{dataset}_Random")

    if run_id is None:
        run_id = seed

    np.random.seed(seed)

    # Load landscape
    all_sequences, all_fitness_raw = load_landscape_data(dataset, data_dir=data_dir)
    # Normalize to [0,1] by dividing by the global max, matching the
    # convention used by AiCE/ALDE/FLEXS/EvoPlay/alphavariant so that
    # `max_fitness` and `simple_regret` are reported on the same scale
    # across all methods. The aggregator (`scripts/aggregate_metrics.py`)
    # also has a fallback rescale, but normalizing here is the right place.
    _global_max_raw = float(np.max(all_fitness_raw))
    all_fitness = all_fitness_raw / _global_max_raw
    n_total = len(all_sequences)
    total_budget = batch_size * n_rounds

    print(f"\n{'='*60}")
    print(f"Random Sampling Baseline on {dataset}")
    print(f"  Seed: {seed}, Batch: {batch_size}, Rounds: {n_rounds}, Total: {total_budget}")
    print(f"  Landscape size: {n_total}")
    print(f"{'='*60}\n")

    start_time = datetime.now()

    # Uniform random sampling without replacement
    sampled_indices = np.random.choice(n_total, size=min(total_budget, n_total), replace=False)
    # Wrap as tensor-like for compat
    import torch
    queried_indices = torch.tensor(sampled_indices, dtype=torch.long)
    initial_indices = queried_indices[:batch_size]

    runtime = (datetime.now() - start_time).total_seconds()

    # Save indices
    subdir = os.path.join(output_path, dataset, "random", "")
    os.makedirs(subdir, exist_ok=True)
    indices_path = os.path.join(subdir, f"Random_seed{seed}_indices.pt")
    torch.save(queried_indices, indices_path)

    result = {
        'seed': seed,
        'run_id': run_id,
        'result_path': indices_path,
        'runtime_seconds': runtime,
        'n_queries': len(queried_indices),
        'config': {
            'method': 'Random',
            'batch_size': batch_size,
            'n_rounds': n_rounds,
        },
    }

    if compute_metrics:
        print("Computing evaluation metrics...")
        # Determine wildtype (first sequence as fallback)
        wildtype = all_sequences[0]

        metrics_result = compute_all_metrics(
            queried_indices=queried_indices,
            all_sequences=all_sequences,
            all_fitness=all_fitness,
            initial_indices=initial_indices,
            y_pred=None,
            y_std=None,
            batch_size=batch_size,
            wildtype=wildtype,
        )

        m_dict = metrics_result.to_dict()
        # Preserve raw value for transparency
        m_dict['max_fitness_raw'] = float(np.max(all_fitness_raw[sampled_indices]))
        m_dict['global_max_raw'] = _global_max_raw
        result['metrics'] = m_dict
        result['fitness_trajectory'] = metrics_result.fitness_trajectory
        result['regret_trajectory'] = metrics_result.regret_trajectory

        print(f"  Max Fitness (norm): {metrics_result.max_fitness:.4f}")
        print(f"  Max Fitness (raw):  {m_dict['max_fitness_raw']:.4f}")
        print(f"  Simple Regret:      {metrics_result.simple_regret:.4f}")
        print(f"  Normalized Fitness (Top-128): {metrics_result.normalized_fitness_median_top128:.4f}")

        metrics_path = os.path.join(subdir, f"metrics_seed{seed}.json")
        with open(metrics_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Saved to: {metrics_path}")

    return result


def load_seeds_from_file(filepath: str, num_seeds: int) -> List[int]:
    seeds = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                seeds.append(int(line))
                if len(seeds) >= num_seeds:
                    break
    return seeds


def save_aggregated_results(results, dataset, output_path):
    import pandas as pd
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(MetricsResult)}
    rename = {'hit_rate': 'hit_rate_value'}
    metrics_results = []
    for r in results:
        if 'metrics' in r:
            kwargs = {rename.get(k, k): v for k, v in r['metrics'].items()}
            kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
            mr = MetricsResult(**kwargs)
            mr.fitness_trajectory = r.get('fitness_trajectory', [])
            mr.regret_trajectory = r.get('regret_trajectory', [])
            metrics_results.append(mr)
    if not metrics_results:
        return

    aggregated = aggregate_run_metrics(metrics_results)
    max_fitness_values = [r['metrics']['max_fitness'] for r in results if 'metrics' in r]
    global_max = max(max_fitness_values) if max_fitness_values else 1.0
    hit_count, hr = global_max_hit_count(max_fitness_values, global_max, tolerance=0.01)
    aggregated['global_max_hit_count'] = {'count': hit_count, 'rate': hr}

    summary_data = []
    for metric, stats in aggregated.items():
        if isinstance(stats, dict):
            if 'mean' in stats:
                summary_data.append({'metric': metric, 'mean': stats['mean'], 'std': stats['std']})
            elif 'count' in stats:
                summary_data.append({'metric': metric, 'mean': stats['rate'], 'std': 0})

    subdir = os.path.join(output_path, dataset, 'random', '')
    os.makedirs(subdir, exist_ok=True)
    pd.DataFrame(summary_data).to_csv(os.path.join(subdir, 'aggregated_metrics.csv'), index=False)
    with open(os.path.join(subdir, 'aggregated_results.json'), 'w') as f:
        json.dump({'aggregated_metrics': aggregated, 'n_runs': len(results),
                   'seeds': [r['seed'] for r in results]}, f, indent=2, default=str)

    print(f"\n{'Metric':<40} {'Mean':>10} {'Std':>10}")
    print("-" * 60)
    for row in summary_data:
        print(f"{row['metric']:<40} {row['mean']:>10.4f} {row['std']:>10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Random sampling baseline for protein optimization benchmark")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (GB1, AAV_med, AAV_hard, GFP_med, GFP_hard)")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=None)
    seed_group.add_argument("--seeds", type=int, nargs='+')
    seed_group.add_argument("--seed_file", type=str)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--n_rounds", type=int, default=5)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--data_dir", type=str, default=None)

    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    if args.output_path is None:
        args.output_path = os.path.join(BENCHMARK_ROOT, "Random", "results", f"{args.dataset}_Random")
    if args.data_dir is None:
        args.data_dir = os.path.join(BENCHMARK_ROOT, 'data')

    if args.seeds is not None:
        seeds = args.seeds
    elif args.seed_file is not None:
        seeds = load_seeds_from_file(args.seed_file, args.num_seeds)
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42]

    print(f"Random baseline on {args.dataset} with {len(seeds)} seed(s): {seeds}")

    results = []
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] seed={seed}")
        result = run_single_experiment(
            dataset=args.dataset, seed=seed, batch_size=args.batch_size,
            n_rounds=args.n_rounds, output_path=args.output_path,
            run_id=i + 1, compute_metrics=not args.skip_metrics, data_dir=args.data_dir)
        results.append(result)

    if len(results) > 1 and not args.skip_metrics:
        save_aggregated_results(results, args.dataset, args.output_path)

    print(f"\nDone. {len(results)} run(s) on {args.dataset}.")


if __name__ == "__main__":
    main()
