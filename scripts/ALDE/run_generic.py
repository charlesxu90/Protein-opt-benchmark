#!/usr/bin/env python
"""
run_generic.py - Execute ALDE optimization on any dataset with comprehensive metrics

Generic dataset runner that auto-detects sequence length and adjusts architecture
accordingly. Unlike dataset-specific scripts, this script determines the dataset name,
wildtype, sequence length, and architecture dynamically from the loaded data.

Configuration:
    - Model: DNN Ensemble
    - Encoding: One-hot
    - Acquisition: Thompson Sampling (TS)
    - Batch size: 96
    - Rounds: 5 (1 initial + 4 iterations)

Architecture auto-selection:
    - seq_len <= 50: hidden layers [30, 30] (suitable for GB1, AAV)
    - seq_len > 50:  hidden layers [128, 128] (suitable for GFP and longer sequences)

Usage:
    # Single run on a dataset
    ALDE/env/bin/python scripts/ALDE/run_generic.py --dataset 4site_GB1 --seed 42

    # Multiple runs with different seeds
    ALDE/env/bin/python scripts/ALDE/run_generic.py --dataset 4site_GB1 --seeds 42 123 456 789 1000

    # Use predefined seeds from file
    ALDE/env/bin/python scripts/ALDE/run_generic.py --dataset 4site_TRPB --seed_file rand_seeds.txt --num_seeds 30

    # Skip metrics computation (faster)
    ALDE/env/bin/python scripts/ALDE/run_generic.py --dataset 4site_PhoQ --seed 42 --skip_metrics

    # Use a different data directory
    ALDE/env/bin/python scripts/ALDE/run_generic.py --dataset 4site_GB1 --seed 42 --data_dir data
"""

from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
import torch
import random
import os
import warnings
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

import sys
# Canonical location is scripts/<Method>/. This file used to be invoked through a
# symlink in <Method>/, where `__file__/..` happened to be the benchmark root; walk
# up to find the root instead, so it runs correctly from either path.
# (Same idiom as scripts/EVOLVEpro/run_generic.py.)
_p = os.path.dirname(os.path.realpath(__file__))
while os.path.dirname(_p) != _p and not os.path.isdir(os.path.join(_p, 'utils')):
    _p = os.path.dirname(_p)
BENCHMARK_ROOT = _p
sys.path.insert(0, BENCHMARK_ROOT)
sys.path.insert(0, os.path.join(BENCHMARK_ROOT, 'ALDE'))  # upstream package lives in the method dir

from src.optimize import BayesianOptimization, BO_ARGS
import src.utils as utils
import sys
from utils.compat import (
    compute_all_metrics,
    aggregate_run_metrics,
    load_landscape_data,
    MetricsResult,
    global_max_hit_count
)

# Standard amino acid alphabet (matches ALDE's ALL_AAS)
ALL_AAS = ("A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
           "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y")


# =============================================================================
# Dataset Objective Wrapper
# =============================================================================

class DatasetObjective:
    """
    Wrapper to make arbitrary datasets compatible with ALDE's BayesianOptimization.

    Mimics the interface of ALDE's src.objectives.Combo class:
        - objective(x) -> (qx, qy)
        - get_domain() -> (lower, upper)
        - get_points() -> (X, y)
    """

    def __init__(self, data_path: str, sequence_col: str = 'seq',
                 fitness_col: str = 'fitness'):
        """
        Load dataset and create one-hot encoded features.

        Args:
            data_path: Path to CSV file with sequence and fitness columns.
            sequence_col: Column name for sequences.
            fitness_col: Column name for fitness values.
        """
        df = pd.read_csv(data_path)
        sequences = df[sequence_col].tolist()
        if fitness_col not in df.columns and 'blue' in df.columns and 'red' in df.columns:
            # Multi-objective ("*_joint") dataset: scalarize to geometric mean
            blue = df['blue'].values.astype(float)
            red = df['red'].values.astype(float)
            fitness_raw = np.sqrt(np.clip(blue, 0, None) * np.clip(red, 0, None))
        else:
            fitness_raw = df[fitness_col].values

        self.sequences = sequences
        self.seq_length = len(sequences[0])
        self.n_aas = len(ALL_AAS)
        self.feat_dim = self.seq_length * self.n_aas

        # One-hot encode all sequences
        aa_to_idx = {aa: i for i, aa in enumerate(ALL_AAS)}
        n_seqs = len(sequences)
        X = torch.zeros(n_seqs, self.seq_length, self.n_aas)
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                if aa in aa_to_idx:
                    X[i, j, aa_to_idx[aa]] = 1.0
        # Flatten to 2D: (n_seqs, seq_length * n_aas)
        self.X = X.reshape(n_seqs, -1)

        # Normalize fitness (same as Combo)
        self.y = torch.tensor(fitness_raw, dtype=torch.float64)
        self.y = self.y / self.y.max()

        # Determine wildtype as the sequence with the highest fitness
        max_idx = int(np.argmax(fitness_raw))
        self.wildtype = sequences[max_idx]

    def objective(self, x, noise=0.0):
        """Query the objective function at point x."""
        qx, qy = utils.query_discrete(self.X, self.y, x)
        return qx.double(), qy.double()

    def get_domain(self):
        """Returns (lower, upper) bounds of the feature space."""
        lower, upper = utils.domain_discrete(self.X)
        return lower.double(), upper.double()

    def get_points(self):
        """Returns (X, y) for all data points."""
        return self.X.double(), self.y.double()

    def get_max(self):
        """Returns the maximum fitness value."""
        return torch.max(self.y).double()


# =============================================================================
# Utility Functions
# =============================================================================

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_hidden_layers(seq_len: int) -> List[int]:
    """
    Auto-select hidden layer sizes based on sequence length.

    Args:
        seq_len: Length of the protein sequences in the dataset.

    Returns:
        List of hidden layer sizes.
    """
    if seq_len <= 50:
        return [30, 30]
    else:
        return [128, 128]


def run_single_experiment(
    dataset_name: str,
    seed: int,
    device: str = "cuda",
    output_path: str = None,
    verbose: int = 2,
    train_iter: int = 300,
    run_id: Optional[int] = None,
    compute_metrics: bool = True,
    data_dir: str = None
) -> Dict[str, Any]:
    """
    Run a single ALDE optimization experiment on the specified dataset.

    Args:
        dataset_name: Name of the dataset (must have data/<name>/data.csv)
        seed: Random seed for reproducibility
        device: Device to use ('cuda' or 'cpu')
        output_path: Base path for saving results
        verbose: Verbosity level (0-3)
        train_iter: Number of training iterations
        run_id: Optional run identifier (defaults to seed)
        compute_metrics: Whether to compute evaluation metrics
        data_dir: Base directory for data files

    Returns:
        Dictionary containing results and metrics
    """
    if data_dir is None:
        data_dir = os.path.join(BENCHMARK_ROOT, 'data')
        data_dir = os.path.abspath(data_dir)

    if output_path is None:
        output_path = os.path.join(BENCHMARK_ROOT, "ALDE", "results", f"{dataset_name}_ALDE")

    # Fixed configuration
    protein = dataset_name
    encoding = "onehot"
    mtype = "DNN_ENSEMBLE"
    acq_fn = "TS"
    batch_size = 96
    n_pseudorand_init = 96
    budget = 384  # 4 iteration rounds (4 * 96)
    dropout = 0
    activation = "lrelu"
    kernel = "RBF"

    if run_id is None:
        run_id = seed

    # Load objective function via custom wrapper
    data_path = os.path.join(data_dir, protein, 'data.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            f"Expected a CSV file at data/{protein}/data.csv with 'seq' and 'fitness' columns."
        )
    obj = DatasetObjective(data_path, sequence_col='seq', fitness_col='fitness')

    # Auto-detect sequence length and architecture
    seq_length = obj.seq_length
    hidden_layers = get_hidden_layers(seq_length)
    wildtype = obj.wildtype

    print(f"\n{'='*60}")
    print(f"Starting ALDE optimization on {protein}")
    print(f"  Seed: {seed}")
    print(f"  Model: {mtype}")
    print(f"  Encoding: {encoding}")
    print(f"  Acquisition: {acq_fn}")
    print(f"  Batch size: {batch_size}")
    print(f"  Initial samples: {n_pseudorand_init}")
    print(f"  Budget: {budget}")
    print(f"  Total samples: {n_pseudorand_init + budget}")
    print(f"  Rounds: 5 (1 init + 4 iterations)")
    print(f"  Sequence length: {seq_length}")
    print(f"  Feature dimension: {seq_length * len(ALL_AAS)}")
    print(f"  Hidden layers: {hidden_layers}")
    print(f"  Wildtype: {wildtype}")
    print(f"{'='*60}\n")

    # Set random seeds
    set_seed(seed)

    obj_fn = obj.objective
    domain = obj.get_domain()
    disc_X = obj.get_points()[0]
    disc_y = obj.get_points()[1]

    # Load complete landscape for metrics
    all_sequences, all_fitness_raw = load_landscape_data(protein, data_dir=data_dir)
    # Normalize fitness to match ALDE's normalization
    all_fitness = all_fitness_raw / np.max(all_fitness_raw)

    # Create output directory
    subdir = os.path.join(output_path, protein, encoding, "")
    os.makedirs(subdir, exist_ok=True)

    # Save a copy of this script for reproducibility
    script_copy_path = os.path.join(subdir, f"run_{protein}_seed{seed}.py")
    if not os.path.exists(script_copy_path):
        try:
            os.system(f'cp {__file__} {script_copy_path}')
        except:
            pass

    # Random initialization
    start_x, start_y, start_indices = utils.samp_discrete(n_pseudorand_init, obj, seed)

    # Random search baseline
    if budget != 0:
        _, _, rand_indices = utils.samp_discrete(budget, obj, seed)
        rand_indices = torch.cat((start_indices, rand_indices), 0)
    else:
        rand_indices = start_indices

    # Save random baseline
    random_baseline_path = os.path.join(subdir, f'Random_{seed}indices.pt')
    torch.save(rand_indices, random_baseline_path)
    print(f"Random baseline saved to: {random_baseline_path}")

    # Set architecture for DNN Ensemble with onehot encoding
    feat_dim = domain[0].size(-1)  # seq_length * 20
    arc = [feat_dim] + hidden_layers + [1]

    # Construct filename
    fname = f"{mtype}-{acq_fn}-{dropout}-{kernel}-{arc[-2:]}-seed{seed}"

    # Configure optimization arguments
    sim_args = BO_ARGS(
        mtype=mtype,
        kernel=kernel,
        acq_fn=acq_fn,
        xi=4,
        architecture=arc,
        activation=activation,
        min_noise=1e-6,
        trainlr=1e-3,
        train_iter=train_iter,
        dropout=dropout,
        mcdropout=0,
        verbose=verbose,
        bb_fn=obj_fn,
        domain=domain,
        disc_X=disc_X,
        disc_y=disc_y,
        noise_std=0,
        n_rand_init=0,
        budget=budget,
        query_cost=1,
        queries_x=start_x,
        queries_y=start_y,
        indices=start_indices,
        savedir=os.path.join(subdir, fname),
        batch_size=batch_size,
        seed_index=seed
    )

    # Run optimization
    start_time = datetime.now()
    BayesianOptimization.run(sim_args, seed)
    runtime = (datetime.now() - start_time).total_seconds()

    # Load results
    result_path = os.path.join(subdir, fname + "indices.pt")
    queried_indices = torch.load(result_path)
    print(f"\nResults saved to: {result_path}")

    # Prepare result dictionary
    result = {
        'seed': seed,
        'run_id': run_id,
        'dataset': dataset_name,
        'result_path': result_path,
        'runtime_seconds': runtime,
        'n_queries': len(queried_indices),
        'config': {
            'model': mtype,
            'encoding': encoding,
            'acquisition': acq_fn,
            'batch_size': batch_size,
            'n_init': n_pseudorand_init,
            'budget': budget,
            'train_iter': train_iter,
            'seq_length': seq_length,
            'hidden_layers': hidden_layers,
        }
    }

    # Compute metrics
    if compute_metrics:
        print("\nComputing evaluation metrics...")

        # Load mu and sigma if available (for uncertainty metrics)
        y_pred = None
        y_std = None

        mu_path = os.path.join(subdir, fname + f"_{budget}mu.pt")
        sigma_path = os.path.join(subdir, fname + f"_{budget}sigma.pt")

        if os.path.exists(mu_path) and os.path.exists(sigma_path):
            try:
                y_pred = torch.load(mu_path).cpu().numpy()
                y_std = torch.load(sigma_path).cpu().numpy()
                y_pred = y_pred / np.max(all_fitness_raw)
                y_std = y_std / np.max(all_fitness_raw)
                print(f"  Loaded predictions from {mu_path}")
            except Exception as e:
                print(f"  Warning: Could not load predictions: {e}")

        metrics_result = compute_all_metrics(
            queried_indices=queried_indices,
            all_sequences=all_sequences,
            all_fitness=all_fitness,
            initial_indices=start_indices,
            y_pred=y_pred,
            y_std=y_std,
            batch_size=batch_size,
            wildtype=wildtype
        )

        result['metrics'] = metrics_result.to_dict()
        result['fitness_trajectory'] = metrics_result.fitness_trajectory
        result['regret_trajectory'] = metrics_result.regret_trajectory

        print("\n" + "-"*40)
        print("Metrics Summary:")
        print("-"*40)
        print(f"  Max Fitness: {metrics_result.max_fitness:.4f}")
        print(f"  Simple Regret: {metrics_result.simple_regret:.4f}")
        print(f"  Global Max Found: {metrics_result.global_max_found}")
        print(f"  Normalized Fitness (Top-128): {metrics_result.normalized_fitness_median_top128:.4f}")
        print(f"  High-Fitness Proximity: {metrics_result.high_fitness_proximity:.4f}")
        print(f"  Novelty: {metrics_result.novelty:.4f}")
        print(f"  Batch Diversity: {metrics_result.batch_diversity:.4f}")

        if y_pred is not None:
            print(f"  Spearman Correlation: {metrics_result.spearman_correlation:.4f}")
            if y_std is not None:
                print(f"  Miscalibration Area: {metrics_result.miscalibration_area:.4f}")

        metrics_json_path = os.path.join(subdir, f"metrics_seed{seed}.json")
        with open(metrics_json_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nMetrics saved to: {metrics_json_path}")

    return result


def load_seeds_from_file(filepath: str, num_seeds: int) -> List[int]:
    """Load seeds from a file (one seed per line)."""
    seeds = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                seeds.append(int(line))
                if len(seeds) >= num_seeds:
                    break
    return seeds


def save_aggregated_results(
    results: List[Dict[str, Any]],
    output_path: str,
    dataset_name: str
) -> None:
    """Save aggregated results across all runs."""
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
        print("No metrics to aggregate.")
        return

    aggregated = aggregate_run_metrics(metrics_results)

    max_fitness_values = [r['metrics']['max_fitness'] for r in results if 'metrics' in r]
    global_max = 1.0
    hit_count, hit_rate = global_max_hit_count(max_fitness_values, global_max, tolerance=0.01)
    aggregated['global_max_hit_count'] = {'count': hit_count, 'rate': hit_rate}

    summary_data = []
    for metric, stats in aggregated.items():
        if isinstance(stats, dict):
            if 'mean' in stats:
                summary_data.append({
                    'metric': metric,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max']
                })
            elif 'count' in stats:
                summary_data.append({
                    'metric': metric,
                    'mean': stats['rate'],
                    'std': 0,
                    'min': stats['count'],
                    'max': len(results)
                })

    summary_df = pd.DataFrame(summary_data)

    summary_path = os.path.join(output_path, dataset_name, 'onehot', 'aggregated_metrics.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nAggregated metrics saved to: {summary_path}")

    aggregated_json_path = os.path.join(output_path, dataset_name, 'onehot', 'aggregated_results.json')
    with open(aggregated_json_path, 'w') as f:
        json.dump({
            'aggregated_metrics': aggregated,
            'n_runs': len(results),
            'seeds': [r['seed'] for r in results],
            'config': results[0].get('config', {}) if results else {}
        }, f, indent=2, default=str)
    print(f"Aggregated results saved to: {aggregated_json_path}")

    print("\n" + "="*70)
    print("AGGREGATED METRICS SUMMARY")
    print("="*70)
    print(f"{'Metric':<40} {'Mean':>10} {'Std':>10}")
    print("-"*70)
    for _, row in summary_df.iterrows():
        if row['metric'] == 'global_max_hit_count':
            print(f"{row['metric']:<40} {row['mean']*100:>9.1f}% ({int(row['min'])}/{int(row['max'])})")
        else:
            print(f"{row['metric']:<40} {row['mean']:>10.4f} {row['std']:>10.4f}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Run ALDE optimization on any dataset with DNN Ensemble + One-hot + Thompson Sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run on 4site_GB1
  ALDE/env/bin/python scripts/ALDE/run_generic.py --dataset 4site_GB1 --seed 42

  # Single run on GB1
  ALDE/env/bin/python scripts/ALDE/run_generic.py --dataset 4site_GB1 --seed 42

  # Multiple runs for randomness evaluation
  ALDE/env/bin/python scripts/ALDE/run_generic.py --dataset 4site_TRPB --seeds 42 123 456 789 1000

  # Load seeds from file
  ALDE/env/bin/python scripts/ALDE/run_generic.py --dataset 4site_PhoQ --seed_file rand_seeds.txt --num_seeds 30

  # Skip metrics computation
  ALDE/env/bin/python scripts/ALDE/run_generic.py --dataset 4site_GB1 --seed 42 --skip_metrics
        """
    )

    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Name of the dataset (must have data/<dataset>/data.csv with 'seq' and 'fitness' columns)"
    )

    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed", type=int, default=None,
        help="Single random seed for reproducibility"
    )
    seed_group.add_argument(
        "--seeds", type=int, nargs='+',
        help="Multiple seeds for evaluating randomness (space-separated)"
    )
    seed_group.add_argument(
        "--seed_file", type=str,
        help="Path to file containing seeds (one per line)"
    )

    parser.add_argument(
        "--num_seeds", type=int, default=5,
        help="Number of seeds to use from seed file (default: 5)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="Device to use for computation (default: cuda)"
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Output directory for results (default: results/<dataset>_ALDE/)"
    )
    parser.add_argument(
        "--verbose", type=int, default=2, choices=[0, 1, 2, 3],
        help="Verbosity level 0-3 (default: 2)"
    )
    parser.add_argument(
        "--train_iter", type=int, default=300,
        help="Number of training iterations (default: 300)"
    )
    parser.add_argument(
        "--skip_metrics", action="store_true",
        help="Skip metrics computation (faster)"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Base directory for data files (default: auto-detected from script location)"
    )

    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    dataset_name = args.dataset

    # Resolve output_path
    if args.output_path is None:
        args.output_path = os.path.join(BENCHMARK_ROOT, "ALDE", "results", f"{dataset_name}_ALDE")

    # Resolve data_dir
    if args.data_dir is None:
        args.data_dir = os.path.join(BENCHMARK_ROOT, 'data')

    # Determine which seeds to use
    if args.seeds is not None:
        seeds = args.seeds
    elif args.seed_file is not None:
        seeds = load_seeds_from_file(args.seed_file, args.num_seeds)
        print(f"Loaded {len(seeds)} seeds from {args.seed_file}")
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [64]

    print(f"\nRunning ALDE on {dataset_name} with {len(seeds)} seed(s): {seeds}")
    print(f"Device: {args.device}")
    print(f"Output path: {args.output_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Compute metrics: {not args.skip_metrics}")

    results = []
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Running experiment with seed={seed}")
        result = run_single_experiment(
            dataset_name=dataset_name,
            seed=seed,
            device=args.device,
            output_path=args.output_path,
            verbose=args.verbose,
            train_iter=args.train_iter,
            run_id=i + 1,
            compute_metrics=not args.skip_metrics,
            data_dir=args.data_dir
        )
        results.append(result)

    if len(results) > 1 and not args.skip_metrics:
        save_aggregated_results(results, args.output_path, dataset_name)

    # Read back seq_length and hidden_layers from the first result's config
    config = results[0].get('config', {})
    seq_length = config.get('seq_length', '?')
    hidden_layers = config.get('hidden_layers', '?')

    print(f"\n{'='*60}")
    print("Experiment Complete")
    print(f"{'='*60}")
    print(f"Total runs: {len(results)}")
    print(f"Dataset: {dataset_name}")
    print(f"Configuration:")
    print(f"  - Model: DNN_ENSEMBLE")
    print(f"  - Encoding: onehot")
    print(f"  - Acquisition: TS (Thompson Sampling)")
    print(f"  - Batch size: 96")
    print(f"  - Rounds: 5 (1 init + 4 iterations)")
    print(f"  - Total samples per run: 480 (96 init + 384 budget)")
    print(f"  - Sequence length: {seq_length}")
    print(f"  - Hidden layers: {hidden_layers}")
    print(f"  - Feature dimension: {seq_length * len(ALL_AAS) if isinstance(seq_length, int) else '?'}")
    print(f"\nResults saved to: {args.output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
