#!/usr/bin/env python
"""
run_AAV_hard.py - Execute AdaLead optimization on AAV hard dataset with comprehensive metrics

Configuration:
    - Model: CNN Ensemble
    - Encoding: One-hot
    - Explorer: AdaLead
    - Batch size: 96
    - Rounds: 15 (1 initial + 14 iterations)

Metrics computed (from multiple reference works):
    - Exploration: High-Fitness Proximity, Novelty, Batch Diversity
    - Functional: Normalized Fitness (Top-K), Max Fitness
    - Model Quality: Spearman Correlation
    - Success: Simple Regret, Global Max Hit Count

Usage:
    # Single run with default seed
    python run_AAV_hard.py

    # Single run with specific seed
    python run_AAV_hard.py --seed 42

    # Multiple runs with different seeds for randomness evaluation
    python run_AAV_hard.py --seeds 42 123 456 789 1000

    # Use predefined seeds from file
    python run_AAV_hard.py --seed_file ../rand_seeds.txt --num_seeds 5

    # Skip metrics computation (faster)
    python run_AAV_hard.py --seed 42 --skip_metrics
"""

from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
import random
import os
import warnings
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from scipy import stats

import flexs
from flexs import baselines
from flexs.utils import sequence_utils as s_utils

# Import unified metrics from utils.compat
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.compat import (
    compute_all_metrics,
    aggregate_run_metrics,
    load_landscape_data,
    MetricsResult,
    global_max_hit_count,
    hamming_distance,
    high_fitness_proximity,
    novelty,
    batch_diversity,
    normalized_fitness_topk,
    max_fitness,
    simple_regret,
    spearman_correlation,
)


# ============================================================================
# AAV Wild-type
# ============================================================================

AAV_WILDTYPE = "DEEEIRTTNPVATEQYGSYSTNLQQGNR"
AAV_LENGTH = 28


# ============================================================================
# AAV Landscape
# ============================================================================

class AAVLandscape(flexs.Landscape):
    """
    AAV protein fitness landscape.

    Uses empirical fitness data from AAV combinatorial library.
    """

    def __init__(
        self,
        data_dir: str = "/home/xux/Desktop/AlphaVariant/Benchmark/data",
        level: str = "hard",
        normalize: bool = False
    ):
        """
        Initialize AAV landscape.

        Args:
            data_dir: Base directory for data files
            level: Difficulty level ('med' or 'hard')
            normalize: Whether to normalize fitness values to [0, 1]
        """
        super().__init__(name=f"AAV_{level}")

        # Load data
        data_path = os.path.join(data_dir, f"AAV_{level}", "data.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"AAV data not found at {data_path}")

        self.data = pd.read_csv(data_path)
        self.level = level

        # Build sequence -> fitness lookup using full sequence
        self.sequences = self.data['seq'].tolist()
        self.fitness_raw = self.data['fitness'].values

        if normalize:
            self.fitness = (self.fitness_raw - np.min(self.fitness_raw)) / (np.max(self.fitness_raw) - np.min(self.fitness_raw))
        else:
            # AAV data is already normalized to [0, 1]
            self.fitness = self.fitness_raw

        self.seq_to_idx = {seq: i for i, seq in enumerate(self.sequences)}
        self.seq_to_fitness = {seq: fit for seq, fit in zip(self.sequences, self.fitness)}

        # Wild-type sequence
        self.wildtype = AAV_WILDTYPE

        # Determine alphabet (unique characters at each position)
        self.seq_length = len(self.sequences[0])
        self._compute_alphabet()

    def _compute_alphabet(self):
        """Compute the alphabet from all sequences."""
        # Get all unique amino acids used across all sequences
        all_chars = set()
        for seq in self.sequences:
            all_chars.update(seq)
        self.alphabet = "".join(sorted(all_chars))

    def _fitness_function(self, sequences) -> np.ndarray:
        """Return fitness values for given sequences."""
        fitness_values = []
        for seq in sequences:
            if seq in self.seq_to_fitness:
                fitness_values.append(self.seq_to_fitness[seq])
            else:
                # Unknown sequence - return 0
                fitness_values.append(0.0)
        return np.array(fitness_values)

    def get_all_sequences(self) -> List[str]:
        """Return all sequences in the landscape."""
        return self.sequences

    def get_all_fitness(self) -> np.ndarray:
        """Return all fitness values."""
        return self.fitness


# ============================================================================
# Local Metrics Computation (using unified utils.compat)
# ============================================================================

def compute_metrics(
    results_df: pd.DataFrame,
    landscape: AAVLandscape,
    model: Optional[Any] = None,
    batch_size: int = 96
) -> MetricsResult:
    """Compute all metrics for an AdaLead run (aligned with ALDE)."""
    result = MetricsResult()

    all_sequences = landscape.get_all_sequences()
    all_fitness = landscape.get_all_fitness()
    global_max = np.max(all_fitness)
    global_min = np.min(all_fitness)
    wildtype = landscape.wildtype

    # Get queried sequences and fitness
    queried_seqs = results_df['sequence'].tolist()
    queried_fitness = results_df['true_score'].values

    # Initial sequences (round 0)
    initial_seqs = results_df[results_df['round'] == 0]['sequence'].tolist()

    # Non-initial sequences
    non_initial_seqs = results_df[results_df['round'] > 0]['sequence'].tolist()

    # === Exploration metrics ===
    result.high_fitness_proximity = high_fitness_proximity(
        queried_seqs, all_sequences, all_fitness
    )

    if non_initial_seqs and initial_seqs:
        result.novelty = novelty(non_initial_seqs, initial_seqs)

    result.batch_diversity = batch_diversity(queried_seqs)

    # === Functional metrics ===
    result.normalized_fitness_median_top128 = normalized_fitness_topk(
        queried_fitness, k=128, min_fitness=global_min, max_fitness=global_max
    )
    result.normalized_fitness_median_top256 = normalized_fitness_topk(
        queried_fitness, k=256, min_fitness=global_min, max_fitness=global_max
    )
    result.max_fitness = float(np.max(queried_fitness))

    # === Model quality metrics ===
    # Get model predictions if model is available
    if model is not None:
        try:
            # Get predictions for all queried sequences
            y_pred = model.get_fitness(queried_seqs)
            y_true = queried_fitness

            result.spearman_correlation = spearman_correlation(y_true, y_pred)
        except Exception as e:
            print(f"Warning: Could not compute model-based metrics: {e}")

    # === Success metrics ===
    result.simple_regret = simple_regret(result.max_fitness, global_max)
    result.global_max_found = (result.max_fitness >= global_max * 0.99)

    # Fitness trajectory per round
    rounds = sorted(results_df['round'].unique())
    cumulative_max = 0.0
    for r in rounds:
        round_max = results_df[results_df['round'] <= r]['true_score'].max()
        cumulative_max = max(cumulative_max, round_max)
        result.fitness_trajectory.append(cumulative_max)
        result.regret_trajectory.append(global_max - cumulative_max)

    return result


# ============================================================================
# Main Experiment Functions
# ============================================================================

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def run_single_experiment(
    seed: int,
    output_path: str = "results/AAV_hard_AdaLead/",
    verbose: int = 2,
    run_id: Optional[int] = None,
    compute_metrics_flag: bool = True,
    data_dir: str = "/home/xux/Desktop/AlphaVariant/Benchmark/data",
    rounds: int = 14,  # 14 optimization rounds (matching 15 total)
    sequences_batch_size: int = 96,
    model_queries_per_batch: int = 2000,
    n_init_samples: int = 96  # Random initial samples (matching ALDE)
) -> Dict[str, Any]:
    """
    Run a single AdaLead optimization experiment on AAV_hard dataset.

    Args:
        seed: Random seed for reproducibility
        output_path: Base path for saving results
        verbose: Verbosity level
        run_id: Optional run identifier
        compute_metrics_flag: Whether to compute evaluation metrics
        data_dir: Base directory for data files
        rounds: Number of optimization rounds (after initial sampling)
        sequences_batch_size: Batch size for each round
        model_queries_per_batch: Model queries allowed per batch
        n_init_samples: Number of random initial samples

    Returns:
        Dictionary containing results and metrics
    """
    import time

    if run_id is None:
        run_id = seed

    total_samples = n_init_samples + rounds * sequences_batch_size

    print(f"\n{'='*60}")
    print(f"Starting AdaLead optimization on AAV_hard")
    print(f"  Seed: {seed}")
    print(f"  Model: CNN Ensemble")
    print(f"  Batch size: {sequences_batch_size}")
    print(f"  Initial samples: {n_init_samples}")
    print(f"  Optimization rounds: {rounds}")
    print(f"  Total samples: {total_samples}")
    print(f"  Model queries per batch: {model_queries_per_batch}")
    print(f"{'='*60}\n")

    # Set random seeds
    set_seed(seed)

    # Load landscape
    landscape = AAVLandscape(data_dir=data_dir, level="hard", normalize=False)

    # Get landscape info
    all_sequences = landscape.get_all_sequences()
    alphabet = landscape.alphabet
    seq_length = landscape.seq_length

    print(f"Landscape: {len(all_sequences)} sequences")
    print(f"Wild-type: {landscape.wildtype}")
    print(f"Alphabet: {alphabet} (length {len(alphabet)})")
    print(f"Sequence length: {seq_length}")
    print(f"Fitness range: [{np.min(landscape.fitness):.4f}, {np.max(landscape.fitness):.4f}]")

    # Create output directory
    subdir = os.path.join(output_path, "AAV_hard", "")
    os.makedirs(subdir, exist_ok=True)

    # Create CNN Ensemble model
    ensemble_models = [
        baselines.models.CNN(
            seq_len=seq_length,
            num_filters=32,
            hidden_size=100,
            alphabet=alphabet,
            loss='MSE',
            epochs=20
        )
        for _ in range(5)
    ]
    model = flexs.Ensemble(ensemble_models)

    # =========================================================================
    # Random initialization (matching ALDE)
    # =========================================================================
    print(f"\nRound 0: Random initialization ({n_init_samples} samples)")

    # Sample random indices
    all_indices = np.arange(len(all_sequences))
    init_indices = np.random.choice(all_indices, size=n_init_samples, replace=False)

    # Get initial sequences and their true fitness
    init_sequences = [all_sequences[i] for i in init_indices]
    init_fitness = landscape.get_fitness(init_sequences)

    # Create initial DataFrame
    results_df = pd.DataFrame({
        'sequence': init_sequences,
        'model_score': np.nan,
        'true_score': init_fitness,
        'round': 0,
        'model_cost': 0,
        'measurement_cost': n_init_samples,
    })

    print(f"  Max fitness in initial batch: {np.max(init_fitness):.4f}")

    # =========================================================================
    # AdaLead optimization rounds
    # =========================================================================
    start_time = datetime.now()
    model.cost = 0

    for r in range(1, rounds + 1):
        round_start = time.time()

        # Train model on all data so far
        model.train(
            results_df['sequence'].to_numpy(),
            results_df['true_score'].to_numpy(),
        )

        # Get measured sequence set
        measured_set = set(results_df['sequence'])

        # Get top sequences for parent selection
        top_fitness = results_df['true_score'].max()
        threshold = 0.05  # Top 5%
        top_mask = results_df['true_score'] >= top_fitness * (1 - np.sign(top_fitness) * threshold)
        parents = results_df[top_mask]['sequence'].to_numpy()

        # Resize parents to batch size
        parents = np.resize(parents, sequences_batch_size)

        # Generate candidates using mutation
        sequences = {}
        previous_cost = model.cost
        eval_batch_size = 20
        mu = 1  # Expected mutations per sequence

        while model.cost - previous_cost < model_queries_per_batch:
            for i in range(0, len(parents), eval_batch_size):
                roots = parents[i:i + eval_batch_size]
                root_fitnesses = model.get_fitness(roots)

                nodes = list(enumerate(roots))

                while (len(nodes) > 0 and
                       model.cost - previous_cost + eval_batch_size < model_queries_per_batch):

                    child_idxs = []
                    children = []

                    for idx, node in nodes:
                        # Generate mutant
                        child = s_utils.generate_random_mutant(
                            node,
                            mu / len(node),
                            alphabet
                        )

                        # Only keep novel sequences
                        if child not in measured_set and child not in sequences:
                            child_idxs.append(idx)
                            children.append(child)

                        if len(children) >= len(nodes):
                            break

                    if not children:
                        break

                    # Evaluate children
                    fitnesses = model.get_fitness(children)
                    sequences.update(zip(children, fitnesses))

                    # Keep children that improve over their root
                    nodes = []
                    for idx, child, fitness in zip(child_idxs, children, fitnesses):
                        if fitness >= root_fitnesses[idx]:
                            nodes.append((idx, child))

                if model.cost - previous_cost >= model_queries_per_batch:
                    break

            if model.cost - previous_cost >= model_queries_per_batch:
                break

        if len(sequences) == 0:
            print(f"  Warning: No new sequences generated in round {r}")
            continue

        # Select top sequences
        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[::-1][:sequences_batch_size]

        selected_seqs = new_seqs[sorted_order]
        selected_preds = preds[sorted_order]

        # Get true fitness
        true_scores = landscape.get_fitness(selected_seqs)

        # Add to results
        round_df = pd.DataFrame({
            'sequence': selected_seqs,
            'model_score': selected_preds,
            'true_score': true_scores,
            'round': r,
            'model_cost': model.cost,
            'measurement_cost': len(results_df) + len(selected_seqs),
        })

        results_df = pd.concat([results_df, round_df], ignore_index=True)

        round_time = time.time() - round_start
        print(f"Round {r}: max={np.max(true_scores):.4f}, "
              f"cumulative_max={results_df['true_score'].max():.4f}, "
              f"time={round_time:.1f}s")

    runtime = (datetime.now() - start_time).total_seconds()

    print(f"\nOptimization completed in {runtime:.1f} seconds")
    print(f"Total sequences evaluated: {len(results_df)}")

    # Save results
    results_path = os.path.join(subdir, f"results_seed{seed}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

    # Prepare result dictionary
    result = {
        'seed': seed,
        'run_id': run_id,
        'result_path': results_path,
        'runtime_seconds': runtime,
        'n_queries': len(results_df),
        'config': {
            'model': 'CNN_ENSEMBLE',
            'rounds': rounds,
            'batch_size': sequences_batch_size,
            'model_queries_per_batch': model_queries_per_batch,
        }
    }

    # Compute metrics
    if compute_metrics_flag:
        print("\nComputing evaluation metrics...")

        metrics_result = compute_metrics(
            results_df=results_df,
            landscape=landscape,
            model=model,
            batch_size=sequences_batch_size
        )

        result['metrics'] = metrics_result.to_dict()
        result['fitness_trajectory'] = metrics_result.fitness_trajectory
        result['regret_trajectory'] = metrics_result.regret_trajectory

        # Print summary (in ALDE order)
        print("\n" + "-"*40)
        print("Metrics Summary (ALDE order):")
        print("-"*40)
        print(f"  high_fitness_proximity: {metrics_result.high_fitness_proximity:.4f}")
        print(f"  novelty: {metrics_result.novelty:.4f}")
        print(f"  batch_diversity: {metrics_result.batch_diversity:.4f}")
        print(f"  normalized_fitness_median_top128: {metrics_result.normalized_fitness_median_top128:.4f}")
        print(f"  normalized_fitness_median_top256: {metrics_result.normalized_fitness_median_top256:.4f}")
        print(f"  max_fitness: {metrics_result.max_fitness:.4f}")
        print(f"  spearman_correlation: {metrics_result.spearman_correlation:.4f}")
        print(f"  epistatic_correlation: {metrics_result.epistatic_correlation:.4f}")
        print(f"  recall_high_order: {metrics_result.recall_high_order:.4f}")
        print(f"  simple_regret: {metrics_result.simple_regret:.4f}")
        print(f"  miscalibration_area: {metrics_result.miscalibration_area:.4f}")
        print(f"  expected_calibration_error: {metrics_result.expected_calibration_error:.4f}")
        print(f"  global_max_found: {metrics_result.global_max_found}")

        # Save metrics to JSON
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
    output_path: str
) -> None:
    """Save aggregated results across all runs."""

    # Extract metrics from results that have them
    metrics_list = [r['metrics'] for r in results if 'metrics' in r]

    if not metrics_list:
        print("No metrics to aggregate.")
        return

    # Aggregate metrics (ALDE order)
    metric_names = [
        'high_fitness_proximity',
        'novelty',
        'batch_diversity',
        'normalized_fitness_median_top128',
        'normalized_fitness_median_top256',
        'max_fitness',
        'spearman_correlation',
        'epistatic_correlation',
        'recall_high_order',
        'simple_regret',
        'miscalibration_area',
        'expected_calibration_error',
    ]

    aggregated = {}
    for name in metric_names:
        values = [m[name] for m in metrics_list]
        aggregated[name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }

    # Global max hit count (now stored as bool in metrics)
    hit_count = sum(1 for m in metrics_list if m.get('global_max_found', False))
    aggregated['global_max_hit_rate'] = {
        'count': hit_count,
        'rate': hit_count / len(metrics_list)
    }

    # Create summary DataFrame
    summary_data = []
    for metric, stat in aggregated.items():
        if 'mean' in stat:
            summary_data.append({
                'metric': metric,
                'mean': stat['mean'],
                'std': stat['std'],
                'min': stat['min'],
                'max': stat['max']
            })
        elif 'count' in stat:
            summary_data.append({
                'metric': metric,
                'mean': stat['rate'],
                'std': 0,
                'min': stat['count'],
                'max': len(results)
            })

    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    summary_path = os.path.join(output_path, 'AAV_hard', 'aggregated_metrics.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nAggregated metrics saved to: {summary_path}")

    # Save complete aggregated results to JSON
    aggregated_json_path = os.path.join(output_path, 'AAV_hard', 'aggregated_results.json')
    with open(aggregated_json_path, 'w') as f:
        json.dump({
            'aggregated_metrics': aggregated,
            'n_runs': len(results),
            'seeds': [r['seed'] for r in results],
            'config': results[0].get('config', {}) if results else {}
        }, f, indent=2, default=str)
    print(f"Aggregated results saved to: {aggregated_json_path}")

    # Print summary table
    print("\n" + "="*70)
    print("AGGREGATED METRICS SUMMARY")
    print("="*70)
    print(f"{'Metric':<45} {'Mean':>10} {'Std':>10}")
    print("-"*70)
    for _, row in summary_df.iterrows():
        if row['metric'] == 'global_max_hit_rate':
            print(f"{row['metric']:<45} {row['mean']*100:>9.1f}% ({int(row['min'])}/{int(row['max'])})")
        else:
            print(f"{row['metric']:<45} {row['mean']:>10.4f} {row['std']:>10.4f}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Run AdaLead optimization on AAV_hard with CNN Ensemble",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run with default seed
  python run_AAV_hard.py

  # Single run with specific seed
  python run_AAV_hard.py --seed 42

  # Multiple runs for randomness evaluation
  python run_AAV_hard.py --seeds 42 123 456 789 1000

  # Load seeds from file
  python run_AAV_hard.py --seed_file ../rand_seeds.txt --num_seeds 5

  # Skip metrics computation
  python run_AAV_hard.py --seed 42 --skip_metrics
        """
    )

    # Seed configuration
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single random seed for reproducibility"
    )
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs='+',
        help="Multiple seeds for evaluating randomness (space-separated)"
    )
    seed_group.add_argument(
        "--seed_file",
        type=str,
        help="Path to file containing seeds (one per line)"
    )

    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="Number of seeds to use from seed file (default: 5)"
    )

    # Experiment configuration
    parser.add_argument(
        "--rounds",
        type=int,
        default=14,
        help="Number of optimization rounds after init (default: 14, total 15)"
    )
    parser.add_argument(
        "--init_samples",
        type=int,
        default=96,
        help="Number of random initial samples (default: 96)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=96,
        help="Sequences per batch (default: 96)"
    )
    parser.add_argument(
        "--model_queries",
        type=int,
        default=2000,
        help="Model queries per batch (default: 2000)"
    )

    # Other options
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/AAV_hard_AdaLead/",
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="Verbosity level 0-3 (default: 2)"
    )
    parser.add_argument(
        "--skip_metrics",
        action="store_true",
        help="Skip metrics computation (faster)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/xux/Desktop/AlphaVariant/Benchmark/data",
        help="Base directory for data files"
    )

    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    # Suppress TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Determine which seeds to use
    if args.seeds is not None:
        seeds = args.seeds
    elif args.seed_file is not None:
        seeds = load_seeds_from_file(args.seed_file, args.num_seeds)
        print(f"Loaded {len(seeds)} seeds from {args.seed_file}")
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        # Default seed
        seeds = [64]

    total_samples = args.init_samples + args.rounds * args.batch_size
    print(f"\nRunning AdaLead on AAV_hard with {len(seeds)} seed(s): {seeds}")
    print(f"Output path: {args.output_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Initial samples: {args.init_samples}")
    print(f"Optimization rounds: {args.rounds}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total samples: {total_samples}")
    print(f"Model queries per batch: {args.model_queries}")
    print(f"Compute metrics: {not args.skip_metrics}")

    # Run experiments
    results = []
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Running experiment with seed={seed}")
        result = run_single_experiment(
            seed=seed,
            output_path=args.output_path,
            verbose=args.verbose,
            run_id=i + 1,
            compute_metrics_flag=not args.skip_metrics,
            data_dir=args.data_dir,
            rounds=args.rounds,
            sequences_batch_size=args.batch_size,
            model_queries_per_batch=args.model_queries,
            n_init_samples=args.init_samples
        )
        results.append(result)

    # Aggregate results if multiple runs
    if len(results) > 1 and not args.skip_metrics:
        save_aggregated_results(results, args.output_path)

    # Final summary
    print(f"\n{'='*60}")
    print("Experiment Complete")
    print(f"{'='*60}")
    print(f"Total runs: {len(results)}")
    print(f"Configuration:")
    print(f"  - Explorer: AdaLead")
    print(f"  - Model: CNN_ENSEMBLE (5 models)")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Rounds: {args.rounds}")
    print(f"  - Model queries per batch: {args.model_queries}")
    print(f"\nResults saved to: {args.output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
