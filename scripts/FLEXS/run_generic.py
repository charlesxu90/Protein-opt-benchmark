#!/usr/bin/env python
"""
run_generic.py - Execute AdaLead optimization on any dataset with comprehensive metrics

A generic, dataset-agnostic runner for the FLEXS/AdaLead method. Works with any
dataset that has a data.csv file with 'seq' and 'fitness' columns.

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
    python run_generic.py --dataset AAV_med

    # Single run with specific seed
    python run_generic.py --dataset GFP_med --seed 42

    # Multiple runs with different seeds for randomness evaluation
    python run_generic.py --dataset GB1 --seeds 42 123 456 789 1000

    # Use predefined seeds from file
    python run_generic.py --dataset AAV_hard --seed_file ../rand_seeds.txt --num_seeds 5

    # Skip metrics computation (faster)
    python run_generic.py --dataset AAV_med --seed 42 --skip_metrics
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
sys.path.insert(0, os.path.join(BENCHMARK_ROOT, 'FLEXS'))  # upstream package lives in the method dir

import flexs
from flexs import baselines
from flexs.utils import sequence_utils as s_utils

# Import unified metrics from utils.compat
import sys
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
# Generic Landscape
# ============================================================================

class GenericLandscape(flexs.Landscape):
    """
    Generic protein fitness landscape.

    Works with any dataset that has a data.csv file with 'seq' and 'fitness' columns.
    Auto-detects sequence length and alphabet from the data. Uses the highest-fitness
    sequence as the wildtype fallback.
    """

    def __init__(
        self,
        dataset: str,
        data_dir: str = os.path.join(BENCHMARK_ROOT, 'data'),
        normalize: bool = False
    ):
        """
        Initialize generic landscape.

        Args:
            dataset: Name of the dataset (subdirectory under data_dir)
            data_dir: Base directory for data files
            normalize: Whether to normalize fitness values to [0, 1]
        """
        super().__init__(name=dataset)

        # Load data
        data_path = os.path.join(data_dir, dataset, "data.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        self.data = pd.read_csv(data_path)
        self.dataset = dataset

        # Prefer AACombo (short combinatorial form) when present, fall back to full seq
        if 'AACombo' in self.data.columns:
            self.sequences = self.data['AACombo'].tolist()
        elif 'Combo' in self.data.columns:
            self.sequences = self.data['Combo'].tolist()
        elif 'seq' in self.data.columns:
            self.sequences = self.data['seq'].tolist()
        else:
            self.sequences = self.data['sequence'].tolist()
        if 'fitness' not in self.data.columns and 'blue' in self.data.columns and 'red' in self.data.columns:
            # Multi-objective ("*_joint") dataset: scalarize to geometric mean
            blue = self.data['blue'].values.astype(float)
            red = self.data['red'].values.astype(float)
            self.fitness_raw = np.sqrt(np.clip(blue, 0, None) * np.clip(red, 0, None))
        else:
            self.fitness_raw = self.data['fitness'].values

        if normalize:
            self.fitness = (self.fitness_raw - np.min(self.fitness_raw)) / (np.max(self.fitness_raw) - np.min(self.fitness_raw))
        else:
            self.fitness = self.fitness_raw

        self.seq_to_idx = {seq: i for i, seq in enumerate(self.sequences)}
        self.seq_to_fitness = {seq: fit for seq, fit in zip(self.sequences, self.fitness)}

        # Wildtype resolution: prefer wt.fasta (authoritative, no fitness leak),
        # then n_muts==0 row, then argmax(fitness) with a warning.
        wt_fasta = os.path.join(data_dir, dataset, "wt.fasta")
        self.wildtype = None
        if os.path.exists(wt_fasta):
            with open(wt_fasta) as fh:
                lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith('>')]
            if lines:
                self.wildtype = "".join(lines)
        if self.wildtype is None and 'n_muts' in self.data.columns and (self.data['n_muts'] == 0).any():
            wt_idx = int(self.data.index[self.data['n_muts'] == 0][0])
            self.wildtype = self.sequences[wt_idx]
        if self.wildtype is None:
            print(f"WARNING: no wt.fasta or n_muts==0 row for {dataset}; falling back "
                  f"to argmax(fitness) as wildtype — LEAKY.")
            self.wildtype = self.sequences[int(np.argmax(self.fitness))]

        # Auto-detect sequence length from data
        self.seq_length = len(self.sequences[0])
        self._compute_alphabet()

        # Per-position AA sets for combinatorial libraries. Sequences live on
        # a discrete combinatorial subspace — only a handful of positions vary,
        # and at each varying position only a subset of AAs occurs. Mutating
        # outside this subspace produces sequences that aren't in the library
        # (fitness=0), wasting the FLEXS query budget. Cache the per-position
        # AA sets so mutations can stay inside the library.
        self._compute_per_position_alphabets()

    def _compute_per_position_alphabets(self):
        """Per-position observed AA sets; identify varying positions."""
        seqs = self.sequences
        L = self.seq_length
        per_pos: List[List[str]] = []
        for p in range(L):
            aas = sorted({s[p] for s in seqs if len(s) > p})
            per_pos.append(aas)
        self.per_position_alphabets = per_pos
        self.varying_positions = [p for p, aas in enumerate(per_pos) if len(aas) > 1]

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
    landscape: GenericLandscape,
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

    # Record per-query indices for post-hoc multi-objective analysis on _joint datasets
    seq_to_idx = {s: i for i, s in enumerate(all_sequences)}
    result.queried_indices = [int(seq_to_idx[s]) for s in queried_seqs if s in seq_to_idx]

    # Initial sequences (round 0)
    initial_seqs = results_df[results_df['round'] == 0]['sequence'].tolist()

    # Non-initial sequences
    non_initial_seqs = results_df[results_df['round'] > 0]['sequence'].tolist()

    # === Exploration metrics ===
    result.high_fitness_proximity = high_fitness_proximity(
        queried_seqs, all_sequences, all_fitness,
        percentile=0.9, distance_fn='hamming'
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
    dataset: str,
    seed: int,
    output_path: str,
    verbose: int = 2,
    run_id: Optional[int] = None,
    compute_metrics_flag: bool = True,
    data_dir: str = os.path.join(BENCHMARK_ROOT, 'data'),
    rounds: int = 4,  # 4 BO rounds -> 5 total batches (480 queries), benchmark-standard
    sequences_batch_size: int = 96,
    model_queries_per_batch: int = 2000,
    n_init_samples: int = 96  # Random initial samples (matching ALDE)
) -> Dict[str, Any]:
    """
    Run a single AdaLead optimization experiment on a generic dataset.

    Args:
        dataset: Name of the dataset (subdirectory under data_dir)
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
    print(f"Starting AdaLead optimization on {dataset}")
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
    landscape = GenericLandscape(dataset=dataset, data_dir=data_dir, normalize=False)

    # Get landscape info
    all_sequences = landscape.get_all_sequences()
    alphabet = landscape.alphabet
    seq_length = landscape.seq_length

    print(f"Landscape: {len(all_sequences)} sequences")
    print(f"Wild-type (best fitness): {landscape.wildtype}")
    print(f"Alphabet: {alphabet} (length {len(alphabet)})")
    print(f"Sequence length: {seq_length}")
    print(f"Fitness range: [{np.min(landscape.fitness):.4f}, {np.max(landscape.fitness):.4f}]")

    # Create output directory
    subdir = os.path.join(output_path, dataset, "")
    os.makedirs(subdir, exist_ok=True)

    # Create CNN Ensemble model. CNN's first Conv1D uses padding='valid', so
    # kernel_size must be <= seq_len. Default kernel is 5; shrink for short
    # combinatorial sequences (e.g., 4-AA AACombo).
    cnn_kernel = min(3, seq_length) if seq_length < 5 else 5
    ensemble_models = [
        baselines.models.CNN(
            seq_len=seq_length,
            num_filters=32,
            hidden_size=100,
            alphabet=alphabet,
            loss='MSE',
            epochs=20,
            kernel_size=cnn_kernel,
        )
        for _ in range(5)
    ]
    model = flexs.Ensemble(ensemble_models)

    # =========================================================================
    # Uniform random initialization (matching ALDE, Random, CLADE, AiCE).
    # The previous percentile-restricted init biased FLEXS toward medium-
    # fitness sequences, which is both a fitness leak and (counter-intuitively)
    # often a worse starting point than a broad random sample of the library.
    # =========================================================================
    print(f"\nRound 0: Uniform random initialization ({n_init_samples} samples)")

    all_fitness = landscape.get_all_fitness()
    n_total = len(all_sequences)
    init_indices = np.random.choice(n_total, size=min(n_init_samples, n_total), replace=False)

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

        # Library-aware mutation: only flip varying positions, and only to AAs
        # that exist at that position in the library. This keeps proposed
        # mutants on the combinatorial subspace the landscape actually covers.
        varying_positions = landscape.varying_positions
        per_pos_aas = landscape.per_position_alphabets
        import random as _random
        def gen_in_library_mutant(node):
            if not varying_positions:
                return node
            p_mut = mu / len(varying_positions)
            chars = list(node)
            for p in varying_positions:
                if _random.random() < p_mut:
                    choices = [a for a in per_pos_aas[p] if a != chars[p]]
                    if choices:
                        chars[p] = _random.choice(choices)
            return "".join(chars)

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
                        # Generate library-aware mutant (only mutates the
                        # varying positions, sampling from observed AAs).
                        child = gen_in_library_mutant(node)

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
        'dataset': dataset,
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
    output_path: str,
    dataset: str
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
    summary_path = os.path.join(output_path, dataset, 'aggregated_metrics.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nAggregated metrics saved to: {summary_path}")

    # Save complete aggregated results to JSON
    aggregated_json_path = os.path.join(output_path, dataset, 'aggregated_results.json')
    with open(aggregated_json_path, 'w') as f:
        json.dump({
            'aggregated_metrics': aggregated,
            'n_runs': len(results),
            'seeds': [r['seed'] for r in results],
            'dataset': dataset,
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
        description="Run AdaLead optimization on any dataset with CNN Ensemble",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run on AAV_med with default seed
  python run_generic.py --dataset AAV_med

  # Single run with specific seed
  python run_generic.py --dataset GFP_med --seed 42

  # Multiple runs for randomness evaluation
  python run_generic.py --dataset GB1 --seeds 42 123 456 789 1000

  # Load seeds from file
  python run_generic.py --dataset AAV_hard --seed_file ../rand_seeds.txt --num_seeds 5

  # Skip metrics computation
  python run_generic.py --dataset AAV_med --seed 42 --skip_metrics
        """
    )

    # Dataset (required)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset (subdirectory under data_dir, e.g. AAV_med, GB1, GFP_med)"
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
        default=4,
        help="Number of optimization rounds after init (default: 4, total 5 = 480 queries)"
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
        default=None,
        help="Output directory for results (default: results/<dataset>_AdaLead/)"
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
        default=os.path.join(BENCHMARK_ROOT, 'data'),
        help="Base directory for data files"
    )

    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    # Suppress TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Set default output path based on dataset name
    if args.output_path is None:
        args.output_path = os.path.join(BENCHMARK_ROOT, "FLEXS", "results", f"{args.dataset}_AdaLead")

    # Determine which seeds to use
    if args.seeds is not None:
        seeds = args.seeds
    elif args.seed_file is not None:
        seeds = load_seeds_from_file(args.seed_file, args.num_seeds)
        print(f"Loaded {len(seeds)} seeds from {args.seed_file}")
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        # Default seed (matching LatProtRL)
        seeds = [42]

    total_samples = args.init_samples + args.rounds * args.batch_size
    print(f"\nRunning AdaLead on {args.dataset} with {len(seeds)} seed(s): {seeds}")
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
            dataset=args.dataset,
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
        save_aggregated_results(results, args.output_path, args.dataset)

    # Final summary
    print(f"\n{'='*60}")
    print("Experiment Complete")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
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
