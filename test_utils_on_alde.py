#!/usr/bin/env python
"""
test_utils_on_alde.py - Test the new benchmark utils against ALDE's implementation

This script:
1. Loads GB1 data using both our utils and ALDE's approach
2. Runs metrics computation using both implementations
3. Compares results to validate consistency
4. Measures performance

Usage:
    python test_utils_on_alde.py
"""

import sys
import os
import time
import numpy as np
import torch

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ALDE_PATH = "/home/xux/Desktop/AlphaVariant/Benchmark/ALDE"
DATA_DIR = "/home/xux/Desktop/AlphaVariant/Benchmark/data"

# Import our utils
from utils import (
    levenshtein_distance,
    hamming_distance,
    high_fitness_proximity,
    high_fitness_proximity_from_landscape,
    novelty,
    batch_diversity,
    normalized_fitness_median_topk,
    max_fitness,
    spearman_correlation,
    simple_regret,
    global_max_hit_count,
    miscalibration_area,
    expected_calibration_error,
    BenchmarkEvaluator,
)
from utils.data import (
    FitnessLandscape,
    count_mutations,
    get_top_fitness_sequences,
    one_hot_encode,
    load_random_seeds,
    AMINO_ACIDS,
)
from utils.io import save_results, create_results_directory

# Import ALDE's metrics for comparison
sys.path.insert(0, ALDE_PATH)
from src import metrics as alde_metrics


def load_gb1_data():
    """Load GB1 landscape data."""
    import pandas as pd

    fitness_path = os.path.join(DATA_DIR, "GB1", "fitness.csv")

    if os.path.exists(fitness_path):
        df = pd.read_csv(fitness_path)
    else:
        # Try alternative path
        fitness_path = os.path.join(ALDE_PATH, "data", "GB1", "fitness.csv")
        df = pd.read_csv(fitness_path)

    if 'Combo' in df.columns:
        sequences = df['Combo'].tolist()
    elif 'AACombo' in df.columns:
        sequences = df['AACombo'].tolist()
    else:
        raise ValueError("Could not find sequence column")

    fitness = df['fitness'].values
    return sequences, fitness


def test_distance_functions():
    """Test distance function implementations."""
    print("\n" + "="*60)
    print("Testing Distance Functions")
    print("="*60)

    # Test cases
    test_pairs = [
        ("VDGV", "VDGV", 0, 0),  # same
        ("VDGV", "VWHS", 3, 3),  # 3 mutations
        ("VDGV", "XXXX", 4, 4),  # all different
        ("ABCD", "BCDE", 4, 4),  # shift (Levenshtein = 2 for equal length is wrong, should use Hamming)
    ]

    all_pass = True
    for s1, s2, expected_hamming, expected_lev in test_pairs:
        # Test our implementations
        our_hamming = hamming_distance(s1, s2)
        our_lev = levenshtein_distance(s1, s2)

        # Test ALDE implementations
        alde_hamming = alde_metrics.hamming_distance(s1, s2)
        alde_lev = alde_metrics.levenshtein_distance(s1, s2)

        hamming_match = our_hamming == alde_hamming
        lev_match = our_lev == alde_lev

        status = "PASS" if hamming_match and lev_match else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  {s1} vs {s2}:")
        print(f"    Hamming: ours={our_hamming}, ALDE={alde_hamming} [{status}]")
        print(f"    Levenshtein: ours={our_lev}, ALDE={alde_lev} [{status}]")

    return all_pass


def test_exploration_metrics(sequences, fitness, initial_indices, queried_indices):
    """Test exploration metrics: high_fitness_proximity, novelty, batch_diversity."""
    print("\n" + "="*60)
    print("Testing Exploration Metrics")
    print("="*60)

    # Get sequences
    initial_seqs = [sequences[i] for i in initial_indices]
    queried_seqs = [sequences[i] for i in queried_indices]
    non_initial_seqs = queried_seqs[len(initial_indices):]

    # Get high-fitness sequences (top 10%)
    threshold = np.percentile(fitness, 90)
    high_fit_mask = fitness >= threshold
    high_fit_seqs = [seq for seq, is_high in zip(sequences, high_fit_mask) if is_high]

    results = {}

    # --- High-Fitness Proximity ---
    print("\n  High-Fitness Proximity:")

    # Our implementation (using from_landscape version that matches ALDE)
    start = time.time()
    our_hfp = high_fitness_proximity_from_landscape(
        queried_seqs, sequences, fitness, percentile=0.9,
        max_high_seqs=128, distance_fn='hamming'
    )
    our_time = time.time() - start

    # ALDE implementation
    start = time.time()
    alde_hfp = alde_metrics.high_fitness_proximity(
        queried_seqs, sequences, fitness, percentile=0.9, distance_fn='hamming'
    )
    alde_time = time.time() - start

    diff = abs(our_hfp - alde_hfp)
    status = "PASS" if diff < 0.01 else f"DIFF={diff:.4f}"
    print(f"    Ours: {our_hfp:.4f} ({our_time:.3f}s)")
    print(f"    ALDE: {alde_hfp:.4f} ({alde_time:.3f}s)")
    print(f"    Status: {status}")
    results['high_fitness_proximity'] = {'ours': our_hfp, 'alde': alde_hfp, 'match': diff < 0.01}

    # --- Novelty ---
    print("\n  Novelty:")

    start = time.time()
    our_nov = novelty(non_initial_seqs, initial_seqs, distance_fn='hamming')
    our_time = time.time() - start

    start = time.time()
    alde_nov = alde_metrics.novelty(non_initial_seqs, initial_seqs, distance_fn='hamming')
    alde_time = time.time() - start

    diff = abs(our_nov - alde_nov)
    status = "PASS" if diff < 0.01 else f"DIFF={diff:.4f}"
    print(f"    Ours: {our_nov:.4f} ({our_time:.3f}s)")
    print(f"    ALDE: {alde_nov:.4f} ({alde_time:.3f}s)")
    print(f"    Status: {status}")
    results['novelty'] = {'ours': our_nov, 'alde': alde_nov, 'match': diff < 0.01}

    # --- Batch Diversity ---
    print("\n  Batch Diversity:")

    start = time.time()
    our_div = batch_diversity(queried_seqs, distance_fn='hamming')
    our_time = time.time() - start

    start = time.time()
    alde_div = alde_metrics.batch_diversity(queried_seqs, distance_fn='hamming', aggregation='median')
    alde_time = time.time() - start

    diff = abs(our_div - alde_div)
    status = "PASS" if diff < 0.01 else f"DIFF={diff:.4f}"
    print(f"    Ours: {our_div:.4f} ({our_time:.3f}s)")
    print(f"    ALDE: {alde_div:.4f} ({alde_time:.3f}s)")
    print(f"    Status: {status}")
    results['batch_diversity'] = {'ours': our_div, 'alde': alde_div, 'match': diff < 0.01}

    return results


def test_functional_metrics(sequences, fitness, queried_indices):
    """Test functional metrics: normalized_fitness, max_fitness."""
    print("\n" + "="*60)
    print("Testing Functional Metrics")
    print("="*60)

    queried_fitness = fitness[queried_indices]
    global_min = np.min(fitness)
    global_max = np.max(fitness)

    results = {}

    # --- Normalized Fitness Top-128 ---
    print("\n  Normalized Fitness (Top-128):")

    our_nf128 = normalized_fitness_median_topk(queried_fitness, k=128,
                                                fitness_min=global_min, fitness_max=global_max)
    alde_nf128 = alde_metrics.normalized_fitness_topk(queried_fitness, k=128,
                                                       min_fitness=global_min, max_fitness=global_max)

    diff = abs(our_nf128 - alde_nf128)
    status = "PASS" if diff < 0.01 else f"DIFF={diff:.4f}"
    print(f"    Ours: {our_nf128:.4f}")
    print(f"    ALDE: {alde_nf128:.4f}")
    print(f"    Status: {status}")
    results['normalized_fitness_top128'] = {'ours': our_nf128, 'alde': alde_nf128, 'match': diff < 0.01}

    # --- Normalized Fitness Top-256 ---
    print("\n  Normalized Fitness (Top-256):")

    our_nf256 = normalized_fitness_median_topk(queried_fitness, k=256,
                                                fitness_min=global_min, fitness_max=global_max)
    alde_nf256 = alde_metrics.normalized_fitness_topk(queried_fitness, k=256,
                                                       min_fitness=global_min, max_fitness=global_max)

    diff = abs(our_nf256 - alde_nf256)
    status = "PASS" if diff < 0.01 else f"DIFF={diff:.4f}"
    print(f"    Ours: {our_nf256:.4f}")
    print(f"    ALDE: {alde_nf256:.4f}")
    print(f"    Status: {status}")
    results['normalized_fitness_top256'] = {'ours': our_nf256, 'alde': alde_nf256, 'match': diff < 0.01}

    # --- Max Fitness ---
    print("\n  Max Fitness:")

    our_max = max_fitness(queried_fitness)
    alde_max = alde_metrics.max_fitness(queried_fitness)

    diff = abs(our_max - alde_max)
    status = "PASS" if diff < 0.001 else f"DIFF={diff:.4f}"
    print(f"    Ours: {our_max:.4f}")
    print(f"    ALDE: {alde_max:.4f}")
    print(f"    Status: {status}")
    results['max_fitness'] = {'ours': our_max, 'alde': alde_max, 'match': diff < 0.001}

    return results


def test_model_quality_metrics(sequences, fitness, y_pred):
    """Test model quality metrics: spearman_correlation."""
    print("\n" + "="*60)
    print("Testing Model Quality Metrics")
    print("="*60)

    results = {}

    # --- Spearman Correlation ---
    print("\n  Spearman Correlation:")

    our_spearman = spearman_correlation(y_pred, fitness)
    alde_spearman = alde_metrics.spearman_correlation(fitness, y_pred)

    diff = abs(our_spearman - alde_spearman)
    status = "PASS" if diff < 0.001 else f"DIFF={diff:.4f}"
    print(f"    Ours: {our_spearman:.4f}")
    print(f"    ALDE: {alde_spearman:.4f}")
    print(f"    Status: {status}")
    results['spearman_correlation'] = {'ours': our_spearman, 'alde': alde_spearman, 'match': diff < 0.001}

    return results


def test_success_metrics(queried_fitness, global_max_fit):
    """Test success metrics: simple_regret, global_max_hit_count."""
    print("\n" + "="*60)
    print("Testing Success Metrics")
    print("="*60)

    best_found = np.max(queried_fitness)

    results = {}

    # --- Simple Regret ---
    print("\n  Simple Regret:")

    our_regret = simple_regret(best_found, global_max_fit)
    alde_regret = alde_metrics.simple_regret(best_found, global_max_fit)

    diff = abs(our_regret - alde_regret)
    status = "PASS" if diff < 0.001 else f"DIFF={diff:.4f}"
    print(f"    Ours: {our_regret:.4f}")
    print(f"    ALDE: {alde_regret:.4f}")
    print(f"    Status: {status}")
    results['simple_regret'] = {'ours': our_regret, 'alde': alde_regret, 'match': diff < 0.001}

    # --- Global Max Hit Count ---
    print("\n  Global Max Hit Count:")

    # Simulate multiple runs
    run_max_fitness = [0.95, 0.98, 1.0, 0.92, 1.0, 0.99, 1.0, 0.97, 0.96, 1.0]
    global_max = 1.0

    our_hit_count = global_max_hit_count(
        ["seq"]*4 + ["global_max"]*6,  # 6 hits
        "global_max"
    )
    alde_hit_count, alde_hit_rate = alde_metrics.global_max_hit_count(
        run_max_fitness, global_max, tolerance=0.01
    )

    # Our function returns count, ALDE returns (count, rate)
    print(f"    Ours: count={our_hit_count}")
    print(f"    ALDE: count={alde_hit_count}, rate={alde_hit_rate:.2f}")
    results['global_max_hit_count'] = {'ours': our_hit_count, 'alde': alde_hit_count}

    return results


def test_uncertainty_metrics(y_true, y_pred, y_std):
    """Test uncertainty metrics: miscalibration_area, expected_calibration_error."""
    print("\n" + "="*60)
    print("Testing Uncertainty Metrics")
    print("="*60)

    results = {}

    # Compute prediction errors
    errors = np.abs(y_pred - y_true)

    # --- Miscalibration Area ---
    print("\n  Miscalibration Area:")

    our_miscal = miscalibration_area(y_std, errors)
    alde_miscal = alde_metrics.miscalibration_area(y_true, y_pred, y_std)

    # Note: implementations may differ in exact calculation
    print(f"    Ours: {our_miscal:.4f}")
    print(f"    ALDE: {alde_miscal:.4f}")
    diff = abs(our_miscal - alde_miscal)
    status = "SIMILAR" if diff < 0.1 else f"DIFF={diff:.4f}"
    print(f"    Status: {status} (implementations may differ)")
    results['miscalibration_area'] = {'ours': our_miscal, 'alde': alde_miscal}

    # --- Expected Calibration Error ---
    print("\n  Expected Calibration Error:")

    # Our ECE is for classification, ALDE's is for regression
    # Compare regression calibration error instead
    from utils.metrics import regression_calibration_error
    our_misc, our_ece = regression_calibration_error(y_pred, y_std, y_true)
    alde_ece = alde_metrics.expected_calibration_error(y_true, y_pred, y_std)

    print(f"    Ours (regression): {our_ece:.4f}")
    print(f"    ALDE: {alde_ece:.4f}")
    diff = abs(our_ece - alde_ece)
    status = "SIMILAR" if diff < 0.1 else f"DIFF={diff:.4f}"
    print(f"    Status: {status} (implementations may differ)")
    results['expected_calibration_error'] = {'ours': our_ece, 'alde': alde_ece}

    return results


def test_benchmark_evaluator(sequences, fitness):
    """Test the BenchmarkEvaluator class."""
    print("\n" + "="*60)
    print("Testing BenchmarkEvaluator Class")
    print("="*60)

    # Create landscape
    landscape = FitnessLandscape(sequences, fitness)

    # Sample initial training set
    np.random.seed(42)
    n_init = 96
    init_indices = np.random.choice(len(sequences), n_init, replace=False)
    init_seqs = [sequences[i] for i in init_indices]

    # Create evaluator
    evaluator = BenchmarkEvaluator(
        landscape=landscape,
        initial_training_sequences=init_seqs,
        wild_type='VDGV',
        method_name='ALDE',
        dataset_name='GB1',
        distance_fn='hamming'  # Use hamming for GB1 (fixed length)
    )

    print(f"\n  Landscape size: {len(landscape)}")
    print(f"  Global max sequence: {landscape.global_max_sequence}")
    print(f"  Global max fitness: {landscape.global_max_fitness:.4f}")
    print(f"  Fitness range: [{landscape.min_fitness:.4f}, {landscape.max_fitness:.4f}]")
    print(f"  High-fitness sequences (top 10%): {len(evaluator.high_fitness_sequences)}")
    print(f"  True high-order mutants: {len(evaluator.true_top_mutants)}")

    # Generate some test sequences (simulate optimization)
    np.random.seed(123)
    n_generated = 384
    gen_indices = np.random.choice(len(sequences), n_generated, replace=False)
    gen_seqs = [sequences[i] for i in gen_indices]
    gen_fitness = fitness[gen_indices]

    # Evaluate batch
    print("\n  Evaluating batch of generated sequences...")
    start = time.time()
    metrics = evaluator.evaluate_batch(gen_seqs, gen_fitness, round_idx=0)
    eval_time = time.time() - start

    print(f"\n  Evaluation time: {eval_time:.3f}s")
    print("\n  Metrics:")
    evaluator.print_metrics_summary(metrics, prefix="  ")

    return True


def main():
    print("="*60)
    print("TESTING BENCHMARK UTILS AGAINST ALDE IMPLEMENTATION")
    print("="*60)

    # Load data
    print("\nLoading GB1 data...")
    sequences, fitness = load_gb1_data()
    print(f"  Loaded {len(sequences)} sequences")
    print(f"  Fitness range: [{fitness.min():.4f}, {fitness.max():.4f}]")

    # Normalize fitness (as ALDE does)
    fitness_normalized = fitness / fitness.max()

    # Create simulated query indices (as if from an optimization run)
    np.random.seed(42)
    n_init = 96
    n_budget = 384
    initial_indices = np.random.choice(len(sequences), n_init, replace=False)
    remaining = list(set(range(len(sequences))) - set(initial_indices))
    budget_indices = np.random.choice(remaining, n_budget, replace=False)
    queried_indices = np.concatenate([initial_indices, budget_indices]).astype(int)

    print(f"\n  Initial samples: {len(initial_indices)}")
    print(f"  Budget samples: {len(budget_indices)}")
    print(f"  Total queries: {len(queried_indices)}")

    # Create simulated predictions
    np.random.seed(123)
    noise = np.random.normal(0, 0.1, len(fitness_normalized))
    y_pred = fitness_normalized + noise
    y_std = np.abs(np.random.normal(0.1, 0.05, len(fitness_normalized)))

    # Run all tests
    all_results = {}

    # Test 1: Distance functions
    distance_pass = test_distance_functions()
    all_results['distance_functions'] = distance_pass

    # Test 2: Exploration metrics
    exploration_results = test_exploration_metrics(
        sequences, fitness_normalized, initial_indices, queried_indices
    )
    all_results['exploration'] = exploration_results

    # Test 3: Functional metrics
    functional_results = test_functional_metrics(
        sequences, fitness_normalized, queried_indices
    )
    all_results['functional'] = functional_results

    # Test 4: Model quality metrics
    quality_results = test_model_quality_metrics(
        sequences, fitness_normalized, y_pred
    )
    all_results['model_quality'] = quality_results

    # Test 5: Success metrics
    queried_fitness = fitness_normalized[queried_indices]
    success_results = test_success_metrics(queried_fitness, fitness_normalized.max())
    all_results['success'] = success_results

    # Test 6: Uncertainty metrics
    uncertainty_results = test_uncertainty_metrics(
        fitness_normalized, y_pred, y_std
    )
    all_results['uncertainty'] = uncertainty_results

    # Test 7: BenchmarkEvaluator
    evaluator_pass = test_benchmark_evaluator(sequences, fitness_normalized)
    all_results['benchmark_evaluator'] = evaluator_pass

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    n_pass = 0
    n_total = 0

    for category, results in all_results.items():
        if isinstance(results, bool):
            status = "PASS" if results else "FAIL"
            n_total += 1
            n_pass += 1 if results else 0
            print(f"  {category}: {status}")
        elif isinstance(results, dict):
            for metric, data in results.items():
                if 'match' in data:
                    status = "PASS" if data['match'] else "FAIL"
                    n_total += 1
                    n_pass += 1 if data['match'] else 0
                    print(f"  {category}/{metric}: {status}")
                else:
                    print(f"  {category}/{metric}: COMPARED")

    print(f"\n  Overall: {n_pass}/{n_total} tests passed")
    print("="*60)

    return n_pass == n_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
