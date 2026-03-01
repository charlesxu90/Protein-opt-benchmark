"""
Compatibility Module for Legacy Method Integrations

This module provides compatibility wrappers that allow existing run_GB1.py
implementations to seamlessly use the new unified metrics utilities.

Each method can replace their local metrics imports with:
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
        miscalibration_area,
        expected_calibration_error,
        area_under_optimization_curve,
        hit_rate_metric,
    )

This provides drop-in compatibility with the ALDE metrics interface.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Sequence
import numpy as np

try:
    import torch
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Tensor = None

from .metrics import (
    hamming_distance,
    levenshtein_distance,
    high_fitness_proximity_from_landscape as _high_fitness_proximity,
    novelty as _novelty,
    batch_diversity as _batch_diversity,
    normalized_fitness_median_topk,
    max_fitness as _max_fitness,
    simple_regret as _simple_regret,
    spearman_correlation as _spearman_correlation,
    miscalibration_area as _miscalibration_area,
    expected_calibration_error as _expected_calibration_error,
    regression_calibration_error,
    global_max_hit_count as _global_max_hit_count,
    area_under_optimization_curve as _area_under_optimization_curve,
    hit_rate as _hit_rate,
)
from .data import (
    load_csv_data,
    count_mutations,
)


# =============================================================================
# Legacy-Compatible MetricsResult
# =============================================================================

@dataclass
class MetricsResult:
    """
    Container for all computed metrics.

    Compatible with ALDE's MetricsResult interface.
    """
    # Exploration metrics
    high_fitness_proximity: float = 0.0
    novelty: float = 0.0
    batch_diversity: float = 0.0

    # Functional metrics
    normalized_fitness_median_top128: float = 0.0
    normalized_fitness_median_top256: float = 0.0
    max_fitness: float = 0.0

    # Model quality metrics
    spearman_correlation: float = 0.0
    epistatic_correlation: float = 0.0
    recall_high_order: float = 0.0

    # Success metrics
    simple_regret: float = 0.0
    global_max_found: bool = False

    # Uncertainty metrics
    miscalibration_area: float = 1.0
    expected_calibration_error: float = 1.0

    # Optimization curve and hit rate
    auoc: float = 0.0
    hit_rate_value: float = 0.0

    # Trajectories (per-round data)
    regret_trajectory: List[float] = field(default_factory=list)
    fitness_trajectory: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'high_fitness_proximity': self.high_fitness_proximity,
            'novelty': self.novelty,
            'batch_diversity': self.batch_diversity,
            'normalized_fitness_median_top128': self.normalized_fitness_median_top128,
            'normalized_fitness_median_top256': self.normalized_fitness_median_top256,
            'max_fitness': self.max_fitness,
            'spearman_correlation': self.spearman_correlation,
            'epistatic_correlation': self.epistatic_correlation,
            'recall_high_order': self.recall_high_order,
            'simple_regret': self.simple_regret,
            'global_max_found': self.global_max_found,
            'miscalibration_area': self.miscalibration_area,
            'expected_calibration_error': self.expected_calibration_error,
            'auoc': self.auoc,
            'hit_rate': self.hit_rate_value,
        }


# =============================================================================
# Legacy-Compatible Wrapper Functions
# =============================================================================

def high_fitness_proximity(
    generated_seqs: List[str],
    all_seqs: List[str],
    all_fitness: np.ndarray,
    percentile: float = 0.9,
    max_high_seqs: int = 128,
    distance_fn: str = 'hamming'
) -> float:
    """
    High-Fitness Proximity (dhigh): Median minimum distance from generated
    sequences to the top percentile fitness sequences.

    Compatible with ALDE's interface.
    """
    return _high_fitness_proximity(
        generated_seqs, all_seqs, all_fitness,
        percentile=percentile,
        max_high_seqs=max_high_seqs,
        distance_fn=distance_fn
    )


def novelty(
    generated_seqs: List[str],
    initial_seqs: List[str],
    distance_fn: str = 'hamming'
) -> float:
    """
    Novelty (dinit): Median minimum distance to initial training set.

    Compatible with ALDE's interface.
    """
    return _novelty(generated_seqs, initial_seqs, distance_fn=distance_fn)


def batch_diversity(
    sequences: List[str],
    distance_fn: str = 'hamming',
    aggregation: str = 'median'
) -> float:
    """
    Batch Diversity: Pairwise distance between sequences.

    Compatible with ALDE's interface.
    """
    # Note: our version only supports median, but that's what ALDE uses
    return _batch_diversity(sequences, distance_fn=distance_fn)


def normalized_fitness_topk(
    fitness_values: np.ndarray,
    k: int = 128,
    min_fitness: Optional[float] = None,
    max_fitness: Optional[float] = None,
    aggregation: str = 'median'
) -> float:
    """
    Normalized Fitness (Median Top-K).

    Compatible with ALDE's interface.
    """
    return normalized_fitness_median_topk(
        fitness_values, k=k,
        fitness_min=min_fitness,
        fitness_max=max_fitness
    )


def max_fitness(fitness_values: np.ndarray) -> float:
    """Max Fitness: The absolute highest fitness value discovered."""
    return _max_fitness(fitness_values)


def simple_regret(best_found: float, global_max: float) -> float:
    """Simple Regret: Gap between true global optimum and best design found."""
    return _simple_regret(best_found, global_max)


def simple_regret_trajectory(
    fitness_trajectory: np.ndarray,
    global_max: float
) -> np.ndarray:
    """Simple Regret Trajectory: Regret at each round."""
    cumulative_best = np.maximum.accumulate(fitness_trajectory)
    return global_max - cumulative_best


def global_max_hit_count(
    run_max_fitness: List[float],
    global_max: float,
    tolerance: float = 0.01
) -> Tuple[int, float]:
    """
    Global Max Hit Count: Number of runs that found the global maximum.

    Compatible with ALDE's interface (returns tuple of count, rate).
    """
    threshold = global_max * (1 - tolerance)
    hit_count = sum(1 for max_fit in run_max_fitness if max_fit >= threshold)
    hit_rate = hit_count / len(run_max_fitness) if run_max_fitness else 0.0
    return hit_count, float(hit_rate)


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman Rank Correlation."""
    return _spearman_correlation(y_pred, y_true)


def miscalibration_area(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Miscalibration Area.

    Compatible with ALDE's interface (takes y_true, y_pred, y_std).
    """
    errors = np.abs(y_true - y_pred)
    return _miscalibration_area(y_std, errors, n_bins=n_bins)


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Expected Calibration Error.

    Compatible with ALDE's interface.
    """
    misc, ece = regression_calibration_error(y_pred, y_std, y_true, n_bins=n_bins)
    return ece


# =============================================================================
# Data Loading
# =============================================================================

def load_landscape_data(
    protein: str,
    data_dir: str = "/home/xux/Desktop/AlphaVariant/Benchmark/data"
) -> Tuple[List[str], np.ndarray]:
    """
    Load complete fitness landscape data.

    Compatible with ALDE's interface.

    Args:
        protein: Protein name (e.g., 'GB1')
        data_dir: Base data directory

    Returns:
        Tuple of (sequences, fitness_values)
    """
    import os

    # Try different file names for compatibility
    file_patterns = [
        f"{data_dir}/{protein}/data.csv",
        f"{data_dir}/{protein}/fitness.csv",
        f"{data_dir}/{protein}.csv",
    ]

    data_path = None
    for pattern in file_patterns:
        if os.path.exists(pattern):
            data_path = pattern
            break

    if data_path is None:
        raise FileNotFoundError(
            f"No fitness data found for {protein}. Tried: {file_patterns}"
        )

    sequences, fitness = load_csv_data(data_path)
    return sequences, fitness


def indices_to_sequences(
    indices: Union['Tensor', np.ndarray],
    all_sequences: List[str]
) -> List[str]:
    """Convert indices to sequences."""
    if HAS_TORCH and isinstance(indices, Tensor):
        indices = indices.cpu().numpy().astype(int)
    return [all_sequences[i] for i in indices]


# =============================================================================
# Main Metrics Computation
# =============================================================================

def compute_all_metrics(
    queried_indices: Union['Tensor', np.ndarray],
    all_sequences: List[str],
    all_fitness: np.ndarray,
    initial_indices: Union['Tensor', np.ndarray],
    y_pred: Optional[np.ndarray] = None,
    y_std: Optional[np.ndarray] = None,
    batch_size: int = 96,
    wildtype: Optional[str] = None
) -> MetricsResult:
    """
    Compute all metrics for an optimization run.

    Compatible with ALDE's compute_all_metrics interface.

    Args:
        queried_indices: Indices of all queried sequences (including initial)
        all_sequences: Complete list of sequences in landscape
        all_fitness: Fitness values for all sequences
        initial_indices: Indices of initial random samples
        y_pred: Model predictions for all sequences (optional)
        y_std: Model uncertainty for all sequences (optional)
        batch_size: Batch size for round-wise analysis
        wildtype: Wild-type sequence for epistasis computation

    Returns:
        MetricsResult with all computed metrics
    """
    result = MetricsResult()

    # Convert to numpy
    if HAS_TORCH:
        if isinstance(queried_indices, Tensor):
            queried_indices = queried_indices.cpu().numpy().astype(int)
        if isinstance(initial_indices, Tensor):
            initial_indices = initial_indices.cpu().numpy().astype(int)

    queried_indices = np.array(queried_indices).astype(int)
    initial_indices = np.array(initial_indices).astype(int)

    # Get sequences and fitness
    queried_seqs = [all_sequences[i] for i in queried_indices]
    initial_seqs = [all_sequences[i] for i in initial_indices]
    queried_fitness = all_fitness[queried_indices]

    # Global bounds
    global_max = float(np.max(all_fitness))
    global_min = float(np.min(all_fitness))

    # --- Exploration Metrics ---
    result.high_fitness_proximity = high_fitness_proximity(
        queried_seqs, all_sequences, all_fitness,
        percentile=0.9, distance_fn='hamming'
    )

    # Novelty: distance of non-initial queries to initial set
    non_initial_seqs = queried_seqs[len(initial_indices):]
    if non_initial_seqs:
        result.novelty = novelty(non_initial_seqs, initial_seqs, distance_fn='hamming')

    result.batch_diversity = batch_diversity(queried_seqs, distance_fn='hamming')

    # --- Functional Metrics ---
    result.normalized_fitness_median_top128 = normalized_fitness_topk(
        queried_fitness, k=128, min_fitness=global_min, max_fitness=global_max
    )
    result.normalized_fitness_median_top256 = normalized_fitness_topk(
        queried_fitness, k=256, min_fitness=global_min, max_fitness=global_max
    )
    result.max_fitness = max_fitness(queried_fitness)

    # --- Success Metrics ---
    result.simple_regret = simple_regret(result.max_fitness, global_max)
    result.global_max_found = (result.max_fitness >= global_max * 0.99)

    # Fitness trajectory
    fitness_traj = []
    for i in range(0, len(queried_fitness), batch_size):
        batch_fitness = queried_fitness[:i + batch_size]
        fitness_traj.append(float(np.max(batch_fitness)))
    result.fitness_trajectory = fitness_traj
    result.regret_trajectory = list(simple_regret_trajectory(np.array(fitness_traj), global_max))

    # --- Model Quality Metrics (if predictions available) ---
    if y_pred is not None:
        queried_set = set(queried_indices)
        holdout_mask = np.array([i not in queried_set for i in range(len(all_fitness))])

        if np.sum(holdout_mask) > 100:
            holdout_true = all_fitness[holdout_mask]
            holdout_pred = y_pred[holdout_mask]

            result.spearman_correlation = spearman_correlation(holdout_true, holdout_pred)

            # Epistatic correlation (simplified)
            if wildtype is not None:
                holdout_seqs = [all_sequences[i] for i, m in enumerate(holdout_mask) if m]
                high_order_mask = np.array([
                    hamming_distance(seq, wildtype) >= 2 for seq in holdout_seqs
                ])
                if np.sum(high_order_mask) > 10:
                    result.epistatic_correlation = spearman_correlation(
                        holdout_true[high_order_mask],
                        holdout_pred[high_order_mask]
                    )

                # Recall of high-order mutants
                n_ho = np.sum(high_order_mask)
                if n_ho > 0:
                    top_k = min(100, n_ho // 2)
                    if top_k > 0:
                        ho_indices = np.where(high_order_mask)[0]
                        ho_true = holdout_true[high_order_mask]
                        ho_pred = holdout_pred[high_order_mask]

                        true_top_k = set(ho_indices[np.argsort(ho_true)[-top_k:]])
                        pred_top_k = set(ho_indices[np.argsort(ho_pred)[-top_k:]])
                        result.recall_high_order = len(true_top_k & pred_top_k) / len(true_top_k)

    # --- Uncertainty Metrics ---
    if y_pred is not None and y_std is not None:
        queried_set = set(queried_indices)
        holdout_mask = np.array([i not in queried_set for i in range(len(all_fitness))])

        if np.sum(holdout_mask) > 100:
            holdout_true = all_fitness[holdout_mask]
            holdout_pred = y_pred[holdout_mask]
            holdout_std = y_std[holdout_mask]

            result.miscalibration_area = miscalibration_area(
                holdout_true, holdout_pred, holdout_std
            )
            result.expected_calibration_error = expected_calibration_error(
                holdout_true, holdout_pred, holdout_std
            )

    return result


def aggregate_run_metrics(
    run_results: List[MetricsResult]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple runs.

    Compatible with ALDE's interface.

    Args:
        run_results: List of MetricsResult from multiple runs

    Returns:
        Dictionary with mean and std for each metric
    """
    if not run_results:
        return {}

    metrics_names = [
        'high_fitness_proximity', 'novelty', 'batch_diversity',
        'normalized_fitness_median_top128', 'normalized_fitness_median_top256',
        'max_fitness', 'spearman_correlation', 'epistatic_correlation',
        'recall_high_order', 'simple_regret', 'miscalibration_area',
        'expected_calibration_error', 'auoc', 'hit_rate_value'
    ]

    aggregated = {}
    for name in metrics_names:
        values = [getattr(r, name) for r in run_results]
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            aggregated[name] = {
                'mean': float(np.mean(valid_values)),
                'std': float(np.std(valid_values)),
                'min': float(np.min(valid_values)),
                'max': float(np.max(valid_values))
            }
        else:
            aggregated[name] = {
                'mean': float('nan'),
                'std': float('nan'),
                'min': float('nan'),
                'max': float('nan')
            }

    # Global max hit count
    hit_count = sum(1 for r in run_results if r.global_max_found)
    aggregated['global_max_hit_count'] = {
        'count': hit_count,
        'rate': hit_count / len(run_results) if run_results else 0.0
    }

    return aggregated


# =============================================================================
# New Metrics Wrappers: AUOC and Hit Rate
# =============================================================================

def area_under_optimization_curve(
    fitness_trajectory: Sequence[float],
    global_max_fitness: float,
    normalize: bool = True
) -> float:
    """Compute AUOC. See utils.metrics.area_under_optimization_curve."""
    return _area_under_optimization_curve(fitness_trajectory, global_max_fitness, normalize)


def hit_rate_metric(
    generated_fitness: Sequence[float],
    threshold: float
) -> float:
    """Compute hit rate. See utils.metrics.hit_rate."""
    return _hit_rate(generated_fitness, threshold)
