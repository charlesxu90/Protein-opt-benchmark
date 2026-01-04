"""
GB1 Benchmark Utilities

This module provides specialized utilities for the GB1 benchmark dataset,
including data loading, evaluation, and result tracking.

GB1 Dataset:
- 4-site combinatorial library (positions 39, 40, 41, 54)
- Wild-type sequence: VDGV
- Total variants: 149,361 (20^4 - 1 + 1 for WT)
- Global maximum: varies by normalization

Methods supported:
- ALDE, EvoPlay, AdaLead, LatProtRL, AICE, δ-Conservative Search, AlphaVariant
"""

import os
import sys
import json
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
from datetime import datetime
import numpy as np

from .data import (
    load_csv_data,
    FitnessLandscape,
    count_mutations,
    get_top_fitness_sequences,
    one_hot_encode,
    load_random_seeds,
    AMINO_ACIDS,
)
from .metrics import (
    hamming_distance,
    levenshtein_distance,
    high_fitness_proximity,
    high_fitness_proximity_from_landscape,
    novelty,
    batch_diversity,
    normalized_fitness_median_topk,
    max_fitness,
    spearman_correlation,
    epistatic_correlation,
    recall_high_order_mutants,
    simple_regret,
    global_max_hit_count,
    global_max_hit_rate,
    miscalibration_area,
    regression_calibration_error,
)
from .io import save_results, NumpyEncoder


# =============================================================================
# GB1 Constants
# =============================================================================

GB1_WILD_TYPE_4SITE = "VDGV"  # 4-site representation (positions 39, 40, 41, 54)
GB1_POSITIONS = [39, 40, 41, 54]  # 1-indexed positions in full sequence
GB1_POSITIONS_0IDX = [38, 39, 40, 53]  # 0-indexed positions

# Full wild-type GB1 sequence (56 aa)
GB1_WILD_TYPE_FULL = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"


# =============================================================================
# Data Loading
# =============================================================================

def load_gb1_landscape(
    data_dir: str = "/home/xux/Desktop/AlphaVariant/Benchmark/data",
    normalize: bool = True,
    use_4site: bool = True
) -> Tuple[List[str], np.ndarray, str]:
    """
    Load GB1 fitness landscape.

    Parameters
    ----------
    data_dir : str
        Base directory for data files
    normalize : bool, default=True
        Whether to normalize fitness values to [0, 1]
    use_4site : bool, default=True
        If True, return 4-site sequences (VDGV format)
        If False, return full 56aa sequences

    Returns
    -------
    Tuple[List[str], np.ndarray, str]
        (sequences, fitness, wildtype_sequence)
    """
    # Try multiple file patterns
    file_patterns = [
        os.path.join(data_dir, "GB1", "data.csv"),
        os.path.join(data_dir, "GB1", "fitness.csv"),
        os.path.join(data_dir, "GB1.csv"),
    ]

    data_path = None
    for pattern in file_patterns:
        if os.path.exists(pattern):
            data_path = pattern
            break

    if data_path is None:
        raise FileNotFoundError(
            f"GB1 data not found. Tried: {file_patterns}"
        )

    sequences, fitness = load_csv_data(data_path)

    # Determine sequence format
    seq_length = len(sequences[0])
    if seq_length == 4:
        # Already 4-site format
        wildtype = GB1_WILD_TYPE_4SITE
    elif seq_length == 56:
        # Full sequence format
        if use_4site:
            # Extract 4-site representation
            sequences = [
                seq[38] + seq[39] + seq[40] + seq[53]
                for seq in sequences
            ]
            wildtype = GB1_WILD_TYPE_4SITE
        else:
            wildtype = GB1_WILD_TYPE_FULL
    else:
        # Unknown format, use as-is
        wildtype = sequences[0]
        warnings.warn(f"Unknown GB1 sequence format (length={seq_length})")

    if normalize:
        fitness = fitness / np.max(fitness)

    return sequences, fitness, wildtype


def create_gb1_landscape(
    data_dir: str = "/home/xux/Desktop/AlphaVariant/Benchmark/data",
    normalize: bool = True
) -> FitnessLandscape:
    """Create a FitnessLandscape object for GB1."""
    sequences, fitness, _ = load_gb1_landscape(data_dir, normalize)
    return FitnessLandscape(sequences, fitness)


# =============================================================================
# Metrics Result Container
# =============================================================================

@dataclass
class GB1MetricsResult:
    """Container for GB1 benchmark metrics."""

    # Exploration metrics
    high_fitness_proximity: float = float('nan')
    novelty: float = float('nan')
    batch_diversity: float = float('nan')

    # Functional metrics
    normalized_fitness_median_top128: float = float('nan')
    normalized_fitness_median_top256: float = float('nan')
    max_fitness: float = float('nan')

    # Model quality metrics
    spearman_correlation: float = float('nan')
    epistatic_correlation: float = float('nan')
    recall_high_order: float = float('nan')

    # Success metrics
    simple_regret: float = float('nan')
    global_max_found: bool = False

    # Uncertainty metrics
    miscalibration_area: float = float('nan')
    expected_calibration_error: float = float('nan')

    # Trajectories (per-round data)
    fitness_trajectory: List[float] = field(default_factory=list)
    regret_trajectory: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
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
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GB1MetricsResult':
        """Create from dictionary."""
        result = cls()
        for key, value in d.items():
            if hasattr(result, key):
                setattr(result, key, value)
        return result


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_gb1_metrics(
    queried_sequences: Sequence[str],
    queried_fitness: Sequence[float],
    all_sequences: Sequence[str],
    all_fitness: np.ndarray,
    initial_sequences: Sequence[str],
    wildtype: str = GB1_WILD_TYPE_4SITE,
    y_pred: Optional[np.ndarray] = None,
    y_std: Optional[np.ndarray] = None,
    batch_size: int = 96,
    high_fitness_percentile: float = 0.9,
    max_high_seqs: int = 128
) -> GB1MetricsResult:
    """
    Compute all GB1 benchmark metrics.

    This is the main function that should be used by all methods to ensure
    consistent metric computation.

    Parameters
    ----------
    queried_sequences : Sequence[str]
        All sequences queried during optimization (including initial)
    queried_fitness : Sequence[float]
        Fitness values of queried sequences
    all_sequences : Sequence[str]
        Complete landscape sequences
    all_fitness : np.ndarray
        Complete landscape fitness values
    initial_sequences : Sequence[str]
        Initial training set sequences
    wildtype : str, default='VDGV'
        Wild-type sequence for mutation counting
    y_pred : np.ndarray, optional
        Model predictions for all sequences (for Spearman/calibration)
    y_std : np.ndarray, optional
        Model uncertainty estimates (for calibration)
    batch_size : int, default=96
        Batch size for trajectory computation
    high_fitness_percentile : float, default=0.9
        Percentile for high-fitness sequences (0.9 = top 10%)
    max_high_seqs : int, default=128
        Maximum high-fitness sequences for proximity computation

    Returns
    -------
    GB1MetricsResult
        Computed metrics
    """
    result = GB1MetricsResult()

    queried_sequences = list(queried_sequences)
    queried_fitness = np.array(queried_fitness)
    all_fitness = np.array(all_fitness)
    initial_sequences = list(initial_sequences)

    # Global fitness bounds
    global_max = float(np.max(all_fitness))
    global_min = float(np.min(all_fitness))

    # --- Exploration Metrics ---
    result.high_fitness_proximity = high_fitness_proximity_from_landscape(
        queried_sequences, all_sequences, all_fitness,
        percentile=high_fitness_percentile,
        max_high_seqs=max_high_seqs,
        distance_fn='hamming'
    )

    # Novelty: distance of non-initial queries to initial set
    n_initial = len(initial_sequences)
    non_initial_seqs = queried_sequences[n_initial:] if len(queried_sequences) > n_initial else []
    if non_initial_seqs and initial_sequences:
        result.novelty = novelty(non_initial_seqs, initial_sequences, distance_fn='hamming')
    else:
        result.novelty = 0.0

    result.batch_diversity = batch_diversity(queried_sequences, distance_fn='hamming')

    # --- Functional Metrics ---
    result.normalized_fitness_median_top128 = normalized_fitness_median_topk(
        queried_fitness, k=128, fitness_min=global_min, fitness_max=global_max
    )
    result.normalized_fitness_median_top256 = normalized_fitness_median_topk(
        queried_fitness, k=256, fitness_min=global_min, fitness_max=global_max
    )
    result.max_fitness = float(np.max(queried_fitness))

    # --- Success Metrics ---
    result.simple_regret = simple_regret(result.max_fitness, global_max)
    result.global_max_found = (result.max_fitness >= global_max * 0.99)

    # Fitness and regret trajectories
    fitness_traj = []
    for i in range(0, len(queried_fitness), batch_size):
        batch_fitness = queried_fitness[:i + batch_size]
        fitness_traj.append(float(np.max(batch_fitness)))
    result.fitness_trajectory = fitness_traj
    result.regret_trajectory = [global_max - f for f in fitness_traj]

    # --- Model Quality Metrics (if predictions available) ---
    if y_pred is not None:
        # Create lookup for queried sequences
        queried_set = set(queried_sequences)

        # Compute on held-out sequences
        holdout_mask = np.array([seq not in queried_set for seq in all_sequences])
        n_holdout = np.sum(holdout_mask)

        if n_holdout > 100:
            holdout_true = all_fitness[holdout_mask]
            holdout_pred = y_pred[holdout_mask]

            result.spearman_correlation = spearman_correlation(holdout_pred, holdout_true)

            # Epistatic correlation (requires wildtype)
            if wildtype:
                holdout_seqs = [s for s, m in zip(all_sequences, holdout_mask) if m]
                # Note: Full epistatic computation requires more complex setup
                # For now, use spearman on high-order mutants as proxy
                high_order_mask = np.array([
                    count_mutations(seq, wildtype) >= 2 for seq in holdout_seqs
                ])
                if np.sum(high_order_mask) > 10:
                    result.epistatic_correlation = spearman_correlation(
                        holdout_pred[high_order_mask],
                        holdout_true[high_order_mask]
                    )

            # Recall of high-order mutants
            if wildtype:
                high_order_seqs = [
                    s for s, m in zip(all_sequences, holdout_mask) if m
                    and count_mutations(s, wildtype) >= 2
                ]
                if high_order_seqs:
                    # Get true top-k high-order mutants
                    high_order_fit = np.array([
                        all_fitness[all_sequences.index(s)] for s in high_order_seqs
                    ])
                    top_k = min(100, len(high_order_seqs) // 2)
                    if top_k > 0:
                        true_top_idx = np.argsort(high_order_fit)[-top_k:]
                        true_top = set([high_order_seqs[i] for i in true_top_idx])

                        # Get predicted top-k
                        high_order_pred = np.array([
                            y_pred[all_sequences.index(s)] for s in high_order_seqs
                        ])
                        pred_top_idx = np.argsort(high_order_pred)[-top_k:]
                        pred_top = set([high_order_seqs[i] for i in pred_top_idx])

                        result.recall_high_order = len(true_top & pred_top) / len(true_top)

    # --- Uncertainty Metrics (if uncertainty available) ---
    if y_pred is not None and y_std is not None:
        queried_set = set(queried_sequences)
        holdout_mask = np.array([seq not in queried_set for seq in all_sequences])

        if np.sum(holdout_mask) > 100:
            holdout_true = all_fitness[holdout_mask]
            holdout_pred = y_pred[holdout_mask]
            holdout_std = y_std[holdout_mask]

            errors = np.abs(holdout_true - holdout_pred)
            result.miscalibration_area = miscalibration_area(holdout_std, errors)

            misc, ece = regression_calibration_error(
                holdout_pred, holdout_std, holdout_true
            )
            result.expected_calibration_error = ece

    return result


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_gb1_metrics(
    run_results: List[GB1MetricsResult],
    global_max_sequence: Optional[str] = None,
    best_sequences: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate GB1 metrics across multiple runs.

    Parameters
    ----------
    run_results : List[GB1MetricsResult]
        Results from multiple runs
    global_max_sequence : str, optional
        Global maximum sequence for hit count
    best_sequences : List[str], optional
        Best sequence from each run

    Returns
    -------
    dict
        Aggregated statistics
    """
    if not run_results:
        return {}

    metric_names = [
        'high_fitness_proximity', 'novelty', 'batch_diversity',
        'normalized_fitness_median_top128', 'normalized_fitness_median_top256',
        'max_fitness', 'spearman_correlation', 'epistatic_correlation',
        'recall_high_order', 'simple_regret', 'miscalibration_area',
        'expected_calibration_error'
    ]

    aggregated = {}
    for name in metric_names:
        values = [getattr(r, name) for r in run_results]
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            aggregated[name] = {
                'mean': float(np.mean(valid_values)),
                'std': float(np.std(valid_values)),
                'min': float(np.min(valid_values)),
                'max': float(np.max(valid_values)),
            }
        else:
            aggregated[name] = {'mean': float('nan'), 'std': float('nan'),
                                'min': float('nan'), 'max': float('nan')}

    # Global max hit count
    hit_count = sum(1 for r in run_results if r.global_max_found)
    n_runs = len(run_results)
    aggregated['global_max_hit_count'] = {
        'count': hit_count,
        'rate': hit_count / n_runs if n_runs > 0 else 0.0,
        'total_runs': n_runs
    }

    return aggregated


def save_gb1_results(
    results: Union[GB1MetricsResult, Dict[str, Any]],
    output_dir: str,
    method: str,
    seed: int,
    run_id: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save GB1 benchmark results.

    Parameters
    ----------
    results : GB1MetricsResult or dict
        Metrics to save
    output_dir : str
        Output directory
    method : str
        Method name
    seed : int
        Random seed used
    run_id : int, optional
        Run identifier
    config : dict, optional
        Configuration to save

    Returns
    -------
    str
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(results, GB1MetricsResult):
        results_dict = results.to_dict()
        results_dict['fitness_trajectory'] = results.fitness_trajectory
        results_dict['regret_trajectory'] = results.regret_trajectory
    else:
        results_dict = results

    output = {
        'method': method,
        'dataset': 'GB1',
        'seed': seed,
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'metrics': results_dict,
        'config': config or {},
    }

    filename = f"metrics_seed{seed}.json"
    if run_id is not None:
        filename = f"metrics_run{run_id}_seed{seed}.json"

    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    return filepath


def print_gb1_metrics_summary(
    metrics: Union[GB1MetricsResult, Dict[str, Any]],
    prefix: str = ""
) -> None:
    """Print formatted metrics summary."""
    if isinstance(metrics, GB1MetricsResult):
        metrics = metrics.to_dict()

    print(f"{prefix}GB1 Metrics Summary:")
    print(f"{prefix}{'='*50}")

    metric_order = [
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
        'global_max_found',
    ]

    for metric in metric_order:
        if metric in metrics:
            value = metrics[metric]
            if isinstance(value, bool):
                print(f"{prefix}  {metric}: {value}")
            elif isinstance(value, float):
                if np.isnan(value):
                    print(f"{prefix}  {metric}: N/A")
                else:
                    print(f"{prefix}  {metric}: {value:.4f}")
            else:
                print(f"{prefix}  {metric}: {value}")

    print(f"{prefix}{'='*50}")
