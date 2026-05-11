"""
metrics.py - Comprehensive evaluation metrics for ALDE optimization

Implements metrics from multiple reference works:
- LatProtRL: High-Fitness Proximity (dhigh), Novelty (dinit)
- Energy Matching: Batch Diversity
- GGS: Normalized Fitness (Median Top-K)
- delta-Conservative Search: Max Fitness
- muProtein: Spearman Correlation, Epistatic Score Correlation, Recall of High-Order Mutants
- VSD: Simple Regret
- EvoPlay: Global Max Hit Count
- ALDE: Miscalibration Area
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings


# ============================================================================
# Distance Functions
# ============================================================================

def hamming_distance(s1: str, s2: str) -> int:
    """
    Compute Hamming distance between two sequences.
    Used by LatProtRL for fixed-length protein sequences.

    Args:
        s1: First sequence
        s2: Second sequence (must be same length as s1)

    Returns:
        Number of position-wise mismatches
    """
    if len(s1) != len(s2):
        raise ValueError(f"Sequences must have same length: {len(s1)} vs {len(s2)}")
    return sum(1 if c1 != c2 else 0 for c1, c2 in zip(s1, s2))


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein (edit) distance between two sequences.
    Used by Energy Matching for variable-length sequences.

    Args:
        s1: First sequence
        s2: Second sequence

    Returns:
        Minimum number of edits (insertions, deletions, substitutions)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# ============================================================================
# Exploration Metrics
# ============================================================================

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

    Reference: LatProtRL

    Args:
        generated_seqs: List of generated/queried sequences
        all_seqs: Complete list of sequences in the landscape
        all_fitness: Fitness values corresponding to all_seqs
        percentile: Percentile threshold for "high fitness" (default 0.9 = top 10%)
        max_high_seqs: Maximum number of high-fitness sequences to use
        distance_fn: 'hamming' or 'levenshtein'

    Returns:
        Median of minimum distances to high-fitness sequences
    """
    # Get high-fitness sequences (top percentile)
    threshold = np.percentile(all_fitness, percentile * 100)
    high_mask = all_fitness >= threshold
    high_seqs = [seq for seq, is_high in zip(all_seqs, high_mask) if is_high]

    # Limit to max_high_seqs for computational efficiency
    if len(high_seqs) > max_high_seqs:
        # Sort by fitness and take top max_high_seqs
        high_indices = np.where(high_mask)[0]
        high_fitness_vals = all_fitness[high_indices]
        top_indices = high_indices[np.argsort(high_fitness_vals)[-max_high_seqs:]]
        high_seqs = [all_seqs[i] for i in top_indices]

    dist_fn = hamming_distance if distance_fn == 'hamming' else levenshtein_distance

    # Compute minimum distance from each generated seq to high-fitness set
    min_distances = []
    for gen_seq in generated_seqs:
        distances = [dist_fn(gen_seq, high_seq) for high_seq in high_seqs]
        min_distances.append(min(distances) if distances else 0)

    return float(np.median(min_distances)) if min_distances else 0.0


def novelty(
    generated_seqs: List[str],
    initial_seqs: List[str],
    distance_fn: str = 'hamming'
) -> float:
    """
    Novelty (dinit): Median minimum distance of generated sequences to
    any sequence in the initial training set.

    Reference: LatProtRL

    Args:
        generated_seqs: List of generated/queried sequences
        initial_seqs: List of sequences in initial training set
        distance_fn: 'hamming' or 'levenshtein'

    Returns:
        Median of minimum distances to initial set
    """
    if not initial_seqs or not generated_seqs:
        return 0.0

    dist_fn = hamming_distance if distance_fn == 'hamming' else levenshtein_distance

    min_distances = []
    for gen_seq in generated_seqs:
        distances = [dist_fn(gen_seq, init_seq) for init_seq in initial_seqs]
        min_distances.append(min(distances))

    return float(np.median(min_distances))


def batch_diversity(
    sequences: List[str],
    distance_fn: str = 'levenshtein',
    aggregation: str = 'median'
) -> float:
    """
    Batch Diversity: Pairwise distance between all sequences in a batch.

    Reference: Energy Matching

    Args:
        sequences: List of sequences in the batch
        distance_fn: 'hamming' or 'levenshtein'
        aggregation: 'median' or 'mean'

    Returns:
        Aggregated pairwise distance
    """
    if len(sequences) < 2:
        return 0.0

    dist_fn = hamming_distance if distance_fn == 'hamming' else levenshtein_distance

    # Compute all pairwise distances
    n = len(sequences)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(dist_fn(sequences[i], sequences[j]))

    if not distances:
        return 0.0

    if aggregation == 'median':
        return float(np.median(distances))
    else:  # mean
        return float(np.mean(distances))


# ============================================================================
# Functional Metrics
# ============================================================================

def normalized_fitness_topk(
    fitness_values: np.ndarray,
    k: int = 128,
    min_fitness: Optional[float] = None,
    max_fitness: Optional[float] = None,
    aggregation: str = 'median'
) -> float:
    """
    Normalized Fitness (Median/Mean Top-K): Aggregated fitness of the top K
    sequences, normalized between 0 and 1.

    Reference: GGS

    Args:
        fitness_values: Array of fitness values for generated sequences
        k: Number of top sequences to consider
        min_fitness: Minimum fitness for normalization (uses data min if None)
        max_fitness: Maximum fitness for normalization (uses data max if None)
        aggregation: 'median' or 'mean'

    Returns:
        Normalized aggregated fitness of top-K sequences
    """
    if len(fitness_values) == 0:
        return 0.0

    # Get top K fitness values
    k = min(k, len(fitness_values))
    top_k_fitness = np.sort(fitness_values)[-k:]

    # Normalize
    if min_fitness is None:
        min_fitness = np.min(fitness_values)
    if max_fitness is None:
        max_fitness = np.max(fitness_values)

    if max_fitness == min_fitness:
        normalized = np.ones_like(top_k_fitness)
    else:
        normalized = (top_k_fitness - min_fitness) / (max_fitness - min_fitness)

    if aggregation == 'median':
        return float(np.median(normalized))
    else:
        return float(np.mean(normalized))


def max_fitness(fitness_values: np.ndarray) -> float:
    """
    Max Fitness: The absolute highest fitness value discovered.

    Reference: delta-Conservative Search

    Args:
        fitness_values: Array of fitness values for discovered sequences

    Returns:
        Maximum fitness value
    """
    if len(fitness_values) == 0:
        return 0.0
    return float(np.max(fitness_values))


# ============================================================================
# Model Quality Metrics
# ============================================================================

def spearman_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Spearman Rank Correlation (rho): Correlation between predicted and true fitness.

    Reference: muProtein

    Args:
        y_true: True fitness values
        y_pred: Predicted fitness values

    Returns:
        Spearman correlation coefficient
    """
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr, _ = stats.spearmanr(y_true, y_pred)

    return float(corr) if not np.isnan(corr) else 0.0


def epistatic_score_correlation(
    sequences: List[str],
    fitness_values: np.ndarray,
    predicted_values: np.ndarray,
    wildtype: str
) -> float:
    """
    Epistatic Score Correlation: Spearman correlation between predicted and
    observed non-additive mutational effects.

    Reference: muProtein

    Epistasis = observed_fitness - sum(single_mutant_effects)

    Args:
        sequences: List of sequences
        fitness_values: True fitness values
        predicted_values: Predicted fitness values
        wildtype: Wild-type sequence for reference

    Returns:
        Spearman correlation of epistatic scores
    """
    if len(sequences) < 10:
        return 0.0

    # Build single mutant effect dictionary
    single_mutant_effects_true = {}
    single_mutant_effects_pred = {}
    wt_fitness = None
    wt_pred = None

    for seq, fit, pred in zip(sequences, fitness_values, predicted_values):
        n_mutations = hamming_distance(seq, wildtype)
        if n_mutations == 0:
            wt_fitness = fit
            wt_pred = pred
        elif n_mutations == 1:
            # Find the mutation
            for i, (wt_aa, mut_aa) in enumerate(zip(wildtype, seq)):
                if wt_aa != mut_aa:
                    key = (i, mut_aa)
                    single_mutant_effects_true[key] = fit
                    single_mutant_effects_pred[key] = pred
                    break

    if wt_fitness is None or len(single_mutant_effects_true) == 0:
        return 0.0

    # Compute epistatic scores for multi-mutants
    epistasis_true = []
    epistasis_pred = []

    for seq, fit, pred in zip(sequences, fitness_values, predicted_values):
        n_mutations = hamming_distance(seq, wildtype)
        if n_mutations <= 1:
            continue

        # Get mutations
        mutations = []
        for i, (wt_aa, mut_aa) in enumerate(zip(wildtype, seq)):
            if wt_aa != mut_aa:
                mutations.append((i, mut_aa))

        # Check if all single mutants are known
        if not all(m in single_mutant_effects_true for m in mutations):
            continue

        # Compute additive expectation
        additive_true = wt_fitness + sum(
            single_mutant_effects_true[m] - wt_fitness for m in mutations
        )
        additive_pred = wt_pred + sum(
            single_mutant_effects_pred[m] - wt_pred for m in mutations
        )

        # Epistasis = observed - additive
        epistasis_true.append(fit - additive_true)
        epistasis_pred.append(pred - additive_pred)

    if len(epistasis_true) < 5:
        return 0.0

    return spearman_correlation(np.array(epistasis_true), np.array(epistasis_pred))


def recall_high_order_mutants(
    sequences: List[str],
    fitness_values: np.ndarray,
    predicted_values: np.ndarray,
    wildtype: str,
    min_mutations: int = 2,
    top_k: int = 100
) -> float:
    """
    Recall of High-Order Mutants: Percentage of true top multi-point mutants
    correctly identified by the model.

    Reference: muProtein

    Args:
        sequences: List of sequences
        fitness_values: True fitness values
        predicted_values: Predicted fitness values
        wildtype: Wild-type sequence for reference
        min_mutations: Minimum number of mutations to be considered "high-order"
        top_k: Number of top sequences to consider

    Returns:
        Recall (0-1) of top high-order mutants
    """
    # Filter to high-order mutants
    high_order_mask = np.array([
        hamming_distance(seq, wildtype) >= min_mutations
        for seq in sequences
    ])

    if np.sum(high_order_mask) < top_k:
        top_k = max(1, np.sum(high_order_mask) // 2)

    if top_k == 0:
        return 0.0

    # Get indices of high-order mutants
    high_order_indices = np.where(high_order_mask)[0]
    high_order_true = fitness_values[high_order_indices]
    high_order_pred = predicted_values[high_order_indices]

    # Get top-k by true fitness
    true_top_k_idx = set(high_order_indices[np.argsort(high_order_true)[-top_k:]])

    # Get top-k by predicted fitness
    pred_top_k_idx = set(high_order_indices[np.argsort(high_order_pred)[-top_k:]])

    # Compute recall
    recall = len(true_top_k_idx & pred_top_k_idx) / len(true_top_k_idx)

    return float(recall)


# ============================================================================
# Success Metrics
# ============================================================================

def simple_regret(
    best_found: float,
    global_max: float
) -> float:
    """
    Simple Regret: Gap between true global optimum and best design found.

    Reference: VSD (Variational Search)

    Args:
        best_found: Best fitness value found by the algorithm
        global_max: True global maximum fitness

    Returns:
        Regret value (>= 0, lower is better)
    """
    return float(global_max - best_found)


def simple_regret_trajectory(
    fitness_trajectory: np.ndarray,
    global_max: float
) -> np.ndarray:
    """
    Simple Regret Trajectory: Regret at each round.

    Args:
        fitness_trajectory: Array of best fitness found at each round
        global_max: True global maximum fitness

    Returns:
        Array of regret values per round
    """
    cumulative_best = np.maximum.accumulate(fitness_trajectory)
    return global_max - cumulative_best


def global_max_hit_count(
    run_max_fitness: List[float],
    global_max: float,
    tolerance: float = 0.01
) -> Tuple[int, float]:
    """
    Global Max Hit Count: Number of runs that successfully found the global maximum.

    Reference: EvoPlay

    Args:
        run_max_fitness: List of maximum fitness achieved in each run
        global_max: True global maximum fitness
        tolerance: Relative tolerance for considering a hit (default 1%)

    Returns:
        Tuple of (hit_count, hit_rate)
    """
    threshold = global_max * (1 - tolerance)
    hit_count = sum(1 for max_fit in run_max_fitness if max_fit >= threshold)
    hit_rate = hit_count / len(run_max_fitness) if run_max_fitness else 0.0

    return hit_count, float(hit_rate)


# ============================================================================
# Uncertainty Metrics
# ============================================================================

def miscalibration_area(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Miscalibration Area: Area between calibration curve and ideal diagonal.
    Measures if model uncertainty (sigma) matches actual prediction error.

    Reference: ALDE

    For a well-calibrated model, X% of predictions should fall within the
    X% confidence interval.

    Args:
        y_true: True fitness values
        y_pred: Mean predictions
        y_std: Predicted standard deviations (uncertainty)
        n_bins: Number of bins for calibration curve

    Returns:
        Miscalibration area (0 = perfectly calibrated, 1 = maximally miscalibrated)
    """
    if len(y_true) == 0 or np.all(y_std == 0):
        return 1.0

    # Compute z-scores
    z_scores = np.abs(y_true - y_pred) / (y_std + 1e-10)

    # Expected confidence levels
    expected_confidences = np.linspace(0.1, 1.0, n_bins)

    # For each expected confidence level, compute observed coverage
    observed_coverages = []
    for conf in expected_confidences:
        # Z-score threshold for this confidence level (assuming normal distribution)
        z_threshold = stats.norm.ppf((1 + conf) / 2)
        # Fraction of predictions within this confidence interval
        coverage = np.mean(z_scores <= z_threshold)
        observed_coverages.append(coverage)

    observed_coverages = np.array(observed_coverages)

    # Miscalibration area = mean absolute difference from diagonal
    miscal_area = np.mean(np.abs(observed_coverages - expected_confidences))

    return float(miscal_area)


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Expected Calibration Error (ECE): Weighted average of calibration errors.

    Args:
        y_true: True fitness values
        y_pred: Mean predictions
        y_std: Predicted standard deviations
        n_bins: Number of bins

    Returns:
        ECE value
    """
    if len(y_true) == 0 or np.all(y_std == 0):
        return 1.0

    # Sort by uncertainty
    sorted_indices = np.argsort(y_std)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    y_std_sorted = y_std[sorted_indices]

    # Bin by uncertainty level
    bin_size = len(y_true) // n_bins
    if bin_size == 0:
        bin_size = 1

    ece = 0.0
    total_samples = 0

    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size if i < n_bins - 1 else len(y_true)

        if start_idx >= len(y_true):
            break

        bin_true = y_true_sorted[start_idx:end_idx]
        bin_pred = y_pred_sorted[start_idx:end_idx]
        bin_std = y_std_sorted[start_idx:end_idx]

        # Mean predicted std (expected error)
        expected_error = np.mean(bin_std)
        # Actual RMSE (observed error)
        observed_error = np.sqrt(np.mean((bin_true - bin_pred) ** 2))

        bin_weight = len(bin_true)
        ece += bin_weight * np.abs(expected_error - observed_error)
        total_samples += bin_weight

    return float(ece / total_samples) if total_samples > 0 else 1.0


# ============================================================================
# Data Classes for Results
# ============================================================================

@dataclass
class MetricsResult:
    """Container for all computed metrics."""

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
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame([self.to_dict()])


# ============================================================================
# Utility Functions
# ============================================================================

def indices_to_sequences(
    indices: Union[Tensor, np.ndarray],
    all_sequences: List[str]
) -> List[str]:
    """Convert indices to sequences."""
    if isinstance(indices, Tensor):
        indices = indices.cpu().numpy().astype(int)
    return [all_sequences[i] for i in indices]


def load_landscape_data(
    protein: str,
    data_dir: str = "data"
) -> Tuple[List[str], np.ndarray]:
    """
    Load complete fitness landscape data.

    Args:
        protein: Protein name (e.g., 'GB1')
        data_dir: Base data directory

    Returns:
        Tuple of (sequences, fitness_values)
    """
    import os

    # Try different file names for compatibility
    fitness_path = f"{data_dir}/{protein}/fitness.csv"
    data_path = f"{data_dir}/{protein}/data.csv"

    if os.path.exists(fitness_path):
        fitness_df = pd.read_csv(fitness_path)
    elif os.path.exists(data_path):
        fitness_df = pd.read_csv(data_path)
    else:
        raise FileNotFoundError(f"No fitness data found at {fitness_path} or {data_path}")

    # Handle different column names (Combo vs AACombo)
    if 'Combo' in fitness_df.columns:
        sequences = fitness_df['Combo'].tolist()
    elif 'AACombo' in fitness_df.columns:
        sequences = fitness_df['AACombo'].tolist()
    else:
        raise ValueError("Sequence column ('Combo' or 'AACombo') not found in data file")

    fitness = fitness_df['fitness'].values
    return sequences, fitness


def compute_all_metrics(
    queried_indices: Union[Tensor, np.ndarray],
    all_sequences: List[str],
    all_fitness: np.ndarray,
    initial_indices: Union[Tensor, np.ndarray],
    y_pred: Optional[np.ndarray] = None,
    y_std: Optional[np.ndarray] = None,
    batch_size: int = 96,
    wildtype: Optional[str] = None
) -> MetricsResult:
    """
    Compute all metrics for an optimization run.

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
    if isinstance(queried_indices, Tensor):
        queried_indices = queried_indices.cpu().numpy().astype(int)
    if isinstance(initial_indices, Tensor):
        initial_indices = initial_indices.cpu().numpy().astype(int)

    # Get sequences
    queried_seqs = [all_sequences[i] for i in queried_indices]
    initial_seqs = [all_sequences[i] for i in initial_indices]
    queried_fitness = all_fitness[queried_indices]

    # Global max
    global_max = np.max(all_fitness)
    global_min = np.min(all_fitness)

    # --- Exploration Metrics ---
    result.high_fitness_proximity = high_fitness_proximity(
        queried_seqs, all_sequences, all_fitness,
        percentile=0.9, distance_fn='hamming'
    )

    # Novelty: distance of non-initial queries to initial set
    non_initial_seqs = queried_seqs[len(initial_indices):]
    if non_initial_seqs:
        result.novelty = novelty(non_initial_seqs, initial_seqs, distance_fn='hamming')

    result.batch_diversity = batch_diversity(queried_seqs, distance_fn='hamming', aggregation='median')

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

    # Regret trajectory
    fitness_traj = []
    for i in range(0, len(queried_fitness), batch_size):
        batch_fitness = queried_fitness[:i + batch_size]
        fitness_traj.append(np.max(batch_fitness))
    result.fitness_trajectory = fitness_traj
    result.regret_trajectory = list(simple_regret_trajectory(np.array(fitness_traj), global_max))

    # --- Model Quality Metrics (if predictions available) ---
    if y_pred is not None:
        # Compute on held-out set (sequences not in queried set)
        queried_set = set(queried_indices)
        holdout_mask = np.array([i not in queried_set for i in range(len(all_fitness))])

        if np.sum(holdout_mask) > 100:
            holdout_true = all_fitness[holdout_mask]
            holdout_pred = y_pred[holdout_mask]

            result.spearman_correlation = spearman_correlation(holdout_true, holdout_pred)

            if wildtype is not None:
                holdout_seqs = [all_sequences[i] for i, m in enumerate(holdout_mask) if m]
                result.epistatic_correlation = epistatic_score_correlation(
                    holdout_seqs, holdout_true, holdout_pred, wildtype
                )
                result.recall_high_order = recall_high_order_mutants(
                    holdout_seqs, holdout_true, holdout_pred, wildtype,
                    min_mutations=2, top_k=100
                )

    # --- Uncertainty Metrics (if uncertainty available) ---
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
        'expected_calibration_error'
    ]

    aggregated = {}
    for name in metrics_names:
        values = [getattr(r, name) for r in run_results]
        aggregated[name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }

    # Global max hit count
    hit_count = sum(1 for r in run_results if r.global_max_found)
    aggregated['global_max_hit_count'] = {
        'count': hit_count,
        'rate': hit_count / len(run_results)
    }

    return aggregated
