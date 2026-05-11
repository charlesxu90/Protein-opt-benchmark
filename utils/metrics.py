"""
Benchmark Metrics for Protein Optimization

This module provides standardized metric implementations for benchmarking
protein optimization methods. All metrics are designed to be consistent
across different methods (ALDE, EvoPlay, AdaLead, LatProtRL, AICE,
δ-Conservative Search, AlphaVariant).

Metrics:
---------
1. High-Fitness Proximity (dhigh): Median minimum distance to top 10% fitness sequences
2. Novelty (dinit): Median minimum Levenshtein distance to initial training set
3. Batch Diversity: Median pairwise Levenshtein distance within generated batch
4. Normalized Fitness (Median Top-K): Median fitness of top K sequences, normalized [0,1]
5. Max Fitness: Absolute highest fitness discovered
6. Spearman Rank Correlation (ρ): Correlation between predicted and true fitness
7. Epistatic Score Correlation: Spearman correlation for non-additive effects
8. Recall of High-Order Mutants: Percentage of true top multi-point mutants identified
9. Simple Regret: Gap between global optimum and best found at round t
10. Global Max Hit Count: Number of runs finding the absolute global maximum
11. Miscalibration Area: Area between calibration curve and diagonal
12. Expected Calibration Error (ECE): Weighted average of calibration errors

References:
-----------
- LatProtRL: dhigh, dinit metrics
- Energy Matching: Batch Diversity
- GGS: Normalized Fitness
- δ-Conservative Search: Max Fitness
- μProtein: Spearman, Epistatic, Recall
- VSD: Simple Regret
- EvoPlay: Global Max Hit Count
- ALDE: Miscalibration Area, ECE
"""

from typing import List, Sequence, Optional, Tuple, Union
import numpy as np
from scipy import stats
from itertools import combinations


# =============================================================================
# Distance Metrics
# =============================================================================

def levenshtein_distance(seq1: str, seq2: str) -> int:
    """
    Compute the Levenshtein (edit) distance between two sequences.

    The Levenshtein distance is the minimum number of single-character edits
    (insertions, deletions, or substitutions) required to transform one
    sequence into another.

    Parameters
    ----------
    seq1 : str
        First sequence
    seq2 : str
        Second sequence

    Returns
    -------
    int
        Levenshtein distance between the sequences

    Examples
    --------
    >>> levenshtein_distance("ACGT", "ACGT")
    0
    >>> levenshtein_distance("ACGT", "ACGTT")
    1
    >>> levenshtein_distance("ACGT", "TGCA")
    4
    """
    if len(seq1) < len(seq2):
        return levenshtein_distance(seq2, seq1)

    if len(seq2) == 0:
        return len(seq1)

    # Use two-row optimization for space efficiency
    previous_row = range(len(seq2) + 1)
    for i, c1 in enumerate(seq1):
        current_row = [i + 1]
        for j, c2 in enumerate(seq2):
            # Cost is 0 if characters match, 1 otherwise
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def hamming_distance(seq1: str, seq2: str) -> int:
    """
    Compute the Hamming distance between two sequences of equal length.

    The Hamming distance counts the number of positions at which the
    corresponding characters are different.

    Parameters
    ----------
    seq1 : str
        First sequence
    seq2 : str
        Second sequence (must be same length as seq1)

    Returns
    -------
    int
        Hamming distance between the sequences

    Raises
    ------
    ValueError
        If sequences have different lengths

    Examples
    --------
    >>> hamming_distance("ACGT", "ACGT")
    0
    >>> hamming_distance("ACGT", "TGCA")
    4
    """
    if len(seq1) != len(seq2):
        raise ValueError(
            f"Sequences must have equal length for Hamming distance. "
            f"Got lengths {len(seq1)} and {len(seq2)}"
        )
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))


def min_distance_to_set(
    sequence: str,
    reference_set: Sequence[str],
    distance_fn: str = 'levenshtein'
) -> int:
    """
    Compute minimum distance from a sequence to any sequence in a reference set.

    Parameters
    ----------
    sequence : str
        Query sequence
    reference_set : Sequence[str]
        Set of reference sequences
    distance_fn : str, default='levenshtein'
        Distance function to use: 'levenshtein' or 'hamming'

    Returns
    -------
    int
        Minimum distance to any sequence in reference set

    Raises
    ------
    ValueError
        If reference_set is empty or distance_fn is invalid
    """
    if len(reference_set) == 0:
        raise ValueError("Reference set cannot be empty")

    if distance_fn == 'levenshtein':
        dist_func = levenshtein_distance
    elif distance_fn == 'hamming':
        dist_func = hamming_distance
    else:
        raise ValueError(f"Unknown distance function: {distance_fn}")

    return min(dist_func(sequence, ref) for ref in reference_set)


# =============================================================================
# Core Benchmark Metrics
# =============================================================================

def high_fitness_proximity(
    generated_sequences: Sequence[str],
    high_fitness_sequences: Sequence[str],
    distance_fn: str = 'levenshtein',
    max_high_seqs: int = 128
) -> float:
    """
    Compute High-Fitness Proximity (d_high) metric.

    Measures the median minimum distance from generated sequences to the
    top fitness sequences (typically top 10%) in the ground-truth dataset.
    Lower values indicate better coverage of the high-fitness region.

    Reference: LatProtRL

    Parameters
    ----------
    generated_sequences : Sequence[str]
        Sequences generated by the optimization method
    high_fitness_sequences : Sequence[str]
        Top fitness sequences from the ground-truth landscape
        (typically top 10% by fitness). If more than max_high_seqs,
        will be truncated to max_high_seqs for efficiency.
    distance_fn : str, default='levenshtein'
        Distance function: 'levenshtein' or 'hamming'
    max_high_seqs : int, default=128
        Maximum number of high-fitness sequences to consider.
        Set to None to use all provided sequences.

    Returns
    -------
    float
        Median of minimum distances to high-fitness sequences

    Examples
    --------
    >>> gen_seqs = ["ACGT", "TGCA"]
    >>> high_fit_seqs = ["ACGT", "ACGC"]
    >>> high_fitness_proximity(gen_seqs, high_fit_seqs)
    0.0
    """
    if len(generated_sequences) == 0:
        return float('nan')
    if len(high_fitness_sequences) == 0:
        raise ValueError("high_fitness_sequences cannot be empty")

    # Limit high fitness sequences for computational efficiency
    high_seqs = list(high_fitness_sequences)
    if max_high_seqs is not None and len(high_seqs) > max_high_seqs:
        # Take the last max_high_seqs (assuming sorted by fitness ascending)
        high_seqs = high_seqs[-max_high_seqs:]

    min_distances = [
        min_distance_to_set(seq, high_seqs, distance_fn)
        for seq in generated_sequences
    ]
    return float(np.median(min_distances))


def high_fitness_proximity_from_landscape(
    generated_sequences: Sequence[str],
    all_sequences: Sequence[str],
    all_fitness: Sequence[float],
    percentile: float = 0.9,
    max_high_seqs: int = 128,
    distance_fn: str = 'levenshtein'
) -> float:
    """
    Compute High-Fitness Proximity directly from landscape data.

    This version matches ALDE's implementation, computing high-fitness
    sequences from the provided landscape.

    Reference: LatProtRL

    Parameters
    ----------
    generated_sequences : Sequence[str]
        Sequences generated by the optimization method
    all_sequences : Sequence[str]
        Complete list of sequences in the landscape
    all_fitness : Sequence[float]
        Fitness values corresponding to all_sequences
    percentile : float, default=0.9
        Percentile threshold for "high fitness" (0.9 = top 10%)
    max_high_seqs : int, default=128
        Maximum number of high-fitness sequences to use
    distance_fn : str, default='levenshtein'
        Distance function: 'levenshtein' or 'hamming'

    Returns
    -------
    float
        Median of minimum distances to high-fitness sequences
    """
    if len(generated_sequences) == 0:
        return float('nan')

    all_fitness = np.array(all_fitness)

    # Get high-fitness sequences (top percentile)
    threshold = np.percentile(all_fitness, percentile * 100)
    high_mask = all_fitness >= threshold
    high_indices = np.where(high_mask)[0]

    # If more than max_high_seqs, take the top ones
    if len(high_indices) > max_high_seqs:
        high_fitness_vals = all_fitness[high_indices]
        top_local_indices = np.argsort(high_fitness_vals)[-max_high_seqs:]
        high_indices = high_indices[top_local_indices]

    high_seqs = [all_sequences[i] for i in high_indices]

    if distance_fn == 'hamming':
        dist_func = hamming_distance
    else:
        dist_func = levenshtein_distance

    # Compute minimum distance from each generated seq to high-fitness set
    min_distances = []
    for gen_seq in generated_sequences:
        distances = [dist_func(gen_seq, high_seq) for high_seq in high_seqs]
        min_distances.append(min(distances) if distances else 0)

    return float(np.median(min_distances)) if min_distances else 0.0


def novelty(
    generated_sequences: Sequence[str],
    initial_training_set: Sequence[str],
    distance_fn: str = 'levenshtein'
) -> float:
    """
    Compute Novelty (d_init) metric.

    Measures the median minimum Levenshtein distance of generated sequences
    to any sequence in the initial training set. Higher values indicate
    more novel/diverse exploration away from the starting point.

    Reference: LatProtRL

    Parameters
    ----------
    generated_sequences : Sequence[str]
        Sequences generated by the optimization method
    initial_training_set : Sequence[str]
        Initial training sequences provided to the method
    distance_fn : str, default='levenshtein'
        Distance function: 'levenshtein' or 'hamming'

    Returns
    -------
    float
        Median of minimum distances to initial training set

    Examples
    --------
    >>> gen_seqs = ["XXXX", "YYYY"]
    >>> init_seqs = ["ACGT", "TGCA"]
    >>> novelty(gen_seqs, init_seqs)  # High novelty
    4.0
    """
    if len(generated_sequences) == 0:
        return float('nan')
    if len(initial_training_set) == 0:
        raise ValueError("initial_training_set cannot be empty")

    min_distances = [
        min_distance_to_set(seq, initial_training_set, distance_fn)
        for seq in generated_sequences
    ]
    return float(np.median(min_distances))


def batch_diversity(
    sequences: Sequence[str],
    distance_fn: str = 'levenshtein'
) -> float:
    """
    Compute Batch Diversity metric.

    Measures the median pairwise Levenshtein distance between all sequences
    in a generated batch. Higher values indicate better mode coverage and
    diversity within the batch.

    Reference: Energy Matching

    Parameters
    ----------
    sequences : Sequence[str]
        Batch of generated sequences
    distance_fn : str, default='levenshtein'
        Distance function: 'levenshtein' or 'hamming'

    Returns
    -------
    float
        Median pairwise distance within the batch.
        Returns nan if batch has fewer than 2 sequences.

    Examples
    --------
    >>> seqs = ["AAAA", "BBBB", "CCCC"]
    >>> batch_diversity(seqs)  # All different
    4.0
    """
    if len(sequences) < 2:
        return float('nan')

    if distance_fn == 'levenshtein':
        dist_func = levenshtein_distance
    elif distance_fn == 'hamming':
        dist_func = hamming_distance
    else:
        raise ValueError(f"Unknown distance function: {distance_fn}")

    pairwise_distances = [
        dist_func(seq1, seq2)
        for seq1, seq2 in combinations(sequences, 2)
    ]
    return float(np.median(pairwise_distances))


def normalized_fitness_median_topk(
    fitness_values: Sequence[float],
    k: int = 128,
    fitness_min: Optional[float] = None,
    fitness_max: Optional[float] = None
) -> float:
    """
    Compute Normalized Fitness (Median Top-K) metric.

    Calculates the median fitness of the top K generated sequences,
    normalized to the [0, 1] range based on the landscape's fitness bounds.

    Reference: GGS

    Parameters
    ----------
    fitness_values : Sequence[float]
        Fitness values of generated sequences
    k : int, default=128
        Number of top sequences to consider (commonly 128 or 256)
    fitness_min : float, optional
        Minimum fitness in the landscape (for normalization)
        If None, uses min of provided fitness_values
    fitness_max : float, optional
        Maximum fitness in the landscape (for normalization)
        If None, uses max of provided fitness_values

    Returns
    -------
    float
        Normalized median fitness of top K sequences (range [0, 1])

    Examples
    --------
    >>> fitness = [0.1, 0.5, 0.8, 0.9, 1.0]
    >>> normalized_fitness_median_topk(fitness, k=3, fitness_min=0.0, fitness_max=1.0)
    0.9
    """
    if len(fitness_values) == 0:
        return float('nan')

    fitness_array = np.array(fitness_values)

    # Get top-k fitness values
    top_k = np.sort(fitness_array)[-k:] if len(fitness_array) >= k else np.sort(fitness_array)
    median_fitness = np.median(top_k)

    # Normalize to [0, 1]
    if fitness_min is None:
        fitness_min = float(np.min(fitness_array))
    if fitness_max is None:
        fitness_max = float(np.max(fitness_array))

    if fitness_max == fitness_min:
        return 1.0 if median_fitness == fitness_max else 0.0

    normalized = (median_fitness - fitness_min) / (fitness_max - fitness_min)
    return float(np.clip(normalized, 0.0, 1.0))


def max_fitness(fitness_values: Sequence[float]) -> float:
    """
    Compute Max Fitness metric.

    Returns the absolute highest experimental fitness value discovered
    by the optimization method during the search.

    Reference: δ-Conservative Search

    Parameters
    ----------
    fitness_values : Sequence[float]
        Fitness values of discovered sequences

    Returns
    -------
    float
        Maximum fitness value

    Examples
    --------
    >>> max_fitness([0.1, 0.5, 0.8, 0.9, 1.0])
    1.0
    """
    if len(fitness_values) == 0:
        return float('nan')
    return float(np.max(fitness_values))


def spearman_correlation(
    predicted: Sequence[float],
    ground_truth: Sequence[float]
) -> float:
    """
    Compute Spearman Rank Correlation (ρ) metric.

    Measures the correlation between predicted fitness and true fitness
    for variants in the landscape. Higher values indicate better ranking
    agreement between predictions and ground truth.

    Reference: μProtein

    Parameters
    ----------
    predicted : Sequence[float]
        Predicted fitness values from the model
    ground_truth : Sequence[float]
        True fitness values from experimental data

    Returns
    -------
    float
        Spearman correlation coefficient (range [-1, 1])

    Examples
    --------
    >>> pred = [1.0, 2.0, 3.0, 4.0, 5.0]
    >>> true = [1.1, 2.1, 2.9, 4.0, 5.1]
    >>> spearman_correlation(pred, true)  # Close to 1.0
    0.9999...
    """
    if len(predicted) != len(ground_truth):
        raise ValueError(
            f"Arrays must have same length. Got {len(predicted)} and {len(ground_truth)}"
        )
    if len(predicted) < 2:
        return float('nan')

    correlation, _ = stats.spearmanr(predicted, ground_truth)
    return float(correlation)


def recall_high_order_mutants_from_seqs(
    sequences: Sequence[str],
    fitness_values: Sequence[float],
    predicted_values: Sequence[float],
    wildtype: str,
    min_mutations: int = 2,
    top_k: int = 100,
) -> float:
    """Sequence-aware recall of high-order mutants.

    Sequence-aware variant of `recall_high_order_mutants`: identifies the
    top-k true and top-k predicted variants among entries with
    `>=min_mutations` differences from `wildtype`, returns intersection size
    over true set.

    Returns 0.0 if no qualifying entries exist or both ranking sets are empty.
    """
    fit_arr = np.asarray(fitness_values, dtype=float)
    pred_arr = np.asarray(predicted_values, dtype=float)
    high_order_mask = np.array([
        sum(s != w for s, w in zip(seq, wildtype)) >= min_mutations
        if len(seq) == len(wildtype) else False
        for seq in sequences
    ])
    n_high = int(np.sum(high_order_mask))
    if n_high == 0:
        return 0.0
    if n_high < top_k:
        top_k = max(1, n_high // 2)
    high_order_indices = np.where(high_order_mask)[0]
    if top_k == 0 or len(high_order_indices) == 0:
        return 0.0

    true_vals = fit_arr[high_order_indices]
    pred_vals = pred_arr[high_order_indices]
    true_top = set(high_order_indices[np.argsort(true_vals)[-top_k:]])
    pred_top = set(high_order_indices[np.argsort(pred_vals)[-top_k:]])
    if not true_top:
        return 0.0
    return len(true_top & pred_top) / len(true_top)


def epistatic_score_correlation(
    sequences: Sequence[str],
    fitness_values: Sequence[float],
    predicted_values: Sequence[float],
    wildtype: str,
) -> float:
    """Sequence-aware epistatic score correlation.

    Computes single-mutant effects from (sequence, fitness) pairs, then derives
    epistasis = observed_fitness - additive_prediction for every multi-mutant
    where all single-mutant effects are known. Returns Spearman ρ between the
    observed and predicted epistasis vectors.

    This is the 4-arg variant used by alphavariant and AiCE; for cases where
    the caller already has epistasis vectors, use `epistatic_correlation`.

    Returns 0.0 if fewer than 5 multi-mutants have all single-mutant effects
    available, or if no wild-type / single-mutant data was provided.
    """
    fit_arr = np.asarray(fitness_values, dtype=float)
    pred_arr = np.asarray(predicted_values, dtype=float)
    if len(sequences) < 10:
        return 0.0

    single_true: dict = {}
    single_pred: dict = {}
    wt_fitness = None
    wt_pred = None

    for seq, fit, pred in zip(sequences, fit_arr, pred_arr):
        if len(seq) != len(wildtype):
            continue
        n_mut = sum(s != w for s, w in zip(seq, wildtype))
        if n_mut == 0:
            wt_fitness = float(fit)
            wt_pred = float(pred)
        elif n_mut == 1:
            for i, (wt_aa, mut_aa) in enumerate(zip(wildtype, seq)):
                if wt_aa != mut_aa:
                    single_true[(i, mut_aa)] = float(fit)
                    single_pred[(i, mut_aa)] = float(pred)
                    break

    if wt_fitness is None or not single_true:
        return 0.0

    epi_true: List[float] = []
    epi_pred: List[float] = []
    for seq, fit, pred in zip(sequences, fit_arr, pred_arr):
        if len(seq) != len(wildtype):
            continue
        muts = [(i, mut) for i, (wt_aa, mut) in enumerate(zip(wildtype, seq))
                if wt_aa != mut]
        if len(muts) < 2:
            continue
        if not all(m in single_true for m in muts):
            continue
        add_true = wt_fitness + sum(single_true[m] - wt_fitness for m in muts)
        add_pred = wt_pred + sum(single_pred[m] - wt_pred for m in muts)
        epi_true.append(float(fit) - add_true)
        epi_pred.append(float(pred) - add_pred)

    if len(epi_true) < 5:
        return 0.0
    return spearman_correlation(np.asarray(epi_true), np.asarray(epi_pred))


def epistatic_correlation(
    predicted_epistasis: Sequence[float],
    observed_epistasis: Sequence[float]
) -> float:
    """
    Compute Epistatic Score (S_epi) Correlation metric.

    Measures the Spearman correlation between predicted and observed
    non-additive mutational effects (epistasis). Epistasis is computed as:
        S_epi = fitness(AB) - fitness(A) - fitness(B) + fitness(WT)

    Reference: μProtein

    Parameters
    ----------
    predicted_epistasis : Sequence[float]
        Predicted epistatic scores from the model
    observed_epistasis : Sequence[float]
        Observed epistatic scores from experimental data

    Returns
    -------
    float
        Spearman correlation coefficient for epistatic effects

    Notes
    -----
    Epistatic effects should be pre-computed as the difference between
    the observed double-mutant fitness and the expected additive effect.
    """
    return spearman_correlation(predicted_epistasis, observed_epistasis)


def compute_epistasis(
    fitness_wt: float,
    fitness_a: float,
    fitness_b: float,
    fitness_ab: float
) -> float:
    """
    Compute epistatic score for a double mutant.

    Epistasis measures the deviation from additivity:
        S_epi = fitness(AB) - fitness(A) - fitness(B) + fitness(WT)

    Parameters
    ----------
    fitness_wt : float
        Fitness of wild-type sequence
    fitness_a : float
        Fitness of single mutant A
    fitness_b : float
        Fitness of single mutant B
    fitness_ab : float
        Fitness of double mutant AB

    Returns
    -------
    float
        Epistatic score (positive = positive epistasis, negative = negative epistasis)
    """
    return fitness_ab - fitness_a - fitness_b + fitness_wt


def recall_high_order_mutants(
    predicted_top_mutants: Sequence[str],
    true_top_mutants: Sequence[str],
    min_mutations: int = 2
) -> float:
    """
    Compute Recall of High-Order Mutants metric.

    Measures the percentage of true top-functioning multi-point mutants
    correctly identified by the optimization framework.

    Reference: μProtein

    Parameters
    ----------
    predicted_top_mutants : Sequence[str]
        Top mutant sequences predicted/generated by the method
    true_top_mutants : Sequence[str]
        True top mutant sequences from the ground-truth landscape
    min_mutations : int, default=2
        Minimum number of mutations to consider as "high-order"

    Returns
    -------
    float
        Recall score (range [0, 1])

    Notes
    -----
    The sequences in true_top_mutants are filtered based on min_mutations
    relative to wild-type if needed. The recall measures what fraction
    of ground-truth high-order mutants were found.
    """
    if len(true_top_mutants) == 0:
        return float('nan')

    predicted_set = set(predicted_top_mutants)
    true_set = set(true_top_mutants)

    intersection = predicted_set & true_set
    return float(len(intersection) / len(true_set))


def simple_regret(
    best_found_fitness: float,
    global_optimum_fitness: float
) -> float:
    """
    Compute Simple Regret (r_t) metric.

    Measures the gap between the fitness of the true global optimum
    and the best design found by the algorithm at round t.
    Lower values indicate better performance.

    Reference: VSD

    Parameters
    ----------
    best_found_fitness : float
        Best fitness value found by the algorithm
    global_optimum_fitness : float
        Fitness of the global optimum in the landscape

    Returns
    -------
    float
        Simple regret (non-negative)

    Examples
    --------
    >>> simple_regret(best_found_fitness=0.95, global_optimum_fitness=1.0)
    0.05
    """
    return float(max(0.0, global_optimum_fitness - best_found_fitness))


def global_max_hit_count(
    best_sequences_per_run: Sequence[str],
    global_max_sequence: str
) -> int:
    """
    Compute Global Max Hit Count metric.

    Counts the number of independent runs that successfully identified
    the absolute global maximum sequence (e.g., TEMH in GB1).
    This metric is typically used for small fitness landscapes like GB1.

    Reference: EvoPlay

    Parameters
    ----------
    best_sequences_per_run : Sequence[str]
        Best sequence found in each independent run
    global_max_sequence : str
        The global maximum sequence in the landscape

    Returns
    -------
    int
        Number of runs that found the global maximum

    Examples
    --------
    >>> runs = ["ACGT", "TEMH", "XXXX", "TEMH", "YYYY"]
    >>> global_max_hit_count(runs, "TEMH")
    2
    """
    return sum(1 for seq in best_sequences_per_run if seq == global_max_sequence)


def global_max_hit_rate(
    best_sequences_per_run: Sequence[str],
    global_max_sequence: str
) -> float:
    """
    Compute Global Max Hit Rate metric.

    Returns the fraction of runs that successfully identified the global maximum.

    Parameters
    ----------
    best_sequences_per_run : Sequence[str]
        Best sequence found in each independent run
    global_max_sequence : str
        The global maximum sequence in the landscape

    Returns
    -------
    float
        Fraction of runs that found the global maximum (range [0, 1])
    """
    if len(best_sequences_per_run) == 0:
        return float('nan')
    count = global_max_hit_count(best_sequences_per_run, global_max_sequence)
    return float(count / len(best_sequences_per_run))


# =============================================================================
# Calibration Metrics
# =============================================================================

def miscalibration_area(
    predicted_uncertainties: Sequence[float],
    prediction_errors: Sequence[float],
    n_bins: int = 10
) -> float:
    """
    Compute Miscalibration Area metric.

    Measures the area between the calibration curve and the ideal diagonal,
    indicating if model uncertainty (σ) matches actual error (MAE).
    Lower values indicate better calibration.

    Reference: ALDE

    Parameters
    ----------
    predicted_uncertainties : Sequence[float]
        Model's predicted uncertainties (standard deviations)
    prediction_errors : Sequence[float]
        Absolute errors between predictions and ground truth
    n_bins : int, default=10
        Number of bins for calibration curve

    Returns
    -------
    float
        Miscalibration area (lower is better, 0 = perfect calibration)

    Notes
    -----
    The calibration curve plots expected error (x-axis) vs observed error
    (y-axis). A well-calibrated model has points along the diagonal.
    """
    if len(predicted_uncertainties) != len(prediction_errors):
        raise ValueError("Arrays must have same length")
    if len(predicted_uncertainties) == 0:
        return float('nan')

    uncertainties = np.array(predicted_uncertainties)
    errors = np.array(prediction_errors)

    # Sort by predicted uncertainty
    sorted_indices = np.argsort(uncertainties)
    sorted_uncertainties = uncertainties[sorted_indices]
    sorted_errors = errors[sorted_indices]

    # Compute calibration curve using bins
    bin_edges = np.linspace(0, len(sorted_uncertainties), n_bins + 1, dtype=int)
    calibration_area = 0.0

    for i in range(n_bins):
        start, end = bin_edges[i], bin_edges[i + 1]
        if end <= start:
            continue

        # Expected error (mean predicted uncertainty in bin)
        expected = np.mean(sorted_uncertainties[start:end])
        # Observed error (mean actual error in bin)
        observed = np.mean(sorted_errors[start:end])

        # Area between calibration curve and diagonal
        calibration_area += abs(expected - observed) * (end - start)

    # Normalize by total number of samples
    return float(calibration_area / len(uncertainties))


def expected_calibration_error(
    predicted_probs: Sequence[float],
    true_outcomes: Sequence[int],
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE) metric.

    Measures the weighted average of calibration errors across bins.
    For each bin, computes |accuracy - confidence| weighted by bin size.
    Lower values indicate better calibration.

    Reference: ALDE

    Parameters
    ----------
    predicted_probs : Sequence[float]
        Predicted probabilities/confidences
    true_outcomes : Sequence[int]
        True binary outcomes (0 or 1)
    n_bins : int, default=10
        Number of bins for calibration

    Returns
    -------
    float
        Expected Calibration Error (lower is better, 0 = perfect)

    Notes
    -----
    This is the standard ECE formulation used in calibration literature.
    For regression tasks, see miscalibration_area instead.
    """
    if len(predicted_probs) != len(true_outcomes):
        raise ValueError("Arrays must have same length")
    if len(predicted_probs) == 0:
        return float('nan')

    probs = np.array(predicted_probs)
    outcomes = np.array(true_outcomes)
    n_samples = len(probs)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            # Accuracy in bin
            accuracy_in_bin = np.mean(outcomes[in_bin])
            # Average confidence in bin
            avg_confidence_in_bin = np.mean(probs[in_bin])
            # Weighted calibration error
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)


def regression_calibration_error(
    predicted_means: Sequence[float],
    predicted_stds: Sequence[float],
    true_values: Sequence[float],
    n_bins: int = 10
) -> Tuple[float, float]:
    """
    Compute calibration metrics for regression with uncertainty estimates.

    For a well-calibrated model, ~68% of true values should fall within
    1 std of prediction, ~95% within 2 std, etc.

    Parameters
    ----------
    predicted_means : Sequence[float]
        Predicted mean values
    predicted_stds : Sequence[float]
        Predicted standard deviations (uncertainties)
    true_values : Sequence[float]
        True target values
    n_bins : int, default=10
        Number of bins for calibration curve

    Returns
    -------
    Tuple[float, float]
        (miscalibration_area, expected_calibration_error)
    """
    means = np.array(predicted_means)
    stds = np.array(predicted_stds)
    truths = np.array(true_values)

    # Compute z-scores
    z_scores = np.abs((truths - means) / np.maximum(stds, 1e-8))

    # Expected coverage at different confidence levels
    confidence_levels = np.linspace(0.1, 0.99, n_bins)
    expected_coverages = []
    observed_coverages = []

    for conf in confidence_levels:
        # For normal distribution, z-score threshold for given confidence
        z_threshold = stats.norm.ppf((1 + conf) / 2)
        expected_coverages.append(conf)
        observed_coverages.append(np.mean(z_scores <= z_threshold))

    expected_coverages = np.array(expected_coverages)
    observed_coverages = np.array(observed_coverages)

    # Miscalibration area
    misc_area = float(np.mean(np.abs(expected_coverages - observed_coverages)))

    # ECE (weighted by bin width)
    ece = float(np.sum(np.abs(expected_coverages - observed_coverages)) / n_bins)

    return misc_area, ece


# =============================================================================
# Utility Functions
# =============================================================================

def compute_all_metrics(
    generated_sequences: Sequence[str],
    generated_fitness: Sequence[float],
    initial_training_sequences: Sequence[str],
    high_fitness_sequences: Sequence[str],
    landscape_fitness_min: float,
    landscape_fitness_max: float,
    global_max_sequence: str,
    global_max_fitness: float,
    predicted_fitness: Optional[Sequence[float]] = None,
    predicted_uncertainties: Optional[Sequence[float]] = None,
    true_fitness_for_pred: Optional[Sequence[float]] = None,
    predicted_epistasis: Optional[Sequence[float]] = None,
    observed_epistasis: Optional[Sequence[float]] = None,
    true_top_mutants: Optional[Sequence[str]] = None,
    best_sequences_per_run: Optional[Sequence[str]] = None,
    distance_fn: str = 'levenshtein'
) -> dict:
    """
    Compute all benchmark metrics in a single call.

    Parameters
    ----------
    generated_sequences : Sequence[str]
        Sequences generated by the optimization method
    generated_fitness : Sequence[float]
        Fitness values of generated sequences
    initial_training_sequences : Sequence[str]
        Initial training set sequences
    high_fitness_sequences : Sequence[str]
        Top fitness sequences (e.g., top 10%)
    landscape_fitness_min : float
        Minimum fitness in the landscape
    landscape_fitness_max : float
        Maximum fitness in the landscape
    global_max_sequence : str
        Global maximum sequence
    global_max_fitness : float
        Fitness of the global maximum
    predicted_fitness : Sequence[float], optional
        Model's fitness predictions (for Spearman correlation)
    predicted_uncertainties : Sequence[float], optional
        Model's uncertainty predictions (for calibration)
    true_fitness_for_pred : Sequence[float], optional
        True fitness values for predictions
    predicted_epistasis : Sequence[float], optional
        Predicted epistatic scores
    observed_epistasis : Sequence[float], optional
        Observed epistatic scores
    true_top_mutants : Sequence[str], optional
        True top mutant sequences (for recall)
    best_sequences_per_run : Sequence[str], optional
        Best sequences from each run (for global max hit count)
    distance_fn : str, default='levenshtein'
        Distance function to use

    Returns
    -------
    dict
        Dictionary containing all computed metrics with standard keys
    """
    results = {}

    # Distance-based metrics
    results['high_fitness_proximity'] = high_fitness_proximity(
        generated_sequences, high_fitness_sequences, distance_fn
    )
    results['novelty'] = novelty(
        generated_sequences, initial_training_sequences, distance_fn
    )
    results['batch_diversity'] = batch_diversity(generated_sequences, distance_fn)

    # Fitness metrics
    results['normalized_fitness_median_top128'] = normalized_fitness_median_topk(
        generated_fitness, k=128,
        fitness_min=landscape_fitness_min, fitness_max=landscape_fitness_max
    )
    results['normalized_fitness_median_top256'] = normalized_fitness_median_topk(
        generated_fitness, k=256,
        fitness_min=landscape_fitness_min, fitness_max=landscape_fitness_max
    )
    results['max_fitness'] = max_fitness(generated_fitness)
    results['simple_regret'] = simple_regret(
        max_fitness(generated_fitness), global_max_fitness
    )

    # Correlation metrics
    if predicted_fitness is not None and true_fitness_for_pred is not None:
        results['spearman_correlation'] = spearman_correlation(
            predicted_fitness, true_fitness_for_pred
        )
    else:
        results['spearman_correlation'] = float('nan')

    # Epistatic correlation
    if predicted_epistasis is not None and observed_epistasis is not None:
        results['epistatic_correlation'] = epistatic_correlation(
            predicted_epistasis, observed_epistasis
        )
    else:
        results['epistatic_correlation'] = float('nan')

    # Recall
    if true_top_mutants is not None:
        # Get top sequences from generated set
        if len(generated_fitness) > 0:
            top_k = min(len(generated_fitness), len(true_top_mutants))
            top_indices = np.argsort(generated_fitness)[-top_k:]
            predicted_top = [generated_sequences[i] for i in top_indices]
            results['recall_high_order'] = recall_high_order_mutants(
                predicted_top, true_top_mutants
            )
        else:
            results['recall_high_order'] = float('nan')
    else:
        results['recall_high_order'] = float('nan')

    # Calibration metrics
    if predicted_uncertainties is not None and true_fitness_for_pred is not None:
        if predicted_fitness is not None:
            errors = np.abs(np.array(predicted_fitness) - np.array(true_fitness_for_pred))
            results['miscalibration_area'] = miscalibration_area(
                predicted_uncertainties, errors
            )
            misc, ece = regression_calibration_error(
                predicted_fitness, predicted_uncertainties, true_fitness_for_pred
            )
            results['expected_calibration_error'] = ece
        else:
            results['miscalibration_area'] = float('nan')
            results['expected_calibration_error'] = float('nan')
    else:
        results['miscalibration_area'] = float('nan')
        results['expected_calibration_error'] = float('nan')

    # Global max hit count
    if best_sequences_per_run is not None:
        results['global_max_hit_count'] = global_max_hit_count(
            best_sequences_per_run, global_max_sequence
        )
    else:
        results['global_max_hit_count'] = float('nan')

    return results


# =============================================================================
# Additional Metrics: AUOC and Hit Rate
# =============================================================================

def area_under_optimization_curve(
    fitness_trajectory: Sequence[float],
    global_max_fitness: float,
    normalize: bool = True
) -> float:
    """
    Compute the Area Under the Optimization Curve (AUOC).

    The optimization curve tracks the best fitness found at each round.
    AUOC summarizes overall optimization performance in a single scalar.

    Parameters
    ----------
    fitness_trajectory : Sequence[float]
        Best fitness per round (cumulative max). Length = n_rounds.
    global_max_fitness : float
        Global maximum fitness in the landscape (for normalization).
    normalize : bool, default=True
        If True, normalize by (n_rounds * global_max_fitness) so AUOC
        is in [0, 1]. A perfect optimizer scores 1.0.

    Returns
    -------
    float
        AUOC value. Higher is better.

    Examples
    --------
    >>> area_under_optimization_curve([0.5, 0.7, 0.9, 1.0], 1.0)
    0.775
    >>> area_under_optimization_curve([1.0, 1.0, 1.0, 1.0], 1.0)
    1.0
    """
    trajectory = np.array(fitness_trajectory, dtype=float)
    if len(trajectory) == 0:
        return float('nan')

    # Ensure cumulative max
    cum_max = np.maximum.accumulate(trajectory)

    # Compute area using trapezoidal rule
    area = float(np.trapz(cum_max, dx=1.0))

    if normalize and global_max_fitness > 0:
        max_area = (len(trajectory) - 1) * global_max_fitness
        if max_area > 0:
            area /= max_area
        else:
            area = float(cum_max[0] / global_max_fitness) if len(trajectory) == 1 else float('nan')

    return area


def hit_rate(
    generated_fitness: Sequence[float],
    threshold: float
) -> float:
    """
    Compute the fraction of generated sequences with fitness >= threshold.

    Parameters
    ----------
    generated_fitness : Sequence[float]
        Fitness values of generated sequences.
    threshold : float
        Fitness threshold to count as a "hit".

    Returns
    -------
    float
        Fraction of sequences meeting the threshold, in [0, 1].

    Examples
    --------
    >>> hit_rate([0.3, 0.5, 0.7, 0.9, 1.0], 0.5)
    0.8
    >>> hit_rate([0.1, 0.2, 0.3], 0.5)
    0.0
    """
    fitness = np.array(generated_fitness, dtype=float)
    if len(fitness) == 0:
        return float('nan')
    return float(np.mean(fitness >= threshold))


# =============================================================================
# Statistical Comparison
# =============================================================================

def paired_comparison(
    values_a: Sequence[float],
    values_b: Sequence[float],
    test: str = 'wilcoxon',
    alternative: str = 'two-sided'
) -> tuple:
    """
    Perform a paired statistical comparison between two methods.

    Parameters
    ----------
    values_a : Sequence[float]
        Metric values from method A (one per seed/run).
    values_b : Sequence[float]
        Metric values from method B (one per seed/run).
    test : str, default='wilcoxon'
        Statistical test to use:
        - 'wilcoxon': Wilcoxon signed-rank test (paired, non-parametric)
        - 'mannwhitney': Mann-Whitney U test (unpaired, non-parametric)
        - 'ttest': Paired t-test (parametric)
        - 'permutation': Permutation test on mean difference
    alternative : str, default='two-sided'
        Alternative hypothesis: 'two-sided', 'greater', or 'less'.

    Returns
    -------
    tuple of (statistic, p_value)
        Test statistic and p-value.

    Examples
    --------
    >>> a = [0.8, 0.85, 0.78, 0.82, 0.79]
    >>> b = [0.7, 0.72, 0.68, 0.71, 0.69]
    >>> stat, pval = paired_comparison(a, b, test='wilcoxon')
    """
    a = np.array(values_a, dtype=float)
    b = np.array(values_b, dtype=float)

    # Remove NaN pairs
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]

    if len(a) < 2:
        return (float('nan'), float('nan'))

    from scipy import stats as scipy_stats

    if test == 'wilcoxon':
        diff = a - b
        # Wilcoxon requires non-zero differences
        if np.all(diff == 0):
            return (0.0, 1.0)
        result = scipy_stats.wilcoxon(diff, alternative=alternative)
        return (float(result.statistic), float(result.pvalue))

    elif test == 'mannwhitney':
        result = scipy_stats.mannwhitneyu(a, b, alternative=alternative)
        return (float(result.statistic), float(result.pvalue))

    elif test == 'ttest':
        result = scipy_stats.ttest_rel(a, b, alternative=alternative)
        return (float(result.statistic), float(result.pvalue))

    elif test == 'permutation':
        observed_diff = float(np.mean(a) - np.mean(b))
        n_permutations = 10000
        combined = np.concatenate([a, b])
        n = len(a)
        rng = np.random.RandomState(42)
        count = 0
        for _ in range(n_permutations):
            perm = rng.permutation(combined)
            perm_diff = np.mean(perm[:n]) - np.mean(perm[n:])
            if alternative == 'two-sided':
                if abs(perm_diff) >= abs(observed_diff):
                    count += 1
            elif alternative == 'greater':
                if perm_diff >= observed_diff:
                    count += 1
            elif alternative == 'less':
                if perm_diff <= observed_diff:
                    count += 1
        p_value = (count + 1) / (n_permutations + 1)
        return (observed_diff, p_value)

    else:
        raise ValueError(f"Unknown test: {test}. Use 'wilcoxon', 'mannwhitney', 'ttest', or 'permutation'")
