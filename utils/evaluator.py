"""
Unified Benchmark Evaluator for Protein Optimization Methods

This module provides the BenchmarkEvaluator class that standardizes
evaluation across all methods (ALDE, EvoPlay, AdaLead, LatProtRL, AICE,
δ-Conservative Search, AlphaVariant) and datasets (GB1, AAV/GFP).

The evaluator ensures consistent metric computation and result reporting
across different experiments.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
import numpy as np
import json
from datetime import datetime

from .metrics import (
    high_fitness_proximity,
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
    expected_calibration_error,
    regression_calibration_error,
    levenshtein_distance,
    area_under_optimization_curve,
    hit_rate,
)
from .data import (
    FitnessLandscape,
    get_top_fitness_sequences,
    get_high_order_mutants,
    count_mutations,
    load_random_seeds,
)


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class RoundMetrics:
    """Metrics for a single round of optimization."""
    round_idx: int
    n_generated: int
    n_unique: int

    # Distance-based metrics
    high_fitness_proximity: float = float('nan')
    novelty: float = float('nan')
    batch_diversity: float = float('nan')

    # Fitness metrics
    normalized_fitness_median_top128: float = float('nan')
    normalized_fitness_median_top256: float = float('nan')
    max_fitness: float = float('nan')
    simple_regret: float = float('nan')

    # Correlation metrics
    spearman_correlation: float = float('nan')
    epistatic_correlation: float = float('nan')

    # Recall
    recall_high_order: float = float('nan')

    # Calibration metrics
    miscalibration_area: float = float('nan')
    expected_calibration_error: float = float('nan')

    # Optimization curve and hit rate
    auoc: float = float('nan')
    hit_rate_value: float = float('nan')

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'round_idx': self.round_idx,
            'n_generated': self.n_generated,
            'n_unique': self.n_unique,
            'high_fitness_proximity': self.high_fitness_proximity,
            'novelty': self.novelty,
            'batch_diversity': self.batch_diversity,
            'normalized_fitness_median_top128': self.normalized_fitness_median_top128,
            'normalized_fitness_median_top256': self.normalized_fitness_median_top256,
            'max_fitness': self.max_fitness,
            'simple_regret': self.simple_regret,
            'spearman_correlation': self.spearman_correlation,
            'epistatic_correlation': self.epistatic_correlation,
            'recall_high_order': self.recall_high_order,
            'miscalibration_area': self.miscalibration_area,
            'expected_calibration_error': self.expected_calibration_error,
            'auoc': self.auoc,
            'hit_rate': self.hit_rate_value,
        }


@dataclass
class RunResults:
    """Results from a complete optimization run."""
    run_id: int
    seed: int
    method: str
    dataset: str
    n_rounds: int
    batch_size: int
    start_time: str
    end_time: Optional[str] = None

    # Per-round metrics
    round_metrics: List[RoundMetrics] = field(default_factory=list)

    # Cumulative results (all generated sequences across rounds)
    all_generated_sequences: List[str] = field(default_factory=list)
    all_generated_fitness: List[float] = field(default_factory=list)

    # Best sequence found
    best_sequence: Optional[str] = None
    best_fitness: float = float('-inf')

    # Final metrics (computed on all generated sequences)
    final_metrics: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'run_id': self.run_id,
            'seed': self.seed,
            'method': self.method,
            'dataset': self.dataset,
            'n_rounds': self.n_rounds,
            'batch_size': self.batch_size,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'round_metrics': [rm.to_dict() for rm in self.round_metrics],
            'best_sequence': self.best_sequence,
            'best_fitness': self.best_fitness,
            'final_metrics': self.final_metrics,
        }


@dataclass
class AggregateResults:
    """Aggregated results across multiple runs."""
    method: str
    dataset: str
    n_runs: int
    seeds: List[int]

    # Aggregated metrics (mean ± std)
    metrics_mean: Dict[str, float] = field(default_factory=dict)
    metrics_std: Dict[str, float] = field(default_factory=dict)
    metrics_median: Dict[str, float] = field(default_factory=dict)

    # Global max hit count (GB1 specific)
    global_max_hit_count: int = 0
    global_max_hit_rate: float = 0.0

    # Per-run results
    run_results: List[RunResults] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'method': self.method,
            'dataset': self.dataset,
            'n_runs': self.n_runs,
            'seeds': self.seeds,
            'metrics_mean': self.metrics_mean,
            'metrics_std': self.metrics_std,
            'metrics_median': self.metrics_median,
            'global_max_hit_count': self.global_max_hit_count,
            'global_max_hit_rate': self.global_max_hit_rate,
        }


# =============================================================================
# Benchmark Evaluator
# =============================================================================

class BenchmarkEvaluator:
    """
    Unified evaluator for protein optimization benchmarks.

    This class provides standardized metric computation for all methods
    across GB1, AAV, and GFP datasets.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape for evaluation
    initial_training_sequences : Sequence[str]
        Initial training set sequences
    wild_type : str
        Wild-type sequence for mutation counting
    high_fitness_percentile : float, default=10.0
        Top percentile to use for high-fitness sequences
    method_name : str, default='unknown'
        Name of the optimization method
    dataset_name : str, default='unknown'
        Name of the dataset

    Attributes
    ----------
    landscape : FitnessLandscape
        Fitness landscape object
    high_fitness_sequences : List[str]
        Top fitness sequences
    true_top_mutants : List[str]
        High-order mutants in top fitness percentile

    Examples
    --------
    >>> from utils.data import FitnessLandscape, load_initial_training_set
    >>> landscape = FitnessLandscape.from_dataset('GB1')
    >>> init_seqs, _ = load_initial_training_set('GB1', n_samples=100)
    >>> evaluator = BenchmarkEvaluator(
    ...     landscape=landscape,
    ...     initial_training_sequences=init_seqs,
    ...     wild_type='VDGV',
    ...     method_name='ALDE',
    ...     dataset_name='GB1'
    ... )
    >>> metrics = evaluator.evaluate_batch(generated_seqs, generated_fitness)
    """

    # Standard metric order for consistent reporting
    METRIC_ORDER = [
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
        'global_max_hit_count',
        'auoc',
        'hit_rate',
    ]

    def __init__(
        self,
        landscape: FitnessLandscape,
        initial_training_sequences: Sequence[str],
        wild_type: str,
        high_fitness_percentile: float = 10.0,
        max_high_fitness_seqs: int = 128,
        method_name: str = 'unknown',
        dataset_name: str = 'unknown',
        distance_fn: str = 'levenshtein'
    ):
        self.landscape = landscape
        self.initial_training_sequences = list(initial_training_sequences)
        self.wild_type = wild_type
        self.high_fitness_percentile = high_fitness_percentile
        self.max_high_fitness_seqs = max_high_fitness_seqs
        self.method_name = method_name
        self.dataset_name = dataset_name
        self.distance_fn = distance_fn

        # Pre-compute high-fitness sequences (for high_fitness_proximity metric)
        # Limit to max_high_fitness_seqs for efficiency (matching ALDE's approach)
        all_high_seqs, all_high_fitness = get_top_fitness_sequences(
            landscape.sequences,
            landscape.fitness,
            percentile=high_fitness_percentile
        )
        # Take only top max_high_fitness_seqs (sorted by fitness)
        if len(all_high_seqs) > max_high_fitness_seqs:
            # all_high_seqs is already sorted by fitness ascending, take last N
            self.high_fitness_sequences = all_high_seqs[-max_high_fitness_seqs:]
            self._high_fitness_values = all_high_fitness[-max_high_fitness_seqs:]
        else:
            self.high_fitness_sequences = all_high_seqs
            self._high_fitness_values = all_high_fitness

        # Pre-compute high-order mutants in top percentile
        self.true_top_mutants = get_high_order_mutants(
            landscape.sequences,
            landscape.fitness,
            wild_type,
            min_mutations=2,
            top_percentile=high_fitness_percentile
        )

        # Cache fitness bounds
        self.fitness_min = landscape.min_fitness
        self.fitness_max = landscape.max_fitness
        self.global_max_sequence = landscape.global_max_sequence
        self.global_max_fitness = landscape.global_max_fitness

    def evaluate_batch(
        self,
        sequences: Sequence[str],
        fitness: Optional[Sequence[float]] = None,
        predicted_fitness: Optional[Sequence[float]] = None,
        predicted_uncertainties: Optional[Sequence[float]] = None,
        round_idx: int = 0
    ) -> RoundMetrics:
        """
        Evaluate a batch of generated sequences.

        Parameters
        ----------
        sequences : Sequence[str]
            Generated sequences to evaluate
        fitness : Sequence[float], optional
            True fitness values. If None, looks up from landscape.
        predicted_fitness : Sequence[float], optional
            Model's predicted fitness values (for correlation/calibration)
        predicted_uncertainties : Sequence[float], optional
            Model's predicted uncertainties (for calibration)
        round_idx : int, default=0
            Round number for tracking

        Returns
        -------
        RoundMetrics
            Computed metrics for this batch
        """
        sequences = list(sequences)
        n_generated = len(sequences)
        n_unique = len(set(sequences))

        # Get true fitness if not provided
        if fitness is None:
            fitness = self.landscape.get_fitness_batch(sequences)
        else:
            fitness = np.array(fitness)

        # Initialize metrics
        metrics = RoundMetrics(
            round_idx=round_idx,
            n_generated=n_generated,
            n_unique=n_unique
        )

        if n_generated == 0:
            return metrics

        # Distance-based metrics
        # high_fitness_sequences is already limited to max_high_fitness_seqs in __init__
        metrics.high_fitness_proximity = high_fitness_proximity(
            sequences, self.high_fitness_sequences, self.distance_fn,
            max_high_seqs=None  # Already limited in __init__
        )
        metrics.novelty = novelty(
            sequences, self.initial_training_sequences, self.distance_fn
        )
        metrics.batch_diversity = batch_diversity(sequences, self.distance_fn)

        # Fitness metrics
        valid_fitness = fitness[~np.isnan(fitness)]
        if len(valid_fitness) > 0:
            metrics.normalized_fitness_median_top128 = normalized_fitness_median_topk(
                valid_fitness, k=128,
                fitness_min=self.fitness_min, fitness_max=self.fitness_max
            )
            metrics.normalized_fitness_median_top256 = normalized_fitness_median_topk(
                valid_fitness, k=256,
                fitness_min=self.fitness_min, fitness_max=self.fitness_max
            )
            metrics.max_fitness = max_fitness(valid_fitness)
            metrics.simple_regret = simple_regret(
                metrics.max_fitness, self.global_max_fitness
            )

        # Correlation metrics (requires predictions)
        if predicted_fitness is not None:
            pred = np.array(predicted_fitness)
            # Only use sequences that are in the landscape
            valid_mask = ~np.isnan(fitness)
            if np.sum(valid_mask) >= 2:
                metrics.spearman_correlation = spearman_correlation(
                    pred[valid_mask], fitness[valid_mask]
                )

        # Calibration metrics (requires predictions and uncertainties)
        if predicted_fitness is not None and predicted_uncertainties is not None:
            pred = np.array(predicted_fitness)
            unc = np.array(predicted_uncertainties)
            valid_mask = ~np.isnan(fitness)
            if np.sum(valid_mask) >= 2:
                errors = np.abs(pred[valid_mask] - fitness[valid_mask])
                metrics.miscalibration_area = miscalibration_area(
                    unc[valid_mask], errors
                )
                misc, ece = regression_calibration_error(
                    pred[valid_mask], unc[valid_mask], fitness[valid_mask]
                )
                metrics.expected_calibration_error = ece

        # Recall of high-order mutants
        if len(self.true_top_mutants) > 0 and len(valid_fitness) > 0:
            # Get high-order sequences from generated batch
            high_order_generated = [
                seq for seq in sequences
                if count_mutations(seq, self.wild_type) >= 2
            ]
            if len(high_order_generated) > 0:
                metrics.recall_high_order = recall_high_order_mutants(
                    high_order_generated, self.true_top_mutants
                )

        return metrics

    def evaluate_run(
        self,
        run_results: RunResults,
        all_sequences: Sequence[str],
        all_fitness: Optional[Sequence[float]] = None,
        all_predictions: Optional[Sequence[float]] = None,
        all_uncertainties: Optional[Sequence[float]] = None
    ) -> RunResults:
        """
        Compute final metrics for a complete run.

        Parameters
        ----------
        run_results : RunResults
            Run results object to update
        all_sequences : Sequence[str]
            All generated sequences across all rounds
        all_fitness : Sequence[float], optional
            True fitness values
        all_predictions : Sequence[float], optional
            Model predictions
        all_uncertainties : Sequence[float], optional
            Model uncertainties

        Returns
        -------
        RunResults
            Updated run results with final metrics
        """
        all_sequences = list(all_sequences)
        if all_fitness is None:
            all_fitness = self.landscape.get_fitness_batch(all_sequences)
        else:
            all_fitness = np.array(all_fitness)

        run_results.all_generated_sequences = all_sequences
        run_results.all_generated_fitness = list(all_fitness)

        # Find best sequence
        valid_mask = ~np.isnan(all_fitness)
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            best_idx = valid_indices[np.argmax(all_fitness[valid_mask])]
            run_results.best_sequence = all_sequences[best_idx]
            run_results.best_fitness = float(all_fitness[best_idx])

        # Compute final metrics on all sequences
        final_metrics = self.evaluate_batch(
            all_sequences,
            all_fitness,
            all_predictions,
            all_uncertainties,
            round_idx=-1  # -1 indicates final/aggregate
        )

        run_results.final_metrics = final_metrics.to_dict()
        run_results.end_time = datetime.now().isoformat()

        return run_results

    def aggregate_runs(
        self,
        run_results_list: List[RunResults]
    ) -> AggregateResults:
        """
        Aggregate results across multiple runs.

        Parameters
        ----------
        run_results_list : List[RunResults]
            List of individual run results

        Returns
        -------
        AggregateResults
            Aggregated statistics
        """
        if len(run_results_list) == 0:
            raise ValueError("No runs to aggregate")

        n_runs = len(run_results_list)
        seeds = [r.seed for r in run_results_list]

        # Collect final metrics from all runs
        all_metrics: Dict[str, List[float]] = {
            metric: [] for metric in self.METRIC_ORDER
            if metric != 'global_max_hit_count'
        }

        best_sequences = []
        for run in run_results_list:
            if run.final_metrics is not None:
                for metric in all_metrics.keys():
                    if metric in run.final_metrics:
                        val = run.final_metrics[metric]
                        if not np.isnan(val):
                            all_metrics[metric].append(val)
            if run.best_sequence is not None:
                best_sequences.append(run.best_sequence)

        # Compute statistics
        metrics_mean = {}
        metrics_std = {}
        metrics_median = {}

        for metric, values in all_metrics.items():
            if len(values) > 0:
                metrics_mean[metric] = float(np.mean(values))
                metrics_std[metric] = float(np.std(values))
                metrics_median[metric] = float(np.median(values))
            else:
                metrics_mean[metric] = float('nan')
                metrics_std[metric] = float('nan')
                metrics_median[metric] = float('nan')

        # Global max hit count
        hit_count = global_max_hit_count(best_sequences, self.global_max_sequence)
        hit_rate = global_max_hit_rate(best_sequences, self.global_max_sequence)

        return AggregateResults(
            method=self.method_name,
            dataset=self.dataset_name,
            n_runs=n_runs,
            seeds=seeds,
            metrics_mean=metrics_mean,
            metrics_std=metrics_std,
            metrics_median=metrics_median,
            global_max_hit_count=hit_count,
            global_max_hit_rate=hit_rate,
            run_results=run_results_list,
        )

    def create_run_results(
        self,
        run_id: int,
        seed: int,
        n_rounds: int,
        batch_size: int
    ) -> RunResults:
        """
        Create a new RunResults object for tracking a run.

        Parameters
        ----------
        run_id : int
            Run identifier
        seed : int
            Random seed for this run
        n_rounds : int
            Number of optimization rounds
        batch_size : int
            Batch size per round

        Returns
        -------
        RunResults
            Initialized results object
        """
        return RunResults(
            run_id=run_id,
            seed=seed,
            method=self.method_name,
            dataset=self.dataset_name,
            n_rounds=n_rounds,
            batch_size=batch_size,
            start_time=datetime.now().isoformat(),
        )

    def print_metrics_summary(
        self,
        metrics: Union[RoundMetrics, Dict[str, Any]],
        prefix: str = ""
    ) -> None:
        """
        Print a formatted summary of metrics.

        Parameters
        ----------
        metrics : RoundMetrics or dict
            Metrics to display
        prefix : str, default=""
            Prefix for each line
        """
        if isinstance(metrics, RoundMetrics):
            metrics = metrics.to_dict()

        print(f"{prefix}Metrics Summary:")
        print(f"{prefix}{'='*50}")
        for metric in self.METRIC_ORDER:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    if np.isnan(value):
                        print(f"{prefix}  {metric}: N/A")
                    else:
                        print(f"{prefix}  {metric}: {value:.4f}")
                else:
                    print(f"{prefix}  {metric}: {value}")
        print(f"{prefix}{'='*50}")

    def print_aggregate_summary(
        self,
        aggregate: AggregateResults
    ) -> None:
        """
        Print a formatted summary of aggregated results.

        Parameters
        ----------
        aggregate : AggregateResults
            Aggregated results to display
        """
        print(f"\n{'='*60}")
        print(f"Aggregate Results: {aggregate.method} on {aggregate.dataset}")
        print(f"Runs: {aggregate.n_runs}")
        print(f"{'='*60}")

        for metric in self.METRIC_ORDER:
            if metric == 'global_max_hit_count':
                print(f"  global_max_hit_count: {aggregate.global_max_hit_count}")
                print(f"  global_max_hit_rate: {aggregate.global_max_hit_rate:.4f}")
            elif metric in aggregate.metrics_mean:
                mean = aggregate.metrics_mean[metric]
                std = aggregate.metrics_std[metric]
                if not np.isnan(mean):
                    print(f"  {metric}: {mean:.4f} ± {std:.4f}")
                else:
                    print(f"  {metric}: N/A")

        print(f"{'='*60}\n")


# =============================================================================
# Factory Functions
# =============================================================================

def create_evaluator(
    dataset: str,
    method: str,
    data_dir: Optional[Union[str, Path]] = None,
    initial_training_sequences: Optional[Sequence[str]] = None,
    wild_type: Optional[str] = None,
    high_fitness_percentile: float = 10.0
) -> BenchmarkEvaluator:
    """
    Factory function to create a BenchmarkEvaluator.

    Parameters
    ----------
    dataset : str
        Dataset name: 'GB1', 'AAV_medium', 'AAV_hard', 'GFP_medium', 'GFP_hard'
    method : str
        Method name for labeling
    data_dir : str or Path, optional
        Data directory
    initial_training_sequences : Sequence[str], optional
        Initial training set. If None, loads default.
    wild_type : str, optional
        Wild-type sequence. Uses dataset default if None.
    high_fitness_percentile : float, default=10.0
        Percentile for high-fitness sequences

    Returns
    -------
    BenchmarkEvaluator
        Configured evaluator
    """
    from .data import load_initial_training_set

    # Load landscape
    landscape = FitnessLandscape.from_dataset(dataset, data_dir)

    # Load or use provided initial training set
    if initial_training_sequences is None:
        initial_training_sequences, _ = load_initial_training_set(
            dataset, data_dir, n_samples=100
        )

    # Dataset-specific wild types
    wild_types = {
        'GB1': 'VDGV',
        'AAV_medium': None,  # Set from data
        'AAV_hard': None,
        'GFP_medium': None,
        'GFP_hard': None,
    }

    if wild_type is None:
        wild_type = wild_types.get(dataset)
        if wild_type is None:
            # Use first sequence as wild-type if not specified
            wild_type = landscape.sequences[0]

    return BenchmarkEvaluator(
        landscape=landscape,
        initial_training_sequences=initial_training_sequences,
        wild_type=wild_type,
        high_fitness_percentile=high_fitness_percentile,
        method_name=method,
        dataset_name=dataset,
    )
