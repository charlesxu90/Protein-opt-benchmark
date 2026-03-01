"""
Benchmark Utilities for Protein Optimization Methods

This package provides standardized metrics and utilities for benchmarking
protein optimization methods including:
- ALDE
- EvoPlay
- AdaLead
- LatProtRL
- AICE
- δ-Conservative Search
- AlphaVariant

Supported Benchmarks:
- GB1
- AAV/GFP medium
- AAV/GFP hard
"""

from .metrics import (
    # Distance metrics
    levenshtein_distance,
    hamming_distance,

    # Core benchmark metrics
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

    # Calibration metrics
    miscalibration_area,
    expected_calibration_error,

    # Optimization curve and hit rate
    area_under_optimization_curve,
    hit_rate,

    # Statistical comparison
    paired_comparison,
)

from .evaluator import BenchmarkEvaluator
from .data import (
    load_landscape_data,
    load_initial_training_set,
    get_top_fitness_sequences,
    get_global_maximum,
    count_mutations,
)
from .io import (
    save_results,
    load_results,
    aggregate_results,
    export_summary_table,
)
from .gb1 import (
    # GB1 constants
    GB1_WILD_TYPE_4SITE,
    GB1_WILD_TYPE_FULL,
    GB1_POSITIONS,

    # GB1 data loading
    load_gb1_landscape,
    create_gb1_landscape,

    # GB1 metrics
    GB1MetricsResult,
    compute_gb1_metrics,
    aggregate_gb1_metrics,
    save_gb1_results,
    print_gb1_metrics_summary,
)

# Compatibility module for drop-in replacement
from . import compat

__all__ = [
    # Distance metrics
    'levenshtein_distance',
    'hamming_distance',

    # Core metrics
    'high_fitness_proximity',
    'high_fitness_proximity_from_landscape',
    'novelty',
    'batch_diversity',
    'normalized_fitness_median_topk',
    'max_fitness',
    'spearman_correlation',
    'epistatic_correlation',
    'recall_high_order_mutants',
    'simple_regret',
    'global_max_hit_count',

    # Calibration metrics
    'miscalibration_area',
    'expected_calibration_error',

    # Optimization curve and hit rate
    'area_under_optimization_curve',
    'hit_rate',

    # Statistical comparison
    'paired_comparison',

    # Evaluator
    'BenchmarkEvaluator',

    # Data utilities
    'load_landscape_data',
    'load_initial_training_set',
    'get_top_fitness_sequences',
    'get_global_maximum',
    'count_mutations',

    # I/O utilities
    'save_results',
    'load_results',
    'aggregate_results',
    'export_summary_table',

    # GB1 utilities
    'GB1_WILD_TYPE_4SITE',
    'GB1_WILD_TYPE_FULL',
    'GB1_POSITIONS',
    'load_gb1_landscape',
    'create_gb1_landscape',
    'GB1MetricsResult',
    'compute_gb1_metrics',
    'aggregate_gb1_metrics',
    'save_gb1_results',
    'print_gb1_metrics_summary',
]
