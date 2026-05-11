"""
Generic Oracle Utilities for ProteinGym / CombinGym style datasets

Wraps the existing FitnessLandscape so that any prepared dataset (data/<name>/data.csv
with `seq, fitness` columns) can be used as an oracle, regardless of sequence
length or family. Generalizes the GB1-specific helpers in `utils/gb1.py`.

Provides:
    - `load_oracle(name)` -> (FitnessLandscape, wildtype, threshold_top10pct)
    - `top_percent_threshold` for EVOLVEpro-style hit rate
    - `mutation_order_distribution` for MULTI-evolve-style reporting
    - `hierarchical_split` for CombinGym Task-1 splits
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union
from pathlib import Path
import numpy as np

from .data import (
    FitnessLandscape,
    load_landscape_data,
    count_mutations,
)


@dataclass
class OracleHandle:
    """Bundle of objects every benchmark run needs from an oracle."""
    landscape: FitnessLandscape
    wildtype: str
    name: str
    top10_threshold: float  # absolute fitness for top-10% (EVOLVEpro hit rate)
    top1_threshold: float


# =============================================================================
# Loading
# =============================================================================

def _detect_wildtype(sequences: Sequence[str], fitness: np.ndarray) -> str:
    """Pick a sensible wild-type for downstream metrics.

    Heuristic: if the dataset has a very common length and one row matches
    the consensus single-mutant base, use the median-fitness sequence as a
    fallback. Otherwise return the first sequence (matches existing
    Random/GreedyWalk convention).

    Most callers should pass an explicit wildtype; this is only the default.
    """
    return sequences[0]


def load_oracle(
    name: str,
    data_dir: Optional[Union[str, Path]] = None,
    wildtype: Optional[str] = None,
) -> OracleHandle:
    """Load any prepared dataset as an oracle.

    Args:
        name: dataset folder name under `data/`. Examples: "GB1", "AAV_med",
            "BLAT_ECOLX", "PhoQ", "eqFP611_blue".
        data_dir: override `data/` root.
        wildtype: explicit wild-type sequence; if None, falls back to
            `_detect_wildtype`.
    """
    sequences, fitness = load_landscape_data(name, data_dir=data_dir)
    landscape = FitnessLandscape(sequences, fitness)
    wt = wildtype if wildtype is not None else _detect_wildtype(sequences, fitness)
    return OracleHandle(
        landscape=landscape,
        wildtype=wt,
        name=name,
        top10_threshold=top_percent_threshold(fitness, 10.0),
        top1_threshold=top_percent_threshold(fitness, 1.0),
    )


# =============================================================================
# Threshold + hit-rate helpers
# =============================================================================

def top_percent_threshold(fitness: Sequence[float], percent: float) -> float:
    """Absolute fitness threshold for the top `percent`% of the landscape.

    Used by EVOLVEpro-style hit rate ("variants with fitness > top-10%-threshold").
    """
    if percent <= 0 or percent > 100:
        raise ValueError(f"percent must be in (0, 100], got {percent}")
    arr = np.asarray(fitness, dtype=float)
    return float(np.quantile(arr, 1.0 - percent / 100.0))


# =============================================================================
# Mutation-order reporting
# =============================================================================

def mutation_order_distribution(
    sequences: Sequence[str],
    wildtype: str,
    max_order: int = 12,
) -> np.ndarray:
    """Histogram of mutation counts vs wild-type.

    Returns:
        Array of length `max_order + 1` where entry k is the count of
        sequences with exactly k mutations from `wildtype`. Sequences with
        more than `max_order` mutations are bucketed at the last index.
    """
    counts = np.zeros(max_order + 1, dtype=int)
    for seq in sequences:
        n = count_mutations(seq, wildtype)
        counts[min(n, max_order)] += 1
    return counts


# =============================================================================
# Hierarchical splits (CombinGym Task 1)
# =============================================================================

def hierarchical_split(
    sequences: Sequence[str],
    wildtype: str,
    train_order: int,
    test_min_order: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """Build a CombinGym-style "K-vs-rest" split.

    Train set = all variants with mutation order <= `train_order`.
    Test set = variants with mutation order >= `test_min_order`
    (default `train_order + 1`).

    Args:
        sequences: full landscape.
        wildtype: reference for mutation counting.
        train_order: highest mutation order included in training.
        test_min_order: lowest mutation order included in test
            (default = train_order + 1).

    Returns:
        (train_indices, test_indices) into the input `sequences`.
    """
    if test_min_order is None:
        test_min_order = train_order + 1

    train_idx: List[int] = []
    test_idx: List[int] = []
    for i, seq in enumerate(sequences):
        n = count_mutations(seq, wildtype)
        if n <= train_order:
            train_idx.append(i)
        if n >= test_min_order:
            test_idx.append(i)
    return train_idx, test_idx


__all__ = [
    "OracleHandle",
    "load_oracle",
    "top_percent_threshold",
    "mutation_order_distribution",
    "hierarchical_split",
]
