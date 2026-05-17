"""
Data Loading Utilities for Protein Optimization Benchmarks

This module provides standardized data loading functions for benchmark datasets:
- GB1: 4-site combinatorial library
- AAV/GFP medium: Medium difficulty viral capsid optimization
- AAV/GFP hard: Hard difficulty variant

Supported data formats:
- CSV files with sequence and fitness columns
- FASTA files with fitness in headers
- Pickle files with pandas DataFrames
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Sequence
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# =============================================================================
# Constants
# =============================================================================

# Standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}

# Dataset-specific constants
GB1_WILD_TYPE = "VDGV"  # Wild-type sequence at positions 39, 40, 41, 54
GB1_POSITIONS = [39, 40, 41, 54]
GB1_GLOBAL_MAX_SEQUENCE = "VWHS"  # Known global maximum (TEMH in some references)
GB1_GLOBAL_MAX_FITNESS = 8.76  # Approximate, update based on actual data

# Default paths (can be overridden)
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_landscape_data(
    dataset: str,
    data_dir: Optional[Union[str, Path]] = None,
    sequence_col: Optional[str] = None,
    fitness_col: str = 'fitness'
) -> Tuple[List[str], np.ndarray]:
    """
    Load complete fitness landscape data for a benchmark dataset.

    Parameters
    ----------
    dataset : str
        Dataset name: 'GB1', 'AAV_medium', 'AAV_hard', 'GFP_medium', 'GFP_hard'
    data_dir : str or Path, optional
        Directory containing dataset files. Defaults to ../data relative to utils/
    sequence_col : str, default='sequence'
        Column name for sequences in CSV files
    fitness_col : str, default='fitness'
        Column name for fitness values in CSV files

    Returns
    -------
    Tuple[List[str], np.ndarray]
        (sequences, fitness_values) where sequences is a list of strings
        and fitness_values is a numpy array of floats

    Raises
    ------
    FileNotFoundError
        If dataset file is not found
    ValueError
        If dataset name is not recognized
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    # Map dataset names to file patterns (in order of preference)
    dataset_files = {
        'GB1': [
            'GB1/data.csv',        # ALDE format
            'GB1/fitness.csv',     # Alternative ALDE format
            'GB1.csv',
            'gb1.csv',
            'GB1_data.csv',
            'GB1/landscape.csv',
        ],
        # Canonical short names (matching directory names)
        'AAV_med': ['AAV_med/data.csv', 'AAV_medium.csv', 'AAV/medium.csv'],
        'AAV_hard': ['AAV_hard/data.csv', 'AAV_hard.csv', 'AAV/hard.csv'],
        'GFP_med': ['GFP_med/data.csv', 'GFP_medium.csv', 'GFP/medium.csv'],
        'GFP_hard': ['GFP_hard/data.csv', 'GFP_hard.csv', 'GFP/hard.csv'],
        # Legacy aliases
        'AAV_medium': ['AAV_med/data.csv', 'AAV_medium.csv', 'AAV/medium.csv'],
        'GFP_medium': ['GFP_med/data.csv', 'GFP_medium.csv', 'GFP/medium.csv'],
    }

    # Fallback: check for data/<name>/data.csv directly
    if dataset not in dataset_files:
        fallback_path = data_dir / dataset / 'data.csv'
        if fallback_path.exists():
            return load_csv_data(fallback_path, sequence_col, fitness_col)

    if dataset not in dataset_files:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Available: {list(dataset_files.keys())}"
        )

    # Try to find the data file
    data_path = None
    for pattern in dataset_files[dataset]:
        candidate = data_dir / pattern
        if candidate.exists():
            data_path = candidate
            break

    if data_path is None:
        raise FileNotFoundError(
            f"Could not find data file for {dataset} in {data_dir}. "
            f"Tried: {dataset_files[dataset]}"
        )

    return load_csv_data(data_path, sequence_col, fitness_col)


def load_csv_data(
    filepath: Union[str, Path],
    sequence_col: Optional[str] = None,
    fitness_col: str = 'fitness'
) -> Tuple[List[str], np.ndarray]:
    """
    Load sequences and fitness values from a CSV file.

    For multi-objective ("*_joint") datasets that expose `blue` and `red`
    columns (and no `fitness` column), the returned fitness is the
    geometric mean sqrt(blue * red). This scalarization rewards Pareto-
    optimal sequences (high in both channels) and penalizes imbalanced
    ones. Raw per-objective values are recoverable via
    `load_joint_objectives`.

    Parameters
    ----------
    filepath : str or Path
        Path to CSV file
    sequence_col : str, optional
        Column name for sequences. If None, auto-detects from common names:
        'sequence', 'seq', 'Combo', 'AACombo', 'variant'
    fitness_col : str, default='fitness'
        Column name for fitness values

    Returns
    -------
    Tuple[List[str], np.ndarray]
        (sequences, fitness_values)
    """
    filepath = Path(filepath)

    # Common sequence column names used across different methods
    SEQUENCE_COL_CANDIDATES = ['sequence', 'seq', 'Combo', 'AACombo', 'variant', 'mutant']

    if HAS_PANDAS:
        df = pd.read_csv(filepath)

        # Auto-detect sequence column if not specified
        if sequence_col is None:
            for col in SEQUENCE_COL_CANDIDATES:
                if col in df.columns:
                    sequence_col = col
                    break
            if sequence_col is None:
                raise ValueError(
                    f"Could not find sequence column. Tried: {SEQUENCE_COL_CANDIDATES}. "
                    f"Available columns: {list(df.columns)}"
                )

        sequences = df[sequence_col].tolist()
        if fitness_col not in df.columns and 'blue' in df.columns and 'red' in df.columns:
            blue = df['blue'].values.astype(float)
            red = df['red'].values.astype(float)
            # Geometric mean, clipped at 0 so negative-valued joints stay finite
            fitness = np.sqrt(np.clip(blue, 0, None) * np.clip(red, 0, None))
        else:
            fitness = df[fitness_col].values.astype(float)
    else:
        # Fallback without pandas
        sequences = []
        fitness = []
        with open(filepath, 'r') as f:
            header = f.readline().strip().split(',')

            # Auto-detect sequence column
            if sequence_col is None:
                for col in SEQUENCE_COL_CANDIDATES:
                    if col in header:
                        sequence_col = col
                        break
                if sequence_col is None:
                    raise ValueError(
                        f"Could not find sequence column. Tried: {SEQUENCE_COL_CANDIDATES}. "
                        f"Available columns: {header}"
                    )

            seq_idx = header.index(sequence_col)
            fit_idx = header.index(fitness_col)
            for line in f:
                parts = line.strip().split(',')
                sequences.append(parts[seq_idx])
                fitness.append(float(parts[fit_idx]))
        fitness = np.array(fitness)

    return sequences, fitness


def load_joint_objectives(
    dataset: str,
    data_dir: Optional[Union[str, Path]] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Load a multi-objective ("*_joint") dataset's raw per-objective values.

    Use this for post-hoc Pareto / hypervolume analysis on trajectories
    that methods produced under the scalarized objective.

    Parameters
    ----------
    dataset : str
        Joint dataset name (e.g. 'eqFP611_joint', 'mTagBFP2_joint')
    data_dir : str or Path, optional

    Returns
    -------
    Tuple[List[str], np.ndarray, np.ndarray]
        (sequences, blue, red) arrays aligned by index.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)
    path = data_dir / dataset / 'data.csv'
    if not path.exists():
        raise FileNotFoundError(f"No data.csv at {path}")
    df = pd.read_csv(path)
    seq_col = next((c for c in ['seq', 'sequence', 'AACombo', 'Combo', 'variant']
                    if c in df.columns), None)
    if seq_col is None or 'blue' not in df.columns or 'red' not in df.columns:
        raise ValueError(f"{path} does not look like a joint dataset (need seq + blue + red)")
    return df[seq_col].tolist(), df['blue'].values.astype(float), df['red'].values.astype(float)


def load_initial_training_set(
    dataset: str,
    data_dir: Optional[Union[str, Path]] = None,
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
    fitness_threshold: Optional[float] = None
) -> Tuple[List[str], np.ndarray]:
    """
    Load or sample an initial training set for a benchmark.

    For reproducibility, this function can:
    1. Load a pre-defined initial training set file
    2. Sample from the landscape with a fixed seed

    Parameters
    ----------
    dataset : str
        Dataset name
    data_dir : str or Path, optional
        Data directory
    n_samples : int, optional
        Number of samples for initial training. If None, loads pre-defined set.
    seed : int, optional
        Random seed for reproducible sampling
    fitness_threshold : float, optional
        Only include sequences with fitness below this threshold
        (to simulate starting from low-fitness variants)

    Returns
    -------
    Tuple[List[str], np.ndarray]
        (sequences, fitness_values) for initial training set
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    # Try to load pre-defined initial set
    init_files = [
        data_dir / dataset / 'initial_train.csv',
        data_dir / f'{dataset}_initial.csv',
        data_dir / dataset / 'train.csv',
    ]

    for init_file in init_files:
        if init_file.exists():
            return load_csv_data(init_file)

    # Sample from full landscape
    sequences, fitness = load_landscape_data(dataset, data_dir)

    if fitness_threshold is not None:
        mask = fitness < fitness_threshold
        sequences = [s for s, m in zip(sequences, mask) if m]
        fitness = fitness[mask]

    if n_samples is not None and n_samples < len(sequences):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(sequences), size=n_samples, replace=False)
        sequences = [sequences[i] for i in indices]
        fitness = fitness[indices]

    return sequences, fitness


def get_top_fitness_sequences(
    sequences: Sequence[str],
    fitness: Sequence[float],
    percentile: float = 10.0,
    top_k: Optional[int] = None
) -> Tuple[List[str], np.ndarray]:
    """
    Get the top fitness sequences from a landscape.

    Parameters
    ----------
    sequences : Sequence[str]
        All sequences
    fitness : Sequence[float]
        Corresponding fitness values
    percentile : float, default=10.0
        Top percentile to return (e.g., 10.0 = top 10%)
    top_k : int, optional
        If specified, return exactly top_k sequences instead of percentile

    Returns
    -------
    Tuple[List[str], np.ndarray]
        (top_sequences, top_fitness)
    """
    fitness = np.array(fitness)
    sequences = list(sequences)

    if top_k is not None:
        k = min(top_k, len(sequences))
    else:
        k = int(len(sequences) * percentile / 100)
        k = max(1, k)  # At least 1

    top_indices = np.argsort(fitness)[-k:]
    top_sequences = [sequences[i] for i in top_indices]
    top_fitness = fitness[top_indices]

    return top_sequences, top_fitness


def get_global_maximum(
    dataset: str,
    data_dir: Optional[Union[str, Path]] = None
) -> Tuple[str, float]:
    """
    Get the global maximum sequence and fitness for a dataset.

    Parameters
    ----------
    dataset : str
        Dataset name
    data_dir : str or Path, optional
        Data directory

    Returns
    -------
    Tuple[str, float]
        (global_max_sequence, global_max_fitness)
    """
    sequences, fitness = load_landscape_data(dataset, data_dir)
    max_idx = np.argmax(fitness)
    return sequences[max_idx], float(fitness[max_idx])


def get_fitness_bounds(
    dataset: str,
    data_dir: Optional[Union[str, Path]] = None
) -> Tuple[float, float]:
    """
    Get the minimum and maximum fitness values for a dataset.

    Parameters
    ----------
    dataset : str
        Dataset name
    data_dir : str or Path, optional
        Data directory

    Returns
    -------
    Tuple[float, float]
        (fitness_min, fitness_max)
    """
    _, fitness = load_landscape_data(dataset, data_dir)
    return float(np.min(fitness)), float(np.max(fitness))


def count_mutations(
    sequence: str,
    wild_type: str
) -> int:
    """
    Count the number of mutations relative to wild-type.

    Parameters
    ----------
    sequence : str
        Query sequence
    wild_type : str
        Wild-type reference sequence

    Returns
    -------
    int
        Number of positions that differ

    Raises
    ------
    ValueError
        If sequences have different lengths
    """
    if len(sequence) != len(wild_type):
        raise ValueError(
            f"Sequences must have same length. "
            f"Got {len(sequence)} and {len(wild_type)}"
        )
    return sum(s != w for s, w in zip(sequence, wild_type))


def filter_by_mutations(
    sequences: Sequence[str],
    fitness: Sequence[float],
    wild_type: str,
    min_mutations: int = 0,
    max_mutations: Optional[int] = None
) -> Tuple[List[str], np.ndarray]:
    """
    Filter sequences by number of mutations from wild-type.

    Parameters
    ----------
    sequences : Sequence[str]
        Sequences to filter
    fitness : Sequence[float]
        Corresponding fitness values
    wild_type : str
        Wild-type reference sequence
    min_mutations : int, default=0
        Minimum number of mutations (inclusive)
    max_mutations : int, optional
        Maximum number of mutations (inclusive)

    Returns
    -------
    Tuple[List[str], np.ndarray]
        (filtered_sequences, filtered_fitness)
    """
    fitness = np.array(fitness)
    filtered_seqs = []
    filtered_fit = []

    for seq, fit in zip(sequences, fitness):
        n_mut = count_mutations(seq, wild_type)
        if n_mut >= min_mutations:
            if max_mutations is None or n_mut <= max_mutations:
                filtered_seqs.append(seq)
                filtered_fit.append(fit)

    return filtered_seqs, np.array(filtered_fit)


def get_high_order_mutants(
    sequences: Sequence[str],
    fitness: Sequence[float],
    wild_type: str,
    min_mutations: int = 2,
    top_percentile: float = 10.0
) -> List[str]:
    """
    Get high-order mutants that are in the top fitness percentile.

    Used for computing the Recall of High-Order Mutants metric.

    Parameters
    ----------
    sequences : Sequence[str]
        All sequences
    fitness : Sequence[float]
        Corresponding fitness values
    wild_type : str
        Wild-type reference
    min_mutations : int, default=2
        Minimum mutations to be considered "high-order"
    top_percentile : float, default=10.0
        Top percentile to consider as "high fitness"

    Returns
    -------
    List[str]
        High-order mutants in the top fitness percentile
    """
    # Filter to high-order mutants
    high_order_seqs, high_order_fit = filter_by_mutations(
        sequences, fitness, wild_type, min_mutations=min_mutations
    )

    if len(high_order_seqs) == 0:
        return []

    # Get top percentile threshold from full landscape
    fitness = np.array(fitness)
    threshold = np.percentile(fitness, 100 - top_percentile)

    # Return high-order mutants above threshold
    return [
        seq for seq, fit in zip(high_order_seqs, high_order_fit)
        if fit >= threshold
    ]


# =============================================================================
# Encoding Functions
# =============================================================================

def one_hot_encode(sequence: str) -> np.ndarray:
    """
    One-hot encode a protein sequence.

    Parameters
    ----------
    sequence : str
        Amino acid sequence

    Returns
    -------
    np.ndarray
        One-hot encoded array of shape (len(sequence), 20)
    """
    encoding = np.zeros((len(sequence), len(AMINO_ACIDS)), dtype=np.float32)
    for i, aa in enumerate(sequence):
        if aa in AA_TO_IDX:
            encoding[i, AA_TO_IDX[aa]] = 1.0
    return encoding


def one_hot_encode_batch(sequences: Sequence[str]) -> np.ndarray:
    """
    One-hot encode a batch of protein sequences.

    Parameters
    ----------
    sequences : Sequence[str]
        List of amino acid sequences (must all be same length)

    Returns
    -------
    np.ndarray
        One-hot encoded array of shape (n_sequences, seq_length, 20)
    """
    if len(sequences) == 0:
        return np.array([])

    seq_length = len(sequences[0])
    encoded = np.zeros(
        (len(sequences), seq_length, len(AMINO_ACIDS)),
        dtype=np.float32
    )

    for i, seq in enumerate(sequences):
        encoded[i] = one_hot_encode(seq)

    return encoded


def decode_one_hot(encoding: np.ndarray) -> str:
    """
    Decode a one-hot encoded sequence back to amino acids.

    Parameters
    ----------
    encoding : np.ndarray
        One-hot encoded array of shape (seq_length, 20)

    Returns
    -------
    str
        Decoded amino acid sequence
    """
    indices = np.argmax(encoding, axis=-1)
    return ''.join(IDX_TO_AA[i] for i in indices)


# =============================================================================
# Fitness Lookup
# =============================================================================

class FitnessLandscape:
    """
    Fitness landscape lookup for efficient sequence-to-fitness queries.

    This class provides O(1) lookup of fitness values for sequences,
    useful when evaluating generated sequences against the ground truth.

    Parameters
    ----------
    sequences : Sequence[str]
        All sequences in the landscape
    fitness : Sequence[float]
        Corresponding fitness values
    """

    def __init__(
        self,
        sequences: Sequence[str],
        fitness: Sequence[float]
    ):
        self.fitness_dict: Dict[str, float] = {
            seq: float(fit) for seq, fit in zip(sequences, fitness)
        }
        self._sequences = list(sequences)
        self._fitness = np.array(fitness)
        self._min = float(np.min(fitness))
        self._max = float(np.max(fitness))

        # Cache global maximum
        max_idx = np.argmax(fitness)
        self._global_max_seq = sequences[max_idx]
        self._global_max_fit = float(fitness[max_idx])

    def __len__(self) -> int:
        return len(self.fitness_dict)

    def __contains__(self, sequence: str) -> bool:
        return sequence in self.fitness_dict

    def get_fitness(self, sequence: str) -> Optional[float]:
        """Get fitness for a sequence, or None if not in landscape."""
        return self.fitness_dict.get(sequence)

    def get_fitness_batch(
        self,
        sequences: Sequence[str],
        default: float = float('nan')
    ) -> np.ndarray:
        """Get fitness values for multiple sequences."""
        return np.array([
            self.fitness_dict.get(seq, default)
            for seq in sequences
        ])

    @property
    def min_fitness(self) -> float:
        return self._min

    @property
    def max_fitness(self) -> float:
        return self._max

    @property
    def global_max_sequence(self) -> str:
        return self._global_max_seq

    @property
    def global_max_fitness(self) -> float:
        return self._global_max_fit

    @property
    def sequences(self) -> List[str]:
        return self._sequences

    @property
    def fitness(self) -> np.ndarray:
        return self._fitness

    @classmethod
    def from_dataset(
        cls,
        dataset: str,
        data_dir: Optional[Union[str, Path]] = None
    ) -> 'FitnessLandscape':
        """Load a FitnessLandscape from a dataset name."""
        sequences, fitness = load_landscape_data(dataset, data_dir)
        return cls(sequences, fitness)


def load_random_seeds(filepath: Optional[Union[str, Path]] = None) -> List[int]:
    """
    Load random seeds from the rand_seeds.txt file.

    Parameters
    ----------
    filepath : str or Path, optional
        Path to seeds file. Defaults to ../rand_seeds.txt

    Returns
    -------
    List[int]
        List of random seeds
    """
    if filepath is None:
        filepath = Path(__file__).parent.parent / "rand_seeds.txt"

    seeds = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(int(line))
    return seeds
