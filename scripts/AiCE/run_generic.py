#!/usr/bin/env python
"""
run_generic.py - Execute AiCE optimization on any dataset with comprehensive metrics

Generic dataset runner for AiCE (AI-guided Combinatorial Editing). Accepts a
--dataset argument and auto-detects sequence length, wildtype, and paths from the
data.

Configuration:
    - Model: ProteinMPNN-based inverse folding
    - Scoring: Frequency-based filtering with beta/gamma thresholds
    - Multi-mutation: LD (Linkage Disequilibrium) and SCA (Statistical Coupling Analysis)
    - Batch size: 96 (for fair comparison with ALDE)
    - Rounds: 15 (1 initial + 14 iterations, simulated via score ranking)

Metrics computed (same as ALDE for comparison):
    - Exploration: High-Fitness Proximity, Novelty, Batch Diversity
    - Functional: Normalized Fitness (Top-K), Max Fitness
    - Model Quality: Spearman Correlation
    - Success: Simple Regret

Usage:
    # Single run on 4site_GB1
    AiCE/env/bin/python scripts/AiCE/run_generic.py --dataset 4site_GB1 --seed 42

    # Multiple runs on GB1
    AiCE/env/bin/python scripts/AiCE/run_generic.py --dataset 4site_GB1 --seeds 42 123 456 789 1000

    # Use predefined seeds from file (50 runs)
    AiCE/env/bin/python scripts/AiCE/run_generic.py --dataset 4site_TRPB --seed_file rand_seeds.txt --num_seeds 30

    # Skip metrics computation (faster)
    AiCE/env/bin/python scripts/AiCE/run_generic.py --dataset 4site_PhoQ --seed 42 --skip_metrics

    # Custom output path
    AiCE/env/bin/python scripts/AiCE/run_generic.py --dataset 4site_GB1 --seed 42 --output_path results/4site_GB1_AiCE/
"""

from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
import torch
import random
import os
import sys
import warnings
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from itertools import combinations
from scipy import stats

# Add scripts directory to path for AiCE modules
# Canonical location is scripts/<Method>/. This file used to be invoked through a
# symlink in <Method>/, where `__file__/..` happened to be the benchmark root; walk
# up to find the root instead, so it runs correctly from either path.
# (Same idiom as scripts/EVOLVEpro/run_generic.py.)
_p = os.path.dirname(os.path.realpath(__file__))
while os.path.dirname(_p) != _p and not os.path.isdir(os.path.join(_p, 'utils')):
    _p = os.path.dirname(_p)
BENCHMARK_ROOT = _p
sys.path.insert(0, BENCHMARK_ROOT)

from utils.compat import (
    compute_all_metrics,
    aggregate_run_metrics,
    load_landscape_data,
    MetricsResult,
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
)


# =============================================================================
# AiCE Configuration
# =============================================================================

@dataclass
class AiCEConfig:
    """Configuration for AiCE scoring."""
    beta: float = 0.8          # Threshold for non-coil regions
    gamma: float = 0.5         # Threshold for coil regions
    ld_threshold: float = 0.5  # LD score threshold for multi-mutations
    sca_percentile: float = 0.9  # SCA score percentile threshold
    num_mpnn_samples: int = 1000  # Number of ProteinMPNN samples
    sampling_temp: float = 0.5   # ProteinMPNN sampling temperature


# =============================================================================
# AiCE Scorer (Generic)
# =============================================================================

class AiCEScorer:
    """
    AiCE scoring for protein mutations.

    Uses frequency-based scoring from ProteinMPNN inverse folding or
    simulated frequencies from the fitness landscape.

    The AiCE methodology:
    1. Single mutations: Score based on position-specific amino acid frequencies
       from inverse folding samples (structure-compatible sequences)
    2. Multi-mutations: Consider LD (linkage disequilibrium) and SCA
       (statistical coupling analysis) to identify compatible combinations
    """

    def __init__(
        self,
        sequences: List[str],
        fitness: np.ndarray,
        config: AiCEConfig = None,
        wildtype: str = "",
        freq_data: Optional[Dict] = None
    ):
        self.sequences = sequences
        # `fitness` is kept only for the test-time evaluation step (it is the
        # oracle). The scorer itself must NOT peek at it for template / LD
        # construction — that was the source of label leakage in the previous
        # implementation. Templates are built from observations passed in via
        # `update_with_observations(...)` plus the structure-only wildtype
        # neighborhood.
        self.fitness = fitness
        self.config = config or AiCEConfig()
        self.wildtype = wildtype
        self.seq_length = len(wildtype)

        # Build sequence to index mapping
        self.seq_to_idx = {seq: idx for idx, seq in enumerate(sequences)}

        # Amino acids
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"

        # Observed (queried) data — empty until update_with_observations is called.
        self._observed_seqs: List[str] = []
        self._observed_fitness: np.ndarray = np.array([], dtype=float)

        # Precompute positions that show variation across the library.
        self.variable_positions = self._find_variable_positions()

        if freq_data is not None:
            # External (e.g. real ProteinMPNN) frequencies — use as-is.
            self.freq_data = freq_data
            self.ld_matrix = np.ones((max(len(self.variable_positions), 1),
                                      max(len(self.variable_positions), 1)))
        else:
            # Initialize templates from structure-only signal (wildtype + WT
            # neighborhood). No fitness labels are used here.
            self.freq_data = self._compute_frequencies_from_seqs(
                self._structural_templates()
            )
            self.ld_matrix = np.ones((max(len(self.variable_positions), 1),
                                      max(len(self.variable_positions), 1)))

        # Precompute all scores for efficiency
        self._precompute_scores()

    def _find_variable_positions(self) -> List[int]:
        """Find positions that have variation in the sequence set."""
        variable_pos = []
        for pos in range(self.seq_length):
            aa_set = set()
            for seq in self.sequences[:1000]:  # Sample for efficiency
                if len(seq) > pos:
                    aa_set.add(seq[pos])
            if len(aa_set) > 1:
                variable_pos.append(pos)
        return variable_pos

    def _structural_templates(self) -> List[str]:
        """
        Structure-only template set: the wildtype plus its Hamming-≤2 neighborhood.
        Independent of fitness labels — safe to use before any observation.
        """
        wt = self.wildtype
        tmpl = [wt]
        for seq in self.sequences:
            try:
                d = hamming_distance(seq, wt)
            except ValueError:
                continue
            if 0 < d <= 2:
                tmpl.append(seq)
        return tmpl

    def _compute_frequencies_from_seqs(self, template_seqs: List[str]) -> Dict:
        """Position-specific amino-acid frequencies from a given template set."""
        freq_data: Dict[int, Dict[str, float]] = {}
        if not template_seqs:
            for pos in range(self.seq_length):
                freq_data[pos] = {aa: 0.0 for aa in self.amino_acids}
            return freq_data
        n = len(template_seqs)
        for pos in range(self.seq_length):
            freq_data[pos] = {}
            for aa in self.amino_acids:
                count = sum(1 for seq in template_seqs if len(seq) > pos and seq[pos] == aa)
                freq_data[pos][aa] = count / n
        return freq_data

    def _compute_ld_matrix_from_seqs(self, template_seqs: List[str]) -> np.ndarray:
        """LD matrix over `variable_positions` computed from a given template set."""
        n_pos = len(self.variable_positions)
        if n_pos == 0:
            return np.ones((1, 1))
        ld_matrix = np.ones((n_pos, n_pos))
        if len(template_seqs) < 10:
            return ld_matrix
        for i, pos_i in enumerate(self.variable_positions):
            for j, pos_j in enumerate(self.variable_positions):
                if i >= j:
                    continue
                pair_counts: Dict[Tuple[str, str], int] = {}
                for seq in template_seqs:
                    if len(seq) > max(pos_i, pos_j):
                        pair = (seq[pos_i], seq[pos_j])
                        pair_counts[pair] = pair_counts.get(pair, 0) + 1
                if pair_counts:
                    total = sum(pair_counts.values())
                    max_count = max(pair_counts.values())
                    ld = max_count / total
                    ld_matrix[i, j] = ld
                    ld_matrix[j, i] = ld
        return ld_matrix

    def update_with_observations(
        self,
        observed_indices: np.ndarray,
        observed_fitness: np.ndarray,
        top_pct: float = 0.5,
    ) -> None:
        """
        Refit templates from the set of *queried* sequences and their *observed*
        fitness. Combines:
          - top-`top_pct` of observations by observed fitness (fitness-informed
            but label-free w.r.t. unqueried sequences — fully legitimate in an
            iterative campaign);
          - the structure-only wildtype neighborhood (Hamming ≤ 2 from WT).
        Then recomputes `freq_data`, `ld_matrix`, and `all_scores`.
        """
        observed_indices = np.asarray(observed_indices, dtype=int)
        observed_fitness = np.asarray(observed_fitness, dtype=float)
        self._observed_seqs = [self.sequences[i] for i in observed_indices]
        self._observed_fitness = observed_fitness

        # Top observations by observed fitness
        if len(self._observed_seqs) >= 2:
            n_top = max(1, int(np.ceil(len(self._observed_seqs) * top_pct)))
            top_order = np.argsort(observed_fitness)[::-1][:n_top]
            top_obs = [self._observed_seqs[i] for i in top_order]
        else:
            top_obs = list(self._observed_seqs)

        # Combine with structural templates (wildtype neighborhood)
        struct_tmpl = self._structural_templates()
        template_seqs = top_obs + struct_tmpl

        self.freq_data = self._compute_frequencies_from_seqs(template_seqs)
        # LD is computed from top observations only — structural templates
        # are dominated by WT, which would saturate LD.
        self.ld_matrix = self._compute_ld_matrix_from_seqs(top_obs)
        self._precompute_scores()

    def _precompute_scores(self) -> None:
        """Precompute scores for all sequences for efficiency."""
        self.all_scores = np.zeros(len(self.sequences))

        for idx, seq in enumerate(self.sequences):
            if len(seq) != self.seq_length:
                continue

            # Position-specific frequency score
            freq_score = 1.0
            n_positions_scored = 0
            for pos, aa in enumerate(seq):
                freq = self.freq_data.get(pos, {}).get(aa, 0.0)
                if freq > 0:
                    freq_score *= freq
                    n_positions_scored += 1

            # Normalize by number of positions to avoid penalizing longer sequences
            if n_positions_scored > 0:
                freq_score = freq_score ** (1.0 / n_positions_scored)

            # LD bonus for multi-mutations
            try:
                n_muts = hamming_distance(seq, self.wildtype)
            except ValueError:
                n_muts = 0

            if n_muts >= 2 and len(self.variable_positions) > 1:
                # Get mutated positions
                mut_positions = [i for i, (a, b) in enumerate(zip(seq, self.wildtype)) if a != b]

                # Find indices in variable_positions
                mut_var_indices = []
                for mp in mut_positions:
                    if mp in self.variable_positions:
                        mut_var_indices.append(self.variable_positions.index(mp))

                if len(mut_var_indices) >= 2:
                    ld_scores = []
                    for p1 in range(len(mut_var_indices)):
                        for p2 in range(p1 + 1, len(mut_var_indices)):
                            if mut_var_indices[p1] < len(self.ld_matrix) and mut_var_indices[p2] < len(self.ld_matrix):
                                ld_scores.append(self.ld_matrix[mut_var_indices[p1], mut_var_indices[p2]])
                    ld_bonus = np.mean(ld_scores) if ld_scores else 1.0
                    freq_score *= ld_bonus

            self.all_scores[idx] = freq_score

    def score_single_mutation(self, pos: int, aa: str) -> float:
        """Score a single mutation based on frequency."""
        if pos in self.freq_data and aa in self.freq_data[pos]:
            return self.freq_data[pos][aa]
        return 0.0

    def score_sequence(self, sequence: str) -> float:
        """
        Score a complete sequence using AiCE methodology.

        The score combines:
        1. Position-specific frequencies (structure compatibility)
        2. LD scores for multi-mutations (evolutionary coupling)
        """
        idx = self.seq_to_idx.get(sequence)
        if idx is not None:
            return self.all_scores[idx]

        # Fallback computation for new sequences
        if len(sequence) != self.seq_length:
            return 0.0

        score = 1.0
        n_positions = 0
        for pos, aa in enumerate(sequence):
            freq = self.score_single_mutation(pos, aa)
            if freq > 0:
                score *= freq
                n_positions += 1

        if n_positions > 0:
            score = score ** (1.0 / n_positions)

        return score

    def rank_sequences(self, n_select: int = 480) -> List[Tuple[str, float, int]]:
        """
        Rank all sequences by AiCE score and select top N.

        Returns list of (sequence, score, original_index) tuples.
        """
        indices = np.argsort(self.all_scores)[::-1][:n_select]
        return [(self.sequences[i], self.all_scores[i], i) for i in indices]

    def select_diverse_batch(
        self,
        n_select: int,
        already_selected: Optional[set] = None,
        diversity_weight: float = 0.3
    ) -> List[int]:
        """
        Select a batch balancing AiCE score and diversity.

        Uses a greedy approach that balances:
        1. AiCE score (exploitation)
        2. Hamming distance diversity (exploration)
        """
        if already_selected is None:
            already_selected = set()

        # Get unselected indices sorted by score
        available_mask = np.ones(len(self.sequences), dtype=bool)
        for idx in already_selected:
            available_mask[idx] = False

        available_indices = np.where(available_mask)[0]
        available_scores = self.all_scores[available_indices]

        # Sort by score
        sorted_order = np.argsort(available_scores)[::-1]
        sorted_indices = available_indices[sorted_order]
        sorted_scores = available_scores[sorted_order]

        # Normalize scores to [0, 1]
        if len(sorted_scores) > 0 and sorted_scores.max() > sorted_scores.min():
            norm_scores = (sorted_scores - sorted_scores.min()) / (sorted_scores.max() - sorted_scores.min() + 1e-10)
        else:
            norm_scores = np.ones_like(sorted_scores)

        selected = []
        selected_seqs = []

        # Consider top candidates (3x batch size for diversity selection)
        n_candidates = min(len(sorted_indices), n_select * 3)
        candidate_indices = sorted_indices[:n_candidates]
        candidate_scores = norm_scores[:n_candidates]

        for i, (idx, score) in enumerate(zip(candidate_indices, candidate_scores)):
            if len(selected) >= n_select:
                break

            seq = self.sequences[idx]

            # Compute diversity bonus
            if selected_seqs:
                try:
                    min_dist = min(hamming_distance(seq, s) for s in selected_seqs)
                    max_possible_dist = self.seq_length
                    diversity_bonus = min_dist / max_possible_dist * diversity_weight
                except ValueError:
                    diversity_bonus = diversity_weight
            else:
                diversity_bonus = diversity_weight  # First selection gets full bonus

            # Combined score with exploration bonus for later selections
            exploration_bonus = i / max(n_candidates, 1) * 0.1  # Small bonus for exploration
            combined = score * (1 - diversity_weight) + diversity_bonus + exploration_bonus

            # Always add if combined score is reasonable
            if combined > 0.1 or len(selected) < n_select // 2:
                selected.append(idx)
                selected_seqs.append(seq)

        # If we don't have enough, fill with top-scoring remaining
        if len(selected) < n_select:
            remaining = [idx for idx in candidate_indices if idx not in set(selected)]
            selected.extend(remaining[:n_select - len(selected)])

        return selected[:n_select]


# =============================================================================
# Additional Metrics Functions
# =============================================================================

def epistatic_score_correlation(
    sequences: List[str],
    true_fitness: np.ndarray,
    pred_fitness: np.ndarray,
    wildtype: str,
    min_mutations: int = 2
) -> float:
    """
    Compute Spearman correlation on epistatic (high-order mutant) sequences.
    """
    # Filter to high-order mutants
    high_order_mask = []
    for seq in sequences:
        try:
            n_mut = hamming_distance(seq, wildtype)
            high_order_mask.append(n_mut >= min_mutations)
        except ValueError:
            high_order_mask.append(False)

    high_order_mask = np.array(high_order_mask)

    if np.sum(high_order_mask) < 10:
        return 0.0

    ho_true = true_fitness[high_order_mask]
    ho_pred = pred_fitness[high_order_mask]

    corr, _ = stats.spearmanr(ho_true, ho_pred)
    return float(corr) if not np.isnan(corr) else 0.0


def recall_high_order_mutants(
    sequences: List[str],
    true_fitness: np.ndarray,
    pred_fitness: np.ndarray,
    wildtype: str,
    min_mutations: int = 2,
    top_k: int = 100
) -> float:
    """
    Compute recall of top-k high-order mutants.
    """
    # Filter to high-order mutants
    high_order_indices = []
    for i, seq in enumerate(sequences):
        try:
            n_mut = hamming_distance(seq, wildtype)
            if n_mut >= min_mutations:
                high_order_indices.append(i)
        except ValueError:
            continue

    if len(high_order_indices) < top_k:
        return 0.0

    high_order_indices = np.array(high_order_indices)
    ho_true = true_fitness[high_order_indices]
    ho_pred = pred_fitness[high_order_indices]

    # Get top-k by true fitness
    true_top_k = set(high_order_indices[np.argsort(ho_true)[-top_k:]])
    # Get top-k by predicted fitness
    pred_top_k = set(high_order_indices[np.argsort(ho_pred)[-top_k:]])

    overlap = len(true_top_k & pred_top_k)
    return overlap / top_k


# =============================================================================
# Generic Data Loading
# =============================================================================

def load_dataset(data_dir: str, dataset: str) -> Tuple[List[str], np.ndarray, pd.DataFrame, str, int]:
    """
    Load a fitness landscape dataset generically.

    Looks for data/<dataset>/data.csv with 'seq' or 'sequence' and 'fitness' columns.
    Auto-detects sequence length and uses the highest-fitness sequence as wildtype.

    Args:
        data_dir: Base directory for data files
        dataset: Name of the dataset (subdirectory name)

    Returns:
        sequences: List of protein sequences
        fitness: Normalized fitness values
        df: Full dataframe
        wildtype: Wildtype sequence (highest fitness sequence)
        seq_length: Detected sequence length
    """
    data_path = os.path.join(data_dir, dataset, "data.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Get sequences
    if 'seq' in df.columns:
        sequences = df['seq'].tolist()
    elif 'sequence' in df.columns:
        sequences = df['sequence'].tolist()
    else:
        raise ValueError(f"No sequence column ('seq' or 'sequence') found in {data_path}")

    # Get fitness values
    if 'fitness' not in df.columns and 'blue' in df.columns and 'red' in df.columns:
        # Multi-objective ("*_joint") dataset: scalarize to geometric mean
        blue = df['blue'].values.astype(float)
        red = df['red'].values.astype(float)
        fitness = np.sqrt(np.clip(blue, 0, None) * np.clip(red, 0, None))
    else:
        fitness = df['fitness'].values

    # Auto-detect sequence length from the data (use mode of lengths)
    seq_lengths = [len(s) for s in sequences]
    seq_length = max(set(seq_lengths), key=seq_lengths.count)

    # Wildtype resolution order — strict-to-lax. argmax(fitness) is LEAKY for
    # AiCE because the scorer's WT-neighborhood template would then point at
    # the global maximum, so we only fall back to it when nothing else exists.
    wt_fasta = os.path.join(data_dir, dataset, "wt.fasta")
    wildtype = None
    if os.path.exists(wt_fasta):
        with open(wt_fasta) as fh:
            lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith('>')]
        if lines:
            wildtype = "".join(lines)
    if wildtype is None and 'n_muts' in df.columns and (df['n_muts'] == 0).any():
        wt_idx = int(df.index[df['n_muts'] == 0][0])
        wildtype = sequences[wt_idx]
    if wildtype is None:
        print(f"WARNING: no wt.fasta or n_muts==0 row for {dataset}; falling back to "
              f"argmax(fitness) as wildtype — LEAKY for AiCE. Drop a "
              f"data/{dataset}/wt.fasta to fix.")
        wildtype = sequences[int(np.argmax(fitness))]

    # Normalize fitness to [0, 1]
    fitness_min = fitness.min()
    fitness_max = fitness.max()
    if fitness_max > fitness_min:
        fitness = (fitness - fitness_min) / (fitness_max - fitness_min)

    return sequences, fitness, df, wildtype, seq_length


# =============================================================================
# Main Experiment
# =============================================================================

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_single_experiment(
    seed: int,
    dataset: str,
    output_path: str,
    verbose: int = 2,
    run_id: Optional[int] = None,
    compute_metrics: bool = True,
    data_dir: str = os.path.join(BENCHMARK_ROOT, 'data'),
    aice_config: Optional[AiCEConfig] = None
) -> Dict[str, Any]:
    """
    Run a single AiCE optimization experiment on a generic dataset.

    Since AiCE is primarily a one-shot prediction method, we simulate
    iterative optimization by:
    1. Starting with random initial samples
    2. Using AiCE scores to guide batch selection
    3. Comparing results against ALDE's iterative approach

    Args:
        seed: Random seed for reproducibility
        dataset: Dataset name (subdirectory under data_dir)
        output_path: Base path for saving results
        verbose: Verbosity level
        run_id: Optional run identifier
        compute_metrics: Whether to compute evaluation metrics
        data_dir: Base directory for data files
        aice_config: AiCE configuration parameters

    Returns:
        Dictionary containing results and metrics
    """
    # Configuration: 96 variants per round, 5 rounds total — matches the
    # benchmark-wide budget used by ALDE, CLADE, FLEXS, etc.
    batch_size = 96
    n_rounds = 5
    n_init = batch_size  # First round
    budget = batch_size * (n_rounds - 1)  # Remaining 4 rounds
    total_samples = n_init + budget  # 96 * 5 = 480

    if run_id is None:
        run_id = seed

    if aice_config is None:
        aice_config = AiCEConfig()

    print(f"\n{'='*60}")
    print(f"Starting AiCE optimization on {dataset}")
    print(f"  Seed: {seed}")
    print(f"  Model: AiCE (ProteinMPNN-based scoring)")
    print(f"  Beta (non-coil threshold): {aice_config.beta}")
    print(f"  Gamma (coil threshold): {aice_config.gamma}")
    print(f"  Batch size: {batch_size}")
    print(f"  Initial samples: {n_init}")
    print(f"  Budget: {budget}")
    print(f"  Total samples: {total_samples}")
    print(f"  Rounds: {n_rounds} (1 init + {n_rounds - 1} iterations)")
    print(f"{'='*60}\n")

    # Set random seeds
    set_seed(seed)

    # Load data
    all_sequences, all_fitness, df, wildtype, seq_length = load_dataset(data_dir, dataset)
    global_max = np.max(all_fitness)
    global_min = np.min(all_fitness)

    if verbose >= 1:
        print(f"Loaded {len(all_sequences)} sequences")
        print(f"Sequence length (detected): {seq_length}")
        print(f"Wildtype: {wildtype[:50]}{'...' if len(wildtype) > 50 else ''}")
        print(f"Fitness range: [{global_min:.4f}, {global_max:.4f}]")

    # Initialize AiCE scorer. Templates start from the structure-only WT
    # neighborhood; observed fitness is folded in via update_with_observations
    # after each round (proper iterative MLDE, no landscape-fitness leak).
    print("Initializing AiCE scorer (structure-only templates; no fitness leak)...")
    scorer = AiCEScorer(all_sequences, all_fitness, aice_config, wildtype=wildtype)

    # Create output directory
    subdir = os.path.join(output_path, dataset, "aice", "")
    os.makedirs(subdir, exist_ok=True)

    # Random initialization (matching ALDE)
    all_indices = list(range(len(all_sequences)))
    random.shuffle(all_indices)
    initial_indices = all_indices[:n_init]

    # Track all queries
    queried_indices = list(initial_indices)
    queried_set = set(queried_indices)

    # Fold the initial random observations into the scorer before round 1.
    scorer.update_with_observations(
        np.asarray(queried_indices, dtype=int),
        all_fitness[np.asarray(queried_indices, dtype=int)],
    )

    # Iterative selection using AiCE scores
    start_time = datetime.now()

    for round_idx in range(n_rounds - 1):  # n_rounds - 1 optimization rounds (after init)
        if verbose >= 2:
            print(f"\n--- Round {round_idx + 1}/{n_rounds - 1} ---")

        # Select next batch using AiCE scores + diversity
        batch_indices = scorer.select_diverse_batch(
            n_select=batch_size,
            already_selected=queried_set,
            diversity_weight=0.2
        )

        queried_indices.extend(batch_indices)
        queried_set.update(batch_indices)

        # Refit the scorer using all observations so far.
        scorer.update_with_observations(
            np.asarray(queried_indices, dtype=int),
            all_fitness[np.asarray(queried_indices, dtype=int)],
        )

        if verbose >= 2:
            batch_fitness = all_fitness[batch_indices]
            print(f"  Batch fitness: max={np.max(batch_fitness):.4f}, "
                  f"mean={np.mean(batch_fitness):.4f}")
            print(f"  Total queried: {len(queried_indices)}")

    runtime = (datetime.now() - start_time).total_seconds()

    # Save results
    queried_indices_tensor = torch.tensor(queried_indices)
    result_path = os.path.join(subdir, f"AiCE-seed{seed}indices.pt")
    torch.save(queried_indices_tensor, result_path)

    # Also save random baseline
    random_indices = all_indices[:total_samples]
    random_baseline_path = os.path.join(subdir, f'Random_{seed}indices.pt')
    torch.save(torch.tensor(random_indices), random_baseline_path)

    if verbose >= 1:
        print(f"\nResults saved to: {result_path}")

    # Prepare result dictionary
    result = {
        'seed': seed,
        'run_id': run_id,
        'result_path': result_path,
        'runtime_seconds': runtime,
        'n_queries': len(queried_indices),
        'config': {
            'model': 'AiCE',
            'protein': dataset,
            'beta': aice_config.beta,
            'gamma': aice_config.gamma,
            'ld_threshold': aice_config.ld_threshold,
            'batch_size': batch_size,
            'n_init': n_init,
            'budget': budget,
        }
    }

    # Compute metrics
    if compute_metrics:
        print("\nComputing evaluation metrics...")

        queried_seqs = [all_sequences[i] for i in queried_indices]
        initial_seqs = [all_sequences[i] for i in initial_indices]
        queried_fitness = all_fitness[queried_indices]

        metrics = MetricsResult()

        # Record per-query indices for post-hoc multi-objective analysis on _joint datasets
        metrics.queried_indices = [int(i) for i in queried_indices]

        # Exploration metrics
        metrics.high_fitness_proximity = high_fitness_proximity(
            queried_seqs, all_sequences, all_fitness, percentile=0.9
        )

        non_initial_seqs = queried_seqs[n_init:]
        if non_initial_seqs:
            metrics.novelty = novelty(non_initial_seqs, initial_seqs)

        metrics.batch_diversity = batch_diversity(queried_seqs)

        # Functional metrics
        metrics.normalized_fitness_median_top128 = normalized_fitness_topk(
            queried_fitness, k=128, min_fitness=global_min, max_fitness=global_max
        )
        metrics.normalized_fitness_median_top256 = normalized_fitness_topk(
            queried_fitness, k=256, min_fitness=global_min, max_fitness=global_max
        )
        metrics.max_fitness = float(np.max(queried_fitness))

        # Model quality metrics
        aice_scores = scorer.all_scores
        metrics.spearman_correlation = spearman_correlation(all_fitness, aice_scores)

        # Epistatic correlation and recall (computed on holdout set)
        queried_set_final = set(queried_indices)
        holdout_mask = np.array([i not in queried_set_final for i in range(len(all_fitness))])

        if np.sum(holdout_mask) > 100:
            holdout_seqs = [all_sequences[i] for i, m in enumerate(holdout_mask) if m]
            holdout_true = all_fitness[holdout_mask]
            holdout_pred = aice_scores[holdout_mask]

            metrics.epistatic_correlation = epistatic_score_correlation(
                holdout_seqs, holdout_true, holdout_pred, wildtype
            )
            metrics.recall_high_order = recall_high_order_mutants(
                holdout_seqs, holdout_true, holdout_pred, wildtype,
                min_mutations=2, top_k=100
            )

        # Success metrics
        metrics.simple_regret = simple_regret(metrics.max_fitness, global_max)
        metrics.global_max_found = (metrics.max_fitness >= global_max * 0.99)

        # Uncertainty metrics
        # AiCE doesn't provide uncertainty estimates, so we use score variance as proxy
        if np.sum(holdout_mask) > 100:
            holdout_true = all_fitness[holdout_mask]
            holdout_pred = aice_scores[holdout_mask]
            # Normalize predictions to same scale as fitness
            if np.max(holdout_pred) > 0:
                holdout_pred_norm = holdout_pred / np.max(holdout_pred)
            else:
                holdout_pred_norm = holdout_pred
            # Use prediction magnitude as uncertainty proxy (higher score = more confident)
            holdout_std = 1.0 - holdout_pred_norm + 0.1  # Inverse confidence as uncertainty

            metrics.miscalibration_area = miscalibration_area(
                holdout_true, holdout_pred_norm, holdout_std
            )
            metrics.expected_calibration_error = expected_calibration_error(
                holdout_true, holdout_pred_norm, holdout_std
            )

        # Trajectories
        fitness_traj = []
        for i in range(0, len(queried_fitness), batch_size):
            batch_fitness = queried_fitness[:i + batch_size]
            fitness_traj.append(np.max(batch_fitness))
        metrics.fitness_trajectory = fitness_traj
        metrics.regret_trajectory = [global_max - f for f in fitness_traj]

        result['metrics'] = metrics.to_dict()
        result['fitness_trajectory'] = metrics.fitness_trajectory
        result['regret_trajectory'] = metrics.regret_trajectory

        # Print summary (in ALDE order)
        print("\n" + "-"*40)
        print("Metrics Summary:")
        print("-"*40)
        print(f"  High-Fitness Proximity: {metrics.high_fitness_proximity:.4f}")
        print(f"  Novelty: {metrics.novelty:.4f}")
        print(f"  Batch Diversity: {metrics.batch_diversity:.4f}")
        print(f"  Normalized Fitness (Top-128): {metrics.normalized_fitness_median_top128:.4f}")
        print(f"  Normalized Fitness (Top-256): {metrics.normalized_fitness_median_top256:.4f}")
        print(f"  Max Fitness: {metrics.max_fitness:.4f}")
        print(f"  Spearman Correlation: {metrics.spearman_correlation:.4f}")
        print(f"  Epistatic Correlation: {metrics.epistatic_correlation:.4f}")
        print(f"  Recall High-Order: {metrics.recall_high_order:.4f}")
        print(f"  Simple Regret: {metrics.simple_regret:.4f}")
        print(f"  Miscalibration Area: {metrics.miscalibration_area:.4f}")
        print(f"  Expected Calibration Error: {metrics.expected_calibration_error:.4f}")
        print(f"  Global Max Found: {metrics.global_max_found}")

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


def save_aggregated_results(results: List[Dict[str, Any]], output_path: str, dataset: str) -> None:
    """Save aggregated results across all runs."""
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(MetricsResult)}
    rename = {'hit_rate': 'hit_rate_value'}
    metrics_results = []
    for r in results:
        if 'metrics' in r:
            kwargs = {rename.get(k, k): v for k, v in r['metrics'].items()}
            kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
            mr = MetricsResult(**kwargs)
            mr.fitness_trajectory = r.get('fitness_trajectory', [])
            mr.regret_trajectory = r.get('regret_trajectory', [])
            metrics_results.append(mr)

    if not metrics_results:
        print("No metrics to aggregate.")
        return

    # Aggregate metrics (in ALDE order)
    metrics_names = [
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
    for name in metrics_names:
        values = [getattr(r, name) for r in metrics_results]
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

    # Count global max hits (special metric - count not mean/std)
    global_max_hit_count = sum(1 for r in metrics_results if r.global_max_found)
    aggregated['global_max_hit_count'] = {
        'count': global_max_hit_count,
        'total_runs': len(metrics_results),
        'hit_rate': global_max_hit_count / len(metrics_results) if metrics_results else 0.0
    }

    # Create summary DataFrame (maintain order)
    summary_data = []
    for name in metrics_names:
        stats_val = aggregated[name]
        summary_data.append({
            'metric': name,
            'mean': stats_val['mean'],
            'std': stats_val['std'],
            'min': stats_val['min'],
            'max': stats_val['max']
        })

    # Add global_max_hit_count as a special row
    summary_data.append({
        'metric': 'global_max_hit_count',
        'mean': float(global_max_hit_count),
        'std': 0.0,
        'min': float(global_max_hit_count),
        'max': float(global_max_hit_count)
    })

    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    summary_path = os.path.join(output_path, dataset, 'aice', 'aggregated_metrics.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nAggregated metrics saved to: {summary_path}")

    # Save complete aggregated results to JSON
    aggregated_json_path = os.path.join(output_path, dataset, 'aice', 'aggregated_results.json')
    with open(aggregated_json_path, 'w') as f:
        json.dump({
            'aggregated_metrics': aggregated,
            'n_runs': len(results),
            'seeds': [r['seed'] for r in results],
            'config': results[0].get('config', {}) if results else {}
        }, f, indent=2, default=str)
    print(f"Aggregated results saved to: {aggregated_json_path}")

    # Print summary table
    print("\n" + "="*70)
    print(f"AGGREGATED METRICS SUMMARY ({dataset})")
    print("="*70)
    print(f"{'Metric':<40} {'Mean':>10} {'Std':>10}")
    print("-"*70)
    for _, row in summary_df.iterrows():
        if row['metric'] == 'global_max_hit_count':
            print(f"{row['metric']:<40} {int(row['mean']):>10} {'(count)':>10}")
        else:
            print(f"{row['metric']:<40} {row['mean']:>10.4f} {row['std']:>10.4f}")
    print("="*70)
    print(f"\nGlobal Max Hit Rate: {global_max_hit_count}/{len(metrics_results)} = {global_max_hit_count/len(metrics_results)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Run AiCE optimization on any dataset with frequency-based scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run on 4site_GB1
  AiCE/env/bin/python scripts/AiCE/run_generic.py --dataset 4site_GB1 --seed 42

  # Multiple runs on GB1
  AiCE/env/bin/python scripts/AiCE/run_generic.py --dataset 4site_GB1 --seeds 42 123 456 789 1000

  # Load seeds from file (50 runs)
  AiCE/env/bin/python scripts/AiCE/run_generic.py --dataset 4site_TRPB --seed_file rand_seeds.txt --num_seeds 30

  # Skip metrics computation
  AiCE/env/bin/python scripts/AiCE/run_generic.py --dataset 4site_PhoQ --seed 42 --skip_metrics

  # Custom output path
  AiCE/env/bin/python scripts/AiCE/run_generic.py --dataset 4site_GB1 --seed 42 --output_path results/4site_GB1_AiCE/
        """
    )

    # Dataset (required)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (subdirectory under data_dir, e.g. 4site_GB1, 4site_PhoQ, 4site_TRPB)"
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

    # AiCE parameters
    parser.add_argument(
        "--beta",
        type=float,
        default=0.8,
        help="Frequency threshold for non-coil regions (default: 0.8)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Frequency threshold for coil regions (default: 0.5)"
    )

    # Other options
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output directory for results (default: results/<dataset>_AiCE/)"
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

    # Default output path based on dataset name
    if args.output_path is None:
        args.output_path = os.path.join(BENCHMARK_ROOT, "AiCE", "results", f"{args.dataset}_AiCE")

    # Create AiCE config
    aice_config = AiCEConfig(
        beta=args.beta,
        gamma=args.gamma
    )

    # Determine which seeds to use
    if args.seeds is not None:
        seeds = args.seeds
    elif args.seed_file is not None:
        seeds = load_seeds_from_file(args.seed_file, args.num_seeds)
        print(f"Loaded {len(seeds)} seeds from {args.seed_file}")
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        # Default seed
        seeds = [64]

    print(f"\nRunning AiCE on {args.dataset} with {len(seeds)} seed(s): {seeds[:5]}{'...' if len(seeds) > 5 else ''}")
    print(f"Output path: {args.output_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Compute metrics: {not args.skip_metrics}")
    print(f"Beta: {args.beta}, Gamma: {args.gamma}")

    # Run experiments
    results = []
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Running experiment with seed={seed}")
        try:
            result = run_single_experiment(
                seed=seed,
                dataset=args.dataset,
                output_path=args.output_path,
                verbose=args.verbose,
                run_id=i + 1,
                compute_metrics=not args.skip_metrics,
                data_dir=args.data_dir,
                aice_config=aice_config
            )
            results.append(result)
        except Exception as e:
            print(f"Error in seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Aggregate results if multiple runs
    if len(results) > 1 and not args.skip_metrics:
        save_aggregated_results(results, args.output_path, args.dataset)

    # Final summary
    print(f"\n{'='*60}")
    print("Experiment Complete")
    print(f"{'='*60}")
    print(f"Total runs: {len(results)}")
    print(f"Configuration:")
    print(f"  - Model: AiCE (ProteinMPNN-based scoring)")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Beta: {args.beta}")
    print(f"  - Gamma: {args.gamma}")
    print(f"  - Batch size: 96")
    print(f"  - Rounds: 5 (1 init + 4 iterations)")
    print(f"  - Total samples per run: 480 (96 init + 384 budget)")
    print(f"\nResults saved to: {args.output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
