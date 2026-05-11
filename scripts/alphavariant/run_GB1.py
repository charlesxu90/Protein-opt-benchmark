#!/usr/bin/env python
"""
run_GB1.py - Execute AlphaVariant optimization on GB1 dataset with comprehensive metrics

Configuration:
    - Model: GPT-based generative model
    - Scorer: aa_onehot based surrogate with iterative updates
    - Training: REINFORCE with prior regularization
    - Batch size: 96
    - Rounds: 4 (iterative approach like ALDE)

Iterative Training Process:
    - Round 1: Random sampling from full 4-site space, train initial surrogate
    - Rounds 2-4: GPT-guided sampling, get ground truth fitness, update surrogate, train GPT

Metrics computed (from multiple reference works):
    - Exploration: High-Fitness Proximity, Novelty, Batch Diversity
    - Functional: Normalized Fitness (Top-K), Max Fitness
    - Success: Simple Regret, Global Max Hit Count

Usage:
    # Single run with default seed (iterative with 4 rounds)
    python run_GB1.py

    # Single run with specific seed
    python run_GB1.py --seed 42

    # Multiple runs for randomness evaluation
    python run_GB1.py --seeds 42 123 456 789 1000

    # Specify number of rounds
    python run_GB1.py --seed 42 --n_rounds 4

    # Use CLADE-2 clustering sampling (default)
    python run_GB1.py --seed 42 --sampling cluster --top_k_cutoff 1000 --n_clusters 10

    # Use ALDE-style active learning with Thompson Sampling
    python run_GB1.py --seed 42 --sampling active --acquisition ts

    # Use ALDE-style active learning with UCB
    python run_GB1.py --seed 42 --sampling active --acquisition ucb --xi 4.0

    # Skip metrics computation (faster)
    python run_GB1.py --seed 42 --skip_metrics
"""

from __future__ import annotations
import argparse
import copy
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

# Add alphavariant to path if running from this directory
sys.path.insert(0, str(Path(__file__).parent))

from popgen.model.gpt import GPT, GPTConfig, save_gpt_model, save_gpt_config
from popgen.utils.utils import set_random_seed, parse_config, read_fasta_as_list, load_hotspot
from popgen.utils.template import PDETemplate
from popgen.utils.dataset import AASeqDictionary, rnn_start_token_vector
from popscorer.scoring_functions import ScoringFunctions, BonusFunctions

# Import unified metrics from utils.compat
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
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
)

from scipy import stats as scipy_stats
import warnings as metrics_warnings

# Note: Core metrics (hamming_distance, high_fitness_proximity, novelty, batch_diversity,
# normalized_fitness_topk, max_fitness, simple_regret, spearman_correlation, global_max_hit_count,
# MetricsResult) are imported from utils.compat above.

# Additional local helper functions for this module:

def max_fitness_metric(fitness_values: np.ndarray) -> float:
    """
    Max Fitness: The absolute highest fitness value discovered.
    Reference: delta-Conservative Search
    """
    if len(fitness_values) == 0:
        return 0.0
    return float(np.max(fitness_values))


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

        mutations = []
        for i, (wt_aa, mut_aa) in enumerate(zip(wildtype, seq)):
            if wt_aa != mut_aa:
                mutations.append((i, mut_aa))

        if not all(m in single_mutant_effects_true for m in mutations):
            continue

        additive_true = wt_fitness + sum(
            single_mutant_effects_true[m] - wt_fitness for m in mutations
        )
        additive_pred = wt_pred + sum(
            single_mutant_effects_pred[m] - wt_pred for m in mutations
        )

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
    """
    high_order_mask = np.array([
        hamming_distance(seq, wildtype) >= min_mutations
        for seq in sequences
    ])

    if np.sum(high_order_mask) < top_k:
        top_k = max(1, np.sum(high_order_mask) // 2)

    if top_k == 0:
        return 0.0

    high_order_indices = np.where(high_order_mask)[0]

    # Handle case when no high-order mutants exist
    if len(high_order_indices) == 0:
        return 0.0

    high_order_true = fitness_values[high_order_indices]
    high_order_pred = predicted_values[high_order_indices]

    true_top_k_idx = set(high_order_indices[np.argsort(high_order_true)[-top_k:]])
    pred_top_k_idx = set(high_order_indices[np.argsort(high_order_pred)[-top_k:]])

    # Safety check for empty sets
    if len(true_top_k_idx) == 0:
        return 0.0

    recall = len(true_top_k_idx & pred_top_k_idx) / len(true_top_k_idx)
    return float(recall)


def miscalibration_area_local(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Miscalibration Area: Area between calibration curve and ideal diagonal.
    Reference: ALDE
    """
    if len(y_true) == 0 or np.all(y_std == 0):
        return 1.0

    z_scores = np.abs(y_true - y_pred) / (y_std + 1e-10)
    expected_confidences = np.linspace(0.1, 1.0, n_bins)

    observed_coverages = []
    for conf in expected_confidences:
        z_threshold = scipy_stats.norm.ppf((1 + conf) / 2)
        coverage = np.mean(z_scores <= z_threshold)
        observed_coverages.append(coverage)

    observed_coverages = np.array(observed_coverages)
    miscal_area = np.mean(np.abs(observed_coverages - expected_confidences))

    return float(miscal_area)


def expected_calibration_error_local(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Expected Calibration Error (ECE): Weighted average of calibration errors.
    Reference: ALDE
    """
    if len(y_true) == 0 or np.all(y_std == 0):
        return 1.0

    sorted_indices = np.argsort(y_std)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    y_std_sorted = y_std[sorted_indices]

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

        expected_error = np.mean(bin_std)
        observed_error = np.sqrt(np.mean((bin_true - bin_pred) ** 2))

        bin_weight = len(bin_true)
        ece += bin_weight * np.abs(expected_error - observed_error)
        total_samples += bin_weight

    return float(ece / total_samples) if total_samples > 0 else 1.0


# ============================================================================
# AlphaVariant GB1 Iterative Trainer
# ============================================================================

class IterativeGB1Trainer:
    """
    Iterative trainer for GB1 benchmark following ALDE-style approach.

    Training Process:
    - Round 1: Random sampling from full 4-site space, train initial surrogate
    - Rounds 2-N: GPT-guided sampling with CLADE-2 style clustering

    Each round:
    1. Sample sequences using GPT (random in round 1)
    2. For rounds before last: Apply top-k cutoff, cluster, sample from clusters
    3. For last round: Sample without cutoff
    4. Get ground truth fitness for sampled sequences
    5. Update surrogate with ALL collected data
    6. Train/fine-tune GPT model using surrogate predictions
    """

    # GB1 hotspot positions (1-indexed)
    HOTSPOT_POSITIONS = [39, 40, 41, 54]
    WILDTYPE_SEQ = 'MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

    def __init__(
        self,
        model_config: Any,
        optim_config: Any,
        template: PDETemplate,
        landscape_path: str,
        save_dir: str,
        batch_size: int = 96,
        n_rounds: int = 4,
        n_steps_per_round: int = 100,
        sigma: float = 60,
        device: str = 'cuda',
        seed: int = 42,
        top_k_cutoff: int = 1000,
        n_clusters: int = 10,
        sampling_strategy: str = 'cluster',
        acquisition: str = 'ts',
        xi: float = 4.0,
        finetune_prior: bool = False,
        n_finetune_epochs: int = 10,
        finetune_lr: float = 1e-4,
        ablation: str = "none",
    ):
        """
        Initialize iterative trainer.

        Args:
            sampling_strategy: 'cluster' (CLADE-2 style) or 'active' (ALDE-style Thompson Sampling)
            acquisition: For 'active' strategy - 'ts' (Thompson Sampling), 'ucb', or 'ei'
            xi: Exploration parameter for UCB acquisition
            finetune_prior: Whether to finetune prior on collected sequences before RL
            n_finetune_epochs: Number of epochs for prior finetuning (default: 10)
            finetune_lr: Learning rate for prior finetuning
            ablation: Component-removal flag. One of:
                'none'           -> full pipeline (default)
                'no-gpt'         -> AV-NoGPT: replace GPT prior with random single-site
                                   mutations around the best variant
                'no-space'       -> AV-NoSpace: disable dynamic space definition
                                   (no top-k cutoff, no clustering — uniform sampling)
                'static-reward'  -> AV-StaticReward: freeze surrogate after round 0
                'no-rl'          -> AV-NoRL: skip REINFORCE; sample from prior + greedy
                                   top-k by surrogate score
        """
        valid_ablations = {"none", "no-gpt", "no-space", "static-reward", "no-rl"}
        if ablation not in valid_ablations:
            raise ValueError(
                f"Unknown ablation: {ablation!r}. Choose from {sorted(valid_ablations)}"
            )
        self.model_config = model_config
        self.optim_config = optim_config
        self.template = template
        self.landscape_path = landscape_path
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.n_rounds = n_rounds
        self.n_steps_per_round = n_steps_per_round
        self.sigma = sigma
        self.device = device
        self.seed = seed
        self.top_k_cutoff = top_k_cutoff
        self.n_clusters = n_clusters
        self.sampling_strategy = sampling_strategy
        self.acquisition = acquisition
        self.xi = xi
        self.finetune_prior = finetune_prior
        self.n_finetune_epochs = n_finetune_epochs
        self.finetune_lr = finetune_lr
        self.ablation = ablation

        self.sd = AASeqDictionary()
        set_random_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load landscape data for ground truth lookup
        self._load_landscape()

        # Initialize surrogate (will be trained on collected data)
        self.surrogate = None
        self.surrogate_trained = False

        # Track all collected data across rounds
        self.collected_combos = []
        self.collected_fitness = []  # Ground truth fitness
        self.collected_seqs = []

        # Track all generated data for metrics
        self.all_generated_seqs = []
        self.all_predicted_fitness = []
        self.all_oracle_fitness = []

        # Round-level data
        self.round_data = []

        # Models (created fresh each round or fine-tuned)
        self.prior_model = None
        self.agent_model = None
        self.optimizer = None

        # TensorBoard writer for logging
        self.writer = SummaryWriter(save_dir)
        self.global_step = 0  # Track global step across all rounds

    def _load_landscape(self):
        """Load the complete GB1 fitness landscape."""
        df = pd.read_csv(self.landscape_path)

        self.combo_to_fitness = {}
        self.combo_to_seq = {}
        self.all_combos = []
        self.all_fitness = []

        for _, row in df.iterrows():
            combo = row['AACombo']
            fitness = row['fitness']
            self.combo_to_fitness[combo] = fitness
            self.all_combos.append(combo)
            self.all_fitness.append(fitness)

            # Construct full sequence
            seq = list(self.WILDTYPE_SEQ)
            for i, pos in enumerate(self.HOTSPOT_POSITIONS):
                seq[pos - 1] = combo[i]
            self.combo_to_seq[combo] = ''.join(seq)

        self.all_fitness = np.array(self.all_fitness)
        self.max_fitness_raw = np.max(self.all_fitness)
        self.min_fitness_raw = np.min(self.all_fitness)

        logger.info(f"Loaded {len(self.all_combos)} variants from landscape")
        logger.info(f"Fitness range: [{self.min_fitness_raw:.4f}, {self.max_fitness_raw:.4f}]")

    def _get_ground_truth_fitness(self, combos: List[str]) -> np.ndarray:
        """Get ground truth fitness for combos."""
        fitness = np.zeros(len(combos))
        for i, combo in enumerate(combos):
            if combo in self.combo_to_fitness:
                fitness[i] = self.combo_to_fitness[combo] / self.max_fitness_raw
            else:
                fitness[i] = 0.0
        return fitness

    def _sample_random_combos(self, n_samples: int, exclude: set = None) -> List[str]:
        """Sample random combos from the full 4-site space."""
        if exclude is None:
            exclude = set()

        # Sample from available combos (not already collected)
        available = [c for c in self.all_combos if c not in exclude]
        if len(available) <= n_samples:
            return available

        indices = np.random.choice(len(available), size=n_samples, replace=False)
        return [available[i] for i in indices]

    def _random_mutation_samples(self, n_samples: int, exclude: set = None) -> List[str]:
        """Random single-site mutations around the best variants discovered so far.

        Used by the AV-NoGPT ablation: replaces the GPT-generated proposals with
        a simple "mutate the best variant at random positions" baseline.
        Falls back to uniform random if no variants have been collected yet.
        """
        if exclude is None:
            exclude = set()
        AAS = "ACDEFGHIKLMNPQRSTVWY"
        available_set = set(self.all_combos)

        if not self.collected_combos:
            return self._sample_random_combos(n_samples, exclude=exclude)

        # Rank collected combos by fitness
        order = np.argsort(-np.asarray(self.collected_fitness))
        seeds = [self.collected_combos[i] for i in order[: max(1, n_samples // 4)]]

        out: List[str] = []
        attempts = 0
        max_attempts = n_samples * 50
        while len(out) < n_samples and attempts < max_attempts:
            attempts += 1
            base = seeds[np.random.randint(len(seeds))]
            pos = np.random.randint(len(base))
            new_aa = AAS[np.random.randint(len(AAS))]
            if new_aa == base[pos]:
                continue
            mutant = base[:pos] + new_aa + base[pos + 1 :]
            if mutant in exclude or mutant in out:
                continue
            if mutant not in available_set:
                continue
            out.append(mutant)
        # Top up with uniform random if we couldn't find enough valid mutants
        if len(out) < n_samples:
            extras = self._sample_random_combos(
                n_samples - len(out), exclude=exclude.union(out)
            )
            out.extend(extras)
        return out

    def _cluster_init_sample(self, n_samples: int, exclude: set = None) -> List[str]:
        """
        CLADE-2 style cluster-based initialization sampling.

        Clusters the full landscape using AA one-hot features and samples
        uniformly from each cluster to ensure diverse initial coverage.

        Args:
            n_samples: Number of samples to select
            exclude: Set of combos to exclude

        Returns:
            List of selected combos
        """
        from sklearn.cluster import KMeans
        from popscorer.fitness.aa_onehot_pred.embed import seqs2feat

        if exclude is None:
            exclude = set()

        # Get available combos
        available_combos = [c for c in self.all_combos if c not in exclude]
        if len(available_combos) <= n_samples:
            return available_combos

        # Get full sequences for feature extraction
        available_seqs = [self._combo_to_full_seq(c) for c in available_combos]

        # Extract features
        logger.info(f"Extracting features for {len(available_seqs)} sequences...")
        X = seqs2feat(available_seqs)

        # Determine number of clusters (use n_clusters or adapt based on n_samples)
        n_clusters = min(self.n_clusters, n_samples, len(available_combos) // 2)
        n_clusters = max(n_clusters, 1)

        # Run KMeans clustering
        logger.info(f"Clustering into {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # Organize sequences by cluster
        clusters = [[] for _ in range(n_clusters)]
        for idx, label in enumerate(cluster_labels):
            clusters[label].append({
                'idx': idx,
                'combo': available_combos[idx],
            })

        # Shuffle within each cluster for randomness
        for cluster in clusters:
            np.random.shuffle(cluster)

        # Calculate samples per cluster (uniform distribution)
        samples_per_cluster = n_samples // n_clusters
        extra_samples = n_samples % n_clusters

        # Sample from each cluster
        selected_combos = []
        cluster_indices = list(range(n_clusters))
        np.random.shuffle(cluster_indices)  # Randomize which clusters get extra samples

        for i, cluster_idx in enumerate(cluster_indices):
            cluster = clusters[cluster_idx]
            # Clusters in the shuffled order get extra samples first
            n_from_cluster = samples_per_cluster + (1 if i < extra_samples else 0)
            n_from_cluster = min(n_from_cluster, len(cluster))

            for j in range(n_from_cluster):
                selected_combos.append(cluster[j]['combo'])

        # If we still need more samples (some clusters were too small), fill from remaining
        if len(selected_combos) < n_samples:
            remaining_combos = set(available_combos) - set(selected_combos)
            remaining_list = list(remaining_combos)
            np.random.shuffle(remaining_list)
            n_needed = n_samples - len(selected_combos)
            selected_combos.extend(remaining_list[:n_needed])

        logger.info(f"Cluster init sampling: selected {len(selected_combos)} from {n_clusters} clusters")
        return selected_combos

    def _combo_to_full_seq(self, combo: str) -> str:
        """Convert 4-letter combo to full 56-aa sequence."""
        if combo in self.combo_to_seq:
            return self.combo_to_seq[combo]
        seq = list(self.WILDTYPE_SEQ)
        for i, pos in enumerate(self.HOTSPOT_POSITIONS):
            seq[pos - 1] = combo[i]
        return ''.join(seq)

    def _seq_to_combo(self, seq: str) -> str:
        """Extract 4-letter combo from full sequence."""
        return ''.join(seq[pos - 1] for pos in self.HOTSPOT_POSITIONS)

    def _create_models(self):
        """Create fresh GPT models."""
        mconf = GPTConfig(
            vocab_size=self.model_config.vocab_size,
            block_size=self.model_config.block_size,
            n_layer=self.model_config.n_layer,
            n_head=self.model_config.n_head,
            n_embd=self.model_config.n_embd,
        )
        self.prior_model = GPT(mconf).to(self.device)
        self.agent_model = copy.deepcopy(self.prior_model)

        for param in self.prior_model.parameters():
            param.requires_grad = False

        if self.optim_config is not None:
            self.optimizer = self.agent_model.configure_optimizers(self.optim_config)
        else:
            self.optimizer = torch.optim.Adam(self.agent_model.parameters(), lr=1e-4)

    def _finetune_prior_model(self, combos: List[str], fitness: np.ndarray, n_epochs: int = None):
        """
        Finetune the prior model on collected sequences using next-token prediction.

        This is done BEFORE RL training to give the prior a better initialization
        based on the sequences we've already collected. Sequences are weighted by
        their fitness so the prior learns to generate high-fitness sequences.

        Args:
            combos: List of 4-letter combos to train on
            fitness: Fitness values for each combo (used for weighting)
            n_epochs: Number of finetuning epochs (defaults to self.n_finetune_epochs)
        """
        if n_epochs is None:
            n_epochs = self.n_finetune_epochs

        # Calculate number of steps based on epochs and dataset size
        n_samples = len(combos)
        steps_per_epoch = max(1, n_samples // self.batch_size)
        n_steps = n_epochs * steps_per_epoch

        if len(combos) == 0:
            logger.warning("No sequences for prior finetuning")
            return

        # Temporarily enable gradients for prior
        for param in self.prior_model.parameters():
            param.requires_grad = True

        # Create optimizer for prior finetuning
        prior_optimizer = torch.optim.Adam(self.prior_model.parameters(), lr=self.finetune_lr)

        # Convert combos to token tensors
        all_tokens = []
        for combo in combos:
            tokens = [self.sd.char_idx.get(c, 0) for c in combo]
            all_tokens.append(tokens)
        all_tokens = torch.LongTensor(all_tokens).to(self.device)

        # Normalize fitness to create weights (softmax-like weighting)
        fitness_tensor = torch.from_numpy(fitness).float().to(self.device)
        # Use temperature-scaled softmax for weighting
        temperature = 1.0
        weights = F.softmax(fitness_tensor / temperature, dim=0)

        n_seqs = len(combos)
        seq_len = all_tokens.size(1)

        logger.info(f"Finetuning prior on {n_seqs} sequences for {n_epochs} epochs ({n_steps} steps)...")

        self.prior_model.train()
        for step in range(n_steps):
            # Sample batch with replacement, weighted by fitness
            batch_indices = torch.multinomial(weights, min(self.batch_size, n_seqs), replacement=True)
            batch_tokens = all_tokens[batch_indices]
            batch_weights = weights[batch_indices]

            # Compute next-token prediction loss
            # Input: [start_token, token_0, token_1, ..., token_{n-2}]
            # Target: [token_0, token_1, ..., token_{n-1}]
            x = rnn_start_token_vector(len(batch_indices), self.device)

            total_loss = 0.0
            for pos in range(seq_len):
                logits, _ = self.prior_model(x)
                # Cross-entropy loss for next token prediction
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                targets = batch_tokens[:, pos]

                # Per-sample loss
                ce_loss = F.nll_loss(log_probs, targets, reduction='none')

                # Weight by fitness
                weighted_loss = (ce_loss * batch_weights).sum() / batch_weights.sum()
                total_loss += weighted_loss

                # Append target token to input for next position
                x = torch.cat([x, targets.unsqueeze(1)], dim=1)

            # Average over sequence length
            loss = total_loss / seq_len

            prior_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.prior_model.parameters(), 1.0)
            prior_optimizer.step()

            # TensorBoard logging
            self.writer.add_scalar('finetune/loss', loss.item(), step + 1)

            if (step + 1) % 20 == 0 or step == 0:
                logger.debug(f"  Finetune step {step+1}/{n_steps}: loss={loss.item():.4f}")

        # Freeze prior again after finetuning
        for param in self.prior_model.parameters():
            param.requires_grad = False

        # Copy finetuned prior to agent
        self.agent_model = copy.deepcopy(self.prior_model)
        for param in self.agent_model.parameters():
            param.requires_grad = True

        # Re-create optimizer for agent
        if self.optim_config is not None:
            self.optimizer = self.agent_model.configure_optimizers(self.optim_config)
        else:
            self.optimizer = torch.optim.Adam(self.agent_model.parameters(), lr=1e-4)

        logger.info(f"Prior finetuning complete. Final loss: {loss.item():.4f}")

    def _train_surrogate(self):
        """Train surrogate model on all collected data."""
        from sklearn.linear_model import Ridge, BayesianRidge
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from popscorer.fitness.aa_onehot_pred.embed import seqs2feat

        if len(self.collected_combos) == 0:
            logger.warning("No training data for surrogate")
            return

        # Get full sequences for feature extraction
        training_seqs = [self._combo_to_full_seq(c) for c in self.collected_combos]
        X = seqs2feat(training_seqs)
        y = np.array(self.collected_fitness)

        # Train ensemble of models
        self.surrogate_models = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models = [
                Ridge(alpha=1.0, random_state=self.seed),
                Ridge(alpha=0.1, random_state=self.seed + 1),
                BayesianRidge(),
                RandomForestRegressor(n_estimators=50, max_depth=8, random_state=self.seed + 2, n_jobs=-1),
                GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=self.seed + 3),
            ]
            for model in models:
                model.fit(X, y)
                self.surrogate_models.append(model)

        self.surrogate_trained = True
        logger.debug(f"Trained surrogate on {len(self.collected_combos)} samples")

    def _predict_surrogate(self, seqs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using surrogate ensemble with uncertainty."""
        from popscorer.fitness.aa_onehot_pred.embed import seqs2feat

        if not self.surrogate_trained or len(self.surrogate_models) == 0:
            return np.zeros(len(seqs)), np.ones(len(seqs))

        X = seqs2feat(seqs)
        predictions = np.zeros((len(seqs), len(self.surrogate_models)))

        for i, model in enumerate(self.surrogate_models):
            predictions[:, i] = model.predict(X)

        mu = np.mean(predictions, axis=1)
        sigma = np.std(predictions, axis=1)

        # UCB for exploration
        ucb = mu + 2.0 * sigma
        return ucb, mu

    def _predict_surrogate_with_samples(self, seqs: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict using surrogate ensemble and return individual model predictions
        for Thompson Sampling.

        Returns:
            Tuple of (mean, std, all_predictions [n_seqs x n_models])
        """
        from popscorer.fitness.aa_onehot_pred.embed import seqs2feat

        if not self.surrogate_trained or len(self.surrogate_models) == 0:
            return np.zeros(len(seqs)), np.ones(len(seqs)), np.zeros((len(seqs), 1))

        X = seqs2feat(seqs)
        predictions = np.zeros((len(seqs), len(self.surrogate_models)))

        for i, model in enumerate(self.surrogate_models):
            predictions[:, i] = model.predict(X)

        mu = np.mean(predictions, axis=1)
        sigma = np.std(predictions, axis=1)

        return mu, sigma, predictions

    def _active_sample(
        self,
        seqs: List[str],
        n_samples: int,
        exclude_combos: set = None,
        acquisition: str = 'ts',
        xi: float = 4.0,
    ) -> Tuple[List[str], List[str]]:
        """
        ALDE-style active learning sampling using Thompson Sampling or UCB.

        Args:
            seqs: List of sequences generated during RL
            n_samples: Number of samples to select
            exclude_combos: Set of combos already collected (to exclude)
            acquisition: Acquisition function ('ts' for Thompson Sampling, 'ucb' for UCB)
            xi: Exploration parameter for UCB

        Returns:
            Tuple of (selected full sequences, selected combos)
        """
        if exclude_combos is None:
            exclude_combos = set()

        # Get unique sequences not already collected
        seen_combos = set()
        unique_seqs = []
        unique_combos = []

        for seq in seqs:
            combo = self._seq_to_combo(seq)
            if combo not in exclude_combos and combo not in seen_combos:
                unique_seqs.append(seq)
                unique_combos.append(combo)
                seen_combos.add(combo)

        if len(unique_seqs) == 0:
            return [], []

        # If we have fewer candidates than needed, return all
        if len(unique_seqs) <= n_samples:
            return unique_seqs, unique_combos

        # Get predictions from surrogate ensemble
        mu, sigma, all_predictions = self._predict_surrogate_with_samples(unique_seqs)

        # Compute acquisition scores
        if acquisition == 'ts':
            # Thompson Sampling: sample from posterior for each model and take mean
            # For each sequence, sample a random model's prediction
            n_seqs = len(unique_seqs)
            n_models = all_predictions.shape[1]

            # Sample which model to use for each sequence
            model_indices = np.random.randint(0, n_models, size=n_seqs)
            acquisition_scores = np.array([
                all_predictions[i, model_indices[i]] for i in range(n_seqs)
            ])
            logger.info(f"Active sampling (Thompson Sampling): {n_seqs} candidates")

        elif acquisition == 'ucb':
            # Upper Confidence Bound: mu + xi * sigma
            acquisition_scores = mu + xi * sigma
            logger.info(f"Active sampling (UCB, xi={xi}): {len(unique_seqs)} candidates")

        elif acquisition == 'ei':
            # Expected Improvement (simplified)
            best_so_far = max(self.collected_fitness) if self.collected_fitness else 0
            z = (mu - best_so_far) / (sigma + 1e-8)
            from scipy.stats import norm
            acquisition_scores = (mu - best_so_far) * norm.cdf(z) + sigma * norm.pdf(z)
            logger.info(f"Active sampling (EI): {len(unique_seqs)} candidates")

        else:
            # Default to UCB
            acquisition_scores = mu + xi * sigma

        # Select top n_samples based on acquisition scores
        top_indices = np.argsort(acquisition_scores)[-n_samples:][::-1]

        selected_seqs = [unique_seqs[i] for i in top_indices]
        selected_combos = [unique_combos[i] for i in top_indices]

        logger.info(f"Active sampling: selected {len(selected_seqs)} samples")
        return selected_seqs, selected_combos

    def nll_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Custom NLL loss returning per-example loss."""
        target_expanded = torch.zeros(inputs.size()).to(inputs.device)
        target_expanded.scatter_(1, targets.contiguous().view(-1, 1).detach(), 1.0)
        loss = torch.sum(target_expanded * inputs, 1)
        return loss

    def sample_from_model(self, model: GPT, num_samples: int) -> List[str]:
        """Sample sequences from GPT model."""
        model.eval()
        sequences = []
        x = rnn_start_token_vector(num_samples, self.device)

        n_positions = len(self.template.positions)

        with torch.no_grad():
            for step in range(n_positions):
                logits, _ = model(x)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                sampled_idx = Categorical(probs=probs).sample().squeeze()

                # Ensure valid amino acids
                if self.template.pos_aa_candidates is not None and step < len(self.template.pos_aa_candidates):
                    aa_candidates = [self.sd.char_idx[c] for c in list(self.template.pos_aa_candidates.values())[step]]
                    sample_idx_mask = sum(sampled_idx == i for i in aa_candidates).bool()
                    sample_to_replace = (sample_idx_mask == False).nonzero(as_tuple=True)

                    if len(sample_to_replace[0]) > 0:
                        aa_candidate_prob = torch.ones(len(aa_candidates)).to(sampled_idx.device) / len(aa_candidates)
                        rep_count = len(sample_to_replace[0])
                        rep_aa_idxes = aa_candidate_prob.multinomial(num_samples=rep_count, replacement=True)
                        for i, idx in enumerate(sample_to_replace[0]):
                            sampled_idx[idx] = aa_candidates[rep_aa_idxes[i].item()]

                sequences.append(sampled_idx.view(-1, 1))
                x = torch.cat(sequences, 1)

        # Convert to combos
        token_seqs = torch.cat(sequences, 1)
        aa_seqs = self.sd.matrix_to_seqs(token_seqs)

        # Convert combos to full sequences
        full_seqs = []
        for combo in aa_seqs:
            full_seqs.append(self._combo_to_full_seq(combo))

        return full_seqs

    def _cluster_sample(
        self,
        seqs: List[str],
        predicted_fitness: np.ndarray,
        n_samples: int,
        exclude_combos: set = None,
        apply_cutoff: bool = True,
    ) -> Tuple[List[str], List[str]]:
        """
        CLADE-2 style clustering-based sampling.

        Args:
            seqs: List of sequences generated during RL
            predicted_fitness: Predicted fitness for each sequence
            n_samples: Number of samples to select
            exclude_combos: Set of combos already collected (to exclude)
            apply_cutoff: Whether to apply top-k cutoff (for rounds before last)

        Returns:
            Tuple of (selected full sequences, selected combos)
        """
        from sklearn.cluster import KMeans
        from popscorer.fitness.aa_onehot_pred.embed import seqs2feat

        if exclude_combos is None:
            exclude_combos = set()

        # Get unique sequences not already collected
        seen_combos = set()
        unique_seqs = []
        unique_fitness = []
        unique_combos = []

        for seq, fit in zip(seqs, predicted_fitness):
            combo = self._seq_to_combo(seq)
            if combo not in exclude_combos and combo not in seen_combos:
                unique_seqs.append(seq)
                unique_fitness.append(fit)
                unique_combos.append(combo)
                seen_combos.add(combo)

        if len(unique_seqs) == 0:
            return [], []

        unique_fitness = np.array(unique_fitness)

        # Apply top-k cutoff for rounds before the last
        if apply_cutoff and len(unique_seqs) > self.top_k_cutoff:
            top_k_indices = np.argsort(unique_fitness)[-self.top_k_cutoff:]
            unique_seqs = [unique_seqs[i] for i in top_k_indices]
            unique_fitness = unique_fitness[top_k_indices]
            unique_combos = [unique_combos[i] for i in top_k_indices]
            logger.info(f"Applied top-{self.top_k_cutoff} cutoff, {len(unique_seqs)} candidates remaining")

        # If we have fewer candidates than needed, return all
        if len(unique_seqs) <= n_samples:
            return unique_seqs, unique_combos

        # Get features for clustering
        X = seqs2feat(unique_seqs)

        # Determine number of clusters
        n_clusters = min(self.n_clusters, len(unique_seqs) // 2, n_samples)
        n_clusters = max(n_clusters, 1)

        # Run KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # Organize sequences by cluster
        clusters = [[] for _ in range(n_clusters)]
        for idx, label in enumerate(cluster_labels):
            clusters[label].append({
                'idx': idx,
                'seq': unique_seqs[idx],
                'combo': unique_combos[idx],
                'fitness': unique_fitness[idx],
            })

        # Sort within each cluster by fitness (descending)
        for cluster in clusters:
            cluster.sort(key=lambda x: x['fitness'], reverse=True)

        # Compute cluster mean fitness for sampling probability
        cluster_mean_fitness = []
        for cluster in clusters:
            if len(cluster) > 0:
                mean_fit = np.mean([item['fitness'] for item in cluster])
            else:
                mean_fit = 0.0
            cluster_mean_fitness.append(mean_fit)

        cluster_mean_fitness = np.array(cluster_mean_fitness)

        # Normalize to get sampling probability (linear probability like CLADE-2)
        if np.sum(cluster_mean_fitness) > 0:
            cluster_prob = cluster_mean_fitness / np.sum(cluster_mean_fitness)
        else:
            cluster_prob = np.ones(n_clusters) / n_clusters

        # Sample from clusters
        selected_seqs = []
        selected_combos = []
        cluster_indices = [0] * n_clusters  # Track position in each cluster

        for _ in range(n_samples):
            # Update probabilities (set to 0 if cluster exhausted)
            for i in range(n_clusters):
                if cluster_indices[i] >= len(clusters[i]):
                    cluster_prob[i] = 0

            # Ensure non-negative and normalize
            cluster_prob = np.maximum(cluster_prob, 0)
            prob_sum = np.sum(cluster_prob)
            if prob_sum == 0:
                break

            # Normalize probability
            cluster_prob = cluster_prob / prob_sum

            # Sample a cluster
            selected_cluster = np.random.choice(n_clusters, p=cluster_prob)

            # Get next item from this cluster
            item = clusters[selected_cluster][cluster_indices[selected_cluster]]
            selected_seqs.append(item['seq'])
            selected_combos.append(item['combo'])
            cluster_indices[selected_cluster] += 1

        logger.info(f"Cluster sampling: selected {len(selected_seqs)} from {n_clusters} clusters")
        return selected_seqs, selected_combos

    def _generate_rl_samples(self, n_samples: int) -> Tuple[List[str], np.ndarray]:
        """
        Generate samples using GPT model during RL training.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Tuple of (sequences, predicted_fitness)
        """
        all_seqs = []
        all_fitness = []

        # Generate in batches
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        for _ in range(n_batches):
            seqs = self.sample_from_model(self.agent_model, self.batch_size)

            # Get surrogate predictions
            if self.surrogate_trained:
                ucb_scores, raw_pred = self._predict_surrogate(seqs)
                all_seqs.extend(seqs)
                all_fitness.extend(raw_pred.tolist())
            else:
                all_seqs.extend(seqs)
                all_fitness.extend([0.0] * len(seqs))

        return all_seqs[:n_samples], np.array(all_fitness[:n_samples])

    def likelihood(self, model: GPT, x: torch.Tensor) -> torch.Tensor:
        """Compute log likelihood of sequences."""
        num_samples, seq_length = x.size()
        log_probs = torch.zeros(num_samples).to(x.device)

        model.eval()
        with torch.no_grad():
            for step in range(1, seq_length):
                logits, _ = model(x[:, :step])
                log_prob = F.log_softmax(logits[:, -1, :], dim=-1).squeeze()
                log_probs += self.nll_loss(log_prob, x[:, step])
        return log_probs

    def _train_gpt_on_surrogate(
        self, n_steps: int, round_idx: int = 0
    ) -> Tuple[List[str], np.ndarray]:
        """Train GPT model using surrogate predictions as rewards.

        Collects all unique sequences generated during training for use in
        the sampling step.

        Args:
            n_steps: Number of training steps
            round_idx: Current round index (for logging)

        Returns:
            Tuple of (all_generated_seqs, all_predicted_fitness)
        """
        self.agent_model.train()

        # Collect all generated sequences during RL training
        all_seqs_dict = {}  # seq -> predicted_fitness (keep best prediction)

        for step in range(n_steps):
            # Sample from agent
            seqs = self.sample_from_model(self.agent_model, self.batch_size)

            # Get unique sequences
            unique_seqs = list(set(seqs))
            if len(unique_seqs) == 0:
                continue

            # Get combos for encoding
            combos = [self._seq_to_combo(s) for s in unique_seqs]

            # Convert to token indices
            token_seqs = []
            for combo in combos:
                tokens = [self.sd.char_idx.get(c, 0) for c in combo]
                token_seqs.append(tokens)
            token_tensor = torch.LongTensor(token_seqs).to(self.device)

            # Get agent likelihood
            self.agent_model.train()
            sample_log_probs = torch.zeros(len(unique_seqs)).to(self.device)
            x = rnn_start_token_vector(len(unique_seqs), self.device)

            for pos in range(token_tensor.size(1)):
                logits, _ = self.agent_model(x)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                log_probs = probs.log()
                sample_log_probs += self.nll_loss(log_probs, token_tensor[:, pos])
                x = torch.cat([x, token_tensor[:, pos:pos+1]], dim=1)

            agent_likelihoods = sample_log_probs

            # Get prior likelihood
            prior_likelihoods = self.likelihood(self.prior_model, token_tensor)

            # Get surrogate predictions (UCB score and raw prediction)
            ucb_scores, raw_scores = self._predict_surrogate(unique_seqs)
            scores = torch.from_numpy(ucb_scores).float().to(self.device)

            # Collect sequences with their predicted fitness
            for seq, pred in zip(unique_seqs, raw_scores):
                if seq not in all_seqs_dict or pred > all_seqs_dict[seq]:
                    all_seqs_dict[seq] = pred

            # Also get ground truth fitness for monitoring (not used in training)
            gt_fitness = self._get_ground_truth_fitness(combos)

            # REINFORCE loss
            augmented_likelihoods = prior_likelihoods + self.sigma * scores
            loss = torch.pow((augmented_likelihoods - agent_likelihoods), 2).mean()
            loss -= 5 * 1e3 * (1 / agent_likelihoods).mean()

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent_model.parameters(), 1.0)
            self.optimizer.step()

            # TensorBoard logging
            self.global_step += 1
            self.writer.add_scalar('train/loss', loss.item(), self.global_step)
            self.writer.add_scalar('train/avg_ucb_score', ucb_scores.mean(), self.global_step)
            self.writer.add_scalar('train/avg_raw_score', raw_scores.mean(), self.global_step)
            self.writer.add_scalar('train/max_ucb_score', ucb_scores.max(), self.global_step)
            self.writer.add_scalar('train/max_raw_score', raw_scores.max(), self.global_step)
            self.writer.add_scalar('train/avg_gt_fitness', gt_fitness.mean(), self.global_step)
            self.writer.add_scalar('train/max_gt_fitness', gt_fitness.max(), self.global_step)
            self.writer.add_scalar('train/agent_likelihood', agent_likelihoods.mean().item(), self.global_step)
            self.writer.add_scalar('train/prior_likelihood', prior_likelihoods.mean().item(), self.global_step)
            self.writer.add_scalar('train/n_unique_seqs', len(unique_seqs), self.global_step)
            self.writer.add_scalar('train/round', round_idx + 1, self.global_step)

            # Log step info periodically
            if (step + 1) % 50 == 0 or step == 0:
                logger.debug(f"  Step {step+1}/{n_steps}: loss={loss.item():.4f}, "
                           f"avg_score={raw_scores.mean():.4f}, max_gt={gt_fitness.max():.4f}")

        # Return collected sequences and their predicted fitness
        collected_seqs = list(all_seqs_dict.keys())
        collected_fitness = np.array([all_seqs_dict[s] for s in collected_seqs])
        logger.info(f"Collected {len(collected_seqs)} unique sequences during RL training")

        return collected_seqs, collected_fitness

    def train(self) -> Tuple[List[str], List[float], List[float]]:
        """
        Run iterative training for n_rounds with configurable sampling strategy.

        Training Flow:
        - Round 1: Initial sampling (random/cluster), train surrogate, create GPT
        - Rounds 2+: Finetune prior (10 epochs), train GPT (collecting sequences),
                     apply sampling strategy on collected sequences, update surrogate

        Sampling Strategies:
        - 'cluster': CLADE-2 style clustering with top-k cutoff
        - 'active': ALDE-style active learning with Thompson Sampling/UCB/EI

        Returns:
            Tuple of (all_sequences, all_predicted_fitness, all_oracle_fitness)
        """
        logger.info(f"Starting iterative training: {self.n_rounds} rounds, {self.batch_size} samples/round")
        logger.info(f"Sampling strategy: {self.sampling_strategy}")
        if self.sampling_strategy == 'cluster':
            logger.info(f"  Clustering params: top_k_cutoff={self.top_k_cutoff}, n_clusters={self.n_clusters}")
        else:
            logger.info(f"  Active learning params: acquisition={self.acquisition}, xi={self.xi}")
        if self.finetune_prior:
            logger.info(f"  Prior finetuning: {self.n_finetune_epochs} epochs before each round")

        for round_idx in range(self.n_rounds):
            round_start = datetime.now()
            is_last_round = (round_idx == self.n_rounds - 1)

            logger.info(f"\n{'='*50}")
            logger.info(f"Round {round_idx + 1}/{self.n_rounds}" + (" (LAST)" if is_last_round else ""))
            logger.info(f"{'='*50}")

            # Determine sampling method
            if round_idx == 0:
                # Round 1: Initial sampling from full 4-site space
                collected_set = set(self.collected_combos)

                if self.ablation == "no-space":
                    # AV-NoSpace: skip dynamic space (no clustering); uniform random
                    logger.info("[ablation=no-space] Uniform random initialization...")
                    new_combos = self._sample_random_combos(self.batch_size, exclude=collected_set)
                elif self.sampling_strategy == 'cluster':
                    # CLADE-2 style: cluster-based initialization for diverse coverage
                    logger.info("Cluster-based initialization from full landscape...")
                    new_combos = self._cluster_init_sample(self.batch_size, exclude=collected_set)
                else:
                    # Random sampling for active learning strategy
                    logger.info("Random sampling from full 4-site space...")
                    new_combos = self._sample_random_combos(self.batch_size, exclude=collected_set)

                new_seqs = [self._combo_to_full_seq(c) for c in new_combos]

            else:
                # Rounds 2+: 1) Finetune prior, 2) Train surrogate, 3) Run RL

                # Step 1: Finetune prior on ALL collected sequences (copies to agent after)
                if self.finetune_prior:
                    logger.info(f"Step 1: Finetuning prior on {len(self.collected_combos)} sequences ({self.n_finetune_epochs} epochs)...")
                    self._finetune_prior_model(
                        combos=self.collected_combos,
                        fitness=np.array(self.collected_fitness),
                        n_epochs=self.n_finetune_epochs,
                    )

                # Step 2: Train new surrogate on ALL collected data.
                # Ablation seam: AV-StaticReward freezes the surrogate after round 0.
                if self.ablation == "static-reward":
                    logger.info("[ablation=static-reward] Skipping surrogate retraining; "
                                "reusing round-0 surrogate")
                else:
                    logger.info(f"Step 2: Training surrogate on {len(self.collected_combos)} samples...")
                    self._train_surrogate()

                # Step 3: Generate candidate sequences. Ablation seam: AV-NoGPT replaces
                # the GPT prior + RL with random single-site mutations around the best
                # variants. AV-NoRL skips the REINFORCE update and samples directly
                # from the prior.
                collected_set = set(self.collected_combos)

                if self.ablation == "no-gpt":
                    logger.info("[ablation=no-gpt] Sampling random single-site mutations "
                                "around best variants...")
                    new_combos = self._random_mutation_samples(
                        self.batch_size, exclude=collected_set,
                    )
                    new_seqs = [self._combo_to_full_seq(c) for c in new_combos]
                elif self.ablation == "no-space":
                    # AV-NoSpace: still use GPT but ignore space-definition step;
                    # uniform random pick over full landscape, ignoring clusters / cutoff.
                    logger.info("[ablation=no-space] Uniform random over full landscape "
                                "(no cutoff, no clustering)...")
                    new_combos = self._sample_random_combos(
                        self.batch_size, exclude=collected_set,
                    )
                    new_seqs = [self._combo_to_full_seq(c) for c in new_combos]
                elif self.ablation == "no-rl":
                    # AV-NoRL: skip REINFORCE; sample from current prior, then greedy
                    # top-k by surrogate score.
                    logger.info(f"[ablation=no-rl] Sampling {self.n_steps_per_round * self.batch_size} "
                                f"sequences from prior (no RL update)...")
                    n_pool = max(self.n_steps_per_round * self.batch_size, self.batch_size * 8)
                    if self.prior_model is None:
                        # Round 1 hasn't created models yet (only initial sampling done)
                        # Fall back to random.
                        new_combos = self._sample_random_combos(self.batch_size, exclude=collected_set)
                    else:
                        generated_seqs = self.sample_from_model(self.prior_model, n_pool)
                        # Filter to landscape and not-yet-queried, then greedy top-k by surrogate
                        scored: Dict[str, float] = {}
                        for s in generated_seqs:
                            combo = self._seq_to_combo(s)
                            if combo in collected_set or combo not in self.combo_to_fitness:
                                continue
                            if combo in scored:
                                continue
                            scored[combo] = 0.0  # placeholder; bulk-score below
                        candidate_combos = list(scored.keys())
                        candidate_seqs = [self._combo_to_full_seq(c) for c in candidate_combos]
                        if candidate_seqs and self.surrogate_trained:
                            _, raw_pred = self._predict_surrogate(candidate_seqs)
                            order = np.argsort(-raw_pred)[: self.batch_size]
                            new_combos = [candidate_combos[i] for i in order]
                        elif candidate_combos:
                            new_combos = candidate_combos[: self.batch_size]
                        else:
                            new_combos = self._sample_random_combos(self.batch_size, exclude=collected_set)
                    new_seqs = [self._combo_to_full_seq(c) for c in new_combos]
                else:
                    # Default path: train GPT with RL, then apply space-definition selection
                    logger.info(f"Step 3: Training GPT for {self.n_steps_per_round} steps (collecting variants)...")
                    generated_seqs, generated_fitness = self._train_gpt_on_surrogate(
                        self.n_steps_per_round, round_idx=round_idx
                    )

                    if self.sampling_strategy == 'cluster':
                        # CLADE-2 style: top-k cutoff + clustering
                        apply_cutoff = not is_last_round

                        logger.info(f"Cluster sampling from RL variants (cutoff={apply_cutoff})...")
                        new_seqs, new_combos = self._cluster_sample(
                            seqs=generated_seqs,
                            predicted_fitness=generated_fitness,
                            n_samples=self.batch_size,
                            exclude_combos=collected_set,
                            apply_cutoff=apply_cutoff,
                        )

                    elif self.sampling_strategy == 'active':
                        # ALDE style: Thompson Sampling or UCB acquisition
                        logger.info(f"Active sampling from RL variants ({self.acquisition})...")
                        new_seqs, new_combos = self._active_sample(
                            seqs=generated_seqs,
                            n_samples=self.batch_size,
                            exclude_combos=collected_set,
                            acquisition=self.acquisition,
                            xi=self.xi,
                        )

                    else:
                        raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

            if len(new_combos) == 0:
                logger.warning("No new samples generated, skipping round")
                continue

            # Get ground truth fitness for new samples
            new_fitness = self._get_ground_truth_fitness(new_combos)

            # Add to collected data
            self.collected_combos.extend(new_combos)
            self.collected_fitness.extend(new_fitness.tolist())
            self.collected_seqs.extend(new_seqs)

            # Store for metrics
            self.all_generated_seqs.extend(new_seqs)
            self.all_oracle_fitness.extend(new_fitness.tolist())

            # Round 1: Train initial surrogate and create GPT models
            if round_idx == 0:
                logger.info(f"Training initial surrogate on {len(self.collected_combos)} samples...")
                self._train_surrogate()

                logger.info("Creating initial GPT models...")
                self._create_models()

            # Get surrogate predictions for new samples (for model quality metrics)
            if self.surrogate_trained:
                ucb_scores, raw_pred = self._predict_surrogate(new_seqs)
                self.all_predicted_fitness.extend(raw_pred.tolist())
            else:
                self.all_predicted_fitness.extend(new_fitness.tolist())

            # Round statistics
            round_runtime = (datetime.now() - round_start).total_seconds()
            max_fitness_so_far = max(self.collected_fitness)
            mean_fitness_round = np.mean(new_fitness)

            round_info = {
                'round': round_idx + 1,
                'n_new_samples': len(new_combos),
                'n_total_samples': len(self.collected_combos),
                'mean_fitness_round': float(mean_fitness_round),
                'max_fitness_so_far': float(max_fitness_so_far),
                'is_last_round': is_last_round,
                'sampling_strategy': self.sampling_strategy if round_idx > 0 else 'random',
                'runtime_seconds': round_runtime,
            }
            self.round_data.append(round_info)

            logger.info(f"Round {round_idx + 1} complete:")
            logger.info(f"  New samples: {len(new_combos)}")
            logger.info(f"  Total samples: {len(self.collected_combos)}")
            logger.info(f"  Mean fitness (round): {mean_fitness_round:.4f}")
            logger.info(f"  Max fitness (all): {max_fitness_so_far:.4f}")
            logger.info(f"  Runtime: {round_runtime:.1f}s")

            # TensorBoard round-level logging
            self.writer.add_scalar('round/mean_fitness', mean_fitness_round, round_idx + 1)
            self.writer.add_scalar('round/max_fitness', max_fitness_so_far, round_idx + 1)
            self.writer.add_scalar('round/n_total_samples', len(self.collected_combos), round_idx + 1)
            self.writer.add_scalar('round/runtime_seconds', round_runtime, round_idx + 1)

            # Log fitness distribution stats
            if len(new_fitness) > 0:
                self.writer.add_scalar('round/min_fitness_batch', float(np.min(new_fitness)), round_idx + 1)
                self.writer.add_scalar('round/std_fitness_batch', float(np.std(new_fitness)), round_idx + 1)

            # Log regret
            regret = 1.0 - max_fitness_so_far  # Assuming normalized fitness
            self.writer.add_scalar('round/simple_regret', regret, round_idx + 1)

        # Save final model
        if self.agent_model is not None:
            save_gpt_model(self.agent_model, self.save_dir, 'Agent_final')

        # Save round data
        round_data_path = os.path.join(self.save_dir, 'round_data.json')
        with open(round_data_path, 'w') as f:
            json.dump(self.round_data, f, indent=2)

        # Close TensorBoard writer
        self.writer.close()
        logger.info(f"TensorBoard logs saved to: {self.save_dir}")

        return self.all_generated_seqs, self.all_predicted_fitness, self.all_oracle_fitness


# Keep old trainer for backward compatibility
class GB1Trainer(IterativeGB1Trainer):
    """Alias for backward compatibility."""
    pass


# ============================================================================
# Main Experiment Functions
# ============================================================================

def create_model(config) -> Tuple[GPT, GPTConfig]:
    """Create a new GPT model from config."""
    mconf = GPTConfig(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
    )
    model = GPT(mconf)
    return model, mconf


def load_landscape_data(data_path: str) -> Tuple[List[str], np.ndarray]:
    """Load complete GB1 fitness landscape."""
    df = pd.read_csv(data_path)

    # Get sequences (AACombo or full seq)
    if 'seq' in df.columns:
        sequences = df['seq'].tolist()
    else:
        sequences = df['AACombo'].tolist()

    fitness = df['fitness'].values
    # Normalize to [0, 1]
    fitness = fitness / np.max(fitness)

    return sequences, fitness


def compute_all_metrics(
    generated_seqs: List[str],
    generated_fitness: List[float],
    all_sequences: List[str],
    all_fitness: np.ndarray,
    batch_size: int = 96,
    predicted_fitness: Optional[List[float]] = None,
    wildtype: Optional[str] = None,
    y_pred_all: Optional[np.ndarray] = None,
    y_std_all: Optional[np.ndarray] = None,
) -> MetricsResult:
    """
    Compute all evaluation metrics (aligned with ALDE).

    Args:
        generated_seqs: List of generated sequences
        generated_fitness: Oracle/ground truth fitness of generated sequences
        all_sequences: Complete landscape sequences
        all_fitness: Complete landscape fitness values
        batch_size: Batch size for trajectory computation
        predicted_fitness: Predicted fitness from surrogate (for model quality metrics)
        wildtype: Wild-type sequence (for epistatic metrics)
        y_pred_all: Model predictions for all sequences (for uncertainty metrics)
        y_std_all: Model uncertainty for all sequences (for uncertainty metrics)

    Returns:
        MetricsResult with all computed metrics
    """
    result = MetricsResult()

    generated_fitness_np = np.array(generated_fitness)
    global_max = np.max(all_fitness)
    global_min = np.min(all_fitness)

    # Get unique sequences
    unique_seqs = list(set(generated_seqs))

    # --- Exploration Metrics ---
    result.high_fitness_proximity = high_fitness_proximity(
        unique_seqs, all_sequences, all_fitness,
        percentile=0.9
    )

    # Use first batch as "initial" sequences for novelty
    initial_seqs = generated_seqs[:batch_size] if len(generated_seqs) >= batch_size else generated_seqs
    later_seqs = generated_seqs[batch_size:] if len(generated_seqs) > batch_size else []
    if later_seqs:
        result.novelty = novelty(later_seqs, initial_seqs)

    result.batch_diversity = batch_diversity(unique_seqs[:256])  # Sample for efficiency

    # --- Functional Metrics ---
    result.normalized_fitness_median_top128 = normalized_fitness_topk(
        generated_fitness_np, k=128, min_fitness=global_min, max_fitness=global_max
    )
    result.normalized_fitness_median_top256 = normalized_fitness_topk(
        generated_fitness_np, k=256, min_fitness=global_min, max_fitness=global_max
    )
    result.max_fitness = max_fitness_metric(generated_fitness_np)

    # --- Model Quality Metrics ---
    if predicted_fitness is not None:
        predicted_fitness_np = np.array(predicted_fitness)

        # Spearman correlation between predicted and oracle fitness
        result.spearman_correlation = spearman_correlation(
            generated_fitness_np, predicted_fitness_np
        )

        # Epistatic correlation (requires wildtype)
        if wildtype is not None and len(generated_seqs) >= 10:
            result.epistatic_correlation = epistatic_score_correlation(
                generated_seqs, generated_fitness_np, predicted_fitness_np, wildtype
            )

            result.recall_high_order = recall_high_order_mutants(
                generated_seqs, generated_fitness_np, predicted_fitness_np, wildtype,
                min_mutations=2, top_k=100
            )

    # --- Success Metrics ---
    result.simple_regret = simple_regret(result.max_fitness, global_max)
    result.global_max_found = (result.max_fitness >= global_max * 0.99)

    # --- Uncertainty Metrics ---
    if y_pred_all is not None and y_std_all is not None:
        # Compute on held-out set (sequences not queried)
        queried_set = set(generated_seqs)
        holdout_mask = np.array([seq not in queried_set for seq in all_sequences])

        if np.sum(holdout_mask) > 100:
            holdout_true = all_fitness[holdout_mask]
            holdout_pred = y_pred_all[holdout_mask]
            holdout_std = y_std_all[holdout_mask]

            result.miscalibration_area = miscalibration_area(
                holdout_true, holdout_pred, holdout_std
            )
            result.expected_calibration_error = expected_calibration_error(
                holdout_true, holdout_pred, holdout_std
            )

    # --- Trajectory (per batch) ---
    fitness_traj = []
    regret_traj = []
    for i in range(0, len(generated_fitness_np), batch_size):
        batch_fitness = generated_fitness_np[:i + batch_size]
        max_so_far = np.max(batch_fitness)
        fitness_traj.append(float(max_so_far))
        regret_traj.append(float(global_max - max_so_far))

    result.fitness_trajectory = fitness_traj
    result.regret_trajectory = regret_traj

    return result


def run_single_experiment(
    seed: int,
    config_path: str,
    output_path: str,
    data_dir: str,
    compute_metrics: bool = True,
    run_id: Optional[int] = None,
    n_rounds: int = 4,
    n_steps_per_round: int = 100,
    top_k_cutoff: int = 1000,
    n_clusters: int = 10,
    sampling_strategy: str = 'cluster',
    acquisition: str = 'ts',
    xi: float = 4.0,
    finetune_prior: bool = True,
    n_finetune_epochs: int = 10,
    finetune_lr: float = 1e-4,
    ablation: str = "none",
) -> Dict[str, Any]:
    """Run a single AlphaVariant iterative optimization experiment on GB1.

    Args:
        seed: Random seed
        config_path: Path to config file
        output_path: Output directory
        data_dir: Data directory
        compute_metrics: Whether to compute metrics
        run_id: Run identifier
        n_rounds: Number of iterative rounds (default: 4)
        n_steps_per_round: GPT training steps per round (default: 100)
        top_k_cutoff: Top-k cutoff for rounds before last (default: 1000)
        n_clusters: Number of clusters for CLADE-2 sampling (default: 10)
        sampling_strategy: 'cluster' (CLADE-2) or 'active' (ALDE-style)
        acquisition: For 'active' strategy - 'ts', 'ucb', or 'ei'
        xi: Exploration parameter for UCB
        finetune_prior: Whether to finetune prior on collected sequences before RL
        n_finetune_epochs: Number of epochs for prior finetuning (default: 10)
        finetune_lr: Learning rate for prior finetuning
        ablation: Component-removal flag (see IterativeGB1Trainer for values)
    """

    if run_id is None:
        run_id = seed

    print(f"\n{'='*60}")
    print(f"Starting AlphaVariant Iterative Optimization on GB1")
    print(f"  Seed: {seed}")
    print(f"  Rounds: {n_rounds}")
    print(f"  Steps per round: {n_steps_per_round}")
    print(f"  Sampling strategy: {sampling_strategy}")
    if sampling_strategy == 'cluster':
        print(f"    Top-k cutoff: {top_k_cutoff}")
        print(f"    N clusters: {n_clusters}")
    else:
        print(f"    Acquisition: {acquisition}")
        print(f"    Xi: {xi}")
    print(f"  Prior finetuning: {finetune_prior}")
    if finetune_prior:
        print(f"    Finetune epochs: {n_finetune_epochs}")
        print(f"    Finetune LR: {finetune_lr}")
    print(f"  Config: {config_path}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    # Load configuration
    config = parse_config(config_path)

    # Set seed
    set_random_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create output directory
    run_dir = os.path.join(output_path, f'seed_{seed}')
    os.makedirs(run_dir, exist_ok=True)

    # Save config copy
    os.system(f'cp {config_path} {os.path.join(run_dir, "config.yaml")}')

    # Get landscape path
    if hasattr(config.task.fn_config.fitness, 'gb1_oracle'):
        landscape_path = config.task.fn_config.fitness.gb1_oracle.data_path
    elif hasattr(config.task.fn_config.fitness, 'gb1_surrogate'):
        landscape_path = config.task.fn_config.fitness.gb1_surrogate.data_path
    else:
        landscape_path = f'{data_dir}/GB1/data.csv'

    # Initialize template
    logger.info("Initializing template with hotspots...")
    fasta_sequences, _ = read_fasta_as_list(config.template.ref_seq_path)
    ref_seq = fasta_sequences[0]
    positions, pos_aa_candidates = load_hotspot(config.template.hotspot_path)
    template = PDETemplate(ref_seq, positions=positions, pos_aa_candidates=pos_aa_candidates)

    logger.info(f"Template positions: {positions}")
    logger.info(f"Reference sequence length: {len(ref_seq)}")

    # Initialize iterative trainer
    trainer = IterativeGB1Trainer(
        model_config=config.model,
        optim_config=config.optim,
        template=template,
        landscape_path=landscape_path,
        save_dir=run_dir,
        batch_size=config.train.batch_size,
        n_rounds=n_rounds,
        n_steps_per_round=n_steps_per_round,
        sigma=config.train.sigma,
        device=config.train.device,
        seed=seed,
        top_k_cutoff=top_k_cutoff,
        n_clusters=n_clusters,
        sampling_strategy=sampling_strategy,
        acquisition=acquisition,
        xi=xi,
        finetune_prior=finetune_prior,
        n_finetune_epochs=n_finetune_epochs,
        finetune_lr=finetune_lr,
        ablation=ablation,
    )

    # Run iterative training
    start_time = datetime.now()
    all_seqs, all_predicted, all_oracle = trainer.train()
    runtime = (datetime.now() - start_time).total_seconds()

    logger.info(f"Training completed in {runtime:.1f} seconds")
    logger.info(f"Generated {len(all_seqs)} sequences ({len(set(all_seqs))} unique)")

    # Prepare result
    result = {
        'seed': seed,
        'run_id': run_id,
        'runtime_seconds': runtime,
        'n_sequences': len(all_seqs),
        'n_unique_sequences': len(set(all_seqs)),
        'n_rounds': n_rounds,
        'n_steps_per_round': n_steps_per_round,
        'round_data': trainer.round_data,
        'config': {
            'batch_size': config.train.batch_size,
            'n_rounds': n_rounds,
            'n_steps_per_round': n_steps_per_round,
            'sigma': config.train.sigma,
        }
    }

    # Compute metrics using ground truth fitness
    if compute_metrics:
        logger.info("Computing evaluation metrics (using ground truth fitness)...")

        all_landscape_seqs, all_landscape_fitness = load_landscape_data(landscape_path)

        # Get wildtype for epistatic metrics (GB1 wildtype combo)
        wildtype = 'MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'

        # Compute all metrics (aligned with ALDE)
        metrics_result = compute_all_metrics(
            generated_seqs=all_seqs,
            generated_fitness=all_oracle,  # Oracle/ground truth fitness
            all_sequences=all_landscape_seqs,
            all_fitness=all_landscape_fitness,
            batch_size=config.train.batch_size,
            predicted_fitness=all_predicted,  # Surrogate predictions (for model quality)
            wildtype=wildtype,
        )

        result['metrics'] = metrics_result.to_dict()
        result['fitness_trajectory'] = metrics_result.fitness_trajectory
        result['regret_trajectory'] = metrics_result.regret_trajectory

        # Print summary (ALDE-aligned order)
        print("\n" + "-"*60)
        print("Metrics Summary (ALDE-aligned, Oracle/Ground Truth):")
        print("-"*60)
        print(f"  [Exploration]")
        print(f"    high_fitness_proximity:        {metrics_result.high_fitness_proximity:.4f}")
        print(f"    novelty:                       {metrics_result.novelty:.4f}")
        print(f"    batch_diversity:               {metrics_result.batch_diversity:.4f}")
        print(f"  [Functional]")
        print(f"    normalized_fitness_top128:     {metrics_result.normalized_fitness_median_top128:.4f}")
        print(f"    normalized_fitness_top256:     {metrics_result.normalized_fitness_median_top256:.4f}")
        print(f"    max_fitness:                   {metrics_result.max_fitness:.4f}")
        print(f"  [Model Quality]")
        print(f"    spearman_correlation:          {metrics_result.spearman_correlation:.4f}")
        print(f"    epistatic_correlation:         {metrics_result.epistatic_correlation:.4f}")
        print(f"    recall_high_order:             {metrics_result.recall_high_order:.4f}")
        print(f"  [Success]")
        print(f"    simple_regret:                 {metrics_result.simple_regret:.4f}")
        print(f"    global_max_found:              {metrics_result.global_max_found}")
        print(f"  [Uncertainty]")
        print(f"    miscalibration_area:           {metrics_result.miscalibration_area:.4f}")
        print(f"    expected_calibration_error:    {metrics_result.expected_calibration_error:.4f}")
        print("-"*60)

        # Save metrics
        metrics_path = os.path.join(run_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Metrics saved to: {metrics_path}")

    return result


def aggregate_run_metrics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across multiple runs (ALDE-aligned order)."""
    # Metrics in ALDE order
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
        values = [r['metrics'][name] for r in results if 'metrics' in r and name in r['metrics']]
        if values:
            aggregated[name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }

    # Global max hit count
    hit_count = sum(1 for r in results if 'metrics' in r and r['metrics'].get('global_max_found', False))
    aggregated['global_max_hit_count'] = {
        'count': hit_count,
        'rate': hit_count / len(results) if results else 0
    }

    return aggregated


def save_aggregated_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save aggregated results across all runs."""
    aggregated = aggregate_run_metrics(results)

    # Create summary DataFrame
    summary_data = []
    for metric, stats in aggregated.items():
        if isinstance(stats, dict):
            if 'mean' in stats:
                summary_data.append({
                    'metric': metric,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max']
                })
            elif 'count' in stats:
                summary_data.append({
                    'metric': metric,
                    'mean': stats['rate'],
                    'std': 0,
                    'min': stats['count'],
                    'max': len(results)
                })

    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    summary_path = os.path.join(output_path, 'aggregated_metrics.csv')
    summary_df.to_csv(summary_path, index=False)

    # Save to JSON
    json_path = os.path.join(output_path, 'aggregated_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'aggregated_metrics': aggregated,
            'n_runs': len(results),
            'seeds': [r['seed'] for r in results],
            'config': results[0].get('config', {}) if results else {}
        }, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*70)
    print("AGGREGATED METRICS SUMMARY")
    print("="*70)
    print(f"{'Metric':<40} {'Mean':>10} {'Std':>10}")
    print("-"*70)
    for _, row in summary_df.iterrows():
        if row['metric'] == 'global_max_hit_count':
            print(f"{row['metric']:<40} {row['mean']*100:>9.1f}% ({int(row['min'])}/{int(row['max'])})")
        else:
            print(f"{row['metric']:<40} {row['mean']:>10.4f} {row['std']:>10.4f}")
    print("="*70)

    logger.info(f"Aggregated metrics saved to: {summary_path}")


def load_seeds_from_file(filepath: str, num_seeds: int) -> List[int]:
    """Load seeds from a file."""
    seeds = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                seeds.append(int(line))
                if len(seeds) >= num_seeds:
                    break
    return seeds


def main():
    parser = argparse.ArgumentParser(
        description="Run AlphaVariant iterative optimization on GB1 benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run with default seed (4 rounds of iterative training)
  python run_GB1.py

  # Single run with specific seed
  python run_GB1.py --seed 42

  # Specify number of rounds
  python run_GB1.py --seed 42 --n_rounds 4

  # Specify steps per round
  python run_GB1.py --seed 42 --n_steps_per_round 100

  # Multiple runs for randomness evaluation
  python run_GB1.py --seeds 42 123 456 789 1000

  # Load seeds from file
  python run_GB1.py --seed_file seeds.txt --num_seeds 5

  # Skip metrics computation
  python run_GB1.py --seed 42 --skip_metrics
        """
    )

    # Seed configuration
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=None, help="Single random seed")
    seed_group.add_argument("--seeds", type=int, nargs='+', help="Multiple seeds")
    seed_group.add_argument("--seed_file", type=str, help="Path to file containing seeds")

    parser.add_argument("--num_seeds", type=int, default=5, help="Number of seeds from file (default: 5)")
    parser.add_argument("--config", type=str, default="examples/GB1/config/train_agent_config.yaml",
                       help="Path to config file")
    parser.add_argument("--output_path", type=str, default="results/GB1_AlphaVariant/",
                       help="Output directory")
    parser.add_argument("--data_dir", type=str, default="/home/xux/Desktop/AlphaVariant/Benchmark/data",
                       help="Base data directory")
    parser.add_argument("--skip_metrics", action="store_true", help="Skip metrics computation")
    parser.add_argument("--n_rounds", type=int, default=4,
                       help="Number of iterative rounds (default: 4)")
    parser.add_argument("--n_steps_per_round", type=int, default=500,
                       help="GPT training steps per round (default: 500)")
    parser.add_argument("--top_k_cutoff", type=int, default=1000,
                       help="Top-k cutoff for CLADE-2 sampling in rounds before last (default: 1000)")
    parser.add_argument("--n_clusters", type=int, default=10,
                       help="Number of clusters for CLADE-2 style sampling (default: 10)")
    parser.add_argument("--sampling", type=str, choices=['cluster', 'active'], default='cluster',
                       help="Sampling strategy: 'cluster' (CLADE-2) or 'active' (ALDE-style) (default: cluster)")
    parser.add_argument("--acquisition", type=str, choices=['ts', 'ucb', 'ei'], default='ts',
                       help="Acquisition function for active sampling: 'ts' (Thompson), 'ucb', 'ei' (default: ts)")
    parser.add_argument("--xi", type=float, default=4.0,
                       help="Exploration parameter for UCB acquisition (default: 4.0)")
    parser.add_argument("--finetune_prior", action="store_true", default=False,
                       help="Finetune prior on collected sequences before RL (default: False)")
    parser.add_argument("--no_finetune_prior", action="store_true",
                       help="Disable prior finetuning (redundant, disabled by default)")
    parser.add_argument("--n_finetune_epochs", type=int, default=10,
                       help="Number of epochs for prior finetuning (default: 10)")
    parser.add_argument("--finetune_lr", type=float, default=1e-4,
                       help="Learning rate for prior finetuning (default: 1e-4)")
    parser.add_argument(
        "--ablation", type=str, default="none",
        choices=["none", "no-gpt", "no-space", "static-reward", "no-rl"],
        help="Component-removal flag for ablation studies. 'none' (default) "
             "runs the full AlphaVariant pipeline.",
    )

    args = parser.parse_args()

    # Handle finetune_prior flag
    if args.no_finetune_prior:
        args.finetune_prior = False
    warnings.filterwarnings("ignore")

    # Determine seeds
    if args.seeds is not None:
        seeds = args.seeds
    elif args.seed_file is not None:
        seeds = load_seeds_from_file(args.seed_file, args.num_seeds)
        print(f"Loaded {len(seeds)} seeds from {args.seed_file}")
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42]

    print(f"\nRunning AlphaVariant Iterative Optimization")
    print(f"  Seeds: {seeds}")
    print(f"  Rounds: {args.n_rounds}")
    print(f"  Steps per round: {args.n_steps_per_round}")
    print(f"  Sampling strategy: {args.sampling}")
    if args.sampling == 'cluster':
        print(f"    Top-k cutoff: {args.top_k_cutoff}")
        print(f"    N clusters: {args.n_clusters}")
    else:
        print(f"    Acquisition: {args.acquisition}")
        print(f"    Xi: {args.xi}")
    print(f"  Prior finetuning: {args.finetune_prior}")
    if args.finetune_prior:
        print(f"    Finetune epochs: {args.n_finetune_epochs}")
        print(f"    Finetune LR: {args.finetune_lr}")
    print(f"  Output path: {args.output_path}")
    print(f"  Compute metrics: {not args.skip_metrics}")

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Run experiments
    results = []
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Running experiment with seed={seed}")
        result = run_single_experiment(
            seed=seed,
            config_path=args.config,
            output_path=args.output_path,
            data_dir=args.data_dir,
            compute_metrics=not args.skip_metrics,
            run_id=i + 1,
            n_rounds=args.n_rounds,
            n_steps_per_round=args.n_steps_per_round,
            top_k_cutoff=args.top_k_cutoff,
            n_clusters=args.n_clusters,
            sampling_strategy=args.sampling,
            acquisition=args.acquisition,
            xi=args.xi,
            finetune_prior=args.finetune_prior,
            n_finetune_epochs=args.n_finetune_epochs,
            finetune_lr=args.finetune_lr,
            ablation=args.ablation,
        )
        results.append(result)

    # Aggregate results
    if len(results) > 1 and not args.skip_metrics:
        save_aggregated_results(results, args.output_path)

    # Final summary
    print(f"\n{'='*60}")
    print("Experiment Complete")
    print(f"{'='*60}")
    print(f"Total runs: {len(results)}")
    print(f"Configuration:")
    print(f"  - Method: AlphaVariant Iterative (GPT + REINFORCE)")
    print(f"  - Rounds: {args.n_rounds}")
    print(f"  - Batch size: 96 samples per round")
    print(f"  - Steps per round: {args.n_steps_per_round}")
    print(f"  - Sampling strategy: {args.sampling}")
    if args.sampling == 'cluster':
        print(f"    top_k={args.top_k_cutoff}, n_clusters={args.n_clusters}")
    else:
        print(f"    acquisition={args.acquisition}, xi={args.xi}")
    print(f"  - Prior finetuning: {args.finetune_prior}")
    if args.finetune_prior:
        print(f"    epochs={args.n_finetune_epochs}, lr={args.finetune_lr}")
    print(f"  - Surrogate: Updated with ALL collected data each round")
    print(f"Results saved to: {args.output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
