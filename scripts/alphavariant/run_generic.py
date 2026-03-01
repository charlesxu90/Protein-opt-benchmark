#!/usr/bin/env python
"""
run_generic.py - Generic dataset runner for AlphaVariant optimization

A dataset-agnostic version of the AlphaVariant benchmark scripts. Automatically
detects sequence length from loaded data, generates appropriate GPT config values,
and uses the highest-fitness sequence as the wildtype fallback.

Configuration:
    - Model: GPT-based generative model (config auto-generated from seq_len)
    - Scorer: aa_onehot based surrogate with iterative updates
    - Training: REINFORCE with prior regularization
    - Batch size: 96
    - Rounds: 15 (iterative approach)

Iterative Training Process:
    - Round 1: Initial sampling from configurable fitness region
    - Rounds 2-15: GPT-guided sampling, get ground truth fitness, update surrogate, train GPT

Usage:
    # Run on AAV_med dataset
    python run_generic.py --dataset AAV_med --seed 42

    # Run on a custom dataset (data/<name>/data.csv must exist with seq,fitness columns)
    python run_generic.py --dataset my_protein --seed 42

    # Use hard-level initialization (bottom 20th percentile)
    python run_generic.py --dataset AAV_hard --level hard --seed 42

    # Override with an existing YAML config
    python run_generic.py --dataset AAV_med --config examples/AAV_med/config/train_agent_config.yaml

    # Multiple seeds
    python run_generic.py --dataset GFP_med --seeds 42 123 456

    # Load seeds from file
    python run_generic.py --dataset GB1 --seed_file ../rand_seeds.txt --num_seeds 5

    # Skip metrics computation (faster)
    python run_generic.py --dataset AAV_med --seed 42 --skip_metrics
"""

from __future__ import annotations
import argparse
import copy
import json
import os
import sys
import tempfile
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

# =============================================================================
# Constants
# =============================================================================

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
ALL_20_AAS = list(AMINO_ACIDS)


# =============================================================================
# Auto-config generation
# =============================================================================

def auto_detect_from_data(data_path: str) -> Dict[str, Any]:
    """
    Auto-detect dataset properties from the data CSV.

    Returns dict with:
        - seq_len: detected sequence length (mode of all lengths)
        - wildtype: sequence with highest fitness
        - wildtype_fitness: fitness of the wildtype
        - n_variants: total number of variants
    """
    df = pd.read_csv(data_path)
    sequences = df['seq'].tolist()
    fitness = df['fitness'].values

    # Detect sequence length (use mode for variable-length datasets)
    lengths = [len(s) for s in sequences]
    seq_len = int(pd.Series(lengths).mode().iloc[0])

    # Use highest-fitness sequence as wildtype fallback
    best_idx = int(np.argmax(fitness))
    wildtype = sequences[best_idx]
    wildtype_fitness = float(fitness[best_idx])

    logger.info(f"Auto-detected from {data_path}:")
    logger.info(f"  Sequence length (mode): {seq_len}")
    logger.info(f"  Number of variants: {len(sequences)}")
    logger.info(f"  Fitness range: [{fitness.min():.4f}, {fitness.max():.4f}]")
    logger.info(f"  Wildtype (best fitness): {wildtype[:40]}{'...' if len(wildtype) > 40 else ''} "
                f"(fitness={wildtype_fitness:.4f})")

    return {
        'seq_len': seq_len,
        'wildtype': wildtype,
        'wildtype_fitness': wildtype_fitness,
        'n_variants': len(sequences),
    }


def generate_config_dict(
    seq_len: int,
    wildtype: str,
    data_path: str,
    dataset_name: str,
    batch_size: int = 96,
    sigma: float = 60,
    device: str = 'cuda:0',
    n_steps: int = 500,
) -> Dict[str, Any]:
    """
    Generate a config dictionary appropriate for the detected sequence length.

    The GPT block_size is set to seq_len + 7 (buffer for start/end tokens).
    Model dimensions are scaled based on sequence length:
      - Short seqs (<=10): smaller model
      - Medium seqs (<=50): standard model
      - Long seqs (>50): larger block_size, same layer/head count
    """
    block_size = seq_len + 7

    # Scale model based on sequence length
    if seq_len <= 10:
        n_layer = 4
        n_head = 4
        n_embd = 128
    elif seq_len <= 100:
        n_layer = 4
        n_head = 4
        n_embd = 128
    else:
        # Longer sequences: same architecture, larger block_size handles it
        n_layer = 4
        n_head = 4
        n_embd = 128

    config = {
        'task': {
            'score_type': 'weight',
            'prop_names': ['fitness'],
            'fn_config': {
                'fitness': {
                    'fn_type': f'{dataset_name}_surrogate',
                    f'{dataset_name}_surrogate': {
                        'data_path': data_path,
                        'seq_len': seq_len,
                        'n_init': batch_size,
                        'n_ensemble': 5,
                        'model_type': 'ensemble',
                        'min_val': -10.0,
                        'update_surrogate': False,
                    },
                },
            },
        },
        'bonus': {
            'bonus_type': 'none',
            'bonus_amplitude': 1,
        },
        'template': {
            'ref_seq_path': None,  # Will be generated dynamically
            'hotspot_path': None,  # Will be generated dynamically
        },
        'train': {
            'max_seq_len': seq_len,
            'batch_size': batch_size,
            'n_steps': n_steps,
            'sigma': sigma,
            'device': device,
            'n_devices': 1,
            'precision': 'bf16-mixed',
            'matmul_precision': 'high',
            'save_per_n_steps': 100,
            'seed': 42,
            'focus_on_hotspots': True,
        },
        'optim': {
            'learning_rate': 0.0001,
            'lr_decay': True,
            'weight_decay': 0.1,
            'beta_1': 0.9,
            'beta_2': 0.95,
            'grad_norm_clip': 1.0,
        },
        'model': {
            'vocab_size': 24,   # 20 AAs + start/end/pad/X
            'block_size': block_size,
            'n_layer': n_layer,
            'n_head': n_head,
            'n_embd': n_embd,
        },
    }

    return config


def dict_to_easydict(d: dict) -> Any:
    """Convert a nested dict to EasyDict for attribute-style access."""
    from easydict import EasyDict
    return EasyDict(d)


def create_temp_fasta(wildtype: str, output_dir: str) -> str:
    """Create a temporary FASTA file containing the wildtype sequence."""
    fasta_path = os.path.join(output_dir, 'wt.fasta')
    os.makedirs(os.path.dirname(fasta_path), exist_ok=True)
    with open(fasta_path, 'w') as f:
        f.write(f">WT\n{wildtype}\n")
    return fasta_path


def create_temp_hotspots(seq_len: int, output_dir: str) -> str:
    """Create a temporary hotspots CSV with all positions and all 20 AAs."""
    hotspot_path = os.path.join(output_dir, 'hotspots.csv')
    os.makedirs(os.path.dirname(hotspot_path), exist_ok=True)
    rows = []
    aa_list_str = repr(ALL_20_AAS)
    for pos in range(1, seq_len + 1):
        rows.append(f'{pos},"{aa_list_str}"')
    with open(hotspot_path, 'w') as f:
        f.write("pos,mut_aas\n")
        f.write("\n".join(rows) + "\n")
    return hotspot_path


# ============================================================================
# AlphaVariant Generic Iterative Trainer
# ============================================================================

class IterativeProteinTrainer:
    """
    Generic iterative trainer for protein optimization benchmarks.

    Training Process:
    - Round 1: Initial sampling from configurable fitness region
    - Rounds 2-N: GPT-guided sampling with clustering

    Each round:
    1. Sample sequences using GPT (from landscape in round 1)
    2. Apply top-k cutoff, cluster, sample from clusters
    3. Get ground truth fitness for sampled sequences
    4. Update surrogate with ALL collected data
    5. Train/fine-tune GPT model using surrogate predictions
    """

    def __init__(
        self,
        model_config: Any,
        optim_config: Any,
        template: PDETemplate,
        landscape_path: str,
        save_dir: str,
        seq_len: int,
        batch_size: int = 96,
        n_rounds: int = 15,
        n_steps_per_round: int = 500,
        sigma: float = 60,
        device: str = 'cuda:0',
        seed: int = 42,
        top_k_cutoff: int = 1000,
        n_clusters: int = 10,
        sampling_strategy: str = 'cluster',
        acquisition: str = 'ts',
        xi: float = 4.0,
        finetune_prior: bool = False,
        n_finetune_epochs: int = 10,
        finetune_lr: float = 1e-4,
        level: str = 'medium',
    ):
        """Initialize iterative trainer."""
        self.model_config = model_config
        self.optim_config = optim_config
        self.template = template
        self.landscape_path = landscape_path
        self.save_dir = save_dir
        self.seq_len = seq_len
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
        self.level = level

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
        self.collected_seqs = []
        self.collected_fitness = []  # Ground truth fitness

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

        # Track best sequence from collected data
        self.best_seq = None
        self.best_fitness = -np.inf

        # TensorBoard writer for logging
        self.writer = SummaryWriter(save_dir)
        self.global_step = 0

    def _load_landscape(self):
        """Load the complete fitness landscape."""
        df = pd.read_csv(self.landscape_path)

        self.seq_to_fitness = {}
        self.all_seqs = []
        self.all_fitness = []

        for _, row in df.iterrows():
            seq = row['seq']
            fitness = row['fitness']
            self.seq_to_fitness[seq] = fitness
            self.all_seqs.append(seq)
            self.all_fitness.append(fitness)

        self.all_fitness = np.array(self.all_fitness)
        self.max_fitness_raw = np.max(self.all_fitness)
        self.min_fitness_raw = np.min(self.all_fitness)

        logger.info(f"Loaded {len(self.all_seqs)} variants from landscape")
        logger.info(f"Fitness range: [{self.min_fitness_raw:.4f}, {self.max_fitness_raw:.4f}]")

    def _get_ground_truth_fitness(self, seqs: List[str]) -> np.ndarray:
        """Get ground truth fitness for sequences."""
        fitness = np.zeros(len(seqs))
        for i, seq in enumerate(seqs):
            if seq in self.seq_to_fitness:
                fitness[i] = self.seq_to_fitness[seq]
            else:
                # If sequence not in landscape, assign low fitness
                fitness[i] = 0.0
        return fitness

    def _sample_initial_seqs(self, n_samples: int, exclude: set = None) -> List[str]:
        """
        Sample initial sequences from a fitness region based on level.
        - medium: 40th-60th percentile
        - hard: bottom 20th percentile
        """
        if exclude is None:
            exclude = set()

        if self.level == 'medium':
            threshold_low = np.percentile(self.all_fitness, 40)
            threshold_high = np.percentile(self.all_fitness, 60)
            mask = (self.all_fitness >= threshold_low) & (self.all_fitness <= threshold_high)
        else:  # hard
            threshold = np.percentile(self.all_fitness, 20)
            mask = self.all_fitness <= threshold

        available_indices = np.where(mask)[0]
        available_seqs = [self.all_seqs[i] for i in available_indices if self.all_seqs[i] not in exclude]

        if len(available_seqs) <= n_samples:
            return available_seqs

        indices = np.random.choice(len(available_seqs), size=n_samples, replace=False)
        return [available_seqs[i] for i in indices]

    def _cluster_init_sample(self, n_samples: int, exclude: set = None) -> List[str]:
        """CLADE-2 style cluster-based initialization sampling."""
        from sklearn.cluster import KMeans
        from popscorer.fitness.aa_onehot_pred.embed import seqs2feat

        if exclude is None:
            exclude = set()

        if self.level == 'medium':
            threshold_low = np.percentile(self.all_fitness, 40)
            threshold_high = np.percentile(self.all_fitness, 60)
            mask = (self.all_fitness >= threshold_low) & (self.all_fitness <= threshold_high)
        else:
            threshold = np.percentile(self.all_fitness, 20)
            mask = self.all_fitness <= threshold

        available_indices = np.where(mask)[0]
        available_seqs = [self.all_seqs[i] for i in available_indices if self.all_seqs[i] not in exclude]

        if len(available_seqs) <= n_samples:
            return available_seqs

        # Extract features
        logger.info(f"Extracting features for {len(available_seqs)} sequences...")
        X = seqs2feat(available_seqs)

        # Determine number of clusters
        n_clusters = min(self.n_clusters, n_samples, len(available_seqs) // 2)
        n_clusters = max(n_clusters, 1)

        # Run KMeans clustering
        logger.info(f"Clustering into {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # Organize sequences by cluster
        clusters = [[] for _ in range(n_clusters)]
        for idx, label in enumerate(cluster_labels):
            clusters[label].append({'idx': idx, 'seq': available_seqs[idx]})

        # Shuffle within each cluster
        for cluster in clusters:
            np.random.shuffle(cluster)

        # Calculate samples per cluster
        samples_per_cluster = n_samples // n_clusters
        extra_samples = n_samples % n_clusters

        # Sample from each cluster
        selected_seqs = []
        cluster_indices = list(range(n_clusters))
        np.random.shuffle(cluster_indices)

        for i, cluster_idx in enumerate(cluster_indices):
            cluster = clusters[cluster_idx]
            n_from_cluster = samples_per_cluster + (1 if i < extra_samples else 0)
            n_from_cluster = min(n_from_cluster, len(cluster))

            for j in range(n_from_cluster):
                selected_seqs.append(cluster[j]['seq'])

        # Fill remaining if needed
        if len(selected_seqs) < n_samples:
            remaining_seqs = set(available_seqs) - set(selected_seqs)
            remaining_list = list(remaining_seqs)
            np.random.shuffle(remaining_list)
            n_needed = n_samples - len(selected_seqs)
            selected_seqs.extend(remaining_list[:n_needed])

        logger.info(f"Cluster init sampling: selected {len(selected_seqs)} from {n_clusters} clusters")
        return selected_seqs

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

    def _finetune_prior_model(self, seqs: List[str], fitness: np.ndarray, n_epochs: int = None):
        """Finetune the prior model on collected sequences using next-token prediction."""
        if n_epochs is None:
            n_epochs = self.n_finetune_epochs

        n_samples = len(seqs)
        steps_per_epoch = max(1, n_samples // self.batch_size)
        n_steps = n_epochs * steps_per_epoch

        if len(seqs) == 0:
            logger.warning("No sequences for prior finetuning")
            return

        # Enable gradients for prior
        for param in self.prior_model.parameters():
            param.requires_grad = True

        prior_optimizer = torch.optim.Adam(self.prior_model.parameters(), lr=self.finetune_lr)

        # Convert sequences to token tensors
        all_tokens = []
        for seq in seqs:
            tokens = [self.sd.char_idx.get(c, 0) for c in seq]
            all_tokens.append(tokens)
        all_tokens = torch.LongTensor(all_tokens).to(self.device)

        # Normalize fitness to create weights
        fitness_tensor = torch.from_numpy(fitness).float().to(self.device)
        temperature = 1.0
        weights = F.softmax(fitness_tensor / temperature, dim=0)

        # Give extra weight to best sequence if present
        if self.best_seq is not None and self.best_seq in seqs:
            best_idx = seqs.index(self.best_seq)
            weights[best_idx] *= 2.0
            weights = weights / weights.sum()

        n_seqs = len(seqs)
        seq_len = all_tokens.size(1)

        logger.info(f"Finetuning prior on {n_seqs} sequences for {n_epochs} epochs ({n_steps} steps)...")

        self.prior_model.train()
        for step in range(n_steps):
            batch_indices = torch.multinomial(weights, min(self.batch_size, n_seqs), replacement=True)
            batch_tokens = all_tokens[batch_indices]
            batch_weights = weights[batch_indices]

            x = rnn_start_token_vector(len(batch_indices), self.device)

            total_loss = 0.0
            for pos in range(seq_len):
                logits, _ = self.prior_model(x)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                targets = batch_tokens[:, pos]
                ce_loss = F.nll_loss(log_probs, targets, reduction='none')
                weighted_loss = (ce_loss * batch_weights).sum() / batch_weights.sum()
                total_loss += weighted_loss
                x = torch.cat([x, targets.unsqueeze(1)], dim=1)

            loss = total_loss / seq_len

            prior_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.prior_model.parameters(), 1.0)
            prior_optimizer.step()

            self.writer.add_scalar('finetune/loss', loss.item(), step + 1)

            if (step + 1) % 20 == 0 or step == 0:
                logger.debug(f"  Finetune step {step+1}/{n_steps}: loss={loss.item():.4f}")

        # Freeze prior again
        for param in self.prior_model.parameters():
            param.requires_grad = False

        # Copy finetuned prior to agent
        self.agent_model = copy.deepcopy(self.prior_model)
        for param in self.agent_model.parameters():
            param.requires_grad = True

        # Re-create optimizer
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

        if len(self.collected_seqs) == 0:
            logger.warning("No training data for surrogate")
            return

        X = seqs2feat(self.collected_seqs)
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
        logger.debug(f"Trained surrogate on {len(self.collected_seqs)} samples")

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
        """Predict using surrogate ensemble and return individual model predictions."""
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
        exclude_seqs: set = None,
        acquisition: str = 'ts',
        xi: float = 4.0,
    ) -> List[str]:
        """ALDE-style active learning sampling using Thompson Sampling or UCB."""
        if exclude_seqs is None:
            exclude_seqs = set()

        # Get unique sequences not already collected
        seen_seqs = set()
        unique_seqs = []

        for seq in seqs:
            if seq not in exclude_seqs and seq not in seen_seqs:
                unique_seqs.append(seq)
                seen_seqs.add(seq)

        if len(unique_seqs) == 0:
            return []

        if len(unique_seqs) <= n_samples:
            return unique_seqs

        # Get predictions from surrogate ensemble
        mu, sigma, all_predictions = self._predict_surrogate_with_samples(unique_seqs)

        # Compute acquisition scores
        if acquisition == 'ts':
            n_seqs = len(unique_seqs)
            n_models = all_predictions.shape[1]
            model_indices = np.random.randint(0, n_models, size=n_seqs)
            acquisition_scores = np.array([
                all_predictions[i, model_indices[i]] for i in range(n_seqs)
            ])
            logger.info(f"Active sampling (Thompson Sampling): {n_seqs} candidates")

        elif acquisition == 'ucb':
            acquisition_scores = mu + xi * sigma
            logger.info(f"Active sampling (UCB, xi={xi}): {len(unique_seqs)} candidates")

        elif acquisition == 'ei':
            best_so_far = max(self.collected_fitness) if self.collected_fitness else 0
            z = (mu - best_so_far) / (sigma + 1e-8)
            from scipy.stats import norm
            acquisition_scores = (mu - best_so_far) * norm.cdf(z) + sigma * norm.pdf(z)
            logger.info(f"Active sampling (EI): {len(unique_seqs)} candidates")

        else:
            acquisition_scores = mu + xi * sigma

        # Select top n_samples
        top_indices = np.argsort(acquisition_scores)[-n_samples:][::-1]
        selected_seqs = [unique_seqs[i] for i in top_indices]

        logger.info(f"Active sampling: selected {len(selected_seqs)} samples")
        return selected_seqs

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

        n_positions = self.seq_len

        with torch.no_grad():
            for step in range(n_positions):
                logits, _ = model(x)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                sampled_idx = Categorical(probs=probs).sample().squeeze()
                sequences.append(sampled_idx.view(-1, 1))
                x = torch.cat(sequences, 1)

        # Convert to sequences
        token_seqs = torch.cat(sequences, 1)
        aa_seqs = self.sd.matrix_to_seqs(token_seqs)

        return aa_seqs

    def _cluster_sample(
        self,
        seqs: List[str],
        predicted_fitness: np.ndarray,
        n_samples: int,
        exclude_seqs: set = None,
        apply_cutoff: bool = True,
    ) -> List[str]:
        """CLADE-2 style clustering-based sampling."""
        from sklearn.cluster import KMeans
        from popscorer.fitness.aa_onehot_pred.embed import seqs2feat

        if exclude_seqs is None:
            exclude_seqs = set()

        # Get unique sequences not already collected
        seen_seqs = set()
        unique_seqs = []
        unique_fitness = []

        for seq, fit in zip(seqs, predicted_fitness):
            if seq not in exclude_seqs and seq not in seen_seqs:
                unique_seqs.append(seq)
                unique_fitness.append(fit)
                seen_seqs.add(seq)

        if len(unique_seqs) == 0:
            return []

        unique_fitness = np.array(unique_fitness)

        # Apply top-k cutoff for rounds before the last
        if apply_cutoff and len(unique_seqs) > self.top_k_cutoff:
            top_k_indices = np.argsort(unique_fitness)[-self.top_k_cutoff:]
            unique_seqs = [unique_seqs[i] for i in top_k_indices]
            unique_fitness = unique_fitness[top_k_indices]
            logger.info(f"Applied top-{self.top_k_cutoff} cutoff, {len(unique_seqs)} candidates remaining")

        if len(unique_seqs) <= n_samples:
            return unique_seqs

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
                'fitness': unique_fitness[idx],
            })

        # Sort within each cluster by fitness (descending)
        for cluster in clusters:
            cluster.sort(key=lambda x: x['fitness'], reverse=True)

        # Compute cluster mean fitness
        cluster_mean_fitness = []
        for cluster in clusters:
            if len(cluster) > 0:
                mean_fit = np.mean([item['fitness'] for item in cluster])
            else:
                mean_fit = 0.0
            cluster_mean_fitness.append(mean_fit)

        cluster_mean_fitness = np.array(cluster_mean_fitness)

        # Normalize to get sampling probability
        if np.sum(cluster_mean_fitness) > 0:
            cluster_prob = cluster_mean_fitness / np.sum(cluster_mean_fitness)
        else:
            cluster_prob = np.ones(n_clusters) / n_clusters

        # Sample from clusters
        selected_seqs = []
        cluster_indices = [0] * n_clusters

        for _ in range(n_samples):
            for i in range(n_clusters):
                if cluster_indices[i] >= len(clusters[i]):
                    cluster_prob[i] = 0

            cluster_prob = np.maximum(cluster_prob, 0)
            prob_sum = np.sum(cluster_prob)
            if prob_sum == 0:
                break

            cluster_prob = cluster_prob / prob_sum
            selected_cluster = np.random.choice(n_clusters, p=cluster_prob)
            item = clusters[selected_cluster][cluster_indices[selected_cluster]]
            selected_seqs.append(item['seq'])
            cluster_indices[selected_cluster] += 1

        logger.info(f"Cluster sampling: selected {len(selected_seqs)} from {n_clusters} clusters")
        return selected_seqs

    def _filter_valid_seqs(self, seqs: List[str]) -> List[str]:
        """Filter sequences to only include valid amino acid sequences of correct length."""
        valid_aas = set(AMINO_ACIDS)
        valid_seqs = []
        for seq in seqs:
            # Remove non-AA characters (B, space, newline, X)
            clean_seq = ''.join(c for c in seq if c in valid_aas)
            if len(clean_seq) == self.seq_len:
                valid_seqs.append(clean_seq)
        return valid_seqs

    def _generate_rl_samples(self, n_samples: int) -> Tuple[List[str], np.ndarray]:
        """Generate samples using GPT model during RL training."""
        all_seqs = []
        all_fitness = []

        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        for _ in range(n_batches):
            seqs = self.sample_from_model(self.agent_model, self.batch_size)
            # Filter to valid sequences
            seqs = self._filter_valid_seqs(seqs)
            if len(seqs) == 0:
                continue

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
                log_prob = F.log_softmax(logits[:, -1, :], dim=-1)
                # Don't squeeze - keep batch dimension
                if log_prob.dim() == 1:
                    log_prob = log_prob.unsqueeze(0)
                log_probs += self.nll_loss(log_prob, x[:, step])
        return log_probs

    def _train_gpt_on_surrogate(
        self, n_steps: int, round_idx: int = 0
    ) -> Tuple[List[str], np.ndarray]:
        """Train GPT model using surrogate predictions as rewards."""
        self.agent_model.train()

        all_seqs_dict = {}

        for step in range(n_steps):
            seqs = self.sample_from_model(self.agent_model, self.batch_size)
            # Filter to valid sequences
            seqs = self._filter_valid_seqs(seqs)
            unique_seqs = list(set(seqs))
            if len(unique_seqs) == 0:
                continue

            # Convert to token indices
            token_seqs = []
            for seq in unique_seqs:
                tokens = [self.sd.char_idx.get(c, 0) for c in seq]
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
            prior_likelihoods = self.likelihood(self.prior_model, token_tensor)

            # Get surrogate predictions
            ucb_scores, raw_scores = self._predict_surrogate(unique_seqs)
            scores = torch.from_numpy(ucb_scores).float().to(self.device)

            # Collect sequences
            for seq, pred in zip(unique_seqs, raw_scores):
                if seq not in all_seqs_dict or pred > all_seqs_dict[seq]:
                    all_seqs_dict[seq] = pred

            # Ground truth for monitoring
            gt_fitness = self._get_ground_truth_fitness(unique_seqs)

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
            self.writer.add_scalar('train/round', round_idx + 1, self.global_step)

            if (step + 1) % 50 == 0 or step == 0:
                logger.debug(f"  Step {step+1}/{n_steps}: loss={loss.item():.4f}, "
                           f"avg_score={raw_scores.mean():.4f}, max_gt={gt_fitness.max():.4f}")

        collected_seqs = list(all_seqs_dict.keys())
        collected_fitness = np.array([all_seqs_dict[s] for s in collected_seqs])
        logger.info(f"Collected {len(collected_seqs)} unique sequences during RL training")

        return collected_seqs, collected_fitness

    def train(self) -> Tuple[List[str], List[float], List[float]]:
        """Run iterative training for n_rounds."""
        logger.info(f"Starting iterative training: {self.n_rounds} rounds, {self.batch_size} samples/round")
        logger.info(f"Sampling strategy: {self.sampling_strategy}")

        for round_idx in range(self.n_rounds):
            round_start = datetime.now()
            is_last_round = (round_idx == self.n_rounds - 1)

            logger.info(f"\n{'='*50}")
            logger.info(f"Round {round_idx + 1}/{self.n_rounds}" + (" (LAST)" if is_last_round else ""))
            logger.info(f"{'='*50}")

            if round_idx == 0:
                # Round 1: Initial sampling from landscape
                collected_set = set(self.collected_seqs)

                if self.sampling_strategy == 'cluster':
                    logger.info("Cluster-based initialization from landscape...")
                    new_seqs = self._cluster_init_sample(self.batch_size, exclude=collected_set)
                else:
                    logger.info("Random sampling from landscape...")
                    new_seqs = self._sample_initial_seqs(self.batch_size, exclude=collected_set)

            else:
                # Rounds 2+: Finetune prior, train surrogate, run RL
                if self.finetune_prior:
                    logger.info(f"Step 1: Finetuning prior on {len(self.collected_seqs)} sequences...")
                    self._finetune_prior_model(
                        seqs=self.collected_seqs,
                        fitness=np.array(self.collected_fitness),
                        n_epochs=self.n_finetune_epochs,
                    )

                logger.info(f"Step 2: Training surrogate on {len(self.collected_seqs)} samples...")
                self._train_surrogate()

                logger.info(f"Step 3: Training GPT for {self.n_steps_per_round} steps...")
                generated_seqs, generated_fitness = self._train_gpt_on_surrogate(
                    self.n_steps_per_round, round_idx=round_idx
                )

                collected_set = set(self.collected_seqs)

                if self.sampling_strategy == 'cluster':
                    apply_cutoff = not is_last_round
                    logger.info(f"Cluster sampling from RL variants (cutoff={apply_cutoff})...")
                    new_seqs = self._cluster_sample(
                        seqs=generated_seqs,
                        predicted_fitness=generated_fitness,
                        n_samples=self.batch_size,
                        exclude_seqs=collected_set,
                        apply_cutoff=apply_cutoff,
                    )

                elif self.sampling_strategy == 'active':
                    logger.info(f"Active sampling from RL variants ({self.acquisition})...")
                    new_seqs = self._active_sample(
                        seqs=generated_seqs,
                        n_samples=self.batch_size,
                        exclude_seqs=collected_set,
                        acquisition=self.acquisition,
                        xi=self.xi,
                    )

                else:
                    raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

            if len(new_seqs) == 0:
                logger.warning("No new samples generated, skipping round")
                continue

            # Get ground truth fitness
            new_fitness = self._get_ground_truth_fitness(new_seqs)

            # Update best sequence tracking
            max_idx = np.argmax(new_fitness)
            if new_fitness[max_idx] > self.best_fitness:
                self.best_fitness = float(new_fitness[max_idx])
                self.best_seq = new_seqs[max_idx]
                if round_idx == 0:
                    logger.info(f"Initial best sequence (fitness: {self.best_fitness:.4f})")
                else:
                    logger.info(f"New best sequence found (fitness: {self.best_fitness:.4f})")

            # Add to collected data
            self.collected_seqs.extend(new_seqs)
            self.collected_fitness.extend(new_fitness.tolist())

            # Store for metrics
            self.all_generated_seqs.extend(new_seqs)
            self.all_oracle_fitness.extend(new_fitness.tolist())

            # Round 1: Train initial surrogate and create GPT models
            if round_idx == 0:
                logger.info(f"Training initial surrogate on {len(self.collected_seqs)} samples...")
                self._train_surrogate()

                logger.info("Creating initial GPT models...")
                self._create_models()

            # Get surrogate predictions for metrics
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
                'n_new_samples': len(new_seqs),
                'n_total_samples': len(self.collected_seqs),
                'mean_fitness_round': float(mean_fitness_round),
                'max_fitness_so_far': float(max_fitness_so_far),
                'is_last_round': is_last_round,
                'runtime_seconds': round_runtime,
            }
            self.round_data.append(round_info)

            logger.info(f"Round {round_idx + 1} complete:")
            logger.info(f"  New samples: {len(new_seqs)}")
            logger.info(f"  Total samples: {len(self.collected_seqs)}")
            logger.info(f"  Mean fitness (round): {mean_fitness_round:.4f}")
            logger.info(f"  Max fitness (all): {max_fitness_so_far:.4f}")
            logger.info(f"  Runtime: {round_runtime:.1f}s")

            # TensorBoard logging
            self.writer.add_scalar('round/mean_fitness', mean_fitness_round, round_idx + 1)
            self.writer.add_scalar('round/max_fitness', max_fitness_so_far, round_idx + 1)
            self.writer.add_scalar('round/n_total_samples', len(self.collected_seqs), round_idx + 1)
            self.writer.add_scalar('round/runtime_seconds', round_runtime, round_idx + 1)

            if len(new_fitness) > 0:
                self.writer.add_scalar('round/min_fitness_batch', float(np.min(new_fitness)), round_idx + 1)
                self.writer.add_scalar('round/std_fitness_batch', float(np.std(new_fitness)), round_idx + 1)

            regret = self.max_fitness_raw - max_fitness_so_far
            self.writer.add_scalar('round/simple_regret', regret, round_idx + 1)

        # Save final model
        if self.agent_model is not None:
            save_gpt_model(self.agent_model, self.save_dir, 'Agent_final')

        # Save round data
        round_data_path = os.path.join(self.save_dir, 'round_data.json')
        with open(round_data_path, 'w') as f:
            json.dump(self.round_data, f, indent=2)

        self.writer.close()
        logger.info(f"TensorBoard logs saved to: {self.save_dir}")

        return self.all_generated_seqs, self.all_predicted_fitness, self.all_oracle_fitness


# ============================================================================
# Main Experiment Functions
# ============================================================================

def load_landscape_data_local(data_path: str) -> Tuple[List[str], np.ndarray]:
    """Load complete fitness landscape."""
    df = pd.read_csv(data_path)
    sequences = df['seq'].tolist()
    fitness = df['fitness'].values
    return sequences, fitness


def compute_all_metrics_local(
    generated_seqs: List[str],
    generated_fitness: List[float],
    all_sequences: List[str],
    all_fitness: np.ndarray,
    batch_size: int = 96,
    predicted_fitness: Optional[List[float]] = None,
    wildtype: Optional[str] = None,
) -> MetricsResult:
    """Compute all evaluation metrics (aligned with ALDE)."""
    result = MetricsResult()

    generated_fitness_np = np.array(generated_fitness)
    global_max = np.max(all_fitness)
    global_min = np.min(all_fitness)

    unique_seqs = list(set(generated_seqs))

    # Exploration Metrics
    result.high_fitness_proximity = high_fitness_proximity(
        unique_seqs, all_sequences, all_fitness, percentile=0.9
    )

    initial_seqs = generated_seqs[:batch_size] if len(generated_seqs) >= batch_size else generated_seqs
    later_seqs = generated_seqs[batch_size:] if len(generated_seqs) > batch_size else []
    if later_seqs:
        result.novelty = novelty(later_seqs, initial_seqs)

    result.batch_diversity = batch_diversity(unique_seqs[:256])

    # Functional Metrics
    result.normalized_fitness_median_top128 = normalized_fitness_topk(
        generated_fitness_np, k=128, min_fitness=global_min, max_fitness=global_max
    )
    result.normalized_fitness_median_top256 = normalized_fitness_topk(
        generated_fitness_np, k=256, min_fitness=global_min, max_fitness=global_max
    )
    result.max_fitness = max_fitness(generated_fitness_np)

    # Model Quality Metrics
    if predicted_fitness is not None:
        predicted_fitness_np = np.array(predicted_fitness)
        result.spearman_correlation = spearman_correlation(
            generated_fitness_np, predicted_fitness_np
        )

    # Success Metrics
    result.simple_regret = simple_regret(result.max_fitness, global_max)
    result.global_max_found = (result.max_fitness >= global_max * 0.99)

    # Trajectory
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
    dataset: str,
    output_path: str,
    data_dir: str,
    config_path: Optional[str] = None,
    compute_metrics: bool = True,
    run_id: Optional[int] = None,
    n_rounds: int = 15,
    n_steps_per_round: int = 500,
    batch_size: int = 96,
    sigma: float = 60,
    top_k_cutoff: int = 1000,
    n_clusters: int = 10,
    sampling_strategy: str = 'cluster',
    acquisition: str = 'ts',
    xi: float = 4.0,
    finetune_prior: bool = False,
    n_finetune_epochs: int = 10,
    finetune_lr: float = 1e-4,
    level: str = 'medium',
    device: str = 'cuda:0',
) -> Dict[str, Any]:
    """Run a single AlphaVariant iterative optimization experiment on any dataset."""

    if run_id is None:
        run_id = seed

    # Resolve landscape path
    landscape_path = os.path.join(data_dir, dataset, 'data.csv')
    if not os.path.exists(landscape_path):
        raise FileNotFoundError(
            f"Dataset file not found: {landscape_path}\n"
            f"Expected data/<dataset>/data.csv with 'seq' and 'fitness' columns."
        )

    # Auto-detect dataset properties
    dataset_info = auto_detect_from_data(landscape_path)
    seq_len = dataset_info['seq_len']
    wildtype = dataset_info['wildtype']

    print(f"\n{'='*60}")
    print(f"Starting AlphaVariant Iterative Optimization on {dataset} ({level})")
    print(f"  Seed: {seed}")
    print(f"  Seq length: {seq_len}")
    print(f"  Rounds: {n_rounds}")
    print(f"  Steps per round: {n_steps_per_round}")
    print(f"  Sampling strategy: {sampling_strategy}")
    print(f"  Config: {config_path if config_path else 'auto-generated'}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    # Set seed
    set_random_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create output directory
    run_dir = os.path.join(output_path, f'seed_{seed}')
    os.makedirs(run_dir, exist_ok=True)

    if config_path is not None:
        # Load from YAML config file (existing behavior)
        config = parse_config(config_path)
        os.system(f'cp {config_path} {os.path.join(run_dir, "config.yaml")}')

        # Initialize template from config
        logger.info("Initializing template from config file...")
        fasta_sequences, _ = read_fasta_as_list(config.template.ref_seq_path)
        ref_seq = fasta_sequences[0]
        positions, pos_aa_candidates = load_hotspot(config.template.hotspot_path)
        template = PDETemplate(ref_seq, positions=positions, pos_aa_candidates=pos_aa_candidates)

        model_config = config.model
        optim_config = config.optim
        effective_batch_size = config.train.batch_size
        effective_sigma = config.train.sigma
        effective_device = config.train.device

    else:
        # Auto-generate config from detected properties
        config_dict = generate_config_dict(
            seq_len=seq_len,
            wildtype=wildtype,
            data_path=landscape_path,
            dataset_name=dataset,
            batch_size=batch_size,
            sigma=sigma,
            device=device,
            n_steps=n_steps_per_round,
        )
        config = dict_to_easydict(config_dict)

        # Save generated config as YAML for reproducibility
        try:
            import yaml
            config_save_path = os.path.join(run_dir, 'config.yaml')
            with open(config_save_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            logger.info(f"Auto-generated config saved to: {config_save_path}")
        except ImportError:
            # Save as JSON if yaml not available
            config_save_path = os.path.join(run_dir, 'config.json')
            with open(config_save_path, 'w') as f:
                json.dump(config_dict, f, indent=2)

        # Create temporary template files in the run directory
        temp_dir = os.path.join(run_dir, 'generated_template')
        os.makedirs(temp_dir, exist_ok=True)
        fasta_path = create_temp_fasta(wildtype, temp_dir)
        hotspot_path = create_temp_hotspots(seq_len, temp_dir)

        logger.info("Initializing template from auto-generated files...")
        fasta_sequences, _ = read_fasta_as_list(fasta_path)
        ref_seq = fasta_sequences[0]
        positions, pos_aa_candidates = load_hotspot(hotspot_path)
        template = PDETemplate(ref_seq, positions=positions, pos_aa_candidates=pos_aa_candidates)

        model_config = config.model
        optim_config = config.optim
        effective_batch_size = batch_size
        effective_sigma = sigma
        effective_device = device

    logger.info(f"Template positions: {len(positions)} positions")
    logger.info(f"Reference sequence length: {len(ref_seq)}")

    # Initialize iterative trainer
    trainer = IterativeProteinTrainer(
        model_config=model_config,
        optim_config=optim_config,
        template=template,
        landscape_path=landscape_path,
        save_dir=run_dir,
        seq_len=seq_len,
        batch_size=effective_batch_size,
        n_rounds=n_rounds,
        n_steps_per_round=n_steps_per_round,
        sigma=effective_sigma,
        device=effective_device,
        seed=seed,
        top_k_cutoff=top_k_cutoff,
        n_clusters=n_clusters,
        sampling_strategy=sampling_strategy,
        acquisition=acquisition,
        xi=xi,
        finetune_prior=finetune_prior,
        n_finetune_epochs=n_finetune_epochs,
        finetune_lr=finetune_lr,
        level=level,
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
        'dataset': dataset,
        'seq_len': seq_len,
        'wildtype': wildtype,
        'runtime_seconds': runtime,
        'n_sequences': len(all_seqs),
        'n_unique_sequences': len(set(all_seqs)),
        'n_rounds': n_rounds,
        'n_steps_per_round': n_steps_per_round,
        'round_data': trainer.round_data,
        'config': {
            'batch_size': effective_batch_size,
            'n_rounds': n_rounds,
            'n_steps_per_round': n_steps_per_round,
            'sigma': effective_sigma,
        }
    }

    # Compute metrics
    if compute_metrics:
        logger.info("Computing evaluation metrics...")

        all_landscape_seqs, all_landscape_fitness = load_landscape_data_local(landscape_path)

        metrics_result = compute_all_metrics_local(
            generated_seqs=all_seqs,
            generated_fitness=all_oracle,
            all_sequences=all_landscape_seqs,
            all_fitness=all_landscape_fitness,
            batch_size=effective_batch_size,
            predicted_fitness=all_predicted,
            wildtype=wildtype,
        )

        result['metrics'] = metrics_result.to_dict()
        result['fitness_trajectory'] = metrics_result.fitness_trajectory
        result['regret_trajectory'] = metrics_result.regret_trajectory

        # Print summary
        print("\n" + "-"*60)
        print(f"Metrics Summary for {dataset} (ALDE-aligned, Oracle/Ground Truth):")
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
        print(f"  [Success]")
        print(f"    simple_regret:                 {metrics_result.simple_regret:.4f}")
        print(f"    global_max_found:              {metrics_result.global_max_found}")
        print("-"*60)

        # Save metrics
        metrics_path = os.path.join(run_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Metrics saved to: {metrics_path}")

    return result


def aggregate_run_metrics_local(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across multiple runs."""
    metrics_names = [
        'high_fitness_proximity',
        'novelty',
        'batch_diversity',
        'normalized_fitness_median_top128',
        'normalized_fitness_median_top256',
        'max_fitness',
        'spearman_correlation',
        'simple_regret',
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

    hit_count = sum(1 for r in results if 'metrics' in r and r['metrics'].get('global_max_found', False))
    aggregated['global_max_hit_count'] = {
        'count': hit_count,
        'rate': hit_count / len(results) if results else 0
    }

    return aggregated


def save_aggregated_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save aggregated results across all runs."""
    aggregated = aggregate_run_metrics_local(results)

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
    summary_path = os.path.join(output_path, 'aggregated_metrics.csv')
    summary_df.to_csv(summary_path, index=False)

    json_path = os.path.join(output_path, 'aggregated_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'aggregated_metrics': aggregated,
            'n_runs': len(results),
            'seeds': [r['seed'] for r in results],
            'dataset': results[0].get('dataset', 'unknown') if results else 'unknown',
            'config': results[0].get('config', {}) if results else {}
        }, f, indent=2, default=str)

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
        description="Run AlphaVariant iterative optimization on any protein dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on AAV medium dataset
  python run_generic.py --dataset AAV_med --seed 42

  # Run on GFP medium with hard-level init
  python run_generic.py --dataset GFP_med --level hard --seed 42

  # Use an existing YAML config instead of auto-generating
  python run_generic.py --dataset AAV_med --config examples/AAV_med/config/train_agent_config.yaml

  # Multiple runs for randomness evaluation
  python run_generic.py --dataset GB1 --seeds 42 123 456 789 1000

  # Load seeds from file
  python run_generic.py --dataset AAV_hard --seed_file ../rand_seeds.txt --num_seeds 5

  # Skip metrics computation
  python run_generic.py --dataset AAV_med --seed 42 --skip_metrics

  # Custom output path
  python run_generic.py --dataset my_protein --output_path results/my_experiment/
        """
    )

    # Dataset (required)
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name (must have data/<dataset>/data.csv with seq,fitness columns)")

    # Seed configuration
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=None, help="Single random seed")
    seed_group.add_argument("--seeds", type=int, nargs='+', help="Multiple seeds")
    seed_group.add_argument("--seed_file", type=str, help="Path to file containing seeds")

    parser.add_argument("--num_seeds", type=int, default=5, help="Number of seeds from file (default: 5)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML config file (optional; auto-generated if not provided)")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output directory (default: results/<dataset>_AlphaVariant/)")
    parser.add_argument("--data_dir", type=str, default="/home/xux/Desktop/AlphaVariant/Benchmark/data",
                       help="Base data directory")
    parser.add_argument("--skip_metrics", action="store_true", help="Skip metrics computation")

    # Training parameters
    parser.add_argument("--level", type=str, choices=['medium', 'hard'], default='medium',
                       help="Difficulty level for initial sampling (default: medium)")
    parser.add_argument("--n_rounds", type=int, default=15,
                       help="Number of iterative rounds (default: 15)")
    parser.add_argument("--n_steps_per_round", type=int, default=500,
                       help="GPT training steps per round (default: 500)")
    parser.add_argument("--batch_size", type=int, default=96,
                       help="Batch size / samples per round (default: 96)")
    parser.add_argument("--sigma", type=float, default=60,
                       help="REINFORCE sigma (default: 60)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device for training (default: cuda:0)")
    parser.add_argument("--top_k_cutoff", type=int, default=1000,
                       help="Top-k cutoff for CLADE-2 sampling (default: 1000)")
    parser.add_argument("--n_clusters", type=int, default=10,
                       help="Number of clusters for sampling (default: 10)")
    parser.add_argument("--sampling", type=str, choices=['cluster', 'active'], default='cluster',
                       help="Sampling strategy (default: cluster)")
    parser.add_argument("--acquisition", type=str, choices=['ts', 'ucb', 'ei'], default='ts',
                       help="Acquisition function for active sampling (default: ts)")
    parser.add_argument("--xi", type=float, default=4.0,
                       help="Exploration parameter for UCB (default: 4.0)")
    parser.add_argument("--finetune_prior", action="store_true", default=False,
                       help="Finetune prior on collected sequences before RL")
    parser.add_argument("--n_finetune_epochs", type=int, default=10,
                       help="Number of epochs for prior finetuning (default: 10)")
    parser.add_argument("--finetune_lr", type=float, default=1e-4,
                       help="Learning rate for prior finetuning (default: 1e-4)")

    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    # Set default output path based on dataset name
    if args.output_path is None:
        args.output_path = f"results/{args.dataset}_AlphaVariant/"

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

    print(f"\nRunning AlphaVariant Iterative Optimization on {args.dataset} ({args.level})")
    print(f"  Seeds: {seeds}")
    print(f"  Rounds: {args.n_rounds}")
    print(f"  Steps per round: {args.n_steps_per_round}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sampling strategy: {args.sampling}")
    print(f"  Config: {args.config if args.config else 'auto-generated'}")
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
            dataset=args.dataset,
            output_path=args.output_path,
            data_dir=args.data_dir,
            config_path=args.config,
            compute_metrics=not args.skip_metrics,
            run_id=i + 1,
            n_rounds=args.n_rounds,
            n_steps_per_round=args.n_steps_per_round,
            batch_size=args.batch_size,
            sigma=args.sigma,
            top_k_cutoff=args.top_k_cutoff,
            n_clusters=args.n_clusters,
            sampling_strategy=args.sampling,
            acquisition=args.acquisition,
            xi=args.xi,
            finetune_prior=args.finetune_prior,
            n_finetune_epochs=args.n_finetune_epochs,
            finetune_lr=args.finetune_lr,
            level=args.level,
            device=args.device,
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
    print(f"Dataset: {args.dataset}")
    print(f"Configuration:")
    print(f"  - Method: AlphaVariant Iterative (GPT + REINFORCE)")
    print(f"  - Rounds: {args.n_rounds}")
    print(f"  - Batch size: {args.batch_size} samples per round")
    print(f"  - Steps per round: {args.n_steps_per_round}")
    print(f"  - Sampling strategy: {args.sampling}")
    print(f"  - Level: {args.level}")
    print(f"Results saved to: {args.output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
