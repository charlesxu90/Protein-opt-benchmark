#!/usr/bin/env python
"""
run_GFP_hard.py - Execute delta-Conservative Search optimization on GFP_hard dataset

Configuration:
    - Model: GFlowNet with MLP
    - Proxy: Dropout-based MLP Regressor
    - Acquisition: UCB with delta-Conservative radius
    - Batch size: 96
    - Rounds: 5 (1 initial + 4 iterations)

Based on delta-Conservative Search method adapted for GFP hard protein variant landscape.
Reference: delta-Conservative Search (BioSeq-GFN-AL)

Metrics computed (ALDE-aligned order):
    1. high_fitness_proximity - Median distance to high-fitness sequences
    2. novelty - Mean minimum distance to reference set
    3. batch_diversity - Mean pairwise distance in batch
    4. normalized_fitness_median_top128 - Normalized median fitness of top-128
    5. normalized_fitness_median_top256 - Normalized median fitness of top-256
    6. max_fitness - Maximum fitness found
    7. spearman_correlation - Proxy-oracle correlation
    8. epistatic_correlation - Epistatic interaction effects
    9. recall_high_order - Recall of top-percentile sequences
    10. simple_regret - Gap to global maximum
    11. miscalibration_area - Uncertainty calibration error
    12. expected_calibration_error - ECE metric

Usage:
    # Single run with default seed
    python run_GFP_hard.py

    # Single run with specific seed
    python run_GFP_hard.py --seed 42

    # Multiple runs with different seeds
    python run_GFP_hard.py --seeds 42 123 456 789 1000

    # Use seeds from file
    python run_GFP_hard.py --seed_file ../rand_seeds.txt --num_seeds 50
"""

from __future__ import annotations
import argparse
import gzip
import pickle
import itertools
import json
import os
import warnings
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm
from scipy import stats

# Local imports
from lib.acquisition_fn import get_acq_fn
from lib.generator import get_generator
from lib.logging import get_logger
from lib.proxy import get_proxy_model
from lib.utils.distance import is_similar, edit_dist
from lib.model.mlp import MLP

import wandb

# Import unified metrics from utils.compat
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
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


# ============================================================================
# Argument Parser
# ============================================================================

parser = argparse.ArgumentParser(
    description="Run delta-Conservative Search optimization on GFP_hard",
    formatter_class=argparse.RawDescriptionHelpFormatter
)

# Paths and logging
parser.add_argument("--save_path", default='results/GFP_hard_delta_cs.pkl.gz')
parser.add_argument("--tb_log_dir", default='results/GFP_hard_delta_cs')
parser.add_argument("--name", default='GFP_hard_delta_cs')
parser.add_argument("--data_dir", default='/home/xux/Desktop/AlphaVariant/Benchmark/data',
                    help="Base directory for data files")

# Multi-round settings
parser.add_argument("--num_rounds", default=15, type=int, help="Number of optimization rounds")
parser.add_argument("--task", default="gfp_hard", type=str)
parser.add_argument("--num_queries_per_round", default=96, type=int, help="Batch size per round")
parser.add_argument("--vocab_size", default=20, type=int, help="Number of amino acids")
parser.add_argument("--max_len", default=237, type=int, help="Sequence length (GFP is 237 AAs)")
parser.add_argument("--gen_max_len", default=237, type=int)

# Seeds and randomness
parser.add_argument("--seed", default=None, type=int, help="Single random seed")
parser.add_argument("--seeds", type=int, nargs='+', help="Multiple seeds for evaluation")
parser.add_argument("--seed_file", type=str, default=None, help="Path to file containing seeds (one per line)")
parser.add_argument("--num_seeds", type=int, default=None, help="Number of seeds to use from seed_file")
parser.add_argument("--run", default=-1, type=int)

# Logging options
parser.add_argument("--enable_tensorboard", action="store_true")
parser.add_argument("--use_wandb", action="store_true")
parser.add_argument("--save_proxy_weights", action="store_true")
parser.add_argument("--skip_metrics", action="store_true", help="Skip metrics computation")

# Acquisition function
parser.add_argument("--acq_fn", default="ucb", type=str, help="Acquisition function: ucb, ei, none")
parser.add_argument("--kappa", default=0.1, type=float, help="UCB kappa parameter")

# delta-CS specific parameters
parser.add_argument("--radius_option", default='adaptive_linear', type=str,
                    choices=['linear', 'adaptive_linear', 'adaptive', 'constant', 'none'],
                    help="Radius scheduling option for delta-CS")
parser.add_argument("--min_radius", default=0.4, type=float, help="Minimum exploration radius")
parser.add_argument("--max_radius", default=0.8, type=float, help="Maximum exploration radius")
parser.add_argument("--K", default=25, type=int, help="Number of sampling iterations")
parser.add_argument("--sigma_coeff", default=1.5, type=float, help="Sigma coefficient for adaptive radius")
parser.add_argument("--rank_coeff", default=0.01, type=float, help="Rank coefficient for weighted sampling")

# Generator parameters
parser.add_argument("--gen_learning_rate", default=1e-4, type=float)
parser.add_argument("--gen_Z_learning_rate", default=1e-3, type=float)
parser.add_argument("--gen_clip", default=10, type=float)
parser.add_argument("--gen_num_iterations", default=3000, type=int)
parser.add_argument("--gen_episodes_per_step", default=16, type=int)
parser.add_argument("--gen_num_hidden", default=256, type=int)
parser.add_argument("--gen_num_layers", default=2, type=int)
parser.add_argument("--gen_reward_norm", default=1, type=float)
parser.add_argument("--gen_reward_exp", default=3, type=float)
parser.add_argument("--gen_reward_min", default=0, type=float)
parser.add_argument("--gen_L2", default=0, type=float)
parser.add_argument("--gen_partition_init", default=50, type=float)
parser.add_argument("--gen_reward_exp_ramping", default=3, type=float)
parser.add_argument("--gen_balanced_loss", default=1, type=float)
parser.add_argument("--gen_output_coef", default=10, type=float)
parser.add_argument("--gen_loss_eps", default=1e-5, type=float)
parser.add_argument("--gen_random_action_prob", default=0.001, type=float)
parser.add_argument("--gen_sampling_temperature", default=4., type=float)
parser.add_argument("--gen_leaf_coef", default=25, type=float)
parser.add_argument("--gen_data_sample_per_step", default=16, type=int)
parser.add_argument("--gen_do_pg", default=0, type=int)
parser.add_argument("--gen_pg_entropy_coef", default=0.1, type=float)
parser.add_argument("--gen_do_explicit_Z", default=1, type=int)
parser.add_argument("--gen_model_type", default="mlp")

# Proxy parameters
parser.add_argument("--proxy_uncertainty", default="dropout", choices=["dropout", "ensemble"])
parser.add_argument("--proxy_learning_rate", default=1e-4, type=float)
parser.add_argument("--proxy_type", default="regression")
parser.add_argument("--proxy_arch", default="mlp")
parser.add_argument("--proxy_num_layers", default=2, type=int)
parser.add_argument("--proxy_dropout", default=0.1, type=float)
parser.add_argument("--proxy_num_hid", default=256, type=int)
parser.add_argument("--proxy_L2", default=1e-4, type=float)
parser.add_argument("--proxy_num_per_minibatch", default=256, type=int)
parser.add_argument("--proxy_early_stop_tol", default=5, type=int)
parser.add_argument("--proxy_early_stop_to_best_params", default=0, type=int)
parser.add_argument("--proxy_num_iterations", default=3000, type=int)
parser.add_argument("--proxy_num_dropout_samples", default=25, type=int)


# ============================================================================
# GFP_hard Oracle and Dataset
# ============================================================================

# GFP wild-type sequence
GFP_WILDTYPE = "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"


class GFPOracle:
    """Oracle wrapper for GFP_hard fitness landscape."""

    def __init__(self, data_dir: str = '/home/xux/Desktop/AlphaVariant/Benchmark/data'):
        self.data_dir = data_dir
        self.alphabet = list('ACDEFGHIKLMNPQRSTVWY')
        self.stoi = {aa: idx for idx, aa in enumerate(self.alphabet)}
        self.itos = {idx: aa for idx, aa in enumerate(self.alphabet)}

        # Load GFP_hard landscape
        self._load_landscape()

    def _load_landscape(self):
        """Load the complete GFP_hard fitness landscape."""
        data_path = os.path.join(self.data_dir, 'GFP_hard', 'data.csv')
        df = pd.read_csv(data_path)

        # Store complete landscape - use 'seq' column for full sequences
        self.sequences = df['seq'].tolist()
        self.fitness = df['fitness'].values

        # Create lookup dictionary for fast access
        self.seq_to_fitness = {seq: fit for seq, fit in zip(self.sequences, self.fitness)}
        self.seq_to_idx = {seq: idx for idx, seq in enumerate(self.sequences)}

        # Normalization factor
        self.max_fitness = np.max(self.fitness)
        self.min_fitness = np.min(self.fitness)

        print(f"Loaded GFP_hard landscape: {len(self.sequences)} sequences")
        print(f"  Fitness range: [{self.min_fitness:.4f}, {self.max_fitness:.4f}]")
        print(f"  Sequence length: {len(self.sequences[0])}")

    def __call__(self, x, return_all=False):
        """Evaluate fitness for sequences."""
        if isinstance(x[0], (list, np.ndarray)):
            # Input is list of indices
            seqs = [''.join([self.itos[i] for i in seq]) for seq in x]
        else:
            # Input is list of strings
            seqs = x

        scores = np.array([self.seq_to_fitness.get(seq, 0.0) for seq in seqs])

        if return_all:
            # Return scores, mu, sigma (for compatibility with UCB)
            return torch.tensor(scores, dtype=torch.float32), \
                   torch.tensor(scores, dtype=torch.float32), \
                   torch.zeros(len(scores))

        return torch.tensor(scores, dtype=torch.float32)

    def get_all_data(self):
        """Return all sequences and fitness values."""
        return self.sequences, self.fitness


class GFPDataset:
    """Dataset class for GFP_hard optimization."""

    def __init__(self, args, oracle: GFPOracle, init_size: int = 96):
        self.args = args
        self.oracle = oracle
        self.rng = np.random.RandomState(args.seed if hasattr(args, 'seed') and args.seed else 42)

        # Get all data
        all_seqs, all_fitness = oracle.get_all_data()
        self.x_all = np.array([[oracle.stoi[c] for c in seq] for seq in all_seqs])
        self.y_all = all_fitness

        # Initialize with random samples from medium fitness range
        self._initialize_data(init_size)

        self.train_added = len(self.train)
        self.val_added = len(self.valid)

    def _initialize_data(self, init_size: int):
        """Initialize training/validation split with samples from medium fitness range."""
        n_total = len(self.x_all)

        # Select from medium fitness range (40th-60th percentile)
        threshold_low = np.percentile(self.y_all, 40)
        threshold_high = np.percentile(self.y_all, 60)
        medium_mask = (self.y_all >= threshold_low) & (self.y_all <= threshold_high)
        medium_indices = np.where(medium_mask)[0]

        # Sample from medium fitness range
        if len(medium_indices) >= init_size:
            indices = self.rng.choice(medium_indices, size=init_size, replace=False)
        else:
            # Fall back to random if not enough medium fitness samples
            indices = self.rng.permutation(n_total)[:init_size]

        x_init = self.x_all[indices]
        y_init = self.y_all[indices]

        # 90/10 train/val split
        n_train = int(0.9 * len(x_init))

        self.train = list(x_init[:n_train])
        self.valid = list(x_init[n_train:])
        self.train_scores = y_init[:n_train]
        self.valid_scores = y_init[n_train:]

    def sample(self, n: int):
        """Random sample from training data."""
        indices = self.rng.randint(0, len(self.train), n)
        return ([self.train[i] for i in indices],
                [self.train_scores[i] for i in indices])

    def weighted_sample(self, n: int, rank_coefficient: float = 0.01):
        """Weighted sampling favoring high-fitness sequences."""
        ranks = np.argsort(np.argsort(-1 * self.train_scores))
        weights = 1.0 / (rank_coefficient * len(self.train_scores) + ranks)

        indices = list(torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=n, replacement=True
        ))

        return ([self.train[i] for i in indices],
                [self.train_scores[i] for i in indices])

    def validation_set(self):
        """Return validation data."""
        return self.valid, self.valid_scores

    def add(self, batch):
        """Add new samples to the dataset."""
        samples, scores = batch
        for x, score in zip(samples, scores):
            if self.rng.uniform() < 0.1:
                self.valid.append(x)
                self.valid_scores = np.concatenate([self.valid_scores, [score]])
            else:
                self.train.append(x)
                self.train_scores = np.concatenate([self.train_scores, [score]])

    def _tostr(self, seqs):
        """Convert index sequences to string."""
        return [''.join([self.oracle.itos[i] for i in seq]) for seq in seqs]

    def _top_k(self, data, k: int):
        """Get top-k sequences by fitness."""
        indices = np.argsort(data[1])[::-1][:k]
        topk_scores = data[1][indices]
        topk_seqs = np.array(data[0])[indices]
        return self._tostr(topk_seqs), topk_scores

    def top_k(self, k: int):
        """Get top-k from all collected data."""
        all_seqs = np.array(self.train + self.valid)
        all_scores = np.concatenate([self.train_scores, self.valid_scores])
        return self._top_k((all_seqs, all_scores), k)

    def top_k_collected(self, k: int):
        """Get top-k from newly collected (non-initial) data."""
        new_seqs = np.array(self.train[self.train_added:] + self.valid[self.val_added:])
        new_scores = np.concatenate([self.train_scores[self.train_added:],
                                     self.valid_scores[self.val_added:]])
        if len(new_seqs) == 0:
            return [], np.array([])
        return self._top_k((new_seqs, new_scores), k)

    def get_ref_data(self):
        """Get reference (initial) data."""
        seqs = np.array(self.train[:self.train_added] + self.valid[:self.val_added])
        scores = np.concatenate([self.train_scores[:self.train_added],
                                 self.valid_scores[:self.val_added]])
        return self._tostr(seqs), scores

    def get_all_data(self, return_as_str=True):
        """Get all collected data."""
        seqs = np.array(self.train + self.valid)
        scores = np.concatenate([self.train_scores, self.valid_scores])
        if return_as_str:
            return self._tostr(seqs), scores
        return seqs, scores

    def get_holdout_data(self, return_as_str=True):
        """Get data not in training/validation sets."""
        collected_seqs = set(self._tostr(self.train + self.valid))
        holdout_x, holdout_y = [], []

        for i, seq in enumerate(self.oracle.sequences):
            if seq not in collected_seqs:
                holdout_x.append(list(self.x_all[i]))
                holdout_y.append(self.y_all[i])

        if return_as_str:
            return self._tostr(holdout_x), holdout_y
        return holdout_x, holdout_y


# ============================================================================
# GFP Tokenizer
# ============================================================================

class GFPVocab:
    """Vocabulary for GFP amino acids."""
    def __init__(self):
        self.alphabet = list('ACDEFGHIKLMNPQRSTVWY')
        self.stoi = {aa: idx for idx, aa in enumerate(self.alphabet)}
        self.itos = {idx: aa for idx, aa in enumerate(self.alphabet)}


class GFPTokenizer:
    """Tokenizer for GFP sequences."""
    def __init__(self, max_len=237):
        self.vocab = GFPVocab()
        self.eos_token = None  # No EOS for fixed-length sequences
        self.max_len = max_len

    def process(self, x):
        """Convert sequences to tensor indices."""
        if isinstance(x[0], str):
            # String sequences
            ret = [[self.vocab.stoi[c] for c in seq] for seq in x]
        else:
            # Already indices
            ret = [list(seq) for seq in x]

        # Pad to max length if needed
        max_len = max(len(s) for s in ret)
        for i in range(len(ret)):
            if len(ret[i]) < max_len:
                ret[i] = ret[i] + [20] * (max_len - len(ret[i]))  # Pad token = 20

        return torch.tensor(ret, dtype=torch.long)

    @property
    def stoi(self):
        return self.vocab.stoi

    @property
    def itos(self):
        return self.vocab.itos


def get_gfp_tokenizer(max_len=237):
    """Create tokenizer for GFP."""
    return GFPTokenizer(max_len=max_len)


# ============================================================================
# Radius Computation for delta-CS
# ============================================================================

def get_current_radius(iter: int, round: int, args, rs=None, y=None, sigma=None):
    """Compute current exploration radius based on option."""
    if args.radius_option == 'linear':
        return (args.max_radius - args.min_radius) * ((round + 1) / args.num_rounds) + args.min_radius

    elif args.radius_option == 'adaptive_linear':
        linear_r = (args.max_radius - args.min_radius) * ((round + 1) / args.num_rounds) + args.min_radius
        linear_r = linear_r * torch.ones(rs.size(0)).to(rs.device)
        return (linear_r - args.sigma_coeff * sigma.view(-1)).clamp(0.1, 1)

    elif args.radius_option == 'adaptive':
        base_r = args.max_radius * torch.ones(rs.size(0)).to(rs.device)
        return (base_r - args.sigma_coeff * sigma.view(-1)).clamp(0.1, 1)

    elif args.radius_option == 'constant':
        return args.max_radius * torch.ones(rs.size(0)).to(rs.device)

    else:
        return 1.0


# ============================================================================
# Rollout Worker for delta-CS
# ============================================================================

class MbStack:
    """Mini-batch stack for efficient oracle evaluation."""
    def __init__(self, f):
        self.stack = []
        self.f = f

    def push(self, x, i):
        self.stack.append((x, i))

    def pop_all(self):
        if not len(self.stack):
            return []
        with torch.no_grad():
            ys = self.f([i[0] for i in self.stack])
        idxs = [i[1] for i in self.stack]
        self.stack = []
        return zip(ys, idxs)


class RolloutWorker:
    """Worker for generating sequences with delta-CS."""

    def __init__(self, args, oracle, tokenizer):
        self.oracle = oracle
        self.max_len = args.max_len
        self.episodes_per_step = args.gen_episodes_per_step
        self.random_action_prob = args.gen_random_action_prob
        self.reward_exp = args.gen_reward_exp
        self.sampling_temperature = args.gen_sampling_temperature
        self.out_coef = args.gen_output_coef

        self.balanced_loss = args.gen_balanced_loss == 1
        self.reward_norm = args.gen_reward_norm
        self.reward_min = torch.tensor(float(args.gen_reward_min))
        self.loss_eps = torch.tensor(float(args.gen_loss_eps)).to(args.device)
        self.leaf_coef = args.gen_leaf_coef
        self.exp_ramping_factor = args.gen_reward_exp_ramping

        self.tokenizer = tokenizer
        if self.exp_ramping_factor > 0:
            self.l2r = lambda x, t=0: (x) ** (1 + (self.reward_exp - 1) * (1 - 1/(1 + t / self.exp_ramping_factor)))
        else:
            self.l2r = lambda x, t=0: (x) ** self.reward_exp

        self.device = args.device
        self.args = args
        self.workers = MbStack(oracle)

    def rollout(self, model, episodes, use_rand_policy=True):
        """Standard rollout without delta-CS guidance."""
        visited = []
        lists = lambda n: [list() for i in range(n)]
        states = [[] for i in range(episodes)]
        traj_states = torch.tensor([[[]] for i in range(episodes)]).to(model.device)
        traj_actions = torch.tensor(lists(episodes)).to(model.device)
        traj_rewards = torch.tensor(lists(episodes)).to(model.device)
        traj_dones = torch.tensor(lists(episodes)).to(model.device)

        for t in range(self.max_len) if episodes > 0 else []:
            x = self.tokenizer.process(states).to(self.device)
            with torch.no_grad():
                logits = model(x, None, coef=self.out_coef)

            if t == 0:
                logits[:, 0] = -1000  # Prevent early stopping

            cat = Categorical(logits=logits / self.sampling_temperature)
            actions = cat.sample()

            if use_rand_policy and self.random_action_prob > 0:
                for i in range(actions.shape[0]):
                    if np.random.uniform(0, 1) < self.random_action_prob:
                        actions[i] = torch.tensor(np.random.randint(0, logits.shape[1])).to(self.device)

            states_tensor = torch.tensor(states).to(actions.device)
            next_states = torch.cat([states_tensor, actions.view(-1, 1)], dim=1)

            if t == self.max_len - 1:
                for ns, i in zip(next_states, torch.arange(next_states.size(0))):
                    self.workers.push(ns.tolist(), i.item())
                rewards = torch.zeros(actions.size(0), 1).to(actions.device)
                dones = torch.ones(actions.size(0), 1).to(actions.device)
            else:
                rewards = torch.zeros(actions.size(0), 1).to(actions.device)
                dones = torch.zeros(actions.size(0), 1).to(actions.device)

            traj_states = torch.cat([traj_states, next_states.unsqueeze(1)], dim=-1)
            traj_actions = torch.cat([traj_actions, actions.unsqueeze(1)], dim=-1)
            traj_rewards = torch.cat([traj_rewards, rewards], dim=-1)
            traj_dones = torch.cat([traj_dones, dones], dim=-1)
            states = next_states.tolist()

        return visited, states, traj_states.tolist(), traj_actions.tolist(), traj_rewards.tolist(), traj_dones.tolist()

    def guided_rollout(self, model, guide_seqs, sample_action_prob):
        """Rollout with delta-CS guidance from existing sequences."""
        visited = []
        lists = lambda n: [list() for i in range(n)]
        episodes = len(guide_seqs)
        states = [[] for i in range(episodes)]
        traj_states = [[[]] for i in range(episodes)]
        traj_actions = lists(episodes)
        traj_rewards = lists(episodes)
        traj_dones = lists(episodes)

        masked_cnt = torch.zeros(episodes).to(self.device)

        for t in range(self.max_len) if episodes > 0 else []:
            x = self.tokenizer.process(states).to(self.device)
            with torch.no_grad():
                logits = model(x, None, coef=self.out_coef)

            if t == 0:
                logits[:, 0] = -1000

            cat = Categorical(logits=logits / self.sampling_temperature)
            actions = cat.sample()

            # Apply delta-CS guidance
            guide_actions = torch.tensor(np.array(guide_seqs))[:, t].to(self.device)
            mask = torch.rand(actions.size(0)).to(self.device) > sample_action_prob.view(-1)
            masked_cnt += mask.int()
            actions[mask] = guide_actions.long()[mask]

            for i, a in enumerate(actions):
                if t == self.max_len - 1:
                    self.workers.push(states[i] + [a.item()], i)
                    r = 0
                    d = 1
                else:
                    r = 0
                    d = 0
                traj_states[i].append(states[i] + [a.item()])
                traj_actions[i].append(a)
                traj_rewards[i].append(r)
                traj_dones[i].append(d)
                states[i] = states[i] + [a.item()]

        return visited, states, traj_states, traj_actions, traj_rewards, traj_dones, self.max_len - masked_cnt

    def execute_train_episode_batch(self, model, it=0, dataset=None, use_rand_policy=True, return_all_visited=False):
        """Execute training batch without delta-CS."""
        lists = lambda n: [list() for i in range(n)]
        visited, states, traj_states, traj_actions, traj_rewards, traj_dones = \
            self.rollout(model, self.episodes_per_step, use_rand_policy=use_rand_policy)

        bulk_trajs = []
        rq = []
        for (r, mbidx) in self.workers.pop_all():
            traj_rewards[mbidx][-1] = self.l2r(r, it)
            rq.append(r.item())
            s = states[mbidx]
            visited.append((s, traj_rewards[mbidx][-1].item(), r.item()))
            bulk_trajs.append((s, traj_rewards[mbidx][-1].item()))

        # Sample from dataset
        if self.args.gen_data_sample_per_step > 0 and dataset is not None:
            n = self.args.gen_data_sample_per_step
            m = len(traj_states)
            x, y = dataset.sample(n)
            n = len(x)
            traj_states += lists(n)
            traj_actions += lists(n)
            traj_rewards += lists(n)
            traj_dones += lists(n)
            bulk_trajs += list(zip([list(i) for i in x],
                                   [self.l2r(torch.tensor(i), it) for i in y]))

            if return_all_visited:
                with torch.no_grad():
                    rs = self.oracle(x).view(-1).tolist()
                visited += list(zip([list(i) for i in x],
                                    [self.l2r(torch.tensor(i), it) for i in y],
                                    [self.l2r(torch.tensor(i), it) for i in rs],
                                    y, rs))

            for i in range(len(x)):
                traj_states[i+m].append([])
                seq = list(x[i])
                for j, a in enumerate(seq):
                    traj_states[i+m].append(traj_states[i+m][-1] + [a])
                    traj_actions[i+m].append(a)
                    traj_rewards[i+m].append(0 if j != self.max_len - 1 else self.l2r(torch.tensor(y[i]), it))
                    traj_dones[i+m].append(float(j == self.max_len - 1))

        return {
            "visited": visited,
            "trajectories": {
                "traj_states": traj_states,
                "traj_actions": traj_actions,
                "traj_rewards": traj_rewards,
                "traj_dones": traj_dones,
                "states": states,
                "bulk_trajs": bulk_trajs
            }
        }

    def execute_train_episode_batch_with_delta(self, model, it=0, dataset=None,
                                                return_all_visited=False, round=0,
                                                use_offline_data=True):
        """Execute training batch with delta-CS guidance."""
        lists = lambda n: [list() for i in range(n)]
        n = self.episodes_per_step
        x, y = dataset.weighted_sample(n, self.args.rank_coeff)

        if self.args.acq_fn.lower() == "ucb":
            with torch.no_grad():
                rs, mu, sigma = self.oracle(x, return_all=True)
        else:
            with torch.no_grad():
                rs = self.oracle(x)
                sigma = torch.zeros(n).to(self.device)

        radius = get_current_radius(iter=it, round=round, args=self.args, rs=rs, y=y, sigma=sigma)

        if isinstance(radius, float):
            radius = torch.tensor([radius] * n).to(self.device)

        visited, states, traj_states, traj_actions, traj_rewards, traj_dones, noised_cnt = \
            self.guided_rollout(model, x, sample_action_prob=radius)

        bulk_trajs = []
        rq = []
        for (r, mbidx) in self.workers.pop_all():
            traj_rewards[mbidx][-1] = self.l2r(r, it)
            rq.append(r.item())
            s = states[mbidx]
            visited.append((s, traj_rewards[mbidx][-1].item(), r.item()))
            bulk_trajs.append((s, traj_rewards[mbidx][-1].item()))

        # Sample from dataset for offline learning
        if self.args.gen_data_sample_per_step > 0 and use_offline_data:
            n = self.args.gen_data_sample_per_step
            m = len(traj_states)
            x, y = dataset.sample(n)
            n = len(x)
            traj_states += lists(n)
            traj_actions += lists(n)
            traj_rewards += lists(n)
            traj_dones += lists(n)
            bulk_trajs += list(zip([list(i) for i in x],
                                   [self.l2r(torch.tensor(i), it) for i in y]))

            if return_all_visited:
                with torch.no_grad():
                    rs = self.oracle(x).view(-1).tolist()
                visited += list(zip([list(i) for i in x],
                                    [self.l2r(torch.tensor(i), it) for i in y],
                                    [self.l2r(torch.tensor(i), it) for i in rs],
                                    y, rs))

            for i in range(len(x)):
                traj_states[i+m].append([])
                seq = list(x[i])
                for j, a in enumerate(seq):
                    traj_states[i+m].append(traj_states[i+m][-1] + [a])
                    traj_actions[i+m].append(a)
                    traj_rewards[i+m].append(0 if j != self.max_len - 1 else self.l2r(torch.tensor(y[i]), it))
                    traj_dones[i+m].append(float(j == self.max_len - 1))

        return {
            "visited": visited,
            "trajectories": {
                "traj_states": traj_states,
                "traj_actions": traj_actions,
                "traj_rewards": traj_rewards,
                "traj_dones": traj_dones,
                "states": states,
                "bulk_trajs": bulk_trajs
            },
            "sigma": sigma.mean().item() if self.args.acq_fn.lower() == "ucb" else 0,
            "radius": radius.mean().item(),
            "radius_max": radius.max().item(),
            "radius_min": radius.min().item(),
            "noised_cnt": noised_cnt.mean().item(),
            "noised_cnt_max": noised_cnt.max().item(),
        }


# ============================================================================
# Training and Sampling Functions
# ============================================================================

def train_generator(args, generator, oracle, tokenizer, dataset, round=0):
    """Train the GFlowNet generator."""
    print("Training generator...")
    visited = []
    rollout_worker = RolloutWorker(args, oracle, tokenizer)
    losses = []

    p_bar = tqdm(range(args.gen_num_iterations + 1))
    for it in p_bar:
        if args.radius_option == 'none':
            rollout_artifacts = rollout_worker.execute_train_episode_batch(
                generator, it, dataset, return_all_visited=True)
        else:
            rollout_artifacts = rollout_worker.execute_train_episode_batch_with_delta(
                generator, it, dataset, return_all_visited=True, round=round)
            p_bar.set_postfix({
                "sigma": f"{rollout_artifacts['sigma']:.3f}",
                "radius": f"{rollout_artifacts['radius']:.3f}"
            })

        visited.extend(rollout_artifacts["visited"])

        loss, loss_info = generator.train_step(rollout_artifacts["trajectories"])
        losses.append(loss.item())

        args.logger.add_scalar("generator_total_loss", loss.item())
        wandb_log = {"generator_total_loss": loss.item()}

        if args.radius_option != 'none':
            wandb_log["sigma"] = rollout_artifacts["sigma"]
            wandb_log["radius"] = rollout_artifacts["radius"]

        for key, val in loss_info.items():
            args.logger.add_scalar(f"generator_{key}", val.item())
            wandb_log[f"generator_{key}"] = val.item()

        if it % 100 == 0:
            rs = torch.tensor([i[-1] for i in rollout_artifacts["trajectories"]["traj_rewards"]]).mean()
            args.logger.add_scalar("gen_reward", rs.item())
            wandb_log["gen_reward"] = rs.item()

        if args.use_wandb:
            wandb.log(wandb_log)

    rollout_worker.sampling_temperature = args.gen_sampling_temperature
    return rollout_worker, losses


def sample_batch(args, rollout_worker, generator, oracle, round=0, dataset=None):
    """Sample a batch of sequences from the trained generator."""
    print("Generating samples...")
    samples = ([], [])
    scores = []
    i = 0
    num_iter = args.K

    while len(samples[0]) < args.num_queries_per_round * num_iter:
        if args.radius_option == 'none':
            rollout_artifacts = rollout_worker.execute_train_episode_batch(
                generator, it=0, use_rand_policy=False)
        else:
            rollout_artifacts = rollout_worker.execute_train_episode_batch_with_delta(
                generator, it=args.gen_num_iterations + 1, dataset=dataset,
                round=round, use_offline_data=False)

        states = rollout_artifacts["trajectories"]["states"]
        vals = oracle(states).reshape(-1)
        samples[0].extend(states)
        samples[1].extend(vals.tolist())
        scores.extend(torch.tensor(rollout_artifacts["trajectories"]["traj_rewards"])[:, -1].numpy().tolist())
        i += 1

    idx_pick = np.argsort(scores)[::-1][:args.num_queries_per_round]
    return (np.array(samples[0])[idx_pick].tolist(),
            np.array(samples[1])[idx_pick]), np.array(scores)[idx_pick]


def construct_proxy(args, tokenizer, dataset=None):
    """Construct the proxy model with acquisition function."""
    proxy = get_proxy_model(args, tokenizer)
    sigmoid = nn.Sigmoid()

    if args.proxy_type == "classification":
        l2r = lambda x: sigmoid(x.clamp(min=args.gen_reward_min)) / args.gen_reward_norm
    elif args.proxy_type == "regression":
        l2r = lambda x: x.clamp(min=args.gen_reward_min) / args.gen_reward_norm

    args.reward_exp_min = max(l2r(torch.tensor(args.gen_reward_min)), 1e-32)
    acq_fn = get_acq_fn(args)
    return acq_fn(args, proxy, l2r, dataset)


# ============================================================================
# Metrics Computation (using unified utils.compat)
# ============================================================================

def mean_pairwise_distances(seqs: List[str]) -> float:
    """Compute mean pairwise Hamming distance."""
    if len(seqs) < 2:
        return 0.0
    dists = []
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            dists.append(hamming_distance(seqs[i], seqs[j]))
    return np.mean(dists) if dists else 0.0


def mean_novelty(seqs: List[str], ref_seqs: List[str]) -> float:
    """Compute mean novelty (minimum distance to reference set)."""
    if not seqs or not ref_seqs:
        return 0.0
    nov = [min(hamming_distance(seq, ref) for ref in ref_seqs) for seq in seqs]
    return np.mean(nov)


def epistatic_correlation_local(seqs: List[str], fitness: np.ndarray, alphabet: str = 'ACDEFGHIKLMNPQRSTVWY') -> float:
    """Compute epistatic correlation (interaction effects between positions)."""
    if len(seqs) < 10:
        return 0.0

    seq_len = len(seqs[0])
    # Build one-hot encoding
    aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}
    X = np.zeros((len(seqs), seq_len * len(alphabet)))
    for i, seq in enumerate(seqs):
        for j, aa in enumerate(seq):
            if aa in aa_to_idx:
                X[i, j * len(alphabet) + aa_to_idx[aa]] = 1

    # Simple linear model residuals as proxy for epistasis
    try:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, fitness)
        predicted = model.predict(X)
        residuals = fitness - predicted
        # Epistatic correlation: correlation between residuals and fitness variance
        corr, _ = stats.spearmanr(np.abs(residuals), fitness)
        return float(corr) if not np.isnan(corr) else 0.0
    except:
        return 0.0


def recall_high_order(collected_seqs: List[str], all_seqs: List[str],
                      all_fitness: np.ndarray, percentile: float = 0.99) -> float:
    """Compute recall of high-order (top percentile) sequences."""
    threshold = np.percentile(all_fitness, percentile * 100)
    high_order_seqs = set(seq for seq, fit in zip(all_seqs, all_fitness) if fit >= threshold)

    if not high_order_seqs:
        return 0.0

    collected_set = set(collected_seqs)
    found = len(high_order_seqs & collected_set)
    return float(found) / len(high_order_seqs)


def miscalibration_area_local(true_scores: np.ndarray, pred_scores: np.ndarray,
                        pred_std: np.ndarray = None) -> float:
    """Compute miscalibration area for uncertainty estimates."""
    if pred_std is None or len(true_scores) < 2:
        return 0.0

    # Compute z-scores
    z_scores = np.abs(true_scores - pred_scores) / (pred_std + 1e-8)

    # Expected vs observed coverage
    expected_coverages = np.linspace(0.1, 0.9, 9)
    miscal = 0.0
    for exp_cov in expected_coverages:
        z_threshold = stats.norm.ppf((1 + exp_cov) / 2)
        obs_cov = np.mean(z_scores <= z_threshold)
        miscal += np.abs(obs_cov - exp_cov)

    return float(miscal / len(expected_coverages))


def expected_calibration_error_local(true_scores: np.ndarray, pred_scores: np.ndarray,
                                n_bins: int = 10) -> float:
    """Compute expected calibration error."""
    if len(true_scores) < n_bins:
        return 0.0

    # Bin by predicted scores
    bin_edges = np.linspace(pred_scores.min(), pred_scores.max(), n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (pred_scores >= bin_edges[i]) & (pred_scores < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (pred_scores >= bin_edges[i]) & (pred_scores <= bin_edges[i + 1])

        if np.sum(mask) > 0:
            bin_true_mean = np.mean(true_scores[mask])
            bin_pred_mean = np.mean(pred_scores[mask])
            bin_weight = np.sum(mask) / len(true_scores)
            ece += bin_weight * np.abs(bin_true_mean - bin_pred_mean)

    return float(ece)


def compute_metrics(dataset: GFPDataset, oracle: GFPOracle, new_batch, round: int,
                    proxy_model=None) -> Dict[str, Any]:
    """Compute comprehensive metrics for a round (ALDE-aligned order)."""
    all_seqs, all_fitness = oracle.get_all_data()
    global_max = np.max(all_fitness)

    # Normalize fitness for metrics
    fitness_min, fitness_max = np.min(all_fitness), np.max(all_fitness)
    normalize = lambda x: (x - fitness_min) / (fitness_max - fitness_min + 1e-8)

    # Get collected data
    top128 = dataset.top_k(128)
    top256 = dataset.top_k(256)
    top128_collected = dataset.top_k_collected(128)
    ref_seqs, _ = dataset.get_ref_data()
    collected_seqs, collected_scores = dataset.get_all_data()

    new_seqs, new_scores, proxy_scores = new_batch

    # Initialize metrics dict in ALDE order
    metrics = {'round': round}

    # 1. high_fitness_proximity
    metrics['high_fitness_proximity'] = high_fitness_proximity(
        collected_seqs, all_seqs, all_fitness)

    # 2. novelty
    if len(top128_collected[0]) > 0:
        metrics['novelty'] = mean_novelty(top128_collected[0], ref_seqs)
    else:
        metrics['novelty'] = mean_novelty(top128[0], ref_seqs)

    # 3. batch_diversity
    metrics['batch_diversity'] = mean_pairwise_distances(new_seqs)

    # 4. normalized_fitness_median_top128
    metrics['normalized_fitness_median_top128'] = float(np.median(normalize(top128[1])))

    # 5. normalized_fitness_median_top256
    if len(top256[1]) > 0:
        metrics['normalized_fitness_median_top256'] = float(np.median(normalize(top256[1])))
    else:
        metrics['normalized_fitness_median_top256'] = 0.0

    # 6. max_fitness
    metrics['max_fitness'] = float(np.max(top128[1]))

    # 7. spearman_correlation
    metrics['spearman_correlation'] = spearman_correlation(
        np.array(new_scores), np.array(proxy_scores))

    # 8. epistatic_correlation
    metrics['epistatic_correlation'] = epistatic_correlation_local(
        collected_seqs, np.array(collected_scores))

    # 9. recall_high_order
    metrics['recall_high_order'] = recall_high_order(
        collected_seqs, all_seqs, all_fitness)

    # 10. simple_regret
    metrics['simple_regret'] = float(global_max - np.max(top128[1]))

    # 11. miscalibration_area (requires uncertainty estimates)
    if proxy_model is not None and hasattr(proxy_model, 'get_uncertainty'):
        try:
            pred_scores, pred_std = proxy_model.get_uncertainty(new_seqs)
            metrics['miscalibration_area'] = miscalibration_area_local(
                np.array(new_scores), pred_scores, pred_std)
        except:
            metrics['miscalibration_area'] = 0.0
    else:
        metrics['miscalibration_area'] = 0.0

    # 12. expected_calibration_error
    metrics['expected_calibration_error'] = expected_calibration_error_local(
        np.array(new_scores), np.array(proxy_scores))

    # 13. global_max_found (for aggregating global_max_hit_count across runs)
    metrics['global_max_found'] = (metrics['max_fitness'] >= global_max * 0.99)

    return metrics


def log_overall_metrics(args, dataset, round, new_batch, oracle=None, rst=None):
    """Log metrics for a round (ALDE-aligned naming)."""
    new_seqs, new_scores, proxy_scores = new_batch

    # Get fitness normalization bounds
    if oracle is not None:
        all_seqs, all_fitness = oracle.get_all_data()
        fitness_min, fitness_max = np.min(all_fitness), np.max(all_fitness)
        global_max = fitness_max
    else:
        fitness_min, fitness_max = 0, 1
        global_max = 1
    normalize = lambda x: (x - fitness_min) / (fitness_max - fitness_min + 1e-8)

    top128 = dataset.top_k(128)
    top256 = dataset.top_k(256)
    args.logger.add_scalar("max_fitness", np.max(top128[1]), use_context=False)
    batch_div = mean_pairwise_distances(new_seqs)
    args.logger.add_scalar("batch_diversity", batch_div, use_context=False)
    args.logger.add_object("top-128-seqs", top128[0])

    spearman_corr = spearman_correlation(np.array(new_scores), np.array(proxy_scores))

    ref_seqs, _ = dataset.get_ref_data()
    collected_seqs, collected_scores = dataset.get_all_data()
    novelty_val = mean_novelty(top128[0], ref_seqs)

    # High-fitness proximity
    if oracle is not None:
        hfp = high_fitness_proximity(collected_seqs, all_seqs, all_fitness)
    else:
        hfp = 0.0

    print(f"========== Round {round} ==========")
    print(f"  High-Fitness Proximity: {hfp:.4f}")
    print(f"  Novelty: {novelty_val:.4f}")
    print(f"  Batch Diversity: {batch_div:.4f}")
    print(f"  Normalized Fitness Median (Top-128): {np.median(normalize(top128[1])):.4f}")
    print(f"  Max Fitness: {np.max(top128[1]):.4f}")
    print(f"  Spearman Correlation: {spearman_corr:.4f}")

    # ALDE-aligned log dict
    log = {
        'round': round,
        'high_fitness_proximity': hfp,
        'novelty': novelty_val,
        'batch_diversity': batch_div,
        'normalized_fitness_median_top128': float(np.median(normalize(top128[1]))),
        'normalized_fitness_median_top256': float(np.median(normalize(top256[1]))) if len(top256[1]) > 0 else 0.0,
        'max_fitness': float(np.max(top128[1])),
        'spearman_correlation': spearman_corr,
        'simple_regret': float(global_max - np.max(top128[1])),
    }

    # Collected metrics
    top128_collected = dataset.top_k_collected(128)
    if len(top128_collected[0]) > 0:
        args.logger.add_scalar("collected_max_fitness", np.max(top128_collected[1]), use_context=False)
        collected_novelty = mean_novelty(top128_collected[0], ref_seqs)

        print(f"  Collected Max Fitness: {np.max(top128_collected[1]):.4f}")
        print(f"  Collected Novelty: {collected_novelty:.4f}")

        log["collected_max_fitness"] = np.max(top128_collected[1])
        log["collected_novelty"] = collected_novelty

    if args.use_wandb:
        wandb.log(log)

    if rst is None:
        rst = pd.DataFrame({'round': round, 'sequence': top128[0], 'true_score': top128[1]})
    else:
        new_df = pd.DataFrame({'round': round, 'sequence': top128[0], 'true_score': top128[1]})
        rst = pd.concat([rst, new_df], ignore_index=True)

    return rst


# ============================================================================
# Main Training Loop
# ============================================================================

def train(args, oracle, dataset):
    """Main training loop for delta-CS on GFP_hard."""
    tokenizer = get_gfp_tokenizer(args.max_len)
    args.logger.set_context("iter_0")

    proxy = construct_proxy(args, tokenizer, dataset=dataset)
    proxy.update(dataset)

    all_metrics = []
    rst = None

    for round in range(args.num_rounds):
        args.logger.set_context(f"iter_{round+1}")

        # Create and train generator
        generator = get_generator(args, tokenizer)
        rollout_worker, losses = train_generator(args, generator, proxy, tokenizer, dataset, round=round)

        # Sample new batch
        batch, proxy_scores = sample_batch(args, rollout_worker, generator, oracle,
                                           round=round, dataset=dataset)

        args.logger.add_object("collected_seqs", batch[0])
        args.logger.add_object("collected_seqs_scores", batch[1])

        # Add to dataset
        dataset.add(batch)

        # Convert to strings for logging
        new_seqs = [''.join([oracle.itos[i] for i in x]) for x in batch[0]]

        # Log metrics
        rst = log_overall_metrics(args, dataset, round+1,
                                  new_batch=(new_seqs, batch[1], proxy_scores),
                                  oracle=oracle, rst=rst)

        # Compute detailed metrics
        if not args.skip_metrics:
            metrics = compute_metrics(dataset, oracle, (new_seqs, batch[1], proxy_scores), round+1)
            all_metrics.append(metrics)

        # Update proxy for next round
        if round != args.num_rounds - 1:
            proxy.update(dataset)

        args.logger.save(args.save_path, args)

    return all_metrics, rst


def run_single_experiment(seed: int, args, output_path: str = "results/GFP_hard_delta_cs/") -> Dict[str, Any]:
    """Run a single experiment with given seed."""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    args.seed = seed
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.logger = get_logger(args)

    print(f"\n{'='*60}")
    print(f"Starting delta-CS optimization on GFP_hard")
    print(f"  Seed: {seed}")
    print(f"  Radius option: {args.radius_option}")
    print(f"  Radius range: [{args.min_radius}, {args.max_radius}]")
    print(f"  Batch size: {args.num_queries_per_round}")
    print(f"  Rounds: {args.num_rounds}")
    print(f"  Device: {args.device}")
    print(f"{'='*60}\n")

    # Initialize oracle and dataset
    oracle = GFPOracle(data_dir=args.data_dir)
    dataset = GFPDataset(args, oracle, init_size=args.num_queries_per_round)

    # Initialize wandb
    if args.use_wandb:
        run = wandb.init(project='delta-cs', group='GFP_hard', config=vars(args), reinit=True)
        wandb.run.name = f"{args.name}_seed{seed}_{wandb.run.id}"

    # Run training
    start_time = datetime.now()
    all_metrics, rst = train(args, oracle, dataset)
    runtime = (datetime.now() - start_time).total_seconds()

    # Prepare results
    result = {
        'seed': seed,
        'runtime_seconds': runtime,
        'config': {
            'radius_option': args.radius_option,
            'min_radius': args.min_radius,
            'max_radius': args.max_radius,
            'batch_size': args.num_queries_per_round,
            'num_rounds': args.num_rounds,
            'gen_num_iterations': args.gen_num_iterations,
        },
        'metrics': all_metrics
    }

    # Final summary
    if all_metrics:
        final_metrics = all_metrics[-1]
        result['final_metrics'] = final_metrics

        print(f"\n{'='*60}")
        print("Final Results")
        print(f"{'='*60}")
        print(f"  Max Fitness: {final_metrics.get('max_fitness', 0):.4f}")
        print(f"  Simple Regret: {final_metrics.get('simple_regret', 0):.4f}")
        print(f"  Normalized Fitness Top-128: {final_metrics.get('normalized_fitness_median_top128', 0):.4f}")
        print(f"{'='*60}\n")

    # Save results
    os.makedirs(os.path.join(output_path, 'GFP_hard'), exist_ok=True)
    metrics_path = os.path.join(output_path, 'GFP_hard', f'metrics_seed{seed}.json')
    with open(metrics_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Results saved to: {metrics_path}")

    if rst is not None:
        rst_path = os.path.join(output_path, 'GFP_hard', f'trajectory_seed{seed}.csv')
        rst.to_csv(rst_path, index=False)

    if args.use_wandb:
        wandb.finish()

    return result


def save_aggregated_results(results: List[Dict[str, Any]], output_path: str):
    """Aggregate and save results from multiple runs."""
    if not results:
        return

    # Collect final metrics (ALDE-aligned order)
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
        values = [r['final_metrics'].get(name, 0) for r in results if 'final_metrics' in r]
        if values:
            aggregated[name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }

    # 13. global_max_hit_count - count how many runs found the global maximum
    global_max_hits = sum(
        1 for r in results
        if 'final_metrics' in r and r['final_metrics'].get('global_max_found', False)
    )
    aggregated['global_max_hit_count'] = {
        'count': global_max_hits,
        'rate': float(global_max_hits / len(results)) if results else 0.0,
        'total_runs': len(results)
    }

    # Save
    agg_path = os.path.join(output_path, 'GFP_hard', 'aggregated_results.json')
    with open(agg_path, 'w') as f:
        json.dump({
            'aggregated_metrics': aggregated,
            'n_runs': len(results),
            'seeds': [r['seed'] for r in results],
            'config': results[0].get('config', {})
        }, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print("AGGREGATED METRICS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Metric':<40} {'Mean':>10} {'Std':>10}")
    print("-"*70)
    for name, stats in aggregated.items():
        if isinstance(stats, dict) and 'mean' in stats:
            print(f"{name:<40} {stats['mean']:>10.4f} {stats['std']:>10.4f}")
    # Print global_max_hit_count separately
    gm_stats = aggregated.get('global_max_hit_count', {})
    print(f"{'global_max_hit_count':<40} {gm_stats.get('count', 0):>10} / {gm_stats.get('total_runs', 0):<10} (rate: {gm_stats.get('rate', 0):.4f})")
    print(f"{'='*70}\n")


def main():
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    # Determine seeds
    if args.seed_file is not None:
        # Load seeds from file
        with open(args.seed_file, 'r') as f:
            all_seeds = [int(line.strip()) for line in f if line.strip()]
        if args.num_seeds is not None:
            seeds = all_seeds[:args.num_seeds]
        else:
            seeds = all_seeds
    elif args.seeds is not None:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42]  # Default seed

    print(f"\nRunning delta-CS on GFP_hard with {len(seeds)} seed(s): {seeds[:5]}{'...' if len(seeds) > 5 else ''}")
    print(f"Radius option: {args.radius_option}")
    print(f"Data directory: {args.data_dir}")

    output_path = os.path.dirname(args.save_path)
    if not output_path:
        output_path = "results/GFP_hard_delta_cs/"

    # Run experiments
    results = []
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Running experiment with seed={seed}")
        result = run_single_experiment(seed, args, output_path)
        results.append(result)

    # Aggregate if multiple runs
    if len(results) > 1:
        save_aggregated_results(results, output_path)

    print(f"\n{'='*60}")
    print("All experiments complete!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
