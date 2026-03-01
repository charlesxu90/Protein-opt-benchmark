#!/usr/bin/env python
"""
run_generic.py - Execute EvoPlay optimization on any dataset with comprehensive metrics

This script implements the EvoPlay algorithm following the ALDE benchmarking paradigm
for fair comparison with other directed evolution methods. It is a generic runner
that accepts a --dataset argument and auto-detects sequence properties.

Configuration:
    - Model: Gaussian Process + MCTS with AlphaZero-style Policy-Value Network
    - Encoding: One-hot
    - Initial samples: 96 (cluster-based sampling)
    - Batch size: 96 variants per round
    - Rounds: auto (15 for seq_len <= 50, 5 for seq_len > 50)
    - MCTS playouts: auto (30 for seq_len <= 50, 10 for seq_len > 50)
    - c_puct: 10

Metrics computed (from multiple reference works):
    - Exploration: High-Fitness Proximity, Novelty, Batch Diversity
    - Functional: Normalized Fitness (Top-K), Max Fitness
    - Success: Simple Regret, Global Max Hit Count

Usage:
    # Single run with default seed
    python run_generic.py --dataset AAV_med

    # Single run with specific seed
    python run_generic.py --dataset GB1 --seed 42

    # Multiple runs with different seeds for randomness evaluation
    python run_generic.py --dataset GFP_med --seeds 42 123 456 789 1000

    # Use predefined seeds from file
    python run_generic.py --dataset AAV_hard --seed_file ../rand_seeds.txt --num_seeds 5

    # Use GPU for policy-value network
    python run_generic.py --dataset AAV_med --seed 42 --use_gpu --gpu_device 1
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
from collections import deque
import copy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'GB1_PhoQ_task'))

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cluster import KMeans
from scipy import stats

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
    miscalibration_area,
    expected_calibration_error,
)


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_wildtype_sequence(protein: str, sequences: List[str] = None, fitness: np.ndarray = None) -> Optional[str]:
    """Get the wild-type sequence for a protein.

    Known wild-types are returned directly. For unknown datasets, the sequence
    with the highest fitness is used as a fallback reference.
    """
    wildtype_map = {
        'GB1': 'VDGV',
        'AAV_med': 'DEEEIRTTNPVATEQYGSYSTNLQQGNR',
        'AAV_hard': 'DEEEIRTTNPVATEQYGSYSTNLQQGNR',
    }
    wt = wildtype_map.get(protein)
    if wt is not None:
        return wt
    # Fallback: use the highest-fitness sequence as wildtype reference
    if sequences is not None and fitness is not None and len(sequences) > 0:
        best_idx = int(np.argmax(fitness))
        return sequences[best_idx]
    return None


def epistatic_score_correlation(
    sequences: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    wildtype: str
) -> float:
    """Compute Spearman correlation for high-order mutants (>=2 mutations from wildtype)."""
    high_order_mask = np.array([hamming_distance(seq, wildtype) >= 2 for seq in sequences])
    if np.sum(high_order_mask) < 10:
        return 0.0
    return spearman_correlation(y_true[high_order_mask], y_pred[high_order_mask])


def recall_high_order_mutants(
    sequences: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    wildtype: str,
    min_mutations: int = 2,
    top_k: int = 100
) -> float:
    """Compute recall of top-k high-order mutants."""
    high_order_mask = np.array([hamming_distance(seq, wildtype) >= min_mutations for seq in sequences])
    n_ho = np.sum(high_order_mask)
    if n_ho < top_k:
        top_k = max(1, n_ho // 2)
    if top_k == 0:
        return 0.0

    ho_indices = np.where(high_order_mask)[0]
    ho_true = y_true[high_order_mask]
    ho_pred = y_pred[high_order_mask]

    true_top_k = set(ho_indices[np.argsort(ho_true)[-top_k:]])
    pred_top_k = set(ho_indices[np.argsort(ho_pred)[-top_k:]])
    return len(true_top_k & pred_top_k) / len(true_top_k) if true_top_k else 0.0


# ============================================================================
# EvoPlay Core Components (metrics now imported from utils.compat)
# ============================================================================

AAS = "ILVAGMFYWEDQNHCRKSTP"


def string_to_one_hot(sequence: str, alphabet: str) -> np.ndarray:
    """Convert sequence to one-hot encoding."""
    out = np.zeros((len(sequence), len(alphabet)))
    for i in range(len(sequence)):
        out[i, alphabet.index(sequence[i])] = 1
    return out


def one_hot_to_string(one_hot: np.ndarray, alphabet: str) -> str:
    """Convert one-hot encoding to sequence string."""
    residue_idxs = np.argmax(one_hot, axis=1)
    return "".join([alphabet[idx] for idx in residue_idxs])


def train_gp_predictor(features: np.ndarray, fitness: np.ndarray, seq_indices: List[int], seed: int):
    """Train a Gaussian Process predictor on selected sequences."""
    X_GP = features[seq_indices]
    Y_GP = fitness[seq_indices]
    regr = GaussianProcessRegressor(random_state=seed)
    regr.fit(X_GP, Y_GP)
    return regr


def run_clustering(features: np.ndarray, n_clusters: int) -> List[np.ndarray]:
    """Run K-means clustering on features."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(features)
    cluster_labels = kmeans.labels_

    Index = []
    for i in range(n_clusters):
        index = np.where(cluster_labels == i)[0]
        Index.append(index)
    return Index


def shuffle_index(Index: List[np.ndarray]) -> List[np.ndarray]:
    """Shuffle indices within each cluster."""
    for i in range(len(Index)):
        np.random.shuffle(Index[i])
    return Index


# ============================================================================
# Sequence Environment
# ============================================================================

class SeqEnv:
    """Sequence environment for MCTS-based mutation."""

    def __init__(
        self,
        seq_len: int,
        alphabet: str,
        model,
        start_seq_pool: List[str],
        first_round_combo: List[str],
        combo_to_feature: Dict[str, np.ndarray]
    ):
        self.seq_len = seq_len
        self.vocab_size = len(alphabet)
        self.alphabet = alphabet
        self.model = model
        self.start_seq_pool = list(start_seq_pool)
        self.combo_to_feature = combo_to_feature

        self.move_count = 0
        starting_seq = self.start_seq_pool[0]
        self.starting_seq = starting_seq
        self.seq = starting_seq
        self.start_seq_pool.remove(starting_seq)

        self.init_state = string_to_one_hot(self.seq, self.alphabet).astype(np.float32)
        self.previous_init_state = string_to_one_hot(self.seq, self.alphabet).astype(np.float32)
        self.init_state_count = 0
        self.unuseful_move = 0
        self.states = {}
        self.repeated_seq_ocurr = False
        self.episode_seqs = list(first_round_combo)
        self._state = None
        self._state_fitness = 0.0
        self.previous_fitness = -float("inf")
        self.availables = []
        self.last_move = -1
        self.init_combo = ""

    def init_seq_state(self):
        """Initialize state for a new episode."""
        self.previous_fitness = -float("inf")
        self.move_count = 0
        self.unuseful_move = 0
        self.repeated_seq_ocurr = False
        self._state = copy.deepcopy(self.init_state)

        combo = one_hot_to_string(self._state, AAS)
        self.init_combo = combo

        try:
            input_feat = self.combo_to_feature[combo]
            input_feat = np.expand_dims(input_feat, axis=0)
            outputs = self.model.predict(input_feat)[0]
        except:
            outputs = 0.0
        self._state_fitness = outputs

        self.availables = list(range(self.seq_len * self.vocab_size))
        self.states = {}
        self.last_move = -1
        self.previous_init_state = copy.deepcopy(self._state)

    def current_state(self) -> np.ndarray:
        """Get current state as transposed one-hot."""
        return self._state.T

    def do_mutate(self, move: int):
        """Perform a mutation action."""
        self.previous_fitness = self._state_fitness
        self.move_count += 1
        self.availables.remove(move)
        pos = move // self.vocab_size
        res = move % self.vocab_size

        if self._state[pos, res] == 1:
            self.unuseful_move = 1
            self._state_fitness = 0.0
        else:
            self._state[pos] = 0
            self._state[pos, res] = 1

            combo = one_hot_to_string(self._state, AAS)
            try:
                input_feat = self.combo_to_feature[combo]
                input_feat = np.expand_dims(input_feat, axis=0)
                outputs = self.model.predict(input_feat)[0]
            except:
                outputs = 0.0
            self._state_fitness = outputs

        current_seq = one_hot_to_string(self._state, AAS)
        if current_seq in self.episode_seqs:
            self.repeated_seq_ocurr = True
            self._state_fitness = 0.0
        else:
            self.episode_seqs.append(current_seq)

        if self._state_fitness > self.previous_fitness:
            self.init_state = copy.deepcopy(self._state)
            self.init_state_count = 0

        self.last_move = move

    def mutation_end(self) -> bool:
        """Check if mutation episode should end."""
        if self.repeated_seq_ocurr:
            return True
        if self.unuseful_move == 1:
            return True
        if self._state_fitness < self.previous_fitness:
            return True
        return False


# ============================================================================
# MCTS Implementation
# ============================================================================

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode:
    """A node in the MCTS tree."""

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS:
    """Monte Carlo Tree Search implementation."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=30):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        node = self._root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            state.do_mutate(action)

        action_probs, leaf_value = self._policy(state)
        end = state.mutation_end()
        if not end:
            node.expand(action_probs)
        else:
            leaf_value = state._state_fitness

        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        if not act_visits:
            return [], []

        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)


class MCTSMutater:
    """AI mutater based on MCTS."""

    def __init__(self, policy_value_function, c_puct=5, n_playout=30, is_selfplay=1):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def reset_mutater(self):
        self.mcts.update_with_move(-1)

    def get_action(self, seq_env, temp=1e-3, return_prob=0):
        move_probs = np.zeros(seq_env.seq_len * seq_env.vocab_size)
        acts, probs = self.mcts.get_move_probs(seq_env, temp)

        if acts:
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            return [], []


# ============================================================================
# Policy-Value Network
# ============================================================================

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Net(nn.Module):
    """Policy-value network module adapted for variable-length sequences."""

    def __init__(self, board_width, board_height):
        super(Net, self).__init__()
        self.board_width = board_width  # sequence length
        self.board_height = board_height  # vocab size (20 amino acids)

        # Use smaller kernels and appropriate padding for sequence length
        # Input shape: (batch, vocab_size, seq_len)
        self.conv1 = nn.Conv1d(board_height, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        # Policy head: 128 channels * seq_len -> vocab_size * seq_len actions
        self.act_conv1 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.act_fc1 = nn.Linear(64 * board_width, board_width * board_height)

        # Value head: 128 channels * seq_len -> scalar value
        self.val_conv1 = nn.Conv1d(128, 32, kernel_size=3, padding=1)
        self.val_fc1 = nn.Linear(32 * board_width, 128)
        self.val_fc2 = nn.Linear(128, 1)

    def forward(self, state_input):
        # state_input shape: (batch, vocab_size, seq_len)
        if state_input.shape[2] < 3:
            raise RuntimeError(f"Input shape {state_input.shape} has seq_len < 3! Expected (batch, {self.board_height}, {self.board_width})")
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Policy head
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 64 * self.board_width)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        # Value head
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 32 * self.board_width)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))

        return x_act, x_val


class PolicyValueNet:
    """Policy-value network wrapper."""

    def __init__(self, board_width, board_height, model_file=None, use_gpu=False, gpu_device=0):
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4

        if self.use_gpu:
            self.device = torch.device(f'cuda:{gpu_device}')
            self.policy_value_net = Net(board_width, board_height).to(self.device)
        else:
            self.device = torch.device('cpu')
            self.policy_value_net = Net(board_width, board_height)

        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file, map_location=self.device)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).to(self.device))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board):
        legal_positions = board.availables
        current_state_0 = np.expand_dims(board.current_state(), axis=0)
        current_state = np.ascontiguousarray(current_state_0)

        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                Variable(torch.from_numpy(current_state)).to(self.device).float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())

        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).to(self.device))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).to(self.device))
            winner_batch = Variable(torch.FloatTensor(winner_batch).to(self.device))
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        self.optimizer.zero_grad()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        log_act_probs, value = self.policy_value_net(state_batch)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss

        loss.backward()
        self.optimizer.step()

        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.item(), entropy.item()


# ============================================================================
# Training Pipeline
# ============================================================================

class EvoPlayTrainer:
    """Training pipeline for EvoPlay."""

    def __init__(
        self,
        start_seq_pool: List[str],
        alphabet: str,
        model,
        combo_to_feature: Dict[str, np.ndarray],
        combo_to_index: Dict[str, int],
        combo_to_fitness: Dict[str, float],
        first_round_index_set: set,
        features: np.ndarray,
        fitness: np.ndarray,
        seed: int,
        n_playout: int = 30,
        c_puct: int = 10,
        use_gpu: bool = False,
        gpu_device: int = 0,
        batch_size: int = 96,
        n_rounds: int = 15
    ):
        self.seq_len = len(start_seq_pool[0])
        self.vocab_size = len(alphabet)
        self.alphabet = alphabet
        self.seed = seed

        self.combo_to_index = combo_to_index
        self.combo_to_fitness = combo_to_fitness
        self.features = features
        self.fitness = fitness
        self.first_round_index_set = first_round_index_set

        first_round_combo = start_seq_pool
        self.seq_env = SeqEnv(
            self.seq_len,
            alphabet,
            model,
            start_seq_pool,
            first_round_combo,
            combo_to_feature
        )

        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.n_playout = n_playout
        self.c_puct = c_puct
        self.buffer_size = 10000
        self.batch_size = 8
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.epochs = 5
        self.kl_targ = 0.02
        self.game_batch_num = 10000

        self.collected_seqs_index_set = set()
        self.updata_predictor_index_set = set()
        self.last_tmp_set_len = 0
        self.update_predictor = 0
        self.remove_list = []

        self.policy_value_net = PolicyValueNet(self.seq_len, self.vocab_size, use_gpu=use_gpu, gpu_device=gpu_device)
        self.mcts_player = MCTSMutater(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1
        )

        # Configuration for rounds-based stopping
        self.target_batch_size = batch_size  # Samples per round
        self.n_rounds = n_rounds
        self.total_target = batch_size * n_rounds  # Total samples to collect

        # Threshold for when to update the predictor model
        self.update_thresholds = self._compute_update_thresholds(batch_size, n_rounds)

    def _compute_update_thresholds(self, batch_size: int, n_rounds: int) -> List[int]:
        """Compute thresholds for predictor updates based on batch size and rounds."""
        thresholds = [batch_size * (i + 1) for i in range(n_rounds)]
        return thresholds

    def start_mutating(self):
        """Start a mutation episode."""
        if (self.seq_env.previous_init_state == self.seq_env.init_state).all():
            self.seq_env.init_state_count += 1
        if self.seq_env.init_state_count >= 10:
            if len(self.seq_env.start_seq_pool) > 0:
                new_start_seq = self.seq_env.start_seq_pool[0]
                while new_start_seq == "" or len(new_start_seq) != self.seq_len:
                    self.seq_env.start_seq_pool.remove(new_start_seq)
                    if len(self.seq_env.start_seq_pool) == 0:
                        break
                    new_start_seq = self.seq_env.start_seq_pool[0]
                if new_start_seq and len(new_start_seq) == self.seq_len:
                    self.seq_env.init_state = string_to_one_hot(new_start_seq, self.alphabet).astype(np.float32)
                    self.seq_env.start_seq_pool.remove(new_start_seq)
                    self.seq_env.init_state_count = 0
                    self.remove_list.append(new_start_seq)

        self.seq_env.init_seq_state()

        states, mcts_probs, reward_z = [], [], []
        generated_seqs = set()

        while True:
            move, move_probs = self.mcts_player.get_action(self.seq_env, temp=self.temp, return_prob=1)

            if move:
                states.append(self.seq_env.current_state())
                mcts_probs.append(move_probs)
                reward_z.append(self.seq_env._state_fitness)
                generated_seqs.add(one_hot_to_string(self.seq_env._state, AAS))
                self.seq_env.do_mutate(move)

            end = self.seq_env.mutation_end()
            if end:
                generated_seqs.add(one_hot_to_string(self.seq_env._state, AAS))
                self.mcts_player.reset_mutater()
                return list(zip(states, mcts_probs, reward_z)), generated_seqs

    def collect_selfplay_data(self):
        """Collect self-play data for training."""
        self.update_predictor = 0

        play_data, gen_seqs = self.start_mutating()
        episode_len = len(play_data)

        if episode_len > 0:
            self.data_buffer.extend(play_data)

            for seq in gen_seqs:
                try:
                    self.collected_seqs_index_set.add(self.combo_to_index[seq])
                except:
                    continue

                tmp_set = self.collected_seqs_index_set.union(self.first_round_index_set)

                for threshold in self.update_thresholds:
                    if len(tmp_set) >= threshold and len(tmp_set) != self.last_tmp_set_len:
                        if threshold < self.total_target:
                            self.update_predictor = 1
                        self.updata_predictor_index_set = tmp_set
                        break

                self.last_tmp_set_len = len(tmp_set)

        return episode_len

    def policy_update(self):
        """Update the policy-value network."""
        if len(self.data_buffer) < self.batch_size:
            return 0, 0

        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:
                break

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        return loss, entropy

    def run(self, verbose: int = 1) -> Tuple[List[int], List[str]]:
        """Run the training pipeline."""
        try:
            for i in range(self.game_batch_num):
                episode_len = self.collect_selfplay_data()

                if verbose >= 2:
                    print(f"Batch {i+1}, episode_len: {episode_len}")

                if self.update_predictor == 1:
                    if verbose >= 1:
                        print(f"Updating predictor with {len(self.updata_predictor_index_set)} sequences")
                    seq_index_list = list(self.updata_predictor_index_set)
                    update_model = train_gp_predictor(self.features, self.fitness, seq_index_list, self.seed)

                    update_predictor_dict = {
                        self.index_to_combo(idx): self.fitness[idx]
                        for idx in seq_index_list
                    }
                    update_predictor_dict = {k: v for k, v in update_predictor_dict.items() if k}
                    update_round_d = sorted(update_predictor_dict.items(), key=lambda x: x[1], reverse=True)
                    new_pool = [k for k, v in update_round_d]
                    if verbose >= 2:
                        print(f"  New start pool: {len(new_pool)} seqs, first: '{new_pool[0] if new_pool else 'EMPTY'}' len={len(new_pool[0]) if new_pool else 0}")
                    if new_pool:
                        self.seq_env.start_seq_pool = new_pool
                    self.seq_env.model = update_model

                if len(self.updata_predictor_index_set) >= self.total_target:
                    if verbose >= 1:
                        print(f"Reached {self.total_target} sequences, stopping")
                    break

                if len(self.data_buffer) > self.batch_size:
                    self.policy_update()

        except KeyboardInterrupt:
            print('\nInterrupted')

        # Return collected indices and sequences, filtering out empty sequences
        all_indices = list(self.first_round_index_set.union(self.collected_seqs_index_set))
        all_seqs = [self.index_to_combo(idx) for idx in all_indices]
        valid_pairs = [(idx, seq) for idx, seq in zip(all_indices, all_seqs) if seq and len(seq) == self.seq_len]
        if valid_pairs:
            all_indices, all_seqs = zip(*valid_pairs)
            all_indices = list(all_indices)
            all_seqs = list(all_seqs)
        else:
            all_indices, all_seqs = [], []
        return all_indices, all_seqs

    def index_to_combo(self, idx: int) -> str:
        """Convert index to combo sequence."""
        combo_to_index_inv = {v: k for k, v in self.combo_to_index.items()}
        return combo_to_index_inv.get(idx, "")


# ============================================================================
# Main Experiment Functions
# ============================================================================

def load_dataset_data(dataset: str, data_dir: str = None) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """
    Load dataset from data/<dataset>/data.csv generically.

    The CSV must have columns 'seq' and 'fitness'.

    Returns:
        features: One-hot encoded features (N x seq_len*vocab_size)
        fitness: Fitness values (N,)
        sequences: List of sequence strings
        seq_len: Detected sequence length
    """
    if data_dir is None:
        data_dir = '/home/xux/Desktop/AlphaVariant/Benchmark/data'

    data_path = os.path.join(data_dir, dataset, 'data.csv')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    fitness_df = pd.read_csv(data_path)

    # Support both 'seq' and 'sequence' column names
    if 'seq' in fitness_df.columns:
        sequences = fitness_df['seq'].tolist()
    elif 'sequence' in fitness_df.columns:
        sequences = fitness_df['sequence'].tolist()
    else:
        raise ValueError(f"Data CSV must have a 'seq' or 'sequence' column. Found: {list(fitness_df.columns)}")

    fitness = fitness_df['fitness'].values

    # Auto-detect sequence length from the data
    seq_lengths = set(len(s) for s in sequences)
    if len(seq_lengths) > 1:
        # Variable-length sequences: filter to the most common length
        from collections import Counter
        length_counts = Counter(len(s) for s in sequences)
        most_common_len = length_counts.most_common(1)[0][0]
        print(f"  Warning: Found {len(seq_lengths)} different sequence lengths. "
              f"Filtering to most common length: {most_common_len}")
        mask = [len(s) == most_common_len for s in sequences]
        sequences = [s for s, m in zip(sequences, mask) if m]
        fitness = fitness[mask]
    seq_len = len(sequences[0])

    # Create one-hot features
    features = []
    for seq in sequences:
        oh = string_to_one_hot(seq, AAS)
        features.append(oh.flatten())
    features = np.array(features)

    print(f"Loaded {dataset} data from: {data_path}")
    print(f"  Total sequences: {len(sequences)}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Fitness range: [{fitness.min():.4f}, {fitness.max():.4f}]")

    return features, fitness, sequences, seq_len


def get_default_n_playout(seq_len: int) -> int:
    """Get default MCTS playout count based on sequence length."""
    if seq_len <= 50:
        return 30
    else:
        return 10


def get_default_n_rounds(seq_len: int) -> int:
    """Get default number of optimization rounds based on sequence length."""
    if seq_len <= 50:
        return 15
    else:
        return 5


def run_single_experiment(
    dataset: str,
    seed: int,
    data_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    verbose: int = 2,
    run_id: Optional[int] = None,
    compute_metrics: bool = True,
    n_playout: Optional[int] = None,
    use_gpu: bool = False,
    gpu_device: int = 0,
    batch_size: int = 96,
    n_rounds: Optional[int] = None
) -> Dict[str, Any]:
    """Run a single EvoPlay experiment on the specified dataset."""

    # Configuration
    n_clusters = 30
    num_first_round = 96

    if run_id is None:
        run_id = seed

    if output_path is None:
        output_path = f"results/{dataset}_EvoPlay/"

    # Load data (auto-detects seq_len)
    features, fitness_normalized, sequences, seq_len = load_dataset_data(dataset, data_dir)

    # Set defaults based on detected sequence length
    if n_playout is None:
        n_playout = get_default_n_playout(seq_len)
    if n_rounds is None:
        n_rounds = get_default_n_rounds(seq_len)

    total_samples = batch_size * n_rounds

    print(f"\n{'='*60}")
    print(f"Starting EvoPlay optimization on {dataset}")
    print(f"  Seed: {seed}")
    print(f"  Model: GP + MCTS with Policy-Value Network")
    print(f"  Encoding: One-hot")
    print(f"  Sequence length: {seq_len}")
    print(f"  Initial samples: {num_first_round}")
    print(f"  Batch size: {batch_size}")
    print(f"  Rounds: {n_rounds}")
    print(f"  MCTS playouts: {n_playout}")
    print(f"  Total target samples: {total_samples}")
    print(f"  GPU: {'cuda:' + str(gpu_device) if use_gpu else 'CPU'}")
    print(f"{'='*60}\n")

    set_seed(seed)

    # Create output directory
    subdir = os.path.join(output_path, dataset, "onehot", "")
    os.makedirs(subdir, exist_ok=True)

    # Filter to medium-fitness sequences (40th-60th percentile) for initial sampling
    threshold_low = np.percentile(fitness_normalized, 40)
    threshold_high = np.percentile(fitness_normalized, 60)
    candidate_mask = (fitness_normalized >= threshold_low) & (fitness_normalized <= threshold_high)
    candidate_indices = np.where(candidate_mask)[0]

    print(f"  Filtering to 40th-60th percentile: {len(candidate_indices)} candidates")
    print(f"  Fitness range: [{threshold_low:.4f}, {threshold_high:.4f}]")

    # Get features for candidates only
    candidate_features = features[candidate_indices]

    # Adjust n_clusters if fewer candidates than clusters
    effective_n_clusters = min(n_clusters, len(candidate_indices))
    if effective_n_clusters < n_clusters:
        print(f"  Adjusted n_clusters from {n_clusters} to {effective_n_clusters} (fewer candidates)")

    # Run K-means clustering on candidate features (hybrid approach)
    Index = run_clustering(candidate_features, effective_n_clusters)
    Index = shuffle_index(Index)
    Prob = np.ones([effective_n_clusters]) / effective_n_clusters

    # Sample initial sequences from clusters
    Fit_list = []
    SEQ_list = []
    SEQ_index = [[] for _ in range(len(Index))]
    num = 0

    while num < num_first_round:
        cluster_id = np.random.choice(np.arange(0, effective_n_clusters), p=Prob)
        while len(Index[cluster_id]) == 0:
            Prob[cluster_id] = 0
            if np.sum(Prob) == 0:
                # All clusters exhausted, break
                break
            Prob = Prob / np.sum(Prob)
            cluster_id = np.random.choice(np.arange(0, effective_n_clusters), p=Prob)

        if np.sum(Prob) == 0:
            print(f"  Warning: All clusters exhausted after {num} initial samples")
            break

        local_idx = Index[cluster_id][0]
        global_idx = candidate_indices[local_idx]  # Map back to global index

        Fit_list.append(fitness_normalized[global_idx])
        SEQ_list.append(sequences[global_idx])
        SEQ_index[cluster_id].append(global_idx)
        Index[cluster_id] = np.delete(Index[cluster_id], [0])
        num += 1

    # Create mappings
    combo_to_feature = dict(zip(sequences, features))
    combo_to_index = dict(zip(sequences, range(len(sequences))))
    combo_to_fitness = dict(zip(sequences, fitness_normalized))

    # Sort initial sequences by fitness (descending)
    combo_to_fitness_first_round = dict(zip(SEQ_list, Fit_list))
    first_round_d = sorted(combo_to_fitness_first_round.items(), key=lambda x: x[1], reverse=True)
    start_pool = [k for k, v in first_round_d]

    # Get wildtype (dynamically populated for unknown datasets)
    wildtype = get_wildtype_sequence(dataset, sequences, fitness_normalized)
    if wildtype and wildtype in start_pool:
        start_pool.remove(wildtype)
        print(f"  Warning: Wild-type found in initial samples, removed from start_pool")

    if not start_pool:
        raise ValueError("No valid starting sequences after excluding wild-type")

    print(f"  Starting sequence: {start_pool[0]} (fitness: {combo_to_fitness_first_round.get(start_pool[0], 'N/A'):.4f})")

    # Get first round indices
    first_round_index = []
    for cluster_id in range(len(SEQ_index)):
        first_round_index.extend(SEQ_index[cluster_id])
    first_round_index_set = set(first_round_index)

    # Train initial GP model
    model_gp = train_gp_predictor(features, fitness_normalized, first_round_index, seed)

    # Create and run trainer
    start_time = datetime.now()

    trainer = EvoPlayTrainer(
        start_seq_pool=start_pool,
        alphabet=AAS,
        model=model_gp,
        combo_to_feature=combo_to_feature,
        combo_to_index=combo_to_index,
        combo_to_fitness=combo_to_fitness,
        first_round_index_set=first_round_index_set,
        features=features,
        fitness=fitness_normalized,
        seed=seed,
        n_playout=n_playout,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        batch_size=batch_size,
        n_rounds=n_rounds
    )

    queried_indices, queried_seqs = trainer.run(verbose=verbose)

    runtime = (datetime.now() - start_time).total_seconds()

    # Save indices
    result_path = os.path.join(subdir, f"EvoPlay_seed{seed}_indices.pt")
    torch.save(torch.tensor(queried_indices), result_path)
    print(f"\nResults saved to: {result_path}")

    # Prepare result dictionary
    result = {
        'seed': seed,
        'run_id': run_id,
        'dataset': dataset,
        'result_path': result_path,
        'runtime_seconds': runtime,
        'n_queries': len(queried_indices),
        'seq_len': seq_len,
        'config': {
            'model': 'GP + MCTS + PolicyValueNet',
            'encoding': 'onehot',
            'n_init': num_first_round,
            'n_playout': n_playout,
            'batch_size': batch_size,
            'n_rounds': n_rounds,
        }
    }

    # Compute metrics at checkpoints
    if compute_metrics:
        print("\nComputing evaluation metrics at checkpoints...")

        queried_fitness = fitness_normalized[queried_indices]
        initial_seqs = [sequences[i] for i in first_round_index]

        global_max = np.max(fitness_normalized)
        global_min = np.min(fitness_normalized)

        # Checkpoints based on batch_size and n_rounds
        checkpoints = [batch_size * (i + 1) for i in range(n_rounds)]
        result['checkpoint_metrics'] = {}

        for checkpoint in checkpoints:
            if len(queried_indices) < checkpoint:
                print(f"  Skipping checkpoint {checkpoint}: only {len(queried_indices)} sequences collected")
                continue

            # Use only first `checkpoint` sequences for this evaluation point
            checkpoint_indices = queried_indices[:checkpoint]
            checkpoint_fitness = queried_fitness[:checkpoint]
            checkpoint_seqs = queried_seqs[:checkpoint]

            # Exploration metrics
            hfp = high_fitness_proximity(checkpoint_seqs, sequences, fitness_normalized)
            nov = novelty(checkpoint_seqs[num_first_round:], initial_seqs) if checkpoint > num_first_round else 0.0
            bd = batch_diversity(checkpoint_seqs)

            # Functional metrics
            nf_top128 = normalized_fitness_topk(checkpoint_fitness, k=128, global_min=global_min, global_max=global_max)
            nf_top256 = normalized_fitness_topk(checkpoint_fitness, k=256, global_min=global_min, global_max=global_max)
            max_fit = float(np.max(checkpoint_fitness))

            # Train GP model on checkpoint data for model quality metrics
            checkpoint_model = train_gp_predictor(features, fitness_normalized, checkpoint_indices, seed)

            # Get predictions on holdout set
            checkpoint_set = set(checkpoint_indices)
            holdout_mask = np.array([i not in checkpoint_set for i in range(len(fitness_normalized))])
            holdout_indices = np.where(holdout_mask)[0]

            # Model quality metrics (computed on holdout set)
            spearman_corr = 0.0
            epistatic_corr = 0.0
            recall_ho = 0.0
            miscal_area = 1.0
            ece = 1.0

            if len(holdout_indices) > 100:
                holdout_features = features[holdout_indices]
                holdout_true = fitness_normalized[holdout_indices]
                holdout_seqs = [sequences[i] for i in holdout_indices]

                # Get predictions with uncertainty
                holdout_pred, holdout_std = checkpoint_model.predict(holdout_features, return_std=True)

                # Spearman correlation
                spearman_corr = spearman_correlation(holdout_true, holdout_pred)

                # Epistatic correlation (if wildtype available)
                if wildtype is not None:
                    epistatic_corr = epistatic_score_correlation(
                        holdout_seqs, holdout_true, holdout_pred, wildtype
                    )
                    recall_ho = recall_high_order_mutants(
                        holdout_seqs, holdout_true, holdout_pred, wildtype,
                        min_mutations=2, top_k=100
                    )

                # Uncertainty metrics
                miscal_area = miscalibration_area(holdout_true, holdout_pred, holdout_std)
                ece = expected_calibration_error(holdout_true, holdout_pred, holdout_std)

            # Success metrics
            sr = simple_regret(max_fit, global_max)
            global_max_found = max_fit >= global_max * 0.99

            # Build checkpoint_metrics
            checkpoint_metrics = {
                'n_sequences': checkpoint,
                'high_fitness_proximity': hfp,
                'novelty': nov,
                'batch_diversity': bd,
                'normalized_fitness_median_top128': nf_top128,
                'normalized_fitness_median_top256': nf_top256,
                'max_fitness': max_fit,
                'spearman_correlation': spearman_corr,
                'epistatic_correlation': epistatic_corr,
                'recall_high_order': recall_ho,
                'simple_regret': sr,
                'miscalibration_area': miscal_area,
                'expected_calibration_error': ece,
                'global_max_found': global_max_found,
            }
            result['checkpoint_metrics'][f'n_{checkpoint}'] = checkpoint_metrics

            print(f"\n  Checkpoint {checkpoint}:")
            print(f"    High-Fitness Proximity: {hfp:.4f}")
            print(f"    Novelty: {nov:.4f}")
            print(f"    Batch Diversity: {bd:.4f}")
            print(f"    Normalized Fitness (Top-128): {nf_top128:.4f}")
            print(f"    Normalized Fitness (Top-256): {nf_top256:.4f}")
            print(f"    Max Fitness: {max_fit:.4f}")
            print(f"    Spearman Correlation: {spearman_corr:.4f}")
            print(f"    Epistatic Correlation: {epistatic_corr:.4f}")
            print(f"    Recall High-Order: {recall_ho:.4f}")
            print(f"    Simple Regret: {sr:.4f}")
            print(f"    Miscalibration Area: {miscal_area:.4f}")
            print(f"    Expected Calibration Error: {ece:.4f}")

        # Set final metrics to the last checkpoint
        final_checkpoint = f'n_{batch_size * n_rounds}'
        if final_checkpoint in result['checkpoint_metrics']:
            result['metrics'] = result['checkpoint_metrics'][final_checkpoint]
        elif result['checkpoint_metrics']:
            last_key = sorted(result['checkpoint_metrics'].keys(), key=lambda x: int(x.split('_')[1]))[-1]
            result['metrics'] = result['checkpoint_metrics'][last_key]
        else:
            result['metrics'] = {}

        # Fitness trajectory
        fitness_traj = []
        for i in range(0, len(queried_fitness), batch_size):
            batch_fitness = queried_fitness[:i + batch_size]
            fitness_traj.append(float(np.max(batch_fitness)))
        result['fitness_trajectory'] = fitness_traj

        # Print metrics summary
        if result['metrics']:
            m = result['metrics']
            print("\n" + "-"*40)
            print("Metrics Summary:")
            print("-"*40)
            print(f"  Max Fitness: {m.get('max_fitness', 0):.4f}")
            print(f"  Simple Regret: {m.get('simple_regret', 0):.4f}")
            print(f"  Global Max Found: {m.get('global_max_found', False)}")
            print(f"  Normalized Fitness (Top-128): {m.get('normalized_fitness_median_top128', 0):.4f}")
            print(f"  High-Fitness Proximity: {m.get('high_fitness_proximity', 0):.4f}")
            print(f"  Novelty: {m.get('novelty', 0):.4f}")
            print(f"  Batch Diversity: {m.get('batch_diversity', 0):.4f}")
            print(f"  Spearman Correlation: {m.get('spearman_correlation', 0):.4f}")
            print(f"  Miscalibration Area: {m.get('miscalibration_area', 1):.4f}")

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
    """Save aggregated results across all runs with per-checkpoint metrics."""

    metrics_results = [r for r in results if 'metrics' in r]

    if not metrics_results:
        print("No metrics to aggregate.")
        return

    # Metrics to aggregate
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

    # Get all checkpoint keys
    all_checkpoint_keys = set()
    for r in results:
        if 'checkpoint_metrics' in r:
            all_checkpoint_keys.update(r['checkpoint_metrics'].keys())

    # Sort checkpoint keys by sample count
    checkpoint_keys = sorted(all_checkpoint_keys, key=lambda x: int(x.split('_')[1]))

    # Aggregate per-checkpoint metrics
    print("\n" + "="*70)
    print("PER-CHECKPOINT METRICS AGGREGATION")
    print("="*70)

    checkpoint_aggregates = {}
    for checkpoint_key in checkpoint_keys:
        checkpoint_results = [r.get('checkpoint_metrics', {}).get(checkpoint_key) for r in results]
        checkpoint_results = [r for r in checkpoint_results if r is not None]

        if not checkpoint_results:
            continue

        aggregated = {}
        for name in metrics_names:
            values = [r.get(name, 0) for r in checkpoint_results]
            aggregated[name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }

        # Global max hit count
        max_fitness_values = [r['max_fitness'] for r in checkpoint_results]
        hit_count, hit_rate = global_max_hit_count(max_fitness_values, 1.0, tolerance=0.01)
        aggregated['global_max_hit_count'] = {'count': hit_count, 'rate': hit_rate}

        checkpoint_aggregates[checkpoint_key] = aggregated

        # Print checkpoint summary
        checkpoint_num = int(checkpoint_key.split('_')[1])
        print(f"\nCheckpoint {checkpoint_num} ({len(checkpoint_results)} runs):")
        print(f"  {'Metric':<40} {'Mean':>10} {'Std':>10}")
        print("-"*70)
        for name in metrics_names:
            stats_val = aggregated[name]
            print(f"  {name:<40} {stats_val['mean']:>10.4f} {stats_val['std']:>10.4f}")
        hit_stats = aggregated['global_max_hit_count']
        print(f"  {'global_max_hit_count':<40} {hit_stats['rate']*100:>9.1f}% ({int(hit_stats['count'])}/{len(checkpoint_results)})")

    # Also aggregate final metrics (for backward compatibility)
    aggregated_final = {}
    for name in metrics_names:
        values = [r['metrics'].get(name, 0) for r in metrics_results]
        aggregated_final[name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }

    max_fitness_values = [r['metrics']['max_fitness'] for r in metrics_results]
    hit_count, hit_rate = global_max_hit_count(max_fitness_values, 1.0, tolerance=0.01)
    aggregated_final['global_max_hit_count'] = {'count': hit_count, 'rate': hit_rate}

    # Create summary DataFrame for final metrics
    summary_data = []
    for metric, stats_val in aggregated_final.items():
        if 'mean' in stats_val:
            summary_data.append({
                'metric': metric,
                'mean': stats_val['mean'],
                'std': stats_val['std'],
                'min': stats_val['min'],
                'max': stats_val['max']
            })
        elif 'count' in stats_val:
            summary_data.append({
                'metric': metric,
                'mean': stats_val['rate'],
                'std': 0,
                'min': stats_val['count'],
                'max': len(results)
            })

    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    summary_path = os.path.join(output_path, dataset, 'onehot', 'aggregated_metrics.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n\nFinal aggregated metrics saved to: {summary_path}")

    # Save complete aggregated results to JSON (including checkpoint metrics)
    aggregated_json_path = os.path.join(output_path, dataset, 'onehot', 'aggregated_results.json')
    with open(aggregated_json_path, 'w') as f:
        json.dump({
            'dataset': dataset,
            'aggregated_metrics_final': aggregated_final,
            'checkpoint_aggregates': checkpoint_aggregates,
            'n_runs': len(results),
            'seeds': [r['seed'] for r in results],
            'config': results[0].get('config', {}) if results else {}
        }, f, indent=2, default=str)
    print(f"Complete aggregated results saved to: {aggregated_json_path}")

    # Print final summary table
    print("\n" + "="*70)
    print("FINAL AGGREGATED METRICS SUMMARY")
    print("="*70)
    print(f"{'Metric':<40} {'Mean':>10} {'Std':>10}")
    print("-"*70)
    for _, row in summary_df.iterrows():
        if row['metric'] == 'global_max_hit_count':
            print(f"{row['metric']:<40} {row['mean']*100:>9.1f}% ({int(row['min'])}/{int(row['max'])})")
        else:
            print(f"{row['metric']:<40} {row['mean']:>10.4f} {row['std']:>10.4f}")
    print("="*70)


def run_experiment_wrapper(args_tuple):
    """Wrapper function for parallel execution."""
    seed, run_id, config = args_tuple
    try:
        result = run_single_experiment(
            dataset=config['dataset'],
            seed=seed,
            data_dir=config['data_dir'],
            output_path=config['output_path'],
            verbose=config['verbose'],
            run_id=run_id,
            compute_metrics=config['compute_metrics'],
            n_playout=config['n_playout'],
            use_gpu=config['use_gpu'],
            gpu_device=config['gpu_device'],
            batch_size=config['batch_size'],
            n_rounds=config['n_rounds']
        )
        return result
    except Exception as e:
        print(f"Error running experiment with seed {seed}: {e}")
        import traceback
        traceback.print_exc()
        return {'seed': seed, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Run EvoPlay optimization on any dataset with GP + MCTS + Policy-Value Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run on AAV_med with default seed
  python run_generic.py --dataset AAV_med

  # Single run on GB1 with specific seed
  python run_generic.py --dataset GB1 --seed 42

  # Multiple runs for randomness evaluation
  python run_generic.py --dataset GFP_med --seeds 42 123 456 789 1000

  # Load seeds from file
  python run_generic.py --dataset AAV_hard --seed_file ../rand_seeds.txt --num_seeds 5

  # Skip metrics computation
  python run_generic.py --dataset AAV_med --seed 42 --skip_metrics

  # Use GPU device 1
  python run_generic.py --dataset AAV_med --seed_file ../rand_seeds.txt --num_seeds 5 --gpu_device 1 --use_gpu

  # Override auto-detected MCTS parameters
  python run_generic.py --dataset AAV_med --seed 42 --n_playout 20 --n_rounds 10
        """
    )

    # Dataset (required)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., GB1, AAV_med, AAV_hard, GFP_med). "
             "Data is loaded from data/<dataset>/data.csv"
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

    # Data and output paths
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/xux/Desktop/AlphaVariant/Benchmark/data",
        help="Data directory (default: /home/xux/Desktop/AlphaVariant/Benchmark/data)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output directory for results (default: results/<dataset>_EvoPlay/)"
    )

    # Batch and rounds configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=96,
        help="Number of variants per round (default: 96)"
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=None,
        help="Number of optimization rounds (default: auto based on seq_len; 15 for <=50, 5 for >50)"
    )

    # Other options
    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="Verbosity level 0-3 (default: 2)"
    )
    parser.add_argument(
        "--n_playout",
        type=int,
        default=None,
        help="Number of MCTS playouts per move (default: auto based on seq_len; 30 for <=50, 10 for >50)"
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU for policy-value network"
    )
    parser.add_argument(
        "--gpu_device",
        type=int,
        default=0,
        help="GPU device ID to use (default: 0)"
    )
    parser.add_argument(
        "--skip_metrics",
        action="store_true",
        help="Skip metrics computation (faster)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential)"
    )

    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    dataset = args.dataset

    # Set default output path based on dataset
    if args.output_path is None:
        output_path = f"results/{dataset}_EvoPlay/"
    else:
        output_path = args.output_path

    # Determine which seeds to use
    if args.seeds is not None:
        seeds = args.seeds
    elif args.seed_file is not None:
        seeds = load_seeds_from_file(args.seed_file, args.num_seeds)
        print(f"Loaded {len(seeds)} seeds from {args.seed_file}")
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [64]

    # Probe the dataset to determine seq_len for default parameter reporting
    # (actual loading happens inside run_single_experiment)
    data_path = os.path.join(args.data_dir, dataset, 'data.csv')
    if os.path.exists(data_path):
        probe_df = pd.read_csv(data_path, nrows=1)
        col = 'seq' if 'seq' in probe_df.columns else 'sequence' if 'sequence' in probe_df.columns else None
        if col:
            probe_seq_len = len(probe_df[col].iloc[0])
        else:
            probe_seq_len = None
    else:
        probe_seq_len = None

    effective_n_playout = args.n_playout if args.n_playout is not None else (
        get_default_n_playout(probe_seq_len) if probe_seq_len else '?')
    effective_n_rounds = args.n_rounds if args.n_rounds is not None else (
        get_default_n_rounds(probe_seq_len) if probe_seq_len else '?')

    print(f"\nRunning EvoPlay on {dataset} with {len(seeds)} seed(s): {seeds[:10]}{'...' if len(seeds) > 10 else ''}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output path: {output_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Rounds: {effective_n_rounds}" + (" (auto)" if args.n_rounds is None else ""))
    print(f"MCTS playouts: {effective_n_playout}" + (" (auto)" if args.n_playout is None else ""))
    if probe_seq_len:
        print(f"Detected sequence length: {probe_seq_len}")
    print(f"Compute metrics: {not args.skip_metrics}")
    print(f"GPU device: {args.gpu_device if args.use_gpu else 'CPU'}")
    print(f"Parallel workers: {args.n_workers}")

    # Configuration dict for parallel execution
    config = {
        'dataset': dataset,
        'data_dir': args.data_dir,
        'output_path': output_path,
        'verbose': args.verbose if args.n_workers == 1 else 0,
        'compute_metrics': not args.skip_metrics,
        'n_playout': args.n_playout,  # None means auto-detect inside run_single_experiment
        'use_gpu': args.use_gpu,
        'gpu_device': args.gpu_device,
        'batch_size': args.batch_size,
        'n_rounds': args.n_rounds,  # None means auto-detect inside run_single_experiment
    }

    # Prepare arguments for parallel execution
    experiment_args = [(seed, i + 1, config) for i, seed in enumerate(seeds)]

    # Run experiments
    results = []
    start_time = datetime.now()

    if args.n_workers > 1:
        # Parallel execution
        print(f"\nStarting parallel execution with {args.n_workers} workers...")

        # Use spawn method for CUDA compatibility
        mp.set_start_method('spawn', force=True)

        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {executor.submit(run_experiment_wrapper, arg): arg[0] for arg in experiment_args}

            completed = 0
            for future in as_completed(futures):
                seed = futures[future]
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                    if 'error' not in result:
                        print(f"[{completed}/{len(seeds)}] Completed seed {seed} - Max fitness: {result.get('metrics', {}).get('max_fitness', 'N/A')}")
                    else:
                        print(f"[{completed}/{len(seeds)}] Failed seed {seed}: {result['error']}")
                except Exception as e:
                    print(f"[{completed}/{len(seeds)}] Exception for seed {seed}: {e}")
                    results.append({'seed': seed, 'error': str(e)})
    else:
        # Sequential execution
        for i, seed in enumerate(seeds):
            print(f"\n[{i+1}/{len(seeds)}] Running experiment with seed={seed}")
            result = run_single_experiment(
                dataset=dataset,
                seed=seed,
                data_dir=args.data_dir,
                output_path=output_path,
                verbose=args.verbose,
                run_id=i + 1,
                compute_metrics=not args.skip_metrics,
                n_playout=args.n_playout,
                use_gpu=args.use_gpu,
                gpu_device=args.gpu_device,
                batch_size=args.batch_size,
                n_rounds=args.n_rounds
            )
            results.append(result)

    total_time = (datetime.now() - start_time).total_seconds()

    # Filter out failed results for aggregation
    successful_results = [r for r in results if 'error' not in r]

    # Aggregate results if multiple runs
    if len(successful_results) > 1 and not args.skip_metrics:
        save_aggregated_results(successful_results, output_path, dataset)

    # Final summary
    print(f"\n{'='*60}")
    print("Experiment Complete")
    print(f"{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"Total runs: {len(results)} ({len(successful_results)} successful, {len(results) - len(successful_results)} failed)")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Average time per run: {total_time/len(results):.1f} seconds")
    print(f"Configuration:")
    print(f"  - Model: GP + MCTS + PolicyValueNet")
    print(f"  - Encoding: onehot")
    print(f"  - Initial samples: 96")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Rounds: {effective_n_rounds}")
    print(f"  - MCTS playouts: {effective_n_playout}")
    print(f"  - Workers: {args.n_workers}")
    print(f"  - GPU: {'cuda:' + str(args.gpu_device) if args.use_gpu else 'CPU'}")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
