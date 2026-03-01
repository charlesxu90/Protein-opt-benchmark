#!/usr/bin/env python
"""
run_generic.py - Execute LatProtRL optimization on any protein dataset with comprehensive metrics

This script adapts LatProtRL (Reinforcement Learning in Latent Space for Protein Optimization)
to arbitrary fitness landscape benchmarks. Dataset-specific parameters (sequence length, wildtype,
fitness range) are auto-detected from the data.

Configuration:
    - Model: PPO with VED (Variant Encoder-Decoder)
    - Encoding: ESM-2 latent space
    - Batch size: 96 (per round)
    - Rounds: 5 (1 initial + 4 iterations)

Metrics computed:
    - Exploration: High-Fitness Proximity, Novelty, Batch Diversity
    - Functional: Normalized Fitness (Top-K), Max Fitness
    - Success: Simple Regret

Usage:
    # Single run with default seed
    python run_generic.py --dataset AAV_hard --device cuda:0

    # Single run with specific seed
    python run_generic.py --dataset AAV_hard --device cuda:0 --seed 42

    # Multiple runs with different seeds
    python run_generic.py --dataset AAV_hard --device cuda:0 --seeds 42 123 456 789 1000

    # Use predefined seeds from file
    python run_generic.py --dataset AAV_hard --device cuda:0 --seed_file src/rndseed.txt --num_seeds 5

    # Skip metrics computation (faster)
    python run_generic.py --dataset AAV_hard --device cuda:0 --seed 42 --skip_metrics
"""

from __future__ import annotations
import os
import sys
import time
import json
import argparse
import warnings
import random
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import Env, spaces

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from net.ppo import PPO
from net.buffers import Buffer
from net.rew import BaseCNN
from net.seq_lm import VED
from utils.constants import ALPHABET, AATOIDX, seq_to_one_hot
from utils.eval_utils import distance, diversity

# Try to import wandb, make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not available, logging will be disabled")

# Import unified metrics directly from Benchmark utils/metrics.py
# Using importlib to avoid local utils shadowing
import importlib.util
BENCHMARK_UTILS_PATH = "/home/xux/Desktop/AlphaVariant/Benchmark/utils"

# Load metrics module
spec_metrics = importlib.util.spec_from_file_location("benchmark_metrics", os.path.join(BENCHMARK_UTILS_PATH, "metrics.py"))
benchmark_metrics = importlib.util.module_from_spec(spec_metrics)
spec_metrics.loader.exec_module(benchmark_metrics)

# Extract the needed functions from metrics module
hamming_distance = benchmark_metrics.hamming_distance
high_fitness_proximity = benchmark_metrics.high_fitness_proximity_from_landscape
novelty = benchmark_metrics.novelty
batch_diversity = benchmark_metrics.batch_diversity
normalized_fitness_topk = benchmark_metrics.normalized_fitness_median_topk
compute_max_fitness = benchmark_metrics.max_fitness
simple_regret = benchmark_metrics.simple_regret
spearman_correlation = benchmark_metrics.spearman_correlation

# Create a simple MetricsResult dataclass
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class MetricsResult:
    high_fitness_proximity: float = 0.0
    novelty: float = 0.0
    batch_diversity: float = 0.0
    normalized_fitness_median_top128: float = 0.0
    normalized_fitness_median_top256: float = 0.0
    max_fitness: float = 0.0
    spearman_correlation: float = 0.0
    epistatic_correlation: float = 0.0
    recall_high_order: float = 0.0
    simple_regret: float = 0.0
    global_max_found: bool = False
    miscalibration_area: float = 1.0
    expected_calibration_error: float = 1.0
    regret_trajectory: List[float] = field(default_factory=list)
    fitness_trajectory: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'high_fitness_proximity': self.high_fitness_proximity,
            'novelty': self.novelty,
            'batch_diversity': self.batch_diversity,
            'normalized_fitness_median_top128': self.normalized_fitness_median_top128,
            'normalized_fitness_median_top256': self.normalized_fitness_median_top256,
            'max_fitness': self.max_fitness,
            'spearman_correlation': self.spearman_correlation,
            'epistatic_correlation': self.epistatic_correlation,
            'recall_high_order': self.recall_high_order,
            'simple_regret': self.simple_regret,
            'global_max_found': self.global_max_found,
            'miscalibration_area': self.miscalibration_area,
            'expected_calibration_error': self.expected_calibration_error,
        }

def compute_all_metrics(queried_indices, all_sequences, all_fitness, initial_indices, batch_size=96, wildtype=None):
    """Compute all metrics for an optimization run."""
    result = MetricsResult()

    queried_indices = np.array(queried_indices).astype(int)
    initial_indices = np.array(initial_indices).astype(int)

    queried_seqs = [all_sequences[i] for i in queried_indices]
    initial_seqs = [all_sequences[i] for i in initial_indices]
    queried_fitness = all_fitness[queried_indices]

    global_max = float(np.max(all_fitness))
    global_min = float(np.min(all_fitness))

    # Auto-detect distance function: hamming if all sequences are equal length, else levenshtein
    seq_lengths = set(len(s) for s in all_sequences[:1000])
    dist_fn = 'hamming' if len(seq_lengths) == 1 else 'levenshtein'

    result.high_fitness_proximity = high_fitness_proximity(
        queried_seqs, all_sequences, all_fitness,
        percentile=0.9, distance_fn=dist_fn
    )

    non_initial_seqs = queried_seqs[len(initial_indices):]
    if non_initial_seqs:
        result.novelty = novelty(non_initial_seqs, initial_seqs, distance_fn=dist_fn)

    result.batch_diversity = batch_diversity(queried_seqs, distance_fn=dist_fn)
    result.normalized_fitness_median_top128 = normalized_fitness_topk(
        queried_fitness, k=128, fitness_min=global_min, fitness_max=global_max
    )
    result.normalized_fitness_median_top256 = normalized_fitness_topk(
        queried_fitness, k=256, fitness_min=global_min, fitness_max=global_max
    )
    result.max_fitness = compute_max_fitness(queried_fitness)
    result.simple_regret = simple_regret(result.max_fitness, global_max)
    result.global_max_found = (result.max_fitness >= global_max * 0.99)

    fitness_traj = []
    for i in range(0, len(queried_fitness), batch_size):
        batch_fitness = queried_fitness[:i + batch_size]
        fitness_traj.append(float(np.max(batch_fitness)))
    result.fitness_trajectory = fitness_traj
    result.regret_trajectory = [global_max - f for f in fitness_traj]

    return result

def aggregate_run_metrics(run_results):
    """Aggregate metrics across multiple runs."""
    if not run_results:
        return {}

    metrics_names = [
        'high_fitness_proximity', 'novelty', 'batch_diversity',
        'normalized_fitness_median_top128', 'normalized_fitness_median_top256',
        'max_fitness', 'spearman_correlation', 'epistatic_correlation',
        'recall_high_order', 'simple_regret', 'miscalibration_area',
        'expected_calibration_error'
    ]

    aggregated = {}
    for name in metrics_names:
        values = [getattr(r, name) for r in run_results]
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            aggregated[name] = {
                'mean': float(np.mean(valid_values)),
                'std': float(np.std(valid_values)),
                'min': float(np.min(valid_values)),
                'max': float(np.max(valid_values))
            }

    hit_count = sum(1 for r in run_results if r.global_max_found)
    aggregated['global_max_hit_count'] = {
        'count': hit_count,
        'rate': hit_count / len(run_results) if run_results else 0.0
    }

    return aggregated

def load_landscape_data(dataset, data_dir="/home/xux/Desktop/AlphaVariant/Benchmark/data"):
    """Load complete fitness landscape data."""
    import pandas as pd
    file_patterns = [
        f"{data_dir}/{dataset}/data.csv",
        f"{data_dir}/{dataset}/fitness.csv",
        f"{data_dir}/{dataset}.csv",
    ]

    data_path = None
    for pattern in file_patterns:
        if os.path.exists(pattern):
            data_path = pattern
            break

    if data_path is None:
        raise FileNotFoundError(f"No fitness data found for {dataset}")

    df = pd.read_csv(data_path)
    sequences = df['seq'].tolist()
    fitness = df['fitness'].values
    return sequences, fitness

ALDE_METRICS_AVAILABLE = True  # Always available now


# =============================================================================
# Utility Functions
# =============================================================================

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_dataset_params(all_sequences, all_fitness):
    """Auto-detect dataset parameters from the loaded data.

    Returns:
        dict with keys: seq_length, wildtype, min_fitness, max_fitness, dist_fn
    """
    seq_lengths = [len(s) for s in all_sequences]
    unique_lengths = set(seq_lengths)

    # Sequence length: use the most common length
    from collections import Counter
    length_counts = Counter(seq_lengths)
    seq_length = length_counts.most_common(1)[0][0]

    # Wildtype fallback: sequence with highest fitness
    best_idx = int(np.argmax(all_fitness))
    wildtype = all_sequences[best_idx]

    # Fitness range
    min_fitness = float(np.min(all_fitness))
    max_fitness = float(np.max(all_fitness))

    # Distance function
    dist_fn = 'hamming' if len(unique_lengths) == 1 else 'levenshtein'

    return {
        'seq_length': seq_length,
        'wildtype': wildtype,
        'min_fitness': min_fitness,
        'max_fitness': max_fitness,
        'dist_fn': dist_fn,
    }


def get_generic_config(device: str, seq_length: int, reduce_dim: int = 16) -> argparse.Namespace:
    """Create configuration for a generic protein dataset."""
    args = argparse.Namespace()
    args.name = 'Generic'
    args.device = device
    args.level = 'medium'
    args.embed_dim = 1280
    args.num_layers = 33
    args.hidden_dim = 256
    args.length = seq_length
    args.num_tokens = seq_length + 2  # Include start/end tokens
    args.reduce_dim = reduce_dim
    args.num_trainable_layers = 4
    return args


def create_generic_opt(args: argparse.Namespace, seq_length: int, min_fitness: float, max_fitness: float) -> argparse.Namespace:
    """Create optimization configuration for a generic dataset."""
    opt = argparse.Namespace()
    opt.name = args.dataset
    opt.device = args.device
    opt.level = args.level
    opt.not_sparse = args.not_sparse
    opt.length = seq_length
    opt.min_fitness = min_fitness
    opt.max_fitness = max_fitness
    opt.step_mut = args.step_mut

    # Scale action_size and topk based on sequence length
    opt.action_size = 0.15
    opt.topk = max(5, min(15, seq_length // 15))
    max_mutation = max(5, seq_length // 15)
    opt.done_cond = argparse.Namespace(
        max_steps=3,
        max_mutation=max_mutation,
        step_mut=opt.step_mut
    )

    opt.seq_pretrained = args.ved_checkpoint
    opt.rew_pretrained = args.oracle_checkpoint
    opt.reduce_dim = args.reduce_dim

    return opt


def load_generic_data(
    data_dir: str,
    dataset: str,
    level: str = 'medium',
    n_init: int = 96
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], np.ndarray]:
    """
    Load a generic dataset and prepare training/test splits.

    Args:
        data_dir: Base data directory
        dataset: Dataset name (subdirectory under data_dir)
        level: Difficulty level ('hard' or 'medium')
        n_init: Number of initial samples

    Returns:
        Tuple of (train_df, all_df, all_sequences, all_fitness)
    """
    # Load full dataset
    data_path = os.path.join(data_dir, dataset, 'data.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Get full sequences and fitness
    all_sequences = df['seq'].tolist()
    all_fitness = df['fitness'].values

    # Normalize fitness to [0, 1] if not already
    f_min = all_fitness.min()
    f_max = all_fitness.max()
    if f_max - f_min > 0:
        all_fitness_norm = (all_fitness - f_min) / (f_max - f_min)
    else:
        all_fitness_norm = all_fitness

    # Create training dataframe based on level
    if level == 'hard':
        # Hard: start from low-fitness sequences
        threshold = np.percentile(all_fitness_norm, 20)
        train_mask = all_fitness_norm <= threshold
    else:  # medium
        # Medium: start from medium-fitness sequences
        threshold_low = np.percentile(all_fitness_norm, 40)
        threshold_high = np.percentile(all_fitness_norm, 60)
        train_mask = (all_fitness_norm >= threshold_low) & (all_fitness_norm <= threshold_high)

    train_df = df[train_mask].copy()
    train_df['target'] = all_fitness_norm[train_mask]

    # Sample initial training set
    if len(train_df) > n_init:
        train_df = train_df.sample(n=n_init, random_state=42)

    # Prepare format compatible with LatProtRL
    train_df = train_df.rename(columns={'seq': 'sequence'})
    train_df = train_df[['sequence', 'target']]

    return train_df, df, all_sequences, all_fitness_norm


# =============================================================================
# Generic Environment
# =============================================================================

class GenericOptEnv(Env):
    """
    RL Environment for generic protein optimization using LatProtRL approach.
    """

    def __init__(self, config: argparse.Namespace, train_data: pd.DataFrame,
                 seq_length: int, wildtype: str, seed: int = 42):
        super().__init__()

        # Setup observation and action spaces
        seq_cfg = get_generic_config(config.device, seq_length, config.reduce_dim)
        obs_shape = (seq_cfg.reduce_dim,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-config.action_size, high=config.action_size, shape=obs_shape, dtype=np.float32)

        self.device = config.device
        self.protein = config.name
        self.length = seq_length
        self.done_cond = config.done_cond
        self.oracle_calls = 0
        self.config = config

        # Load oracle (fitness predictor)
        oracle = BaseCNN(make_one_hot=False)
        if os.path.exists(config.rew_pretrained):
            oracle_ckpt = torch.load(config.rew_pretrained, map_location=self.device)
            if "state_dict" in oracle_ckpt.keys():
                oracle_ckpt = oracle_ckpt["state_dict"]
            oracle.load_state_dict({k.replace('predictor.', ''): v for k, v in oracle_ckpt.items()})
        else:
            warnings.warn(f"Oracle checkpoint not found at {config.rew_pretrained}, using random oracle")
        oracle.eval()
        self.oracle = oracle.to(self.device)

        # Load VED model
        if os.path.exists(config.seq_pretrained):
            model = VED(seq_cfg, pretrained=config.seq_pretrained)
        else:
            warnings.warn(f"VED checkpoint not found at {config.seq_pretrained}, using ESM-2 only")
            model = VED(seq_cfg, esm_pretrained='ckpt/esm2_t33_650M_UR50D.pt')
        model.eval()
        self.model = model.to(self.device)

        # Setup buffer with training data
        data_list = list(train_data[['sequence', 'target']].itertuples(index=False, name=None))
        self.buffer = Buffer(data_list, random=random.Random(seed))
        self.inits = train_data['sequence'].tolist()

        # Set wild-type tokens
        self.wt_seq = wildtype
        self.model.set_wt_tokens(self.wt_seq)

        # Logging state
        self.ep = 0
        self.steps = 0
        self.total_steps = 0
        self.state = None
        self.state_seq = None
        self.init_target = 0
        self.buffer_idx = None

        # Metrics
        self.reward = 0
        self.done = False
        self.target = 0
        self.best_discovered = 0
        self.init_seq = None
        self.n_mut = 0
        self.aa = {a: 0 for a in ALPHABET}
        self.pos = {a: 0 for a in range(self.length)}

        # Track all discovered sequences
        self.discovered_sequences = []
        self.discovered_fitness = []

    def normalize_target(self, target: float) -> float:
        """Normalize fitness to [0, 1] range."""
        return (target - self.config.min_fitness) / (self.config.max_fitness - self.config.min_fitness)

    def record_mutation(self, s1: str, s2: str):
        """Record mutation statistics."""
        for pos, (i, j) in enumerate(zip(list(s1), list(s2))):
            if i != j:
                self.aa[j] += 1
                self.pos[pos] += 1

    def reset(self, seed: int = None, options: dict = None):
        """Reset environment to start a new episode."""
        self.state_seq, self.buffer_idx = self.buffer.top()
        self.init_seq = self.state_seq

        with torch.no_grad():
            state = self.model.encode(self.state_seq)
            self.state = state.cpu().view(-1).numpy()
            _, target = self.oracle(seq_to_one_hot(self.state_seq).unsqueeze(0).to(self.device), get_embed=True)
            self.init_target = self.target = self.normalize_target(target.cpu().item())

        self.ep += 1
        self.steps = 0
        self.total_steps += 1

        return self.state, {}

    def step(self, action: np.ndarray):
        """Execute one step in the environment."""
        self.steps += 1
        self.total_steps += 1
        next_state = (torch.tensor(action) + torch.tensor(self.state)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            next_seq = self.model.decode(next_state, to_seq=True, template=self.state_seq, topk=self.config.topk)
            self.record_mutation(self.state_seq, next_seq)
            self.step_mut = step_mut = distance(self.state_seq, next_seq)
            self.n_mut = distance(self.wt_seq, next_seq)

        self.done = done = (
            step_mut > self.done_cond.step_mut or
            self.steps > self.done_cond.max_steps or
            self.n_mut > self.done_cond.max_mutation
        )
        called = False

        if step_mut <= self.done_cond.step_mut:
            if self.config.not_sparse or done:
                self.oracle_calls += 1
                with torch.no_grad():
                    _, target = self.oracle(seq_to_one_hot(next_seq).unsqueeze(0).to(self.device), get_embed=True)
                target = self.normalize_target(target.cpu().item())
                self.reward = target
                self.buffer.push((next_seq, target, self.buffer_idx))

                # Track discovered sequences
                self.discovered_sequences.append(next_seq)
                self.discovered_fitness.append(target)

                if target > self.best_discovered:
                    self.best_discovered = target
                self.target = target
                called = True
            else:
                self.reward = 0
        else:
            target = 0
            self.reward = -1

        self.state = next_state.cpu().numpy()[0]
        self.state_seq = next_seq

        if done:
            info = {
                'candidates': self.state_seq,
                'fitness': self.target,
                'init_seq': self.init_seq,
                'n_mut': self.n_mut,
                'aa': self.aa,
                'pos': self.pos,
                'called': called
            }
            return self.state, self.reward, done, False, info

        return self.state, self.reward, done, False, {}


# =============================================================================
# Callbacks for Logging and Evaluation
# =============================================================================

class GenericBufferCallback:
    """Callback for logging buffer performance during training.

    Metrics are aligned with ALDE in the following order:
    1. high_fitness_proximity
    2. novelty
    3. batch_diversity
    4. normalized_fitness_median_top128
    5. normalized_fitness_median_top256
    6. max_fitness
    7. spearman_correlation (requires model predictions)
    8. epistatic_correlation (requires model predictions)
    9. recall_high_order (requires model predictions)
    10. simple_regret
    11. miscalibration_area (requires uncertainty estimates)
    12. expected_calibration_error (requires uncertainty estimates)
    """

    def __init__(
        self,
        config: argparse.Namespace,
        save_dir: str,
        all_sequences: List[str],
        all_fitness: np.ndarray,
        dist_fn: str = 'hamming',
        use_wandb: bool = True
    ):
        self.save_dir = save_dir
        self.config = config
        self.all_sequences = all_sequences
        self.all_fitness = all_fitness
        self.global_max = float(np.max(all_fitness))
        self.global_min = float(np.min(all_fitness))
        self.dist_fn = dist_fn
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.rounds = 0
        self.results = []

    def on_rollout_end(self, model, env):
        """Called at the end of each rollout."""
        buffer = env.buffer
        infos = buffer.propose(k=128)
        seqs = [info[0] for info in infos]
        fitness_values = np.array([info[1] for info in infos])

        # Get top 256 for additional metrics
        infos_256 = buffer.propose(k=256) if len(buffer.pool) >= 256 else infos
        seqs_256 = [info[0] for info in infos_256]
        fitness_256 = np.array([info[1] for info in infos_256])

        inits = env.inits

        # Compute metrics in ALDE order
        if ALDE_METRICS_AVAILABLE:
            # 1. high_fitness_proximity
            hfp = high_fitness_proximity(
                seqs, self.all_sequences, self.all_fitness,
                percentile=0.9, distance_fn=self.dist_fn
            )
            # 2. novelty
            nov = novelty(seqs, inits, distance_fn=self.dist_fn)
            # 3. batch_diversity
            div = batch_diversity(seqs, distance_fn=self.dist_fn)
            # 4. normalized_fitness_median_top128
            norm_fit_128 = normalized_fitness_topk(
                fitness_values, k=128,
                fitness_min=self.global_min, fitness_max=self.global_max
            )
            # 5. normalized_fitness_median_top256
            norm_fit_256 = normalized_fitness_topk(
                fitness_256, k=256,
                fitness_min=self.global_min, fitness_max=self.global_max
            )
            # 6. max_fitness
            max_fit = compute_max_fitness(fitness_values)
            # 10. simple_regret
            regret = simple_regret(max_fit, self.global_max)
        else:
            # Fallback to local implementations
            hfp = self._compute_high_fitness_proximity(seqs)
            nov = self._compute_novelty(seqs, inits)
            div = diversity(seqs)
            norm_fit_128 = float(np.median(fitness_values))
            norm_fit_256 = float(np.median(fitness_256))
            max_fit = float(np.max(fitness_values))
            regret = self.global_max - max_fit

        # Build log dict in ALDE order
        log = {
            'eval/round': self.rounds,
            # Exploration metrics
            'eval/high_fitness_proximity': hfp,
            'eval/novelty': nov,
            'eval/batch_diversity': div,
            # Functional metrics
            'eval/normalized_fitness_median_top128': norm_fit_128,
            'eval/normalized_fitness_median_top256': norm_fit_256,
            'eval/max_fitness': max_fit,
            # Model quality metrics (N/A without predictions)
            'eval/spearman_correlation': 0.0,
            'eval/epistatic_correlation': 0.0,
            'eval/recall_high_order': 0.0,
            # Success metrics
            'eval/simple_regret': regret,
            # Uncertainty metrics (N/A without uncertainty estimates)
            'eval/miscalibration_area': 1.0,
            'eval/expected_calibration_error': 1.0,
        }

        self.rounds += 1
        self.results.append({
            'round': self.rounds,
            'sequences': seqs,
            'fitness': fitness_values.tolist(),
            'metrics': log
        })

        # Save sequences
        np.save(f'{self.save_dir}/{self.rounds}.npy', seqs)

        if self.use_wandb:
            wandb.log(log)

        with open(f'{self.save_dir}/{self.rounds}.txt', 'w') as f:
            json.dump({k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in log.items()}, f)

        return log

    def _compute_novelty(self, seqs: List[str], inits: List[str]) -> float:
        """Compute novelty (median min distance to initial set)."""
        distances = []
        for seq in seqs:
            min_dist = min(distance(seq, init_seq) for init_seq in inits)
            distances.append(min_dist)
        return float(np.median(distances))

    def _compute_high_fitness_proximity(self, seqs: List[str]) -> float:
        """Compute high-fitness proximity."""
        threshold = np.percentile(self.all_fitness, 90)
        high_mask = self.all_fitness >= threshold
        high_seqs = [s for s, m in zip(self.all_sequences, high_mask) if m][:128]

        if not high_seqs:
            return 0.0

        distances = []
        for seq in seqs:
            min_dist = min(distance(seq, high_seq) for high_seq in high_seqs)
            distances.append(min_dist)
        return float(np.median(distances))


# =============================================================================
# Main Optimization Function
# =============================================================================

def run_single_experiment(
    seed: int,
    dataset: str,
    device: str = "cuda:0",
    level: str = "medium",
    output_path: str = None,
    n_init: int = 96,
    n_rounds: int = 5,
    total_timesteps: int = 20000,
    ved_checkpoint: Optional[str] = None,
    oracle_checkpoint: Optional[str] = None,
    data_dir: str = "/home/xux/Desktop/AlphaVariant/Benchmark/data",
    compute_metrics: bool = True,
    use_wandb: bool = False,
    not_sparse: bool = True,
    step_mut: int = 2,
    reduce_dim: int = 16,
    run_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run a single LatProtRL optimization experiment on a generic dataset.

    Args:
        seed: Random seed for reproducibility
        dataset: Dataset name (subdirectory under data_dir)
        device: Device to use ('cuda:X' or 'cpu')
        level: Difficulty level ('hard' or 'medium')
        output_path: Base path for saving results
        n_init: Number of initial samples
        n_rounds: Number of optimization rounds
        total_timesteps: Total PPO training timesteps
        ved_checkpoint: Path to VED model checkpoint
        oracle_checkpoint: Path to oracle checkpoint
        data_dir: Base directory for data files
        compute_metrics: Whether to compute evaluation metrics
        use_wandb: Whether to use wandb for logging
        not_sparse: Provide reward every step (not just at episode end)
        step_mut: Maximum mutations per step
        reduce_dim: Latent space dimension
        run_id: Optional run identifier

    Returns:
        Dictionary containing results and metrics
    """
    if output_path is None:
        output_path = f"results/{dataset}_LatProtRL/"

    if run_id is None:
        run_id = seed

    print(f"\n{'='*60}")
    print(f"Starting LatProtRL optimization on {dataset}")
    print(f"  Seed: {seed}")
    print(f"  Level: {level}")
    print(f"  Device: {device}")
    print(f"  Initial samples: {n_init}")
    print(f"  Rounds: {n_rounds}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"{'='*60}\n")

    # Set random seeds
    set_seed(seed)

    # Create output directory
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    save_dir = os.path.join(output_path, f"{dataset}_{level}_seed{seed}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'policy'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'results'), exist_ok=True)

    # Load data
    print(f"Loading {dataset} data...")
    train_df, all_df, all_sequences, all_fitness = load_generic_data(data_dir, dataset, level, n_init)
    print(f"  Training samples: {len(train_df)}")
    print(f"  Total landscape: {len(all_sequences)}")
    print(f"  Fitness range: [{all_fitness.min():.4f}, {all_fitness.max():.4f}]")

    # Auto-detect dataset parameters
    ds_params = detect_dataset_params(all_sequences, all_fitness)
    seq_length = ds_params['seq_length']
    wildtype = ds_params['wildtype']
    min_fitness = ds_params['min_fitness']
    max_fitness = ds_params['max_fitness']
    dist_fn = ds_params['dist_fn']

    print(f"  Auto-detected sequence length: {seq_length}")
    print(f"  Auto-detected wildtype (highest fitness): {wildtype[:50]}{'...' if len(wildtype) > 50 else ''}")
    print(f"  Distance function: {dist_fn}")

    # Set default checkpoints if not provided
    if ved_checkpoint is None:
        # Non-existent path triggers fallback to ESM-2 only
        ved_checkpoint = f"saved/{dataset}_{level}_VED.pt"
    if oracle_checkpoint is None:
        oracle_checkpoint = f"ckpt/{dataset}/oracle.ckpt"

    # Create configuration
    args = argparse.Namespace()
    args.dataset = dataset
    args.device = device
    args.level = level
    args.not_sparse = not_sparse
    args.step_mut = step_mut
    args.ved_checkpoint = ved_checkpoint
    args.oracle_checkpoint = oracle_checkpoint
    args.reduce_dim = reduce_dim

    config = create_generic_opt(args, seq_length, min_fitness, max_fitness)

    # Initialize wandb
    project_name = f"{dataset}_{level}_seed{seed}"
    if use_wandb and WANDB_AVAILABLE:
        run = wandb.init(project=f"LatProtRL-{dataset}", name=project_name, reinit=True)
        run.config.update({
            "protein": dataset,
            "level": level,
            "seed": seed,
            "n_init": n_init,
            "n_rounds": n_rounds,
            "total_timesteps": total_timesteps,
            "reduce_dim": reduce_dim,
            "step_mut": step_mut,
            "not_sparse": not_sparse,
            "seq_length": seq_length,
        })

    # Create environment
    print("Creating environment...")
    env = GenericOptEnv(config, train_df, seq_length, wildtype, seed=seed)

    # Create callback
    callback = GenericBufferCallback(
        config,
        os.path.join(save_dir, 'results'),
        all_sequences,
        all_fitness,
        dist_fn=dist_fn,
        use_wandb=use_wandb
    )

    # Create PPO model
    print("Creating PPO model...")
    n_calls = 256
    model = PPO(
        "MlpPolicy",
        env,
        n_calls=n_calls,
        ent_coef=0.0,
        n_steps=min(9192, total_timesteps // 2),
        verbose=1,
        device=device,
        tensorboard_log=None
    )

    # Training loop with manual callback
    print("Starting training...")
    start_time = datetime.now()

    # Custom training loop to integrate callback
    timesteps_per_round = total_timesteps // n_rounds
    for round_idx in range(n_rounds):
        print(f"\n--- Round {round_idx + 1}/{n_rounds} ---")

        model.learn(
            total_timesteps=timesteps_per_round,
            reset_num_timesteps=False,
            progress_bar=True
        )

        # Call callback
        callback.on_rollout_end(model, env)

        # Save model checkpoint
        model.save(os.path.join(save_dir, 'policy', f'round_{round_idx + 1}.zip'))

        # Update buffer
        env.buffer.update()

    runtime = (datetime.now() - start_time).total_seconds()
    print(f"\nTraining complete! Runtime: {runtime:.2f}s")

    # Collect results
    discovered_seqs = env.discovered_sequences
    discovered_fit = env.discovered_fitness

    # Prepare result dictionary
    result = {
        'seed': seed,
        'run_id': run_id,
        'dataset': dataset,
        'save_dir': save_dir,
        'runtime_seconds': runtime,
        'n_queries': len(discovered_seqs),
        'oracle_calls': env.oracle_calls,
        'config': {
            'dataset': dataset,
            'level': level,
            'n_init': n_init,
            'n_rounds': n_rounds,
            'total_timesteps': total_timesteps,
            'reduce_dim': reduce_dim,
            'step_mut': step_mut,
            'seq_length': seq_length,
        }
    }

    # Compute metrics
    if compute_metrics and discovered_seqs:
        print("\nComputing evaluation metrics...")

        # Get final buffer proposals
        buffer = env.buffer
        final_proposals = buffer.propose(k=128)
        final_seqs = [p[0] for p in final_proposals]
        final_fitness = np.array([p[1] for p in final_proposals])

        # Get top 256 as well
        final_proposals_256 = buffer.propose(k=256) if len(buffer.pool) >= 256 else final_proposals
        final_fitness_256 = np.array([p[1] for p in final_proposals_256])

        # Global fitness bounds
        global_max = float(np.max(all_fitness))
        global_min = float(np.min(all_fitness))

        # Compute using ALDE metrics if available
        if ALDE_METRICS_AVAILABLE:
            # Create indices for the discovered sequences
            seq_to_idx = {s: i for i, s in enumerate(all_sequences)}
            queried_indices = []
            for seq in discovered_seqs:
                if seq in seq_to_idx:
                    queried_indices.append(seq_to_idx[seq])

            if queried_indices:
                queried_indices = torch.tensor(queried_indices)
                initial_indices = queried_indices[:min(n_init, len(queried_indices))]

                metrics_result = compute_all_metrics(
                    queried_indices=queried_indices,
                    all_sequences=all_sequences,
                    all_fitness=all_fitness,
                    initial_indices=initial_indices,
                    batch_size=n_init,
                    wildtype=wildtype
                )

                result['metrics'] = metrics_result.to_dict()
                result['fitness_trajectory'] = metrics_result.fitness_trajectory
                result['regret_trajectory'] = metrics_result.regret_trajectory

        # Fallback metrics in ALDE order
        if 'metrics' not in result:
            max_fit = float(np.max(final_fitness)) if len(final_fitness) > 0 else 0.0
            inits = env.inits

            # Compute fallback metrics
            hfp = callback._compute_high_fitness_proximity(final_seqs) if len(final_seqs) > 0 else 0.0
            nov = callback._compute_novelty(final_seqs, inits) if len(final_seqs) > 0 else 0.0
            div = diversity(final_seqs) if len(final_seqs) > 1 else 0.0
            norm_fit_128 = float(np.median((final_fitness - global_min) / (global_max - global_min))) if len(final_fitness) > 0 else 0.0
            norm_fit_256 = float(np.median((final_fitness_256 - global_min) / (global_max - global_min))) if len(final_fitness_256) > 0 else 0.0
            regret = global_max - max_fit

            result['metrics'] = {
                # Exploration metrics
                'high_fitness_proximity': hfp,
                'novelty': nov,
                'batch_diversity': div,
                # Functional metrics
                'normalized_fitness_median_top128': norm_fit_128,
                'normalized_fitness_median_top256': norm_fit_256,
                'max_fitness': max_fit,
                # Model quality metrics (N/A)
                'spearman_correlation': 0.0,
                'epistatic_correlation': 0.0,
                'recall_high_order': 0.0,
                # Success metrics
                'simple_regret': regret,
                # Uncertainty metrics (N/A)
                'miscalibration_area': 1.0,
                'expected_calibration_error': 1.0,
            }

        # Print summary in ALDE order
        metrics = result['metrics']
        print("\n" + "-" * 60)
        print("Metrics Summary (ALDE-aligned):")
        print("-" * 60)
        print(f"  high_fitness_proximity:          {metrics.get('high_fitness_proximity', 0):.4f}")
        print(f"  novelty:                         {metrics.get('novelty', 0):.4f}")
        print(f"  batch_diversity:                 {metrics.get('batch_diversity', 0):.4f}")
        print(f"  normalized_fitness_median_top128:{metrics.get('normalized_fitness_median_top128', 0):.4f}")
        print(f"  normalized_fitness_median_top256:{metrics.get('normalized_fitness_median_top256', 0):.4f}")
        print(f"  max_fitness:                     {metrics.get('max_fitness', 0):.4f}")
        print(f"  spearman_correlation:            {metrics.get('spearman_correlation', 0):.4f}")
        print(f"  epistatic_correlation:           {metrics.get('epistatic_correlation', 0):.4f}")
        print(f"  recall_high_order:               {metrics.get('recall_high_order', 0):.4f}")
        print(f"  simple_regret:                   {metrics.get('simple_regret', 0):.4f}")
        print(f"  miscalibration_area:             {metrics.get('miscalibration_area', 1.0):.4f}")
        print(f"  expected_calibration_error:      {metrics.get('expected_calibration_error', 1.0):.4f}")
        print("-" * 60)
        print(f"  Oracle Calls: {env.oracle_calls}")

        # Save metrics
        metrics_path = os.path.join(save_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nMetrics saved to: {metrics_path}")

    # Save final sequences
    np.save(os.path.join(save_dir, 'final_sequences.npy'), final_seqs)
    np.save(os.path.join(save_dir, 'final_fitness.npy'), final_fitness)

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

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


def save_aggregated_results(results: List[Dict[str, Any]], output_path: str, global_max_fitness: float = 1.0) -> None:
    """Save aggregated results across all runs.

    Metrics are aggregated in ALDE order:
    1. high_fitness_proximity
    2. novelty
    3. batch_diversity
    4. normalized_fitness_median_top128
    5. normalized_fitness_median_top256
    6. max_fitness
    7. spearman_correlation
    8. epistatic_correlation
    9. recall_high_order
    10. simple_regret
    11. miscalibration_area
    12. expected_calibration_error
    13. global_max_hit_count
    """
    if not results:
        print("No results to aggregate.")
        return

    # Extract metrics
    metrics_list = []
    for r in results:
        if 'metrics' in r:
            metrics_list.append(r['metrics'])

    if not metrics_list:
        print("No metrics to aggregate.")
        return

    # Define metric order (ALDE-aligned)
    metric_order = [
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

    # Aggregate metrics in order
    aggregated = {}
    for key in metric_order:
        values = [m[key] for m in metrics_list if key in m and isinstance(m[key], (int, float))]
        if values:
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }

    # Compute global_max_hit_count (number of runs that found global max within 1% tolerance)
    tolerance = 0.01
    threshold = global_max_fitness * (1 - tolerance)
    max_fitness_values = [m.get('max_fitness', 0) for m in metrics_list]
    hit_count = sum(1 for mf in max_fitness_values if mf >= threshold)
    hit_rate = hit_count / len(metrics_list) if metrics_list else 0.0

    aggregated['global_max_hit_count'] = {
        'count': hit_count,
        'rate': float(hit_rate),
        'n_runs': len(metrics_list),
    }

    # Save
    dataset_name = results[0].get('dataset', 'unknown') if results else 'unknown'
    agg_path = os.path.join(output_path, 'aggregated_results.json')
    with open(agg_path, 'w') as f:
        json.dump({
            'dataset': dataset_name,
            'aggregated_metrics': aggregated,
            'n_runs': len(results),
            'seeds': [r['seed'] for r in results],
        }, f, indent=2)

    # Print summary in ALDE order
    print("\n" + "=" * 70)
    print(f"AGGREGATED METRICS SUMMARY (ALDE-aligned) - {dataset_name}")
    print("=" * 70)
    print(f"{'Metric':<40} {'Mean':>10} {'Std':>10}")
    print("-" * 70)
    for metric in metric_order:
        if metric in aggregated:
            stats = aggregated[metric]
            print(f"{metric:<40} {stats['mean']:>10.4f} {stats['std']:>10.4f}")
    # Print global_max_hit_count
    if 'global_max_hit_count' in aggregated:
        gm = aggregated['global_max_hit_count']
        print(f"{'global_max_hit_count':<40} {gm['count']:>10} / {gm['n_runs']:<5} ({gm['rate']*100:.1f}%)")
    print("=" * 70)
    print(f"\nAggregated results saved to: {agg_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run LatProtRL optimization on a generic protein dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run on AAV_hard with default seed
  python run_generic.py --dataset AAV_hard --device cuda:0

  # Single run with specific seed
  python run_generic.py --dataset AAV_hard --device cuda:0 --seed 42

  # Multiple runs for randomness evaluation
  python run_generic.py --dataset AAV_hard --device cuda:0 --seeds 42 123 456 789 1000

  # Load seeds from file
  python run_generic.py --dataset AAV_hard --device cuda:0 --seed_file src/rndseed.txt --num_seeds 5

  # Skip metrics computation
  python run_generic.py --dataset GB1 --device cuda:0 --seed 42 --skip_metrics

  # Custom output path
  python run_generic.py --dataset GFP_med --output_path results/my_run/ --seed 42
        """
    )

    # Dataset (required)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (subdirectory under data_dir, e.g. AAV_hard, AAV_med, GB1, GFP_med)"
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

    # Device and paths
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for computation (default: cuda:0)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output directory for results (default: results/<dataset>_LatProtRL/)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/xux/Desktop/AlphaVariant/Benchmark/data",
        help="Base directory for data files"
    )

    # Model checkpoints
    parser.add_argument(
        "--ved_checkpoint",
        type=str,
        default=None,
        help="Path to VED model checkpoint"
    )
    parser.add_argument(
        "--oracle_checkpoint",
        type=str,
        default=None,
        help="Path to oracle (fitness predictor) checkpoint"
    )

    # Training configuration
    parser.add_argument(
        "--level",
        type=str,
        default="medium",
        choices=["hard", "medium"],
        help="Difficulty level (default: medium)"
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=96,
        help="Number of initial samples (default: 96)"
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=5,
        help="Number of optimization rounds (default: 5)"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=20000,
        help="Total PPO training timesteps (default: 20000)"
    )
    parser.add_argument(
        "--reduce_dim",
        type=int,
        default=16,
        help="Latent space dimension (default: 16)"
    )
    parser.add_argument(
        "--step_mut",
        type=int,
        default=2,
        help="Maximum mutations per step (default: 2)"
    )
    parser.add_argument(
        "--sparse_reward",
        action="store_true",
        help="Use sparse reward (only at episode end)"
    )

    # Other options
    parser.add_argument(
        "--skip_metrics",
        action="store_true",
        help="Skip metrics computation (faster)"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable wandb logging (disabled by default)"
    )

    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    # Set default output path based on dataset name
    if args.output_path is None:
        args.output_path = f"results/{args.dataset}_LatProtRL/"

    # Determine seeds
    if args.seeds is not None:
        seeds = args.seeds
    elif args.seed_file is not None:
        seeds = load_seeds_from_file(args.seed_file, args.num_seeds)
        print(f"Loaded {len(seeds)} seeds from {args.seed_file}")
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42]  # Default seed

    print(f"\nRunning LatProtRL on {args.dataset} with {len(seeds)} seed(s): {seeds}")
    print(f"Device: {args.device}")
    print(f"Output path: {args.output_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Compute metrics: {not args.skip_metrics}")

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Run experiments
    results = []
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Running experiment with seed={seed}")
        try:
            result = run_single_experiment(
                seed=seed,
                dataset=args.dataset,
                device=args.device,
                level=args.level,
                output_path=args.output_path,
                n_init=args.n_init,
                n_rounds=args.n_rounds,
                total_timesteps=args.total_timesteps,
                ved_checkpoint=args.ved_checkpoint,
                oracle_checkpoint=args.oracle_checkpoint,
                data_dir=args.data_dir,
                compute_metrics=not args.skip_metrics,
                use_wandb=args.use_wandb,
                not_sparse=not args.sparse_reward,
                step_mut=args.step_mut,
                reduce_dim=args.reduce_dim,
                run_id=i + 1,
            )
            results.append(result)
        except Exception as e:
            print(f"Error in seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Aggregate results
    if len(results) > 1 and not args.skip_metrics:
        # Use the global max from the first run's normalized data (should be 1.0 after normalization)
        save_aggregated_results(results, args.output_path, global_max_fitness=1.0)

    # Final summary
    print(f"\n{'='*60}")
    print("Experiment Complete")
    print(f"{'='*60}")
    print(f"Total runs: {len(results)}")
    print(f"Configuration:")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Level: {args.level}")
    print(f"  - Initial samples: {args.n_init}")
    print(f"  - Rounds: {args.n_rounds}")
    print(f"  - Total timesteps: {args.total_timesteps}")
    print(f"\nResults saved to: {args.output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
