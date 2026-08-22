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
    - Rounds: 5 (benchmark-standard -> 480 queries)

Iterative Training Process:
    - Round 1: Initial sampling from configurable fitness region
    - Rounds 2-5: GPT-guided sampling, get ground truth fitness, update surrogate, train GPT

Live datasets: 4site_GB1, 4site_PhoQ, 4site_TRPB (lookup landscape) and
ms_AAV, ms_CreiLOV, ms_PAB1 (CNN oracle, needs --oracle + --prior_model_path).

Usage:
    Run from the alphavariant/ package dir — --config, --prior_model_path and
    --output_path are resolved relative to the working directory:

        cd alphavariant
        PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python

    # Four-site (lookup landscape)
    $PY ../scripts/alphavariant/run_generic.py --dataset 4site_GB1 --seed 621

    # Multi-site (CNN oracle, generative proposal over the varying positions)
    $PY ../scripts/alphavariant/run_generic.py --dataset ms_CreiLOV --seed 621 \
        --oracle --prior_model_path priors/ms_CreiLOV/prior_model.pt

    # Custom dataset (data/<name>/data.csv must exist with seq,fitness columns)
    $PY ../scripts/alphavariant/run_generic.py --dataset my_protein --seed 621

    # Hard-level initialization (bottom 20th percentile)
    $PY ../scripts/alphavariant/run_generic.py --dataset 4site_PhoQ --level hard --seed 621

    # Override the auto-generated config with an existing YAML
    $PY ../scripts/alphavariant/run_generic.py --dataset 4site_GB1 \
        --config examples/Savinase/config/train_agent_config.yaml

    # Multiple seeds
    $PY ../scripts/alphavariant/run_generic.py --dataset ms_CreiLOV --seeds 621 100 383

    # Load seeds from the shared benchmark seed file (30-seed standard)
    $PY ../scripts/alphavariant/run_generic.py --dataset 4site_GB1 \
        --seed_file ../rand_seeds.txt --num_seeds 30

    # Skip metrics computation (faster iteration)
    $PY ../scripts/alphavariant/run_generic.py --dataset 4site_GB1 --seed 621 --skip_metrics
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

# Canonical location is scripts/<Method>/. This file used to be invoked through a
# symlink in <Method>/, where `__file__/..` happened to be the benchmark root; walk
# up to find the root instead, so it runs correctly from either path.
# (Same idiom as scripts/EVOLVEpro/run_generic.py.)
_p = os.path.dirname(os.path.realpath(__file__))
while os.path.dirname(_p) != _p and not os.path.isdir(os.path.join(_p, 'utils')):
    _p = os.path.dirname(_p)
BENCHMARK_ROOT = _p
sys.path.insert(0, BENCHMARK_ROOT)
sys.path.insert(0, os.path.join(BENCHMARK_ROOT, 'alphavariant'))  # upstream package lives in the method dir

from popgen.model.gpt import GPT, GPTConfig, save_gpt_model, save_gpt_config
from popgen.utils.utils import set_random_seed, parse_config, read_fasta_as_list, load_hotspot
from popgen.utils.template import PDETemplate
from popgen.utils.dataset import AASeqDictionary, rnn_start_token_vector
from popscorer.scoring_functions import ScoringFunctions, BonusFunctions

# Import unified metrics from utils.compat (BENCHMARK_ROOT is already on sys.path)
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
# Dataset helpers (multi-objective scalarization + wildtype resolution)
# =============================================================================

def _scalarized_fitness(df: 'pd.DataFrame') -> np.ndarray:
    """Return the per-row fitness array, scalarizing blue+red for *_joint datasets."""
    if 'fitness' in df.columns:
        return df['fitness'].values.astype(float)
    if 'blue' in df.columns and 'red' in df.columns:
        b = df['blue'].values.astype(float)
        r = df['red'].values.astype(float)
        return np.sqrt(np.clip(b, 0, None) * np.clip(r, 0, None))  # Clip negatives to zero before sqrt
    raise ValueError(f"No fitness column and no (blue, red) pair in dataframe; cols={df.columns.tolist()}")


def _resolve_wildtype(dataset_dir: str, df: 'pd.DataFrame',
                      sequences: List[str], fitness: np.ndarray) -> str:
    """Resolve WT non-leakily: wt.fasta -> n_muts==0 -> argmax(fitness) (warning)."""
    wt_fasta = os.path.join(dataset_dir, "wt.fasta")
    if os.path.exists(wt_fasta):
        with open(wt_fasta) as fh:
            lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith('>')]
        if lines:
            return "".join(lines)
    if 'n_muts' in df.columns and (df['n_muts'] == 0).any():
        wt_idx = int(df.index[df['n_muts'] == 0][0])
        return sequences[wt_idx]
    logger.warning(
        f"No wt.fasta or n_muts==0 row in {dataset_dir}; falling back to "
        f"argmax(fitness) as wildtype — LEAKY for methods that bias toward "
        f"the WT neighborhood."
    )
    return sequences[int(np.argmax(fitness))]


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
    # Prefer AACombo (short combinatorial form) when present
    _seq_col = ('AACombo' if 'AACombo' in df.columns
                else 'Combo' if 'Combo' in df.columns
                else 'seq' if 'seq' in df.columns
                else 'sequence')
    sequences = df[_seq_col].tolist()
    fitness = _scalarized_fitness(df)

    # Detect sequence length (use mode for variable-length datasets)
    lengths = [len(s) for s in sequences]
    seq_len = int(pd.Series(lengths).mode().iloc[0])

    # Wildtype resolution: prefer wt.fasta (authoritative), then n_muts==0,
    # then argmax(fitness) with warning. argmax is a fitness leak for methods
    # that bias toward the WT or its neighborhood.
    dataset_dir = os.path.dirname(data_path)
    wildtype = _resolve_wildtype(dataset_dir, df, sequences, fitness)
    best_idx = sequences.index(wildtype) if wildtype in sequences else int(np.argmax(fitness))
    wildtype_fitness = float(fitness[best_idx]) if best_idx < len(fitness) else 0.0

    logger.info(f"Auto-detected from {data_path}:")
    logger.info(f"  Sequence length (mode): {seq_len}")
    logger.info(f"  Number of variants: {len(sequences)}")
    logger.info(f"  Fitness range: [{fitness.min():.4f}, {fitness.max():.4f}]")
    logger.info(f"  Wildtype: {wildtype[:40]}{'...' if len(wildtype) > 40 else ''} "
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


def create_temp_hotspots(seq_len: int, output_dir: str,
                         sequences: Optional[List[str]] = None) -> str:
    """Create a hotspots CSV defining the GPT search space.

    When `sequences` is provided, derive per-position AA candidates from the
    library itself (positions with >1 observed AA) — this constrains the
    GPT decoder to the actual combinatorial subspace (e.g. 2^13 = 8192 for
    eqFP611_joint, 20^4 for the 4site_* datasets). Without this, the default
    of "all positions × all 20 AAs" lets the GPT propose 20^L sequences,
    almost none of which exist in the library, and the agent never proposes
    a novel in-library candidate.
    """
    hotspot_path = os.path.join(output_dir, 'hotspots.csv')
    os.makedirs(os.path.dirname(hotspot_path), exist_ok=True)
    rows = []
    if sequences:
        for pos in range(seq_len):
            aas = sorted({s[pos] for s in sequences if len(s) > pos})
            if len(aas) > 1:
                rows.append(f'{pos+1},"{aas}"')  # hotspots file is 1-indexed
        if not rows:
            logger.warning("No varying positions detected; falling back to all-20-AA template")
            aa_list_str = repr(ALL_20_AAS)
            rows = [f'{p},"{aa_list_str}"' for p in range(1, seq_len + 1)]
    else:
        aa_list_str = repr(ALL_20_AAS)
        rows = [f'{p},"{aa_list_str}"' for p in range(1, seq_len + 1)]
    with open(hotspot_path, 'w') as f:
        f.write("pos,mut_aas\n")
        f.write("\n".join(rows) + "\n")
    logger.info(f"Wrote hotspots: {len(rows)} positions -> {hotspot_path}")
    return hotspot_path


# ============================================================================
# CNN surrogate helper modules
# ============================================================================

class _Permute201(torch.nn.Module):
    """Permute (B, L, V) -> (B, V, L) inside an nn.Sequential."""
    def forward(self, x):
        return x.permute(0, 2, 1)


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
        surrogate_kind: str = 'ensemble',
        prior_model_path: Optional[str] = None,
        constrain_alphabet: bool = False,
        features_kind: str = 'onehot',
        wt_seq: Optional[str] = None,
        n_gpt_ensemble: int = 1,
        plm_zeroshot_pool_frac: float = 1.0,
        plm_zeroshot_explore_frac: float = 0.1,
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
        self.surrogate_kind = surrogate_kind
        self.prior_model_path = prior_model_path
        self.constrain_alphabet = constrain_alphabet
        self.features_kind = features_kind
        self.wt_seq = wt_seq
        self._esm_extractor = None
        self.n_gpt_ensemble = n_gpt_ensemble
        self.plm_zeroshot_pool_frac = plm_zeroshot_pool_frac
        self.plm_zeroshot_explore_frac = plm_zeroshot_explore_frac
        self.plm_zeroshot_temperature = 0.0  # set by caller; 0 = strict top-K
        self.plm_active_alpha = 0.0  # set by caller; 0 = off
        # PLM reward shaping in REINFORCE: total_reward = z(ucb) + lambda(r) * z(plm).
        # lambda decays per round so early rounds (noisy surrogate) lean on PLM,
        # late rounds (data-rich surrogate) ignore it.
        self.plm_reward_lambda = 0.0       # initial lambda; 0 = off
        self.plm_reward_decay = 'linear'   # 'linear' | 'exponential' | 'none'
        # Hybrid sampling (UCB + PLM + clustering).
        self.hybrid_alpha = 0.3            # PLM weight; 0 = pure UCB+clustering
        self.hybrid_n_clusters = 12        # KMeans K for diversity
        self.hybrid_alloc = 'weighted'     # 'weighted' | 'roundrobin'
        self.hybrid_temperature = 1.0      # softmax T for weighted allocator
        self.hybrid_min_per_cluster = 1    # diversity floor (min slots per cluster)
        # Staged PLM-fraction sampling: reserve plm_sampling_frac of each batch
        # for PLM-top picks during the first `plm_sampling_until_round` rounds.
        self.plm_sampling_frac = 0.0
        self.plm_sampling_until_round = 0  # rounds (1-indexed); 0 disables
        # PLM reward shaping cutoff: lambda forced to 0 after this round.
        self.plm_reward_until_round = 0    # 0 = honor only existing decay schedule
        # SHAP pruning starts at this round (1-indexed; 0 falls back to min_samples).
        self.shap_prune_start_round = 0
        self._esm_mlm = None  # lazy-loaded EsmForMaskedLM for zero-shot scoring
        self._plm_logprobs = None  # (n_varying, vocab_size) np.float64 — cached WT-marginal log-probs
        # MutCompute (structure-based) zero-shot scoring as an alternative to ESM-2 PLM.
        # When use_mutcompute=True the _score_zeroshot(...) dispatcher routes all
        # PLM-named call sites (reward shaping, sampling fraction, active blending)
        # through _score_mutcompute(...) instead of _score_zeroshot_esm(...).
        self.use_mutcompute = False
        self.mutcompute_offset = None      # None = auto-detect; int = manual override
        # Multi-site: learned CNN oracle (utils.oracle_landscape) as ground-truth
        # fitness instead of the data.csv lookup (which returns 0 for any unmeasured
        # sequence and collapses generation on huge spaces). Set post-construction.
        self.use_oracle = False
        self.oracle_landscape = None
        # Cap generated variants to at most this many mutations from the reference
        # (refSeq = best-so-far "start variant"); None disables. Keeps multi-site
        # generation near the data manifold. Set post-construction.
        self.max_n_mut = None
        # Ensemble blend of MC + ESM in the zero-shot dispatcher. Only applies when
        # use_mutcompute=True. α=0.0 → pure MC (Plan C default); α=1.0 → pure ESM;
        # 0 < α < 1 → blend z(MC) and z(ESM) within the candidate pool.
        self.zeroshot_blend = 0.0
        # Plan E: round-staggered override. When zeroshot_early_blend is not None,
        # the first `zeroshot_early_rounds` RL rounds (round_idx ∈ [1, 1+N]) use
        # `zeroshot_early_blend` instead of `zeroshot_blend`. Default behaviour:
        # round 2 (first RL round) = pure ESM, rounds 3+ = pure MC.
        self.zeroshot_early_blend = None
        self.zeroshot_early_rounds = 0
        # Round tracking for round-staggered scoring; set at the start of each train() iteration.
        self._current_round_idx = 0
        self._mc_pos_log_probs = None      # cache: dict {var_idx: log-prob dict per AA}
        self._mc_wt_aas = None             # cache: list of WT AAs at varying positions
        # SHAP-based per-round alphabet pruning (Savinase-style hotspot reselection).
        self.shap_prune_alphabet = False
        self.shap_prune_threshold = 0.0  # keep AAs whose mean SHAP > this
        self.shap_prune_min_alphabet = 3  # never let a position collapse below this
        self.shap_prune_min_samples = 50  # skip pruning until enough data
        self.shap_prune_topk_keep = 10  # AAs of top-K best variants are always retained
        self._initial_alphabets = None  # immutable copy of round-0 alphabets, for rebound

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

        # Prefer AACombo (short combinatorial form) when present
        _seq_col = ('AACombo' if 'AACombo' in df.columns
                    else 'Combo' if 'Combo' in df.columns
                    else 'seq' if 'seq' in df.columns
                    else 'sequence')

        self.seq_to_fitness = {}
        self.all_seqs = df[_seq_col].tolist()
        # Vectorized scalarization handles both single-objective ('fitness') and
        # multi-objective (*_joint with 'blue' + 'red') datasets.
        fitness_arr = _scalarized_fitness(df)
        for seq, fit in zip(self.all_seqs, fitness_arr):
            self.seq_to_fitness[seq] = float(fit)
        self.all_fitness = np.asarray(fitness_arr, dtype=float)
        self.max_fitness_raw = np.max(self.all_fitness)
        self.min_fitness_raw = np.min(self.all_fitness)

        # Per-position observed alphabet (for constrain_alphabet filter)
        seq_len = len(self.all_seqs[0])
        self.per_position_alphabets: List[set] = []
        for p in range(seq_len):
            self.per_position_alphabets.append({s[p] for s in self.all_seqs if len(s) > p})
        self.varying_positions = [p for p, aas in enumerate(self.per_position_alphabets) if len(aas) > 1]

        logger.info(f"Loaded {len(self.all_seqs)} variants from landscape")
        logger.info(f"Fitness range: [{self.min_fitness_raw:.4f}, {self.max_fitness_raw:.4f}]")
        logger.info(f"Varying positions: {len(self.varying_positions)} / {seq_len}; "
                    f"per-position alphabet sizes: {[len(a) for a in self.per_position_alphabets]}")

    def _ensure_combo_map(self):
        """Build the AACombo(short)->full-sequence mapping for oracle scoring.

        When AlphaVariant generates in the short AACombo space (e.g. CreiLOV: 15
        varying positions) but the oracle was trained on the FULL sequence (119aa),
        the combo must be expanded back onto the wild-type before scoring -- otherwise
        the oracle sees a 15-mer padded to 119 with 'A' and the reward signal degrades
        (verified: Spearman 0.86 vs 1.0). For datasets where the generation length
        already equals the oracle length (AAV/PAB1: AACombo == seq), the map is a
        no-op and sequences pass through unchanged.
        """
        if getattr(self, '_combo_map_ready', False):
            return
        self._combo_map_ready = True
        self._combo_full_wt = None
        self._combo_positions = None
        if not (self.use_oracle and self.oracle_landscape is not None):
            return
        try:
            oracle_len = int(self.oracle_landscape.seq_len)
        except Exception:
            return
        if self.seq_len >= oracle_len:
            return  # generation already in full-sequence space; nothing to expand
        data_dir = os.path.dirname(self.landscape_path)
        wt_path = os.path.join(data_dir, 'wt.fasta')
        vp_path = os.path.join(data_dir, 'varying_positions.txt')
        if not (os.path.exists(wt_path) and os.path.exists(vp_path)):
            logger.warning("AACombo->full expansion unavailable (missing wt.fasta or "
                           "varying_positions.txt); scoring short combos directly.")
            return
        full_wt = ''.join(l.strip() for l in open(wt_path)
                          if l.strip() and not l.startswith('>'))
        positions = [int(x) for x in open(vp_path).read().strip().split(',') if x.strip()]
        if len(positions) != self.seq_len or len(full_wt) != oracle_len:
            logger.warning(f"AACombo->full map size mismatch (combo={self.seq_len}, "
                           f"positions={len(positions)}, full_wt={len(full_wt)}, "
                           f"oracle={oracle_len}); scoring short combos directly.")
            return
        self._combo_full_wt = full_wt
        self._combo_positions = positions
        logger.info(f"AACombo->full expansion active: {self.seq_len} varying positions "
                    f"mapped onto {oracle_len}aa wild-type for oracle scoring.")

    def _combo_to_full(self, seqs: List[str]) -> List[str]:
        """Expand short AACombo sequences onto the full wild-type (no-op if not needed)."""
        self._ensure_combo_map()
        if self._combo_positions is None:
            return list(seqs)
        out = []
        for combo in seqs:
            chars = list(self._combo_full_wt)
            for i, p in enumerate(self._combo_positions):
                if i < len(combo):
                    chars[p] = combo[i]
            out.append(''.join(chars))
        return out

    def _get_ground_truth_fitness(self, seqs: List[str]) -> np.ndarray:
        """Get ground truth fitness for sequences."""
        if self.use_oracle and self.oracle_landscape is not None:
            # Learned-oracle (multi-site): score ANY sequence in RAW fitness units
            # (native scale). The REINFORCE reward + surrogate + sigma are calibrated
            # for raw fitness (as in lookup mode); feeding normalized [0,1] here shrinks
            # the fitness signal (catastrophically for large-range datasets like CreiLOV).
            # Expand short AACombo generations onto the full WT first (CreiLOV); no-op
            # when generation length already matches the oracle (AAV/PAB1).
            return np.asarray(
                self.oracle_landscape.get_fitness(self._combo_to_full(list(seqs))),
                dtype=float)
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
        - uniform: uniformly random across the whole landscape (matches Random/ALDE/FLEXS/etc.)
        - medium:  40th-60th percentile
        - hard:    bottom 20th percentile
        """
        if exclude is None:
            exclude = set()

        if self.level == 'plm_zeroshot':
            return self._zeroshot_init_sample(n_samples, exclude=exclude)

        if self.level == 'uniform':
            mask = np.ones(len(self.all_fitness), dtype=bool)
        elif self.level == 'medium':
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

        if self.level == 'uniform':
            mask = np.ones(len(self.all_fitness), dtype=bool)
        elif self.level == 'medium':
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
        """Create GPT models. Optionally load a pretrained MSA-prior."""
        if self.prior_model_path is not None and os.path.exists(self.prior_model_path):
            # Load pretrained prior (trained on a family MSA via train_prior).
            # The agent is initialized as a copy of the prior, then fine-tuned
            # via REINFORCE on the surrogate's predictions.
            #
            # We load the state_dict directly with map_location=self.device so
            # the weights never live on CPU first. The CPU->GPU transfer step
            # via .to(device) triggers a NVML init in PyTorch 2.x that fails
            # on hosts where the NVML userland and kernel versions don't
            # match, even though the rest of CUDA works.
            from popgen.model.gpt import GPTConfig as _GPTConfig
            from pathlib import Path
            import json as _json
            cfg_path = Path(self.prior_model_path).with_suffix('.json')
            with open(cfg_path) as fh:
                mconf = _GPTConfig(**_json.load(fh))
            # Build prior on the target device and load weights via the
            # standard load_state_dict path. Loading to CPU first and then
            # doing param.copy_() can leave the caching allocator in a state
            # that triggers an NVML init assert mid-run; the standard
            # load_state_dict goes through the same code path as the random
            # init's `.to(device)` call, which is known to work on this host.
            self.prior_model = GPT(mconf).to(self.device)
            _ckpt = torch.load(self.prior_model_path, map_location=self.device)
            _sd = _ckpt['model'] if isinstance(_ckpt, dict) and 'model' in _ckpt else _ckpt
            self.prior_model.load_state_dict(_sd)
            # Agent = exact copy of the loaded prior. Use copy.deepcopy on
            # the device-resident prior_model, matching the random-init code
            # path that's known to work on this host.
            self.agent_model = copy.deepcopy(self.prior_model)
            logger.info(f"Loaded MSA-pretrained prior from {self.prior_model_path}")
        else:
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

    # ---- Per-position alphabet filter (FLEXS-style) -----------------------

    def _is_on_alphabet(self, seq: str) -> bool:
        """True if every position's AA is in the observed per-position alphabet."""
        if not hasattr(self, 'per_position_alphabets') or not self.per_position_alphabets:
            return True
        if len(seq) != len(self.per_position_alphabets):
            return False
        return all(seq[p] in self.per_position_alphabets[p] for p in range(len(seq)))

    def _filter_on_alphabet(self, seqs, scores=None):
        """Drop sequences whose AAs violate observed per-position alphabets."""
        if not self.constrain_alphabet:
            return (seqs, scores) if scores is not None else seqs
        keep_idx = [i for i, s in enumerate(seqs) if self._is_on_alphabet(s)]
        kept_seqs = [seqs[i] for i in keep_idx]
        if scores is not None:
            kept_scores = np.asarray(scores)[keep_idx]
            return kept_seqs, kept_scores
        return kept_seqs

    # ---- ESM-2 feature extractor (ftMLDE-style) ---------------------------

    def _get_esm_extractor(self):
        """Lazy-load ESM-2 model and tokenizer."""
        if self._esm_extractor is not None:
            return self._esm_extractor
        from transformers import EsmModel, EsmTokenizer
        model_name = 'facebook/esm2_t12_35M_UR50D'
        logger.info(f"Loading ESM-2 ({model_name}) for surrogate features...")
        tok = EsmTokenizer.from_pretrained(model_name)
        mdl = EsmModel.from_pretrained(model_name).eval()
        dev = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        mdl = mdl.to(dev)
        self._esm_extractor = {'tokenizer': tok, 'model': mdl, 'device': dev, 'cache': {}}
        return self._esm_extractor

    def _embed_seqs_esm(self, seqs):
        """ESM-2 mean-pool over varying positions, using WT-substituted full sequences."""
        if self.wt_seq is None:
            # Fallback: embed the short combo directly
            wt_full = None
        else:
            wt_full = self.wt_seq

        ext = self._get_esm_extractor()
        tok, model, dev, cache = ext['tokenizer'], ext['model'], ext['device'], ext['cache']

        # Reconstruct full sequences (or use combo as-is when no WT)
        if wt_full is not None and hasattr(self, 'varying_positions') and self.varying_positions:
            full_seqs = []
            for combo in seqs:
                arr = list(wt_full)
                for i, p in enumerate(self.varying_positions):
                    if i < len(combo):
                        arr[p] = combo[i]
                full_seqs.append(''.join(arr))
            positions = self.varying_positions
        else:
            full_seqs = list(seqs)
            positions = list(range(len(seqs[0])))

        # Check cache + queue
        feats = [None] * len(full_seqs)
        to_compute_idx, to_compute_seqs = [], []
        for i, s in enumerate(full_seqs):
            if s in cache:
                feats[i] = cache[s]
            else:
                to_compute_idx.append(i)
                to_compute_seqs.append(s)

        # Batch through ESM-2
        if to_compute_seqs:
            bs = 32
            for b in range(0, len(to_compute_seqs), bs):
                batch = to_compute_seqs[b:b + bs]
                with torch.no_grad():
                    tokens = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=1024)
                    tokens = {k: v.to(dev) for k, v in tokens.items()}
                    out = model(**tokens)
                    # Mean over varying positions (+1 offset for [CLS] token)
                    h = out.last_hidden_state  # (B, L+2, D)
                    pos_feats = h[:, [p + 1 for p in positions], :].mean(dim=1)  # (B, D)
                for j, s in enumerate(batch):
                    vec = pos_feats[j].cpu().numpy()
                    cache[s] = vec
                    feats[to_compute_idx[b + j]] = vec
        return np.stack(feats, axis=0)

    # ---- SHAP-based per-round alphabet pruning ---------------------------
    _SHAP_AAS = "ACDEFGHIKLMNPQRSTVWY"

    def _update_alphabet_via_shap(self, round_idx: int) -> bool:
        """
        Savinase-style hotspot reselection: fit XGBoost on (one-hot mutation
        features, fitness), compute SHAP, prune per-position AAs whose mean
        SHAP <= threshold. Always keeps AAs in the top-K best collected
        variants and enforces a minimum alphabet size per position.

        Combo-encoding assumption: each collected sequence has one AA per
        varying position (the AACombo column), so we index by `j` in the
        combo, not the protein-coordinate `p`.
        """
        if not self.shap_prune_alphabet or round_idx == 0:
            return False
        # Explicit start-round gate overrides the min_samples heuristic when set.
        start_r = int(getattr(self, 'shap_prune_start_round', 0) or 0)
        if start_r > 0 and (round_idx + 1) < start_r:
            logger.info(f"SHAP prune skipped: round {round_idx + 1} < start_round {start_r}")
            return False
        n = len(self.collected_seqs)
        if start_r <= 0 and n < self.shap_prune_min_samples:
            logger.info(f"SHAP prune skipped: {n} samples < min {self.shap_prune_min_samples}")
            return False
        if self._initial_alphabets is None:
            self._initial_alphabets = [set(a) for a in self.per_position_alphabets]

        try:
            import xgboost as xgb
        except Exception as e:
            logger.warning(f"SHAP prune disabled: xgboost import failed ({e})")
            return False

        seqs = list(self.collected_seqs)
        Y = np.array(self.collected_fitness, dtype=np.float32)
        if float(np.std(Y)) < 1e-8:
            logger.info("SHAP prune skipped: collected fitness has zero variance")
            return False

        positions = self.varying_positions
        AAs = self._SHAP_AAS
        n_aa = len(AAs)
        aa2k = {aa: k for k, aa in enumerate(AAs)}
        # Residues at the varying positions, regardless of whether collected
        # sequences are short combos (len == #varying positions, so s[j] is already
        # the varying residue: AAV/PAB1/CreiLOV) or full-length (hundreds of residues
        # with NON-contiguous varying positions, where we must index by the protein
        # coordinate positions[j], not s[j]).
        def _combo_view(s):
            if len(s) == len(positions):
                return s
            if positions and positions[-1] < len(s):
                return ''.join(s[p] for p in positions)
            return s[:len(positions)]
        combo_seqs = [_combo_view(s) for s in seqs]
        X = np.zeros((len(seqs), len(positions) * n_aa), dtype=np.float32)
        for i, cv in enumerate(combo_seqs):
            for j in range(min(len(positions), len(cv))):
                k = aa2k.get(cv[j])
                if k is not None:
                    X[i, j * n_aa + k] = 1.0

        try:
            model = xgb.XGBRegressor(
                n_estimators=200, max_depth=4,
                learning_rate=0.08, subsample=0.85, colsample_bytree=0.85,
                random_state=self.seed, n_jobs=4, verbosity=0,
            )
            model.fit(X, Y)
            # XGBoost native TreeSHAP — avoids the shap-library / xgboost-3.x
            # leaf-format incompatibility. Last column is the bias term.
            dmat = xgb.DMatrix(X)
            contribs = model.get_booster().predict(dmat, pred_contribs=True)
            shap_vals = contribs[:, :-1]  # (n, n_features)
        except Exception as e:
            logger.warning(f"SHAP prune disabled this round: fit/explain failed ({e})")
            return False

        # AAs to always keep: those in the top-K best collected variants.
        top_k = min(self.shap_prune_topk_keep, len(seqs))
        top_indices = np.argsort(-Y)[:top_k]
        must_keep = [set() for _ in positions]
        for ti in top_indices:
            cv = combo_seqs[ti]
            for j in range(min(len(positions), len(cv))):
                must_keep[j].add(cv[j])

        sizes_before = [len(self.per_position_alphabets[p]) for p in positions]
        new_alphabets = [set(a) for a in self.per_position_alphabets]
        dropped_aas_per_pos = []

        for j, p in enumerate(positions):
            # Per-AA mean SHAP, conditional on presence at position j.
            aa_mean_shap = {}
            for k, aa in enumerate(AAs):
                col = j * n_aa + k
                has = X[:, col] > 0
                if has.sum() >= 2:  # need at least 2 occurrences for stable mean
                    aa_mean_shap[aa] = float(shap_vals[has, col].mean())

            kept = set()
            for aa in self._initial_alphabets[p]:
                if aa in must_keep[j]:
                    kept.add(aa)
                elif aa not in aa_mean_shap:
                    # Not yet observed (or too rare) -> keep so we can still explore it.
                    kept.add(aa)
                elif aa_mean_shap[aa] > self.shap_prune_threshold:
                    kept.add(aa)

            # Enforce minimum alphabet size: top up by mean SHAP if we pruned too aggressively.
            if len(kept) < self.shap_prune_min_alphabet:
                ranked = sorted(aa_mean_shap.items(), key=lambda kv: -kv[1])
                for aa, _ in ranked:
                    if aa in self._initial_alphabets[p]:
                        kept.add(aa)
                    if len(kept) >= self.shap_prune_min_alphabet:
                        break

            dropped = set(self.per_position_alphabets[p]) - kept
            dropped_aas_per_pos.append(sorted(dropped))
            new_alphabets[p] = kept

        self.per_position_alphabets = new_alphabets
        # Gate constrain_alphabet to oracle/multi-site only. On 4-site combinatorial
        # tasks all positions vary; the filter rejects valid proposals without aiding
        # exploration and regresses max fitness (PhoQ: 70 → 35).
        if getattr(self, 'use_oracle', False):
            self.constrain_alphabet = True

        # Propagate the pruned alphabet into the GPT hotspot config so generation in
        # the following rounds samples on the pruned subspace. Without this, the GPT
        # keeps proposing on the original observed alphabet while the per-position
        # filter enforces the pruned one, so proposals get rejected -- catastrophically
        # for long sequences (~100% dropped, rounds 2-5 stall). pos_aa_candidates
        # is keyed by 1-indexed protein position (positions[] is 0-indexed).
        # Also gated to oracle mode for the same reason as constrain_alphabet above.
        tmpl = getattr(self, 'template', None)
        cand = getattr(tmpl, 'pos_aa_candidates', None) if tmpl is not None else None
        if getattr(self, 'use_oracle', False) and cand is not None:
            synced = 0
            for j, p in enumerate(positions):
                key = p + 1
                if key in cand:
                    cand[key] = sorted(new_alphabets[p])
                    synced += 1
            logger.info(f"SHAP prune: synced {synced} pruned positions into the GPT "
                        f"hotspot config (generation now samples the pruned subspace)")

        sizes_after = [len(self.per_position_alphabets[p]) for p in positions]
        logger.info(
            f"SHAP prune (round {round_idx + 1}, n={n}): "
            f"sizes {sizes_before} -> {sizes_after}; "
            f"dropped/pos {dropped_aas_per_pos}"
        )
        return True

    # ---- PLM reward-shaping schedule -------------------------------------

    def _plm_reward_lambda_for_round(self, round_idx: int) -> float:
        """
        Compute lambda for round_idx (0-indexed). Round 0 is the random init —
        no RL there. Round 1 = first RL round (highest lambda). The schedule
        decays so the surrogate dominates by the last round once it has data.
        """
        lam0 = float(self.plm_reward_lambda)
        if lam0 <= 0.0 or round_idx <= 0:
            return 0.0
        # Hard cutoff: if plm_reward_until_round is set (1-indexed), force lambda=0
        # for rounds beyond it. round_idx is 0-indexed so the user-facing round
        # number is round_idx + 1.
        until_r = int(getattr(self, 'plm_reward_until_round', 0) or 0)
        if until_r > 0 and (round_idx + 1) > until_r:
            return 0.0
        N = max(self.n_rounds, 2)
        r = max(round_idx, 1)
        decay = (self.plm_reward_decay or 'none').lower()
        if decay == 'linear':
            denom = max(N - 2, 1)
            return lam0 * max(0.0, (N - 1 - r) / denom)
        if decay == 'exponential':
            return lam0 * (0.5 ** (r - 1))
        return lam0  # 'none' -> constant

    # ---- Staged PLM-fraction sampling helpers ----------------------------

    def _plm_quota_for_round(self, round_idx: int) -> int:
        """How many slots of the current batch should come from PLM-top picks."""
        frac = float(getattr(self, 'plm_sampling_frac', 0.0))
        until = int(getattr(self, 'plm_sampling_until_round', 0))
        if frac <= 0.0 or until <= 0:
            return 0
        if (round_idx + 1) > until:
            return 0
        return int(round(self.batch_size * frac))

    def _plm_top_picks(self, pool, n: int, exclude: set = None):
        """Top-n combos by PLM WT-marginal score, deduped against `exclude`."""
        if n <= 0 or not pool:
            return []
        if exclude is None:
            exclude = set()
        if self.wt_seq is None or not getattr(self, 'varying_positions', None):
            logger.warning("PLM-quota requested but wt_seq / varying_positions unavailable; skipping")
            return []
        seen = set()
        uniq = []
        for s in pool:
            if s not in exclude and s not in seen:
                uniq.append(s); seen.add(s)
        if not uniq:
            return []
        scores = self._score_zeroshot(uniq)
        order = np.argsort(-scores)
        picks = [uniq[i] for i in order[:n]]
        logger.info(
            f"  PLM-quota: picked top-{len(picks)} by ESM WT-marginals "
            f"(score range [{scores[order[:len(picks)]].min():.2f}, {scores[order[0]]:.2f}])"
        )
        return picks

    # ---- ESM-2 zero-shot fitness prior (WT-marginal scoring) -------------

    def _get_esm_mlm(self):
        """Lazy-load EsmForMaskedLM for zero-shot scoring (separate from feature model)."""
        if self._esm_mlm is not None:
            return self._esm_mlm
        from transformers import EsmForMaskedLM, EsmTokenizer
        model_name = 'facebook/esm2_t12_35M_UR50D'
        logger.info(f"Loading ESM-2 MLM ({model_name}) for zero-shot scoring...")
        tok = EsmTokenizer.from_pretrained(model_name)
        mdl = EsmForMaskedLM.from_pretrained(model_name).eval()
        dev = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        mdl = mdl.to(dev)
        self._esm_mlm = {'tokenizer': tok, 'model': mdl, 'device': dev}
        return self._esm_mlm

    def _compute_plm_logprobs_matrix(self):
        """Compute WT-marginal log-prob matrix at varying positions (one forward, cached)."""
        if self._plm_logprobs is not None:
            return
        if self.wt_seq is None or not getattr(self, 'varying_positions', None):
            raise RuntimeError(
                "plm_zeroshot requires self.wt_seq and self.varying_positions "
                "(no wt.fasta / no varying positions detected)."
            )
        ext = self._get_esm_mlm()
        tok, model, dev = ext['tokenizer'], ext['model'], ext['device']
        positions = self.varying_positions

        wt_arr = list(self.wt_seq)
        for p in positions:
            wt_arr[p] = tok.mask_token
        masked_wt = "".join(wt_arr)

        with torch.no_grad():
            tokens = tok(masked_wt, return_tensors='pt', padding=True, truncation=True, max_length=1024)
            tokens = {k: v.to(dev) for k, v in tokens.items()}
            logits = model(**tokens).logits[0]  # (L+2, V)
            log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()

        # Slice to varying positions only: shape (n_varying, V). Offset +1 for [CLS].
        self._plm_logprobs = log_probs[[p + 1 for p in positions], :]
        self._plm_aa_to_tok = {aa: tok.convert_tokens_to_ids(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
        self._plm_unk_id = tok.unk_token_id

    # ---- MutCompute (structure-based) zero-shot scoring ------------------
    _AA_THREE2ONE = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    def _load_mutcompute(self):
        """
        Load and cache MutCompute per-position log-probabilities for the
        varying positions in this landscape.

        File expected at landscape_dir/mutcompute.csv (column 'pos' uses some
        PDB-style numbering; column 'wtAA' is three-letter code; per-AA
        probability columns are 'prALA'...'prTYR' which we rename to A..Y).

        Position numbering may have an offset relative to self.wt_seq's
        0-indexed positions; we auto-detect it by scanning offsets
        in [-50, 100] and picking the one whose wtAA column best matches
        self.wt_seq at the varying positions. The caller may override via
        self.mutcompute_offset.
        """
        if self._mc_pos_log_probs is not None:
            return
        if self.wt_seq is None or not getattr(self, 'varying_positions', None):
            raise RuntimeError(
                "MutCompute scoring requires self.wt_seq and self.varying_positions; "
                "wt.fasta or AACombo column may be missing for this dataset."
            )

        import pandas as pd
        landscape_dir = os.path.dirname(self.landscape_path)
        mc_path = os.path.join(landscape_dir, "mutcompute.csv")
        if not os.path.isfile(mc_path):
            raise FileNotFoundError(
                f"--use_mutcompute requested but {mc_path} not present."
            )

        df = pd.read_csv(mc_path, index_col=0)
        # Rename pr<THREE> probability columns to single-letter AAs.
        for three, one in self._AA_THREE2ONE.items():
            col = 'pr' + three
            if col in df.columns:
                df = df.rename(columns={col: one})
        df['wtAA_1'] = df['wtAA'].map(self._AA_THREE2ONE)

        positions = self.varying_positions  # 0-indexed in self.wt_seq
        wt_aas = [self.wt_seq[p] for p in positions]

        # Determine offset: pos_in_csv = (0-indexed varying p) + offset.
        if self.mutcompute_offset is not None:
            offset = int(self.mutcompute_offset)
        else:
            best_offset, best_matches = None, -1
            for cand in range(-50, 101):
                m = 0
                for p, wt_aa in zip(positions, wt_aas):
                    row = df.loc[df['pos'] == (p + cand)]
                    if len(row) and row['wtAA_1'].iloc[0] == wt_aa:
                        m += 1
                if m > best_matches:
                    best_matches = m; best_offset = cand
                if m == len(positions):
                    break
            if best_matches < len(positions):
                logger.warning(
                    f"MutCompute offset auto-detect: best match {best_matches}/"
                    f"{len(positions)} varying positions at offset {best_offset}. "
                    f"WT alignment may be imperfect; proceeding."
                )
            offset = best_offset
        self._mc_offset = offset

        # Per varying position, build {AA: log P} dict from the matching CSV row.
        AAs = "ACDEFGHIKLMNPQRSTVWY"
        per_pos = []
        for p, wt_aa in zip(positions, wt_aas):
            row = df.loc[df['pos'] == (p + offset)]
            if len(row) == 0:
                logger.warning(
                    f"MutCompute: no row for protein position {p} "
                    f"(csv pos={p + offset}); contributions at this position will be 0."
                )
                per_pos.append(None); continue
            row = row.iloc[0]
            if row['wtAA_1'] != wt_aa:
                logger.warning(
                    f"MutCompute wtAA mismatch at protein position {p}: "
                    f"wt.fasta says {wt_aa}, csv says {row['wtAA_1']}; using csv probs anyway."
                )
            log_p = {}
            for aa in AAs:
                p_aa = float(row.get(aa, 0.0)) if aa in row.index else 0.0
                log_p[aa] = np.log(max(p_aa, 1e-12))
            per_pos.append(log_p)
        self._mc_pos_log_probs = per_pos
        self._mc_wt_aas = wt_aas
        logger.info(
            f"Loaded MutCompute table ({mc_path}); offset={offset}, "
            f"varying-position WT AAs: {wt_aas}"
        )

    def _score_mutcompute(self, combo_seqs: List[str]) -> np.ndarray:
        """
        MutCompute log-likelihood-ratio score, summed across varying positions.

        For each combo, contribute log(P_mut) - log(P_ref) per varying position
        where the combo's AA differs from the WT AA. Positions where combo AA
        equals WT contribute 0. Positions without MutCompute data contribute 0.
        """
        self._load_mutcompute()
        per_pos = self._mc_pos_log_probs
        wt_aas = self._mc_wt_aas
        n_pos = len(per_pos)

        scores = np.zeros(len(combo_seqs), dtype=np.float64)
        for i, combo in enumerate(combo_seqs):
            s = 0.0
            for j in range(min(n_pos, len(combo))):
                if per_pos[j] is None:
                    continue
                mut_aa = combo[j]
                ref_aa = wt_aas[j]
                if mut_aa == ref_aa:
                    continue
                if mut_aa not in per_pos[j]:
                    s += -10.0  # heavy penalty for non-canonical AA
                else:
                    s += per_pos[j][mut_aa] - per_pos[j][ref_aa]
            scores[i] = s
        return scores

    def _effective_zeroshot_blend(self) -> float:
        """
        Compute the effective blend coefficient α for the *current round*.

        Plan E round-staggered behaviour: when `zeroshot_early_blend` is set
        and the current RL round (round_idx ∈ {1, 2, …}) falls within the
        first `zeroshot_early_rounds` of those, use `zeroshot_early_blend`
        instead of the default `zeroshot_blend`. Round 0 is the random init
        (no RL); scoring there falls back to default `zeroshot_blend`.
        """
        early_alpha = getattr(self, "zeroshot_early_blend", None)
        n_early = int(getattr(self, "zeroshot_early_rounds", 0) or 0)
        r = int(getattr(self, "_current_round_idx", 0))
        if early_alpha is not None and n_early > 0 and 1 <= r <= n_early:
            return float(early_alpha)
        return float(getattr(self, "zeroshot_blend", 0.0) or 0.0)

    def _score_zeroshot(self, seqs: List[str]) -> np.ndarray:
        """
        Dispatch wrapper: route to MutCompute, ESM-2, or an ensemble blend.

        - use_mutcompute=False → pure ESM-2 (Plan A/B path).
        - use_mutcompute=True, effective α=0 → pure MC (Plan C default).
        - use_mutcompute=True, effective α=1 → pure ESM-2 (=Plan B PLM-reward).
        - 0 < effective α < 1 → blend z(MC) and z(ESM): the two scores are
          z-normalised within the candidate pool, then linearly combined as
          `(1−α)·z(MC) + α·z(ESM)`. Plan D D1 ensemble path.
        - Plan E: effective α can switch per round via _effective_zeroshot_blend.
        """
        if not getattr(self, "use_mutcompute", False):
            return self._score_zeroshot_esm(seqs)
        a = self._effective_zeroshot_blend()
        a = max(0.0, min(1.0, a))
        if a <= 0.0:
            return self._score_mutcompute(seqs)
        if a >= 1.0:
            return self._score_zeroshot_esm(seqs)
        mc = self._score_mutcompute(seqs)
        esm = self._score_zeroshot_esm(seqs)
        def _z(x):
            x = np.asarray(x, dtype=np.float64)
            s = float(np.std(x))
            return (x - float(np.mean(x))) / s if s > 1e-12 else np.zeros_like(x)
        return (1.0 - a) * _z(mc) + a * _z(esm)

    def _score_zeroshot_esm(self, combo_seqs: List[str]) -> np.ndarray:
        """
        WT-marginal zero-shot fitness prior (Meier et al., 2021).

        Independent-position approximation: ignores epistasis between varying
        positions but is O(1) forward passes regardless of pool size (cached
        across calls via self._plm_logprobs).
        """
        self._compute_plm_logprobs_matrix()
        log_probs = self._plm_logprobs  # (n_varying, V)
        aa_to_tok = self._plm_aa_to_tok
        unk_id = self._plm_unk_id

        scores = np.zeros(len(combo_seqs), dtype=np.float64)
        for i, combo in enumerate(combo_seqs):
            s = 0.0
            for j in range(min(len(combo), log_probs.shape[0])):
                tid = aa_to_tok.get(combo[j])
                if tid is None or tid == unk_id:
                    s += -10.0
                else:
                    s += float(log_probs[j, tid])
            scores[i] = s
        return scores

    def _zeroshot_init_sample(self, n_samples: int, exclude: set = None) -> List[str]:
        """Select the initial batch by ESM-2 WT-marginal score, with a small random tail."""
        if exclude is None:
            exclude = set()
        available_indices = [i for i, s in enumerate(self.all_seqs) if s not in exclude]
        if len(available_indices) <= n_samples:
            return [self.all_seqs[i] for i in available_indices]

        # Optionally subsample the pool to cap ESM cost on huge libraries.
        pool = available_indices
        frac = float(self.plm_zeroshot_pool_frac)
        if frac < 1.0:
            target = max(int(len(pool) * frac), n_samples * 10)
            target = min(target, len(pool))
            pool = list(np.random.choice(pool, size=target, replace=False))

        pool_seqs = [self.all_seqs[i] for i in pool]
        _src = "MutCompute" if getattr(self, "use_mutcompute", False) else "ESM-2 WT-marginals"
        logger.info(f"plm_zeroshot: scoring {len(pool_seqs)} candidates with {_src}...")
        scores = self._score_zeroshot(pool_seqs)

        explore_n = int(round(n_samples * float(self.plm_zeroshot_explore_frac)))
        score_n = n_samples - explore_n

        T = float(self.plm_zeroshot_temperature)
        if T > 0.0 and score_n > 0:
            # Temperature-softmax sampling without replacement (Gumbel top-k).
            logits = scores / T
            logits = logits - logits.max()
            g = -np.log(-np.log(np.random.rand(len(pool_seqs)).clip(1e-12, 1 - 1e-12)))
            keys = logits + g
            score_indices = np.argpartition(-keys, score_n - 1)[:score_n]
            score_sel = [pool_seqs[k] for k in score_indices]
            score_summary = (
                f"T={T}; "
                f"selected_score_range [{scores[score_indices].min():.2f}, {scores[score_indices].max():.2f}]"
            )
        else:
            order = np.argsort(-scores)
            score_sel = [pool_seqs[k] for k in order[:score_n]]
            score_summary = (
                f"top-{score_n}; "
                f"selected_score_range [{scores[order[:score_n]].min():.2f}, {scores[order[0]]:.2f}]"
            )

        selected = list(score_sel)
        if explore_n > 0:
            sel_set = set(selected)
            remaining = [s for s in pool_seqs if s not in sel_set]
            if remaining:
                idx = np.random.choice(len(remaining), size=min(explore_n, len(remaining)), replace=False)
                selected.extend(remaining[k] for k in idx)

        logger.info(
            f"plm_zeroshot: PLM-pick={score_n} ({score_summary}), random={len(selected) - score_n}; "
            f"all-pool score_range [{scores.min():.2f}, {scores.max():.2f}]"
        )
        return selected[:n_samples]

    # ---- CNN surrogate (FLEXS-style ensemble for uncertainty) -------------
    _AMINO_ACIDS_ORD = "ACDEFGHIKLMNPQRSTVWY"

    def _seqs_to_onehot(self, seqs: List[str]) -> "torch.Tensor":
        """Convert sequences to one-hot tensor (B, seq_len, 20)."""
        aa2idx = {aa: i for i, aa in enumerate(self._AMINO_ACIDS_ORD)}
        L = len(seqs[0])
        X = torch.zeros(len(seqs), L, len(self._AMINO_ACIDS_ORD), dtype=torch.float32)
        for i, s in enumerate(seqs):
            for j, aa in enumerate(s):
                if aa in aa2idx:
                    X[i, j, aa2idx[aa]] = 1.0
        return X

    @staticmethod
    def _make_cnn(seq_len: int, vocab: int = 20, hidden: int = 64) -> "torch.nn.Module":
        import torch.nn as nn
        k = min(3, seq_len)
        return nn.Sequential(
            _Permute201(),  # (B, L, V) -> (B, V, L)
            nn.Conv1d(vocab, hidden, kernel_size=k, padding='same'),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=k, padding='same'),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def _train_cnn_surrogate(self):
        """Train a 5-model CNN ensemble on collected (seq, fitness) data."""
        import torch.nn as nn

        if len(self.collected_seqs) < 5:
            self.surrogate_trained = False
            return

        dev = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        X = self._seqs_to_onehot(self.collected_seqs).to(dev)
        y = torch.tensor(self.collected_fitness, dtype=torch.float32, device=dev)
        seq_len = X.shape[1]

        self.surrogate_models = []
        for ens_i in range(5):
            torch.manual_seed(self.seed + ens_i + 1000)
            model = self._make_cnn(seq_len).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            loss_fn = nn.MSELoss()
            model.train()
            n_epochs = 80
            bs = min(64, len(self.collected_seqs))
            for ep in range(n_epochs):
                perm = torch.randperm(len(self.collected_seqs), device=dev)
                for b in range(0, len(self.collected_seqs), bs):
                    idx = perm[b:b + bs]
                    opt.zero_grad()
                    pred = model(X[idx]).squeeze(-1)
                    loss = loss_fn(pred, y[idx])
                    loss.backward()
                    opt.step()
            model.eval()
            self.surrogate_models.append(model)

        self.surrogate_trained = True
        self._surrogate_device = dev
        logger.debug(f"Trained CNN surrogate ensemble on {len(self.collected_seqs)} samples")

    def _predict_cnn_surrogate(self, seqs: List[str]) -> "Tuple[np.ndarray, np.ndarray, np.ndarray]":
        """Predict using CNN ensemble. Returns (mu, sigma, per-model predictions)."""
        if not self.surrogate_trained or not self.surrogate_models:
            return np.zeros(len(seqs)), np.ones(len(seqs)), np.zeros((len(seqs), 1))
        dev = self._surrogate_device
        X = self._seqs_to_onehot(seqs).to(dev)
        preds = np.zeros((len(seqs), len(self.surrogate_models)))
        with torch.no_grad():
            for i, m in enumerate(self.surrogate_models):
                preds[:, i] = m(X).squeeze(-1).cpu().numpy()
        mu = preds.mean(axis=1)
        sigma = preds.std(axis=1)
        return mu, sigma, preds

    # ---- End CNN surrogate ------------------------------------------------

    def _featurize(self, seqs):
        """Return numpy features for surrogate per self.features_kind."""
        if self.features_kind == 'esm2':
            return self._embed_seqs_esm(seqs)
        if self.features_kind == 'ev_onehot':
            return self._ev_onehot_feat(seqs)
        from popscorer.fitness.aa_onehot_pred.embed import seqs2feat
        return seqs2feat(seqs)

    def _ev_onehot_feat(self, seqs):
        """one-hot features augmented with the EVmutation (plmc Potts) statistical-
        energy score as an extra column. The EV score is a homolog-derived zero-shot
        fitness proxy that generalizes far better than one-hot on large free-mutation
        spaces. Requires data/<dataset>/plmc/uniref100.model_params + wt.fasta.
        The EV column is standardized (z-score, scaler fit once on the first/training
        call) so the Ridge ensemble members handle its native (~-96..3) scale."""
        seqs = list(seqs)
        if getattr(self, '_ev_predictor', None) is None:
            from popscorer.fitness.ev_onehot_pred.predictor import EVPredictor
            self._ev_predictor = EVPredictor(os.path.dirname(self.landscape_path))
            self._ev_mu, self._ev_sigma = None, None
        from popscorer.fitness.aa_onehot_pred.embed import seqs2feat
        oh = seqs2feat(seqs)
        # The EV (plmc Potts) model scores full-length sequences; expand short AACombo
        # generations onto the full WT first (CreiLOV: 15-mer -> 119aa). No-op when seqs
        # are already full length (AAV/PAB1: AACombo == seq).
        ev = np.asarray(self._ev_predictor.seq2score(self._combo_to_full(seqs)),
                        dtype=np.float32).reshape(-1, 1)
        if self._ev_mu is None:          # fit scaler once (on the first = training call)
            self._ev_mu = float(ev.mean())
            self._ev_sigma = float(ev.std() + 1e-8)
        ev = (ev - self._ev_mu) / self._ev_sigma
        return np.concatenate([oh, ev], axis=1)

    def _train_surrogate(self):
        """Train surrogate model on all collected data."""
        if self.surrogate_kind == 'cnn':
            return self._train_cnn_surrogate()

        from sklearn.linear_model import Ridge, BayesianRidge
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

        if len(self.collected_seqs) == 0:
            logger.warning("No training data for surrogate")
            return

        X = self._featurize(self.collected_seqs)
        y = np.array(self.collected_fitness)

        # Train ensemble of models. Ablation: --single_surrogate uses one model only,
        # so the predictive std (ensemble disagreement) is 0 and UCB collapses to the
        # mean -- i.e. no ensemble scoring / no uncertainty exploration bonus.
        self.surrogate_models = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if getattr(self, 'single_surrogate', False):
                models = [RandomForestRegressor(n_estimators=50, max_depth=8,
                                                random_state=self.seed, n_jobs=-1)]
            else:
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
        if self.surrogate_kind == 'cnn':
            mu, sigma, _ = self._predict_cnn_surrogate(seqs)
            return mu + 2.0 * sigma, mu

        if not self.surrogate_trained or len(self.surrogate_models) == 0:
            return np.zeros(len(seqs)), np.ones(len(seqs))

        X = self._featurize(seqs)
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
        if self.surrogate_kind == 'cnn':
            return self._predict_cnn_surrogate(seqs)

        if not self.surrogate_trained or len(self.surrogate_models) == 0:
            return np.zeros(len(seqs)), np.ones(len(seqs)), np.zeros((len(seqs), 1))

        X = self._featurize(seqs)
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

        # Optional PLM bias: z-normalize both signals so alpha is unitless.
        alpha = float(getattr(self, 'plm_active_alpha', 0.0) or 0.0)
        if alpha > 0.0 and self.wt_seq is not None and getattr(self, 'varying_positions', None):
            plm_scores = self._score_zeroshot(unique_seqs)

            def _z(x):
                m, s = float(np.mean(x)), float(np.std(x))
                return (x - m) / s if s > 1e-12 else np.zeros_like(x)

            blended = _z(acquisition_scores) + alpha * _z(plm_scores)
            corr = float(np.corrcoef(acquisition_scores, plm_scores)[0, 1]) if len(unique_seqs) > 1 else 0.0
            logger.info(
                f"Active sampling: PLM bias alpha={alpha}, "
                f"corr(acq, plm)={corr:+.3f}, "
                f"plm range [{plm_scores.min():.2f}, {plm_scores.max():.2f}]"
            )
            acquisition_scores = blended

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
        """Sample sequences from GPT model, then snap to the library subspace.

        Without this snap step, GPT generates over all 24 tokens × seq_len
        positions, which for full-length proteins (e.g. 233-aa eqFP611) produces
        sequences that exist nowhere in the enumerated landscape. The oracle
        then returns fitness=0 for every proposal and RL provides no signal.

        We mirror the "restore full sequence" logic from
        `popgen/model/agent_trainer.py::train` (around line 212): keep the
        GPT's choice at hotspot positions when it lies in the observed
        per-position alphabet, otherwise pick a random allowed AA. Non-hotspot
        positions are set to the wild-type reference. For combinatorial
        landscapes where every position varies (e.g. 4-site GB1) this is
        a no-op.
        """
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

        token_seqs = torch.cat(sequences, 1)
        aa_seqs = self.sd.matrix_to_seqs(token_seqs)

        tmpl = getattr(self, "template", None)
        if tmpl is not None and getattr(tmpl, "positions", None):
            # Reference = the "start variant": anchor generation on the best-so-far
            # sequence (refSeq tracks the start; updated each round to self.best_seq).
            # Non-hotspot positions are set to it, and the max_n_mut cap is measured
            # relative to it. Falls back to the template WT in round 0.
            ref = tmpl.refSeq
            # If generation is in the short AACombo space but refSeq is the full WT
            # (e.g. CreiLOV: 15-mer combos vs 119aa refSeq), build a combo-space
            # reference so snapping + the max_n_mut cap operate in the generated space
            # instead of being skipped by the length-mismatch guard below.
            if len(ref) != self.seq_len:
                self._ensure_combo_map()
                cp = getattr(self, "_combo_positions", None)
                if cp is not None and len(cp) == self.seq_len:
                    ref = ''.join(self._combo_full_wt[p] for p in cp)
            if getattr(self, "best_seq", None) and len(self.best_seq) == len(ref):
                ref = self.best_seq
            # positions in hotspots.csv are 1-indexed; pos_aa_candidates is a
            # dict keyed by 1-based position with lists of observed AAs.
            allowed = {(p - 1): list(tmpl.pos_aa_candidates.get(p, []))
                       for p in tmpl.positions}
            hotspot0 = sorted(allowed.keys())
            max_mut = getattr(self, "max_n_mut", None)
            rng = np.random
            snapped: List[str] = []
            for s in aa_seqs:
                if len(s) != len(ref):
                    snapped.append(s)
                    continue
                chars = list(ref)
                for pos0 in hotspot0:
                    cands = allowed.get(pos0) or []
                    gpt_aa = s[pos0]
                    if cands:
                        chars[pos0] = gpt_aa if gpt_aa in cands \
                                      else cands[rng.randint(len(cands))]
                # Cap mutations vs the reference (start variant) within hotspots:
                # randomly revert excess mutated positions back to the reference so
                # the generated variant is <= max_n_mut mutations from the start.
                if max_mut is not None:
                    mut_pos = [p for p in hotspot0 if chars[p] != ref[p]]
                    if len(mut_pos) > max_mut:
                        drop = rng.choice(mut_pos, size=len(mut_pos) - max_mut,
                                          replace=False)
                        for p in drop:
                            chars[p] = ref[p]
                snapped.append(''.join(chars))
            aa_seqs = snapped

        return aa_seqs

    def _hybrid_sample(
        self,
        seqs: List[str],
        n_samples: int,
        exclude_seqs: set = None,
        acquisition: str = 'ts',
        xi: float = 4.0,
    ) -> List[str]:
        """
        Hybrid selection combining UCB, PLM, and clustering.

        Algorithm:
        1. Compute per-candidate score = z(ucb_or_TS) + alpha * z(plm).
        2. KMeans cluster the candidate pool by one-hot features.
        3. Within each cluster, sort by combined score (descending).
        4. Distribute n_samples slots across clusters in round-robin, taking
           the next-best candidate from each cluster until quota is filled.

        This ensures (a) diverse coverage across sequence-space basins via
        clustering, (b) within-basin quality via the blended score, and
        (c) biological plausibility via the PLM term.
        """
        from sklearn.cluster import KMeans

        if exclude_seqs is None:
            exclude_seqs = set()
        seen = set()
        unique_seqs = []
        for s in seqs:
            if s not in exclude_seqs and s not in seen:
                unique_seqs.append(s)
                seen.add(s)
        if len(unique_seqs) == 0:
            return []
        if len(unique_seqs) <= n_samples:
            return unique_seqs

        # --- Surrogate score ---
        mu, sigma, all_predictions = self._predict_surrogate_with_samples(unique_seqs)
        if acquisition == 'ts':
            n_seqs = len(unique_seqs)
            n_models = all_predictions.shape[1]
            idx_m = np.random.randint(0, n_models, size=n_seqs)
            ucb_scores = np.array([all_predictions[i, idx_m[i]] for i in range(n_seqs)])
        elif acquisition == 'ucb':
            ucb_scores = mu + xi * sigma
        elif acquisition == 'ei':
            best_so_far = max(self.collected_fitness) if self.collected_fitness else 0
            from scipy.stats import norm
            z_arr = (mu - best_so_far) / (sigma + 1e-8)
            ucb_scores = (mu - best_so_far) * norm.cdf(z_arr) + sigma * norm.pdf(z_arr)
        else:
            ucb_scores = mu + xi * sigma

        # --- PLM score (cached after first call) ---
        alpha = float(self.hybrid_alpha)
        plm_scores = None
        if alpha > 0.0 and self.wt_seq is not None and getattr(self, 'varying_positions', None):
            try:
                plm_scores = self._score_zeroshot(unique_seqs)
            except Exception as e:
                logger.warning(f"Hybrid: PLM scoring failed ({e}); falling back to alpha=0")
                plm_scores = None

        def _z(x):
            s = float(np.std(x))
            return (x - float(np.mean(x))) / s if s > 1e-12 else np.zeros_like(x)
        if plm_scores is not None:
            combined = _z(ucb_scores) + alpha * _z(plm_scores)
            corr = float(np.corrcoef(ucb_scores, plm_scores)[0, 1]) if len(unique_seqs) > 1 else 0.0
        else:
            combined = ucb_scores
            corr = 0.0

        # --- Clustering on one-hot features ---
        try:
            from popscorer.fitness.aa_onehot_pred.embed import seqs2feat
            X = seqs2feat(unique_seqs)
        except Exception:
            # Fallback: build one-hot here directly.
            AAs = "ACDEFGHIKLMNPQRSTVWY"
            aa2 = {a: i for i, a in enumerate(AAs)}
            L = len(unique_seqs[0])
            X = np.zeros((len(unique_seqs), L * 20), dtype=np.float32)
            for i, s in enumerate(unique_seqs):
                for j, a in enumerate(s):
                    k = aa2.get(a)
                    if k is not None:
                        X[i, j * 20 + k] = 1.0

        K = int(self.hybrid_n_clusters)
        K = max(1, min(K, len(unique_seqs) // 2, n_samples))
        kmeans = KMeans(n_clusters=K, random_state=self.seed, n_init=10)
        labels = kmeans.fit_predict(X)

        # Group + per-cluster sort by combined score (desc).
        clusters = [[] for _ in range(K)]
        for i, lab in enumerate(labels):
            clusters[lab].append(i)
        for cl in clusters:
            cl.sort(key=lambda i: -combined[i])

        alloc_mode = getattr(self, 'hybrid_alloc', 'weighted')
        picks = []

        if alloc_mode == 'roundrobin':
            # Legacy: equal slots per cluster, take top-i from each in round-robin.
            cluster_order = sorted(range(K),
                                   key=lambda k: -(combined[clusters[k][0]] if clusters[k] else -np.inf))
            ptrs = [0] * K
            while len(picks) < n_samples:
                progressed = False
                for k in cluster_order:
                    if ptrs[k] < len(clusters[k]):
                        picks.append(unique_seqs[clusters[k][ptrs[k]]])
                        ptrs[k] += 1
                        progressed = True
                        if len(picks) >= n_samples:
                            break
                if not progressed:
                    break
            allocation_summary = [n_samples // K] * K

        else:
            # Weighted slot allocation: number of picks per cluster ∝
            # softmax(cluster_max_score / T), with a min-per-cluster floor so
            # every cluster contributes at least one pick (diversity guarantee).
            T = max(float(getattr(self, 'hybrid_temperature', 1.0)), 1e-6)
            min_per = int(getattr(self, 'hybrid_min_per_cluster', 1))

            # Cluster quality = max combined score among cluster members.
            non_empty = [k for k in range(K) if clusters[k]]
            quality = np.full(K, -np.inf)
            for k in non_empty:
                quality[k] = combined[clusters[k][0]]

            # Reserve min_per slots per non-empty cluster, distribute rest.
            n_eff = len(non_empty)
            reserved = min(min_per * n_eff, n_samples)
            remaining = max(n_samples - reserved, 0)

            slots = np.zeros(K, dtype=int)
            for k in non_empty:
                slots[k] = min_per

            if remaining > 0 and n_eff > 0:
                q = quality.copy()
                q[~np.isfinite(q)] = -1e9
                q = q[non_empty] / T
                q = q - q.max()
                p = np.exp(q)
                p = p / p.sum()
                raw = remaining * p
                add = np.floor(raw).astype(int)
                leftover = remaining - int(add.sum())
                if leftover > 0:
                    fracs = raw - add
                    order = np.argsort(-fracs)
                    for j in range(leftover):
                        add[order[j % n_eff]] += 1
                for idx, k in enumerate(non_empty):
                    slots[k] += int(add[idx])

            # Cap by cluster size, redistribute the deficit to next-best clusters.
            deficit = 0
            for k in non_empty:
                if slots[k] > len(clusters[k]):
                    deficit += slots[k] - len(clusters[k])
                    slots[k] = len(clusters[k])
            if deficit > 0:
                rank = sorted(non_empty, key=lambda k: -quality[k])
                idx = 0
                while deficit > 0 and idx < 10 * len(rank):
                    k = rank[idx % len(rank)]
                    if slots[k] < len(clusters[k]):
                        slots[k] += 1
                        deficit -= 1
                    idx += 1

            # Take top-slots[k] from each cluster.
            for k in range(K):
                for j in range(int(slots[k])):
                    if j < len(clusters[k]):
                        picks.append(unique_seqs[clusters[k][j]])
            picks = picks[:n_samples]
            allocation_summary = slots.tolist()

        cluster_sizes = [len(c) for c in clusters]
        logger.info(
            f"Hybrid sampling: K={K}, alpha={alpha}, alloc={alloc_mode}, "
            f"corr(ucb,plm)={corr:+.3f}; cluster sizes (top-5) "
            f"{sorted(cluster_sizes, reverse=True)[:5]}; "
            f"slots (top-5) {sorted(allocation_summary, reverse=True)[:5]}; "
            f"selected {len(picks)}"
        )
        return picks

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

    def _generate_pool_no_rl(self, n_pool: int) -> Tuple[List[str], np.ndarray]:
        """Ablation (--no_rl): generate a candidate pool by sampling from the prior/agent
        GPT *without* REINFORCE, then score with the surrogate. Tests whether the policy-
        gradient step adds value over plain generate-and-prioritize."""
        self.agent_model.eval()
        pool: Dict[str, None] = {}
        tries = 0
        while len(pool) < n_pool and tries < 500:
            seqs = self._filter_valid_seqs(self.sample_from_model(self.agent_model, self.batch_size))
            for s in set(seqs):
                pool.setdefault(s, None)
            tries += 1
        seqs = list(pool.keys())
        if not seqs:
            return [], np.array([])
        _ucb, mu = self._predict_surrogate(seqs)
        logger.info(f"  no-RL pool: {len(seqs)} unique candidates sampled from prior, scored by surrogate")
        return seqs, np.asarray(mu, dtype=float)

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

            # Get agent likelihood — single teacher-forced forward pass instead
            # of the position-by-position loop. The two are mathematically
            # equivalent: for each position pos in [0, L-1], log p(token_pos |
            # start, token_0..pos-1) is the logit at sequence position `pos`
            # when the model is fed (start, token_0..pos-1, ..., token_L-2).
            # The loop variant was O(L²) in activation memory because all L
            # forward passes' activations were retained for backward; the
            # single pass is O(L) and fits in 4-layer 233-context GPU memory.
            self.agent_model.train()
            start = rnn_start_token_vector(len(unique_seqs), self.device)
            x_full = torch.cat([start, token_tensor[:, :-1]], dim=1)
            logits, _ = self.agent_model(x_full)
            log_probs = F.log_softmax(logits, dim=-1)
            # nll_loss returns inputs[target] = log-prob of the selected token (NEGATIVE).
            # Summing across positions gives a NEGATIVE total. Keeping the negative sign
            # ensures the 5e3*(1/x) regularizer pushes likelihoods AWAY from 0 (entropy
            # bonus, prevents mode collapse). Negating would recreate the loss=inf failure
            # mode: agent → 0 ⟹ 1/x → +∞.
            sample_log_probs = log_probs.gather(
                2, token_tensor.unsqueeze(-1)
            ).squeeze(-1).sum(dim=1)
            agent_likelihoods = sample_log_probs
            prior_likelihoods = self.likelihood(self.prior_model, token_tensor)

            # Get surrogate predictions
            ucb_scores, raw_scores = self._predict_surrogate(unique_seqs)

            # PLM reward shaping: blend in z(plm_log_prob) with a round-decayed weight.
            lam_round = self._plm_reward_lambda_for_round(round_idx)
            if lam_round > 0.0 and self.wt_seq is not None and getattr(self, 'varying_positions', None):
                try:
                    plm_scores = self._score_zeroshot(unique_seqs)
                    def _z(x):
                        s = float(np.std(x))
                        return (x - float(np.mean(x))) / s if s > 1e-12 else np.zeros_like(x)
                    blended = _z(ucb_scores) + lam_round * _z(plm_scores)
                    reward_scores = blended
                    if step == 0:
                        corr = float(np.corrcoef(ucb_scores, plm_scores)[0, 1]) if len(unique_seqs) > 1 else 0.0
                        logger.info(
                            f"  PLM reward shaping (round {round_idx + 1}): "
                            f"lambda={lam_round:.3f}, corr(ucb,plm)={corr:+.3f}, "
                            f"plm_range [{plm_scores.min():.2f}, {plm_scores.max():.2f}]"
                        )
                except Exception as e:
                    logger.warning(f"PLM reward shaping failed this step ({e}); falling back to ucb only")
                    reward_scores = ucb_scores
            else:
                reward_scores = ucb_scores
            scores = torch.from_numpy(np.ascontiguousarray(reward_scores)).float().to(self.device)

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
            total_norm = torch.nn.utils.clip_grad_norm_(self.agent_model.parameters(), 1.0)
            # Non-finite guard: the exploration regularizer 5e3*(1/agent_likelihoods)
            # can hit +/-inf on short sequences (e.g. 28-aa AAV) when a likelihood
            # approaches 0, producing NaN grads that corrupt the GPT (later sampling
            # then fails the Categorical simplex check). Skip such degenerate steps;
            # all finite steps update exactly as before.
            if torch.isfinite(loss) and torch.isfinite(total_norm):
                self.optimizer.step()
            else:
                if step % 50 == 0:
                    logger.warning(f"  skipped non-finite REINFORCE step {step+1} "
                                   f"(loss={loss.item():.3g}, grad_norm={float(total_norm):.3g})")

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

    def _train_gpt_ensemble(
        self, n_steps: int, round_idx: int = 0
    ) -> Tuple[List[str], np.ndarray]:
        """Train K GPT ensemble members with data + surrogate bagging.

        For each ensemble member k (k = 1..K):
          1) Bootstrap-sample ``collected_seqs`` (with replacement).
          2) Train a fresh surrogate on the bootstrap sample.
          3) Reset agent_model from prior with a member-specific sub-seed.
          4) Run REINFORCE on the bootstrap-fitted surrogate.
          5) Merge proposals into the combined pool.

        After all K members finish, the surrogate is set to the concatenated
        list of all K×5 sub-models so that downstream ``_active_sample`` /
        ``_predict_surrogate`` calls see a properly-bagged ensemble — the
        across-bootstrap variance becomes the natural epistemic uncertainty
        used for Thompson Sampling / UCB.

        Final score per pooled sequence = bagged surrogate mean (raw), so the
        caller's active sampler picks via the disagreement-aware acquisition.
        """
        K = max(1, int(getattr(self, "n_gpt_ensemble", 1)))
        if K == 1:
            return self._train_gpt_on_surrogate(n_steps, round_idx)

        combined_pool: Dict[str, None] = {}
        prior_state = {kk: v.detach().clone() for kk, v in self.prior_model.state_dict().items()}

        # Save original collected data; we'll temporarily replace it with bootstraps
        orig_collected_seqs = list(self.collected_seqs)
        orig_collected_fitness = list(self.collected_fitness)
        n = len(orig_collected_seqs)

        all_sub_models: list = []  # accumulate K × 5 sub-models for super-ensemble

        for k in range(K):
            # ---- 1A: Bootstrap collected data for this member ----
            rng = np.random.RandomState(self.seed + 7919 * (k + 1) + round_idx * 31)
            idx = rng.choice(n, size=n, replace=True)
            self.collected_seqs = [orig_collected_seqs[i] for i in idx]
            self.collected_fitness = [orig_collected_fitness[i] for i in idx]

            # ---- 3A: Train surrogate on bootstrap sample ----
            self._train_surrogate()
            member_models = list(self.surrogate_models)  # snapshot
            all_sub_models.extend(member_models)

            # Reset agent to prior, re-seed for diversity
            self.agent_model.load_state_dict({kk: v.clone() for kk, v in prior_state.items()})
            torch.manual_seed(self.seed + 7919 * (k + 1) + round_idx * 31)
            try:
                self.optimizer = self.agent_model.configure_optimizers(self.optim_config)
            except Exception:
                self.optimizer = torch.optim.Adam(self.agent_model.parameters(), lr=1e-4)

            logger.info(f"  Ensemble member {k + 1}/{K} (bootstrap n={n}) starting REINFORCE...")
            seqs, _scores = self._train_gpt_on_surrogate(n_steps, round_idx)
            for s in seqs:
                if s not in combined_pool:
                    combined_pool[s] = None
            logger.info(f"  Ensemble member {k + 1}/{K} done — pool size: {len(combined_pool)}")

        # Restore original collected data
        self.collected_seqs = orig_collected_seqs
        self.collected_fitness = orig_collected_fitness

        # ---- 2B / 3A combined: install super-ensemble surrogate ----
        # All K × 5 sub-models become "the" surrogate for downstream selection.
        # Across-bootstrap variance becomes proper epistemic uncertainty.
        self.surrogate_models = all_sub_models
        self.surrogate_trained = True

        seqs_out = list(combined_pool.keys())
        if not seqs_out:
            return [], np.array([])
        # Final scores = bagged raw mean (mu), so _active_sample will compute UCB/TS
        # using the K×5 super-ensemble's std as uncertainty.
        _ucb, mu = self._predict_surrogate(seqs_out)
        scores_out = np.asarray(mu, dtype=float)
        logger.info(
            f"GPT ensemble ({K} members, {len(all_sub_models)} surrogate sub-models) "
            f"produced {len(seqs_out)} unique proposals"
        )
        return seqs_out, scores_out

    def train(self) -> Tuple[List[str], List[float], List[float]]:
        """Run iterative training for n_rounds."""
        logger.info(f"Starting iterative training: {self.n_rounds} rounds, {self.batch_size} samples/round")
        logger.info(f"Sampling strategy: {self.sampling_strategy}")

        for round_idx in range(self.n_rounds):
            round_start = datetime.now()
            is_last_round = (round_idx == self.n_rounds - 1)
            # Track round for round-staggered zero-shot blending (Plan E).
            self._current_round_idx = round_idx

            logger.info(f"\n{'='*50}")
            logger.info(f"Round {round_idx + 1}/{self.n_rounds}" + (" (LAST)" if is_last_round else ""))
            logger.info(f"{'='*50}")

            # Free unused cached blocks between rounds. On hosts where the
            # NVML kernel/userland versions disagree, PyTorch's caching
            # allocator can wedge mid-run after the GPT training in Round 1;
            # explicitly releasing cached blocks before each subsequent round
            # avoids that failure path.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if round_idx == 0:
                # Round 1: Initial sampling from landscape
                collected_set = set(self.collected_seqs)

                # Optional PLM-quota: reserve top-by-PLM picks from the entire library.
                plm_n = self._plm_quota_for_round(round_idx)
                plm_picks = self._plm_top_picks(self.all_seqs, plm_n, exclude=collected_set)
                remaining = self.batch_size - len(plm_picks)
                excl = collected_set | set(plm_picks)

                if self.sampling_strategy == 'cluster':
                    logger.info(f"Cluster-based initialization (+{len(plm_picks)} PLM-quota)...")
                    rest = self._cluster_init_sample(remaining, exclude=excl) if remaining > 0 else []
                else:
                    logger.info(f"Random sampling from landscape (+{len(plm_picks)} PLM-quota)...")
                    rest = self._sample_initial_seqs(remaining, exclude=excl) if remaining > 0 else []
                new_seqs = list(plm_picks) + list(rest)

            else:
                # Rounds 2+: optionally prune the per-position alphabet via SHAP,
                # then finetune prior, train surrogate, run RL.
                if self.shap_prune_alphabet:
                    self._update_alphabet_via_shap(round_idx)

                if self.finetune_prior:
                    logger.info(f"Step 1: Finetuning prior on {len(self.collected_seqs)} sequences...")
                    self._finetune_prior_model(
                        seqs=self.collected_seqs,
                        fitness=np.array(self.collected_fitness),
                        n_epochs=self.n_finetune_epochs,
                    )

                logger.info(f"Step 2: Training surrogate on {len(self.collected_seqs)} samples...")
                self._train_surrogate()

                if getattr(self, 'no_rl', False):
                    logger.info("Step 3: --no_rl set; generating pool from prior (no REINFORCE)...")
                    generated_seqs, generated_fitness = self._generate_pool_no_rl(
                        max(self.top_k_cutoff * 5, self.batch_size * 50)
                    )
                else:
                    logger.info(f"Step 3: Training GPT for {self.n_steps_per_round} steps...")
                    generated_seqs, generated_fitness = self._train_gpt_ensemble(
                        self.n_steps_per_round, round_idx=round_idx
                    )

                # Drop GPT proposals that violate per-position observed alphabets
                if self.constrain_alphabet:
                    before = len(generated_seqs)
                    generated_seqs, generated_fitness = self._filter_on_alphabet(
                        generated_seqs, generated_fitness
                    )
                    logger.info(f"  Alphabet-constraint filter: kept {len(generated_seqs)}/{before} proposals")

                collected_set = set(self.collected_seqs)

                # Optional PLM-quota: take top-by-PLM picks from the RL pool.
                plm_n = self._plm_quota_for_round(round_idx)
                plm_picks = self._plm_top_picks(generated_seqs, plm_n, exclude=collected_set)
                remaining = self.batch_size - len(plm_picks)
                excl = collected_set | set(plm_picks)

                if self.sampling_strategy == 'cluster':
                    apply_cutoff = not is_last_round
                    logger.info(f"Cluster sampling (+{len(plm_picks)} PLM-quota, cutoff={apply_cutoff})...")
                    rest = self._cluster_sample(
                        seqs=generated_seqs,
                        predicted_fitness=generated_fitness,
                        n_samples=remaining,
                        exclude_seqs=excl,
                        apply_cutoff=apply_cutoff,
                    ) if remaining > 0 else []
                    new_seqs = list(plm_picks) + list(rest)

                elif self.sampling_strategy == 'active':
                    logger.info(f"Active sampling (+{len(plm_picks)} PLM-quota, {self.acquisition})...")
                    rest = self._active_sample(
                        seqs=generated_seqs,
                        n_samples=remaining,
                        exclude_seqs=excl,
                        acquisition=self.acquisition,
                        xi=self.xi,
                    ) if remaining > 0 else []
                    new_seqs = list(plm_picks) + list(rest)

                elif self.sampling_strategy == 'hybrid':
                    logger.info(f"Hybrid sampling (+{len(plm_picks)} PLM-quota)...")
                    rest = self._hybrid_sample(
                        seqs=generated_seqs,
                        n_samples=remaining,
                        exclude_seqs=excl,
                        acquisition=self.acquisition,
                        xi=self.xi,
                    ) if remaining > 0 else []
                    new_seqs = list(plm_picks) + list(rest)

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
    # Prefer AACombo (short combinatorial form) when present
    _seq_col = ('AACombo' if 'AACombo' in df.columns
                else 'Combo' if 'Combo' in df.columns
                else 'seq' if 'seq' in df.columns
                else 'sequence')
    sequences = df[_seq_col].tolist()
    fitness = _scalarized_fitness(df)
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
    surrogate_kind: str = 'ensemble',
    single_surrogate: bool = False,
    no_rl: bool = False,
    prior_model_path: Optional[str] = None,
    constrain_alphabet: bool = False,
    features_kind: str = 'onehot',
    n_gpt_ensemble: int = 1,
    plm_zeroshot_pool_frac: float = 1.0,
    plm_zeroshot_explore_frac: float = 0.1,
    plm_zeroshot_temperature: float = 0.0,
    plm_active_alpha: float = 0.0,
    shap_prune_alphabet: bool = False,
    shap_prune_threshold: float = 0.0,
    shap_prune_min_alphabet: int = 3,
    shap_prune_min_samples: int = 50,
    shap_prune_topk_keep: int = 10,
    plm_reward_lambda: float = 0.0,
    plm_reward_decay: str = 'linear',
    hybrid_alpha: float = 0.3,
    hybrid_n_clusters: int = 12,
    hybrid_alloc: str = 'weighted',
    hybrid_temperature: float = 1.0,
    hybrid_min_per_cluster: int = 1,
    plm_sampling_frac: float = 0.0,
    plm_sampling_until_round: int = 0,
    plm_reward_until_round: int = 0,
    shap_prune_start_round: int = 0,
    use_mutcompute: bool = False,
    mutcompute_offset: Optional[int] = None,
    zeroshot_blend: float = 0.0,
    zeroshot_early_blend: Optional[float] = None,
    zeroshot_early_rounds: int = 0,
    use_oracle: bool = False,
    oracle_dir: Optional[str] = None,
    max_n_mut: Optional[int] = None,
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
        # Derive the hotspots (search-space definition) directly from the
        # library so the GPT decoder is constrained to the actual
        # combinatorial subspace covered by the dataset (e.g. 2^13 = 8192
        # for eqFP611_joint, 20^4 for the 4site_* datasets).
        try:
            _df = pd.read_csv(landscape_path)
            _seq_col = ('AACombo' if 'AACombo' in _df.columns
                        else 'Combo' if 'Combo' in _df.columns
                        else 'seq' if 'seq' in _df.columns
                        else 'sequence')
            _library_seqs = _df[_seq_col].tolist()
        except Exception as _e:
            logger.warning(f"Could not load library for hotspots derivation: {_e}")
            _library_seqs = None
        hotspot_path = create_temp_hotspots(seq_len, temp_dir,
                                            sequences=_library_seqs)

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
        surrogate_kind=surrogate_kind,
        prior_model_path=prior_model_path,
        constrain_alphabet=constrain_alphabet,
        features_kind=features_kind,
        wt_seq=wildtype,
        n_gpt_ensemble=n_gpt_ensemble,
        plm_zeroshot_pool_frac=plm_zeroshot_pool_frac,
        plm_zeroshot_explore_frac=plm_zeroshot_explore_frac,
    )
    trainer.plm_zeroshot_temperature = plm_zeroshot_temperature
    trainer.plm_active_alpha = plm_active_alpha
    trainer.shap_prune_alphabet = shap_prune_alphabet
    trainer.shap_prune_threshold = shap_prune_threshold
    trainer.shap_prune_min_alphabet = shap_prune_min_alphabet
    trainer.shap_prune_min_samples = shap_prune_min_samples
    trainer.shap_prune_topk_keep = shap_prune_topk_keep
    trainer.plm_reward_lambda = plm_reward_lambda
    trainer.plm_reward_decay = plm_reward_decay
    trainer.hybrid_alpha = hybrid_alpha
    trainer.hybrid_n_clusters = hybrid_n_clusters
    trainer.hybrid_alloc = hybrid_alloc
    trainer.hybrid_temperature = hybrid_temperature
    trainer.hybrid_min_per_cluster = hybrid_min_per_cluster
    trainer.plm_sampling_frac = plm_sampling_frac
    trainer.plm_sampling_until_round = plm_sampling_until_round
    trainer.plm_reward_until_round = plm_reward_until_round
    trainer.shap_prune_start_round = shap_prune_start_round
    trainer.single_surrogate = single_surrogate
    trainer.no_rl = no_rl
    trainer.use_mutcompute = use_mutcompute
    trainer.mutcompute_offset = mutcompute_offset
    trainer.zeroshot_blend = zeroshot_blend
    trainer.zeroshot_early_blend = zeroshot_early_blend
    trainer.zeroshot_early_rounds = zeroshot_early_rounds

    # Multi-site learned-oracle mode: swap the data.csv lookup for the CNN oracle and
    # set the normalization reference to the oracle's [0,1] scale.
    trainer.max_n_mut = max_n_mut
    trainer.use_oracle = use_oracle
    if use_oracle:
        # data_dir is <benchmark>/data (symlink-safe root resolution).
        _bench_root = os.path.dirname(os.path.abspath(data_dir))
        sys.path.insert(0, _bench_root)
        from utils.oracle_landscape import OracleLandscape
        _odir = oracle_dir or os.path.join(_bench_root, 'oracles')
        trainer.oracle_landscape = OracleLandscape(dataset, oracle_dir=_odir, device=device)
        # Normalization reference = the oracle's (measured-data) fitness range, so the
        # surrogate/reward behave exactly as in lookup mode (which used measured max/min).
        trainer.max_fitness_raw = trainer.oracle_landscape.fit_max
        trainer.min_fitness_raw = trainer.oracle_landscape.fit_min
        logger.info(f"Oracle mode: {trainer.oracle_landscape} "
                    f"(reward in raw units, range [{trainer.min_fitness_raw:.3g},"
                    f"{trainer.max_fitness_raw:.3g}])")

    # Run iterative training
    start_time = datetime.now()
    all_seqs, all_predicted, all_oracle = trainer.train()
    runtime = (datetime.now() - start_time).total_seconds()

    logger.info(f"Training completed in {runtime:.1f} seconds")
    logger.info(f"Generated {len(all_seqs)} sequences ({len(set(all_seqs))} unique)")

    # Multi-site oracle mode: write a results_oracle-compatible JSON using the SAME
    # metric definitions as scripts/run_oracle_benchmark.py (max_fitness_norm,
    # top128_mean_norm, ...) so AlphaVariant drops into the 9-method comparison.
    if use_oracle:
        _bench_root = os.path.dirname(os.path.abspath(data_dir))
        sys.path.insert(0, os.path.join(_bench_root, 'scripts'))
        from run_oracle_benchmark import compute_metrics as _oracle_metrics
        fraw = np.asarray(all_oracle, dtype=float)          # raw oracle units (reward scale)
        ls = trainer.oracle_landscape
        fn = (fraw - ls.fit_min) / (ls.scale + 1e-12)       # normalized [0,1] for metrics
        metrics = _oracle_metrics(list(all_seqs), fn, fraw, wildtype,
                                  trainer.varying_positions, batch_size, n_rounds)
        res = {'method': 'AlphaVariant', 'dataset': dataset, 'seed': seed,
               'n_queries': len(all_seqs), 'oracle_test_spearman': ls.test_spearman,
               'runtime_seconds': runtime, 'metrics': metrics}
        sub = os.path.join(output_path, dataset, 'AlphaVariant')
        os.makedirs(sub, exist_ok=True)
        with open(os.path.join(sub, f'seed{seed}.json'), 'w') as f:
            json.dump(res, f, indent=2)
        print(f"  [oracle] {dataset}/AlphaVariant/seed{seed}: "
              f"max={metrics['max_fitness_norm']:.4f} "
              f"top128={metrics['top128_mean_norm']:.4f}")
        return res

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
            'prior_model_path': prior_model_path,
            'finetune_prior': finetune_prior,
            'constrain_alphabet': constrain_alphabet,
            'features': features_kind,
            'surrogate': surrogate_kind,
            'sampling': sampling_strategy,
            'acquisition': acquisition,
            'level': level,
            'n_hotspot_positions': len(positions),
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

        # Map generated sequences -> landscape row indices (in query order across
        # rounds) so MOO aggregators can recover (blue, red) tuples via
        # `metrics["queried_indices"]`. Sequences not found in the enumerated
        # landscape (e.g. unconstrained generation outside the library) are skipped.
        try:
            _seq_to_idx = {s: i for i, s in enumerate(all_landscape_seqs)}
            _queried_indices = [_seq_to_idx[s] for s in all_seqs if s in _seq_to_idx]
            result['metrics']['queried_indices'] = _queried_indices
            result['metrics']['n_in_landscape'] = len(_queried_indices)
            result['metrics']['n_off_landscape'] = len(all_seqs) - len(_queried_indices)
            logger.info(
                f"queried_indices: {len(_queried_indices)}/{len(all_seqs)} "
                f"generated sequences found in landscape "
                f"({len(all_seqs) - len(_queried_indices)} off-library)"
            )
        except Exception as _e:
            logger.warning(f"Could not record queried_indices: {_e}")

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
Examples (run from the alphavariant/ package dir — --config, --prior_model_path
and --output_path are relative to the working directory):

  # Four-site lookup landscape
  python ../scripts/alphavariant/run_generic.py --dataset 4site_GB1 --seed 621

  # Multi-site CNN oracle
  python ../scripts/alphavariant/run_generic.py --dataset ms_CreiLOV --seed 621 \
      --oracle --prior_model_path priors/ms_CreiLOV/prior_model.pt

  # Hard-level init
  python ../scripts/alphavariant/run_generic.py --dataset 4site_PhoQ --level hard --seed 621

  # Use an existing YAML config instead of auto-generating
  python ../scripts/alphavariant/run_generic.py --dataset 4site_GB1 \
      --config examples/Savinase/config/train_agent_config.yaml

  # Multiple seeds
  python ../scripts/alphavariant/run_generic.py --dataset 4site_GB1 --seeds 621 100 383

  # The 30-seed benchmark standard
  python ../scripts/alphavariant/run_generic.py --dataset 4site_GB1 \
      --seed_file ../rand_seeds.txt --num_seeds 30

  # Skip metrics computation
  python ../scripts/alphavariant/run_generic.py --dataset 4site_GB1 --seed 621 --skip_metrics

  # Custom output path
  python ../scripts/alphavariant/run_generic.py --dataset ms_PAB1 --output_path results/my_experiment/
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
    parser.add_argument("--data_dir", type=str, default=os.path.join(BENCHMARK_ROOT, "data"),
                       help="Base data directory")
    parser.add_argument("--skip_metrics", action="store_true", help="Skip metrics computation")

    # Training parameters
    parser.add_argument("--surrogate", type=str, choices=['ensemble', 'cnn'], default='ensemble',
                        help="Surrogate model: 'ensemble' (Ridge+RF+GBT, default) or 'cnn' (FLEXS-style CNN ensemble)")
    parser.add_argument("--features", type=str, choices=['onehot', 'esm2', 'ev_onehot'], default='onehot',
                        help="Surrogate features: 'onehot' (popscorer aa_onehot, default), 'ev_onehot' "
                        "(one-hot + EVmutation/plmc statistical-energy column), or 'esm2' "
                        "(facebook/esm2_t12_35M_UR50D mean-pooled over varying positions of WT-substituted sequence)")
    parser.add_argument("--constrain_alphabet", action="store_true", default=False,
                        help="Drop GPT proposals whose AAs violate the observed per-position alphabet "
                        "(library subspace constraint, FLEXS-style).")
    parser.add_argument("--n_gpt_ensemble", type=int, default=1,
                        help="Number of GPT ensemble members per round (bagging). Each member trains "
                        "n_steps_per_round REINFORCE steps from a fresh copy of the prior with a "
                        "different sub-seed; proposals are merged into one pool.")
    parser.add_argument("--level", type=str,
                       choices=['uniform', 'medium', 'hard', 'plm_zeroshot'], default='uniform',
                       help="Difficulty level for initial sampling (default: uniform — "
                            "matches all other benchmark methods; non-uniform levels are a fitness leak). "
                            "'plm_zeroshot' uses ESM-2 WT-marginal log-likelihood as a zero-shot fitness "
                            "prior to bias the first batch toward biologically plausible variants — "
                            "intended for noisy / nearly-flat initial-sample regimes (e.g. TEV).")
    parser.add_argument("--plm_zeroshot_pool_frac", type=float, default=1.0,
                       help="When --level plm_zeroshot, fraction of the library to score with ESM-2 "
                            "before picking top-batch_size. Default 1.0 = score everything; lower it "
                            "(e.g. 0.25) for very large libraries to cap ESM cost.")
    parser.add_argument("--plm_zeroshot_explore_frac", type=float, default=0.1,
                       help="Fraction of the initial batch reserved for uniform-random exploration "
                            "alongside the PLM-top selection (default 0.1 -> 10%% random for diversity).")
    parser.add_argument("--plm_zeroshot_temperature", type=float, default=0.0,
                       help="If >0, sample initial PLM-pick from softmax(score/T) via Gumbel top-k "
                            "instead of strict top-K (0 = strict top-K). Higher T = more diverse / "
                            "more weight on low-PLM candidates.")
    parser.add_argument("--plm_active_alpha", type=float, default=0.0,
                       help="If >0, blend ESM-2 WT-marginal log-prob into the active-sampling "
                            "acquisition score in rounds 2+: combined = z(acq) + alpha * z(plm). "
                            "Both terms are z-normalized within the candidate pool so alpha is "
                            "unitless. 0 = off.")
    parser.add_argument("--plm_reward_lambda", type=float, default=0.0,
                       help="If >0, blend PLM log-prob into REINFORCE reward at every RL step: "
                            "reward = z(ucb) + lambda(round) * z(plm). 0 = off.")
    parser.add_argument("--plm_reward_decay", type=str, default='linear',
                       choices=['linear', 'exponential', 'none'],
                       help="Schedule for lambda across rounds (linear default): linear ramps "
                            "to 0 at the last RL round; exponential halves each round; none = constant.")
    parser.add_argument("--hybrid_alpha", type=float, default=0.3,
                       help="Weight for PLM term in hybrid sampling: score=z(ucb)+alpha*z(plm). "
                            "0 = pure UCB + clustering.")
    parser.add_argument("--hybrid_n_clusters", type=int, default=12,
                       help="Number of KMeans clusters for diversity in hybrid sampling.")
    parser.add_argument("--hybrid_alloc", type=str, default='weighted',
                       choices=['weighted', 'roundrobin'],
                       help="How to allocate the 96 batch slots across clusters. "
                            "'weighted' (default): slots_k ∝ softmax(cluster_max_score / T) with "
                            "a min-per-cluster floor — concentrates picks on high-quality clusters "
                            "while preserving diversity. 'roundrobin' (legacy): equal slots per cluster.")
    parser.add_argument("--hybrid_temperature", type=float, default=1.0,
                       help="Softmax temperature for the weighted allocator. Lower T = more "
                            "concentrated on the best cluster; higher T = more uniform.")
    parser.add_argument("--hybrid_min_per_cluster", type=int, default=1,
                       help="Minimum number of picks per non-empty cluster (diversity floor).")
    parser.add_argument("--plm_sampling_frac", type=float, default=0.0,
                       help="Reserve this fraction of each batch for top-by-PLM picks. 0 = off.")
    parser.add_argument("--plm_sampling_until_round", type=int, default=0,
                       help="PLM-fraction is applied for rounds 1..N where N = this value. "
                            "0 = disable. Use 2 to limit to init + first RL round.")
    parser.add_argument("--plm_reward_until_round", type=int, default=0,
                       help="Force PLM reward lambda to 0 after this round (1-indexed). "
                            "0 = honor only the existing --plm_reward_decay schedule.")
    parser.add_argument("--shap_prune_start_round", type=int, default=0,
                       help="Start SHAP-based alphabet pruning from this round onward "
                            "(1-indexed; round 1 = init). 0 = use --shap_prune_min_samples gate.")
    parser.add_argument("--use_mutcompute", action="store_true", default=False,
                       help="Replace ESM-2 WT-marginal scoring with MutCompute structure-based "
                            "log-likelihood-ratio scoring for all zero-shot call sites "
                            "(reward shaping, sampling fraction, active blending). Requires "
                            "data/<dataset>/mutcompute.csv to be present.")
    parser.add_argument("--mutcompute_offset", type=int, default=None,
                       help="Manual override for the offset between 0-indexed varying positions "
                            "in wt.fasta and the 'pos' column in mutcompute.csv. Default None = "
                            "auto-detect by aligning the wtAA column against wt.fasta.")
    parser.add_argument("--zeroshot_blend", type=float, default=0.0,
                       help="Ensemble blend of MutCompute and ESM-2 in the zero-shot dispatcher "
                            "(only active when --use_mutcompute is set). 0.0 = pure MC (Plan C "
                            "default); 1.0 = pure ESM (=Plan B PLM-reward); 0 < α < 1 → "
                            "(1-α)·z(MC) + α·z(ESM). Used by Plan D D1 ensemble experiments.")
    parser.add_argument("--zeroshot_early_blend", type=float, default=None,
                       help="Plan E: round-staggered override. For the first "
                            "--zeroshot_early_rounds RL rounds (round_idx ∈ [1, N]), use this "
                            "blend value instead of --zeroshot_blend. None = no override.")
    parser.add_argument("--zeroshot_early_rounds", type=int, default=0,
                       help="Plan E: number of early RL rounds to use --zeroshot_early_blend for "
                            "(round-indexed from 1 = first RL round; round 0 is init). Default 0 "
                            "disables the round-staggered override.")
    parser.add_argument("--oracle", action="store_true", default=False,
                       help="Multi-site: use the learned CNN oracle (oracles/<dataset>/oracle.pt) "
                            "as ground-truth fitness and write a results_oracle-compatible JSON.")
    parser.add_argument("--oracle_dir", type=str, default=None,
                       help="Override oracle checkpoint dir (default <benchmark>/oracles).")
    parser.add_argument("--max_n_mut", type=int, default=None,
                       help="Cap generated variants to <= this many mutations from the "
                            "reference (refSeq = best-so-far start variant). None disables. "
                            "Use ~5 for multi-site to keep generation near the data manifold.")
    parser.add_argument("--shap_prune_alphabet", action="store_true", default=False,
                       help="Savinase-style hotspot reselection: at the start of each round 2+, "
                            "fit XGBoost on (one-hot mutation features, fitness), compute SHAP, "
                            "and drop per-position AAs whose mean SHAP <= threshold. AAs in the "
                            "top-K best variants are always retained; minimum alphabet size per "
                            "position is enforced. Sets --constrain_alphabet implicitly.")
    parser.add_argument("--shap_prune_threshold", type=float, default=0.0,
                       help="AAs with mean SHAP > this threshold are kept (default 0.0).")
    parser.add_argument("--shap_prune_min_alphabet", type=int, default=3,
                       help="Minimum alphabet size per varying position (default 3).")
    parser.add_argument("--shap_prune_min_samples", type=int, default=50,
                       help="Minimum collected samples required before pruning kicks in (default 50).")
    parser.add_argument("--shap_prune_topk_keep", type=int, default=10,
                       help="AAs appearing in the top-K best collected variants are always retained (default 10).")
    parser.add_argument("--n_rounds", type=int, default=5,
                       help="Number of iterative rounds (default: 5 -> 480 queries, benchmark-standard)")
    parser.add_argument("--n_steps_per_round", type=int, default=500,
                       help="GPT training steps per round (default: 500)")
    parser.add_argument("--batch_size", type=int, default=96,
                       help="Batch size / samples per round (default: 96)")
    parser.add_argument("--sigma", type=float, default=60,
                       help="REINFORCE sigma (default: 60)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device for training (default: cuda:0)")
    parser.add_argument("--single_surrogate", action="store_true", default=False,
                       help="Ablation: use a single RandomForest surrogate instead of the "
                            "5-model ensemble (predictive std -> 0, UCB -> mean; no ensemble scoring).")
    parser.add_argument("--no_rl", action="store_true", default=False,
                       help="Ablation: skip REINFORCE; each round, generate a candidate pool "
                            "from the prior GPT and prioritize by the surrogate (generate-and-select only).")
    parser.add_argument("--top_k_cutoff", type=int, default=1000,
                       help="Top-k cutoff for CLADE-2 sampling (default: 1000)")
    parser.add_argument("--n_clusters", type=int, default=10,
                       help="Number of clusters for sampling (default: 10)")
    parser.add_argument("--sampling", type=str, choices=['cluster', 'active', 'hybrid'], default='cluster',
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
    parser.add_argument("--prior_model_path", type=str, default=None,
                       help="Path to a pretrained GPT prior (e.g. trained on a family MSA). "
                            "If set, used as the starting GPT instead of random init.")

    parser.add_argument(
        "--ablation", type=str, default="none",
        choices=["none", "no-gpt", "no-space", "static-reward", "no-rl"],
        help="Component-removal flag (only 'none' is supported on this dataset; "
             "GB1 has full ablation support)",
    )

    if False:
        pass
    args = parser.parse_args()
    if getattr(args, 'ablation', 'none') != 'none':
        raise NotImplementedError(
            f"--ablation={args.ablation} is currently only implemented for run_GB1.py. "
            "Extend the per-dataset trainer to support ablation seams; see run_GB1.py for the pattern."
        )
    warnings.filterwarnings("ignore")

    # Set default output path based on dataset name
    if args.output_path is None:
        args.output_path = os.path.join(BENCHMARK_ROOT, "alphavariant", "results", f"{args.dataset}_AlphaVariant")

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
            surrogate_kind=args.surrogate,
            constrain_alphabet=args.constrain_alphabet,
            features_kind=args.features,
            n_gpt_ensemble=args.n_gpt_ensemble,
            prior_model_path=args.prior_model_path,
            plm_zeroshot_pool_frac=args.plm_zeroshot_pool_frac,
            plm_zeroshot_explore_frac=args.plm_zeroshot_explore_frac,
            plm_zeroshot_temperature=args.plm_zeroshot_temperature,
            plm_active_alpha=args.plm_active_alpha,
            shap_prune_alphabet=args.shap_prune_alphabet,
            shap_prune_threshold=args.shap_prune_threshold,
            shap_prune_min_alphabet=args.shap_prune_min_alphabet,
            shap_prune_min_samples=args.shap_prune_min_samples,
            shap_prune_topk_keep=args.shap_prune_topk_keep,
            plm_reward_lambda=args.plm_reward_lambda,
            plm_reward_decay=args.plm_reward_decay,
            hybrid_alpha=args.hybrid_alpha,
            hybrid_n_clusters=args.hybrid_n_clusters,
            hybrid_alloc=args.hybrid_alloc,
            hybrid_temperature=args.hybrid_temperature,
            hybrid_min_per_cluster=args.hybrid_min_per_cluster,
            plm_sampling_frac=args.plm_sampling_frac,
            plm_sampling_until_round=args.plm_sampling_until_round,
            plm_reward_until_round=args.plm_reward_until_round,
            shap_prune_start_round=args.shap_prune_start_round,
            use_mutcompute=args.use_mutcompute,
            mutcompute_offset=args.mutcompute_offset,
            zeroshot_blend=args.zeroshot_blend,
            zeroshot_early_blend=args.zeroshot_early_blend,
            zeroshot_early_rounds=args.zeroshot_early_rounds,
            use_oracle=args.oracle,
            oracle_dir=args.oracle_dir,
            max_n_mut=args.max_n_mut,
            single_surrogate=args.single_surrogate,
            no_rl=args.no_rl,
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
