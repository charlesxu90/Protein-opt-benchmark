# Protein Optimization Benchmark Suite

A comprehensive benchmarking framework for evaluating protein optimization methods on fitness landscape datasets.

## Overview

This benchmark suite provides standardized evaluation of directed evolution and protein optimization algorithms. It includes unified metrics computation, data loading utilities, and result aggregation tools to ensure fair comparison across methods.

## Datasets

| Dataset | Description | Variants | Sequence Length | Task Difficulty |
|---------|-------------|----------|-----------------|-----------------|
| **GB1** | IgG-binding domain of protein G, 4-site combinatorial library (positions 39, 40, 41, 54) | 149,361 | 4 aa (56 aa full) | Standard |
| **AAV Medium** | Adeno-associated virus capsid optimization | TBD | TBD | Medium |
| **AAV Hard** | AAV capsid with restricted starting conditions | TBD | TBD | Hard |
| **GFP Medium** | Green fluorescent protein optimization | TBD | TBD | Medium |
| **GFP Hard** | GFP with restricted starting conditions | TBD | TBD | Hard |

### GB1 Dataset Details

- **Wild-type sequence**: VDGV (4-site) / MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE (full)
- **Mutable positions**: V39, D40, G41, V54
- **Fitness range**: 0.0 - 8.76 (raw) / 0.0 - 1.0 (normalized)
- **Global maximum**: Fitness = 1.0 (normalized)

## Methods

| Method | Type | Key Features | GB1 | AAV/GFP |
|--------|------|--------------|-----|---------|
| **ALDE** | Active Learning | DNN Ensemble, Thompson Sampling, Bayesian optimization | ✓ | - |
| **EvoPlay** | MCTS + RL | AlphaZero-style Policy-Value Network, Gaussian Process | ✓ | ✓ |
| **AdaLead** | Active Learning | CNN Ensemble, Adaptive mutation, FLEXS framework | ✓ | ✓ |
| **LatProtRL** | Reinforcement Learning | PPO, ESM-2 latent space, VED encoder-decoder | ✓ | ✓ |
| **AICE** | Inverse Folding | ProteinMPNN scoring, Frequency-based filtering, LD matrix | ✓ | ✓ |
| **δ-Conservative Search** | GFlowNet | Flow-based generation, Delta-conservative radius, UCB | ✓ | ✓ |
| **AlphaVariant** | Generative | GPT model, REINFORCE training, Ensemble surrogate | ✓ | ✓ |

### Method Configurations (GB1)

All methods use:
- **Batch size**: 96 sequences per round
- **Rounds**: 5 (1 initial + 4 iterations)
- **Total queries**: 480 sequences
- **Encoding**: One-hot (most methods)

### AlphaVariant Configurations (Plan C — shipped)

AlphaVariant ships the **Plan C** configuration: base GPT-REINFORCE + surrogate
ensemble with **MutCompute** (structure-based zero-shot) reward shaping and
**SHAP**-based per-position alphabet pruning. The flags differ slightly between
the two benchmark families. Run from the `alphavariant/` method directory.

> **Environment**: AlphaVariant requires the env's `libstdc++` on the path, otherwise
> matplotlib/torch fail with a `CXXABI` error:
> ```bash
> export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:$LD_LIBRARY_PATH
> ```

**Four-site combinatorial benchmark** (`4site_GB1`, `4site_PhoQ`, `4site_TEV`, `4site_TRPB`)
— lookup-table landscape:

```bash
python run_generic.py --dataset 4site_PhoQ --seed 42 \
    --use_mutcompute --plm_reward_lambda 0.5 --shap_prune_alphabet
```

**Multi-site learned-oracle benchmark** (`ms_AAV`, `ms_CreiLOV`, `ms_GFP`, `ms_PAB1`)
— GGS/LatProtRL-style CNN oracle landscape, generative proposal over varying positions:

```bash
python run_generic.py --dataset ms_GFP --seed 42 \
    --oracle --level uniform \
    --prior_model_path priors/ms_GFP/prior_model.pt \
    --use_mutcompute --shap_prune_alphabet \
    --n_rounds 5 --n_steps_per_round 500 --device cuda:0 \
    --data_dir ../data
```

| Flag | Four-site | Multi-site | Role |
|------|:---------:|:----------:|------|
| `--use_mutcompute` | ✓ | ✓ | Use MutCompute (not ESM-2) as the zero-shot scorer |
| `--plm_reward_lambda 0.5` | ✓ | — | Blend MutCompute z-score into the reward (λ decays over rounds) |
| `--shap_prune_alphabet` | ✓ | ✓ | SHAP per-position alphabet pruning (gated to oracle mode for constrain) |
| `--oracle --prior_model_path …` | — | ✓ | Score via the trained CNN oracle; GPT prior from aligned homologs |
| `--level uniform` | — | ✓ | Uniform hotspot weighting |

Shared defaults (both families): `--batch_size 96 --n_rounds 5 --n_steps_per_round 500 --sigma 60`.
Multi-site priors are trained per dataset via `scripts/alphavariant/train_ms_prior.py` and
saved to `alphavariant/priors/<dataset>/prior_model.{pt,json}`. Sweep launchers:
`scripts/alphavariant/_sweep_av_oracle.sh` (multi-site).

## Metrics

All metrics are computed consistently across methods using the unified `utils/` module.

| Metric | Description | Reference | Range |
|--------|-------------|-----------|-------|
| **High-Fitness Proximity (d_high)** | Median min distance from generated sequences to top 10% fitness sequences | LatProtRL | 0+ (lower is better) |
| **Novelty (d_init)** | Median min distance to initial training set | LatProtRL | 0+ (higher is better) |
| **Batch Diversity** | Median pairwise Hamming distance within generated batch | Energy Matching | 0+ (higher is better) |
| **Normalized Fitness (Top-128)** | Median fitness of top 128 sequences, normalized to [0,1] | GGS | [0, 1] |
| **Normalized Fitness (Top-256)** | Median fitness of top 256 sequences, normalized to [0,1] | GGS | [0, 1] |
| **Max Fitness** | Absolute highest fitness discovered | δ-CS | [0, 1] |
| **Spearman Correlation (ρ)** | Rank correlation between predicted and true fitness | μProtein | [-1, 1] |
| **Epistatic Correlation** | Spearman correlation of non-additive mutational effects | μProtein | [-1, 1] |
| **Recall of High-Order Mutants** | % of true top multi-point mutants correctly identified | μProtein | [0, 1] |
| **Simple Regret (r_t)** | Gap between global optimum and best found at round t | VSD | 0+ (lower is better) |
| **Global Max Hit Count** | Number of runs finding the global maximum (GB1 only) | EvoPlay | Count |
| **Miscalibration Area** | Area between calibration curve and ideal diagonal | ALDE | [0, 1] |
| **Expected Calibration Error** | Weighted average of calibration errors | ALDE | [0, 1] |

## Project Structure

```
Benchmark/
├── README.md                 # This file
├── INTEGRATION.md            # Integration guide for methods
├── Prompts.md                # Task prompts
├── rand_seeds.txt            # 500 random seeds for reproducibility
│
├── utils/                    # Unified benchmark utilities
│   ├── __init__.py           # Package exports
│   ├── metrics.py            # Core metric implementations
│   ├── data.py               # Data loading utilities
│   ├── evaluator.py          # BenchmarkEvaluator class
│   ├── io.py                 # Results I/O and aggregation
│   ├── gb1.py                # GB1-specific utilities
│   └── compat.py             # Drop-in compatibility for ALDE interface
│
├── data/                     # Benchmark datasets
│   └── GB1/
│       └── data.csv          # GB1 fitness landscape
│
├── ALDE/                     # ALDE implementation
│   ├── run_GB1.py
│   └── src/
│
├── EvoPlay/                  # EvoPlay implementation
│   ├── run_GB1.py
│   └── code/
│
├── FLEXS/                    # AdaLead implementation (FLEXS framework)
│   ├── run_GB1_adalead.py
│   └── flexs/
│
├── LatProtRL/                # LatProtRL implementation
│   ├── run_GB1.py
│   └── net/
│
├── AiCE/                     # AICE implementation
│   ├── run_GB1.py
│   └── scripts/
│
├── delta_cs/                 # δ-Conservative Search implementation
│   └── BioSeq-GFN-AL/
│       ├── run_GB1.py
│       └── lib/
│
└── alphavariant/             # AlphaVariant implementation
    ├── run_GB1.py
    └── popgen/
```

## Quick Start

### Using Unified Utilities

```python
import sys
sys.path.insert(0, '/path/to/Benchmark')

# Option 1: Use GB1-specific utilities
from utils import (
    load_gb1_landscape,
    compute_gb1_metrics,
    print_gb1_metrics_summary,
)

sequences, fitness, wildtype = load_gb1_landscape('/path/to/data')
metrics = compute_gb1_metrics(
    queried_sequences=my_sequences,
    queried_fitness=my_fitness,
    all_sequences=sequences,
    all_fitness=fitness,
    initial_sequences=init_seqs,
)
print_gb1_metrics_summary(metrics)

# Option 2: Use compatibility layer (drop-in for ALDE interface)
from utils.compat import (
    compute_all_metrics,
    load_landscape_data,
    MetricsResult,
)
```

### Running Experiments

```bash
# Single run with specific seed
python run_GB1.py --seed 42

# Multiple runs for statistical evaluation
python run_GB1.py --seeds 42 123 456 789 1000

# Use predefined seeds from file
python run_GB1.py --seed_file ../rand_seeds.txt --num_seeds 10

# Skip metrics computation (faster, for debugging)
python run_GB1.py --seed 42 --skip_metrics
```

## Installation

### Requirements

```
numpy>=1.20
scipy>=1.7
pandas>=1.3
torch>=1.9  # For neural network methods
```

### Optional Dependencies

```
scikit-learn>=1.0  # For ensemble models
wandb              # For experiment tracking
```

## TODOs

### High Priority

- [ ] Implement AAV/GFP medium benchmark dataset
- [ ] Implement AAV/GFP hard benchmark dataset
- [ ] Add cross-method comparison scripts
- [ ] Create visualization utilities for trajectories

### Methods

- [ ] Validate EvoPlay integration with unified utils
- [ ] Validate AdaLead integration with unified utils
- [ ] Validate LatProtRL integration with unified utils
- [ ] Validate AICE integration with unified utils
- [ ] Validate δ-Conservative Search integration with unified utils
- [ ] Validate AlphaVariant integration with unified utils

### Metrics & Analysis

- [ ] Add per-round metric tracking visualization
- [ ] Implement statistical significance tests (Wilcoxon, bootstrap)
- [ ] Add convergence analysis utilities
- [ ] Create LaTeX table export for papers

### Documentation

- [ ] Add detailed API documentation
- [ ] Create Jupyter notebook tutorials
- [ ] Add troubleshooting guide
- [ ] Document hyperparameter sensitivity

### Infrastructure

- [ ] Add CI/CD pipeline for testing
- [ ] Create Docker container for reproducibility
- [ ] Add multi-GPU support for large-scale experiments
- [ ] Implement checkpoint/resume functionality

## Usage Examples

### Computing Metrics for a Run

```python
from utils.compat import compute_all_metrics, load_landscape_data
import numpy as np

# Load landscape
sequences, fitness = load_landscape_data('GB1', data_dir='./data')
fitness = fitness / np.max(fitness)  # Normalize

# Your optimization results
queried_indices = np.array([...])  # Indices of queried sequences
initial_indices = np.array([...])  # Indices of initial samples

# Compute metrics
metrics = compute_all_metrics(
    queried_indices=queried_indices,
    all_sequences=sequences,
    all_fitness=fitness,
    initial_indices=initial_indices,
    wildtype='VDGV',
    batch_size=96
)

print(f"Max Fitness: {metrics.max_fitness:.4f}")
print(f"Simple Regret: {metrics.simple_regret:.4f}")
```

### Aggregating Multiple Runs

```python
from utils.compat import aggregate_run_metrics, global_max_hit_count

# Collect results from multiple runs
all_results = [metrics_run1, metrics_run2, ...]

# Aggregate statistics
aggregated = aggregate_run_metrics(all_results)

print(f"Max Fitness: {aggregated['max_fitness']['mean']:.4f} ± {aggregated['max_fitness']['std']:.4f}")

# Global max hit rate
hit_count, hit_rate = global_max_hit_count(
    [r.max_fitness for r in all_results],
    global_max=1.0,
    tolerance=0.01
)
print(f"Global Max Hit Rate: {hit_rate*100:.1f}% ({hit_count}/{len(all_results)})")
```

## References

- **ALDE**: Active Learning for Directed Evolution
- **EvoPlay**: AlphaZero-inspired protein evolution
- **LatProtRL**: Reinforcement Learning in Latent Space
- **AICE**: AI-guided Combinatorial Editing
- **δ-Conservative Search**: GFlowNet with conservative radius
- **AlphaVariant**: GPT-based generative optimization
- **GGS**: Normalized fitness metrics
- **μProtein**: Model quality metrics
- **VSD**: Variational Search Distribution

## License

[Add license information]

## Citation

```bibtex
@misc{proteinbenchmark2024,
  title={Protein Optimization Benchmark Suite},
  author={...},
  year={2024},
  url={...}
}
```
