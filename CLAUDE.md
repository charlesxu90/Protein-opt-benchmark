# CLAUDE.md - AlphaVariant Benchmark Repository

## Overview

This is a **Protein Optimization Benchmark Suite** for evaluating and comparing protein design/optimization methods on fitness landscape datasets. It benchmarks 7 methods (ALDE, EvoPlay, LatProtRL, FLEXS/AdaLead, AiCE, delta_cs, AlphaVariant) using unified metrics.

## Directory Structure

```
Benchmark/
├── scripts/              # Centralized run scripts (SOURCE of truth)
│   ├── add_script_link.sh  # Creates symlinks to method repos
│   ├── ALDE/
│   ├── EvoPlay/
│   ├── LatProtRL/
│   ├── FLEXS/
│   ├── AiCE/
│   ├── delta_cs/BioSeq-GFN-AL/
│   └── alphavariant/
├── utils/                # Unified benchmark utilities
│   ├── metrics.py        # 13+ standardized metrics
│   ├── data.py           # Data loading utilities
│   ├── gb1.py            # GB1-specific utilities
│   ├── compat.py         # Legacy compatibility layer
│   ├── evaluator.py      # BenchmarkEvaluator class
│   └── io.py             # Results I/O
├── data/                 # Fitness landscape datasets
│   ├── GB1/data.csv      # 149,361 sequences
│   ├── AAV_med/data.csv
│   ├── AAV_hard/data.csv
│   └── GFP_med/data.csv
├── ALDE/                 # Method repositories (contain symlinks)
├── EvoPlay/
├── LatProtRL/
├── FLEXS/
├── AiCE/
├── delta_cs/
├── alphavariant/
└── rand_seeds.txt        # 500 random seeds for reproducibility
```

## Symbolic Link System

**Important:** Scripts are centralized in `scripts/` and symlinked to method directories.

```bash
# Create/refresh all symbolic links (uses absolute paths)
./scripts/add_script_link.sh

# This creates links like:
# ALDE/run_GB1.py -> /home/xux/Desktop/AlphaVariant/Benchmark/scripts/ALDE/run_GB1.py
```

**Why:** Changes to scripts are tracked in git without tracking entire method repositories.

**To edit a script:** Edit the source in `scripts/<method>/`, then run `add_script_link.sh` if needed.

## Key Files

| File | Purpose |
|------|---------|
| `utils/metrics.py` | All metric implementations |
| `utils/compat.py` | Drop-in compatibility for ALDE-style code |
| `utils/gb1.py` | GB1 constants and utilities |
| `scripts/add_script_link.sh` | Symlink creation script |
| `rand_seeds.txt` | 500 seeds for reproducibility |
| `INTEGRATION.md` | How to integrate methods with utils |

## Running Benchmarks

```bash
# Run from method directory (scripts are symlinked there)
cd ALDE && python run_GB1.py --seed 42
cd EvoPlay && python run_GB1.py --seed 42
cd LatProtRL && python run_GB1.py --seed 42
cd alphavariant && python run_GB1.py --seed 42

# Run with multiple seeds
python run_GB1.py --seeds 42 123 456
python run_GB1.py --seed_file ../rand_seeds.txt --num_seeds 10
```

## Methods & Environments

| Method | Directory | Python | Description |
|--------|-----------|--------|-------------|
| ALDE | `ALDE/` | 3.11 | DNN Ensemble + Thompson Sampling |
| EvoPlay | `EvoPlay/` | 3.8 | AlphaZero-style MCTS + RL |
| LatProtRL | `LatProtRL/` | 3.9 | PPO in ESM-2 latent space |
| AdaLead | `FLEXS/` | 3.7 | CNN Ensemble within FLEXS |
| AiCE | `AiCE/` | 3.8 | ProteinMPNN inverse folding |
| delta_cs | `delta_cs/` | 3.7 | GFlowNet + conservative search |
| AlphaVariant | `alphavariant/` | - | GPT + REINFORCE |

Each method has its own `env/` conda environment.

## Using Unified Utils

```python
import sys
sys.path.insert(0, '/home/xux/Desktop/AlphaVariant/Benchmark')

# GB1-specific utilities
from utils import load_gb1_landscape, compute_gb1_metrics
sequences, fitness, wt = load_gb1_landscape()

# Compatibility layer (ALDE-style interface)
from utils.compat import compute_all_metrics, MetricsResult
```

## GB1 Dataset Constants

- Wild type (4-site): `VDGV`
- Positions: `[39, 40, 41, 54]`
- Total variants: 149,361
- Benchmark: 96 sequences/round × 5 rounds = 480 queries

## Common Tasks

**Add new run script:**
1. Create script in `scripts/<method>/run_<dataset>.py`
2. Run `./scripts/add_script_link.sh`

**Check results:**
```bash
ls <method>/results/
# Contains: GB1_seed42.json, summary.json, etc.
```

**Activate method environment:**
```bash
conda activate ./<method>/env
```

## Git Tracking

- `scripts/` directory is tracked (contains actual scripts)
- Method directories (`ALDE/`, `EvoPlay/`, etc.) are NOT tracked (contain symlinks + downloaded code)
- Only `scripts/` changes are committed
