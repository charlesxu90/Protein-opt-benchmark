# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Protein Optimization Benchmark Suite comparing 7 methods (ALDE, EvoPlay, LatProtRL, FLEXS/AdaLead, AiCE, delta_cs, AlphaVariant) on fitness landscape datasets (GB1, AAV_med, AAV_hard, GFP_med) using 13 unified metrics.

## Running Benchmarks

```bash
# Each method has its own conda env — activate before running
conda activate ./<method>/env

# Run from method directory (scripts are symlinked there)
cd ALDE && python run_GB1.py --seed 42
cd EvoPlay && python run_GB1.py --seed 42

# Multiple seeds
python run_GB1.py --seeds 42 123 456
python run_GB1.py --seed_file ../rand_seeds.txt --num_seeds 10

# Skip metrics (faster iteration)
python run_GB1.py --seed 42 --skip_metrics
```

Methods and Python versions: ALDE (3.11), EvoPlay (3.8), LatProtRL (3.9), FLEXS (3.7), AiCE (3.8), delta_cs (3.7), AlphaVariant (unspecified).

### Per-method conda env locations

The launcher (`scripts/hpc/launch.py --use-method-env`) and profiler resolve the right python from `scripts/hpc/method_resources.yaml`. Current layout on this machine:

| Method        | Env path                                                  |
|---------------|-----------------------------------------------------------|
| ALDE          | `ALDE/env`                                                |
| EvoPlay       | `EvoPlay/env`                                             |
| FLEXS         | `FLEXS/env`                                               |
| AiCE          | `AiCE/env`                                                |
| LatProtRL     | `LatProtRL/env/latprotrl_env` (nested)                    |
| delta_cs      | `delta_cs/env/delta_cs_env` (nested)                      |
| alphavariant  | `/home/xux/miniforge3/envs/alphavariant-env` (absolute)   |
| Random / GreedyWalk | reuses `ALDE/env` (pure-Python, no GPU)             |
| EVOLVEpro / ftMLDE / MULTIevolve | NOT BUILT YET — see `scripts/setup_baseline_envs.sh` |

`conda_env` in the YAML accepts relative-to-benchmark, absolute, and `~/...` paths. Override per host without editing the launcher.

## Symlink System (Critical)

Scripts live in `scripts/` (git-tracked) and are symlinked into method dirs (git-ignored). **Always edit scripts in `scripts/<method>/`, never in method dirs.**

```bash
# Refresh all symlinks after adding/modifying scripts
./scripts/add_script_link.sh
```

Script naming: `scripts/<method>/run_<dataset>.py` (e.g., `scripts/ALDE/run_GB1.py`).
Special case: delta_cs scripts are in `scripts/delta_cs/BioSeq-GFN-AL/`.

## Git Tracking

- **Tracked:** `scripts/`, `utils/`, `data/`, `CLAUDE.md`, `INTEGRATION.md`, `rand_seeds.txt`
- **Ignored:** Method directories (`ALDE/`, `EvoPlay/`, `FLEXS/`, etc.), `results/`, `env/`
- Only `scripts/` changes are committed — method repos contain symlinks + downloaded code

## Architecture

### `utils/` — Unified benchmark library

All run scripts import from this package via `sys.path.insert(0, '<benchmark_root>')`.

| Module | Role |
|--------|------|
| `metrics.py` | 13 metric implementations (distance, fitness, calibration, correlation) |
| `data.py` | Dataset loading, `FitnessLandscape` class for O(1) lookup, encoding utils |
| `compat.py` | Drop-in replacement for ALDE-style `from src.metrics import ...` |
| `evaluator.py` | `BenchmarkEvaluator` class for standardized evaluation |
| `gb1.py` | GB1-specific constants (`GB1_WILD_TYPE_4SITE = "VDGV"`) and helpers |
| `io.py` | Results save/load/aggregate/export |

### Two import patterns in run scripts

**Legacy (compat):** Most existing scripts use this — matches ALDE's original interface:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.compat import compute_all_metrics, MetricsResult, ...
```

**Direct:** For new scripts or major refactors:
```python
from utils import load_gb1_landscape, compute_gb1_metrics, BenchmarkEvaluator
```

### Datasets

All in `data/<name>/data.csv` with `sequence` and `fitness` columns.
- GB1: 149,361 variants, 4-site (`VDGV` wild-type, positions [39,40,41,54])
- AAV_med, AAV_hard, GFP_med: variable-length protein sequences

Standard benchmark budget: 96 sequences/round × 5 rounds = 480 queries.

## Adding a New Run Script

1. Create `scripts/<method>/run_<dataset>.py`
2. Import utils via compat or direct pattern (see above)
3. Run `./scripts/add_script_link.sh` to create symlink
4. Run from method dir: `cd <method> && python run_<dataset>.py --seed 42`

## HPC (iBex / Shaheen)

Job-array launcher under `scripts/hpc/`. One array task = one seed.

```bash
# Local smoke test (5 seeds, foreground)
python scripts/hpc/launch.py --method ALDE --dataset GB1 --seeds 5 --cluster local

# iBex submission (50 seeds, GPU array)
python scripts/hpc/launch.py --method ALDE --dataset GB1 --seeds 50 --cluster ibex

# Shaheen (account required)
python scripts/hpc/launch.py --method alphavariant --dataset PhoQ --seeds 50 \
    --cluster shaheen --account k01234

# Forward extra args to the run script (e.g. ablations)
python scripts/hpc/launch.py --method alphavariant --dataset GB1 --seeds 50 \
    --cluster ibex --extra-args "--ablation no-gpt"

# Dry-run prints the rendered sbatch script without submitting
python scripts/hpc/launch.py --method ALDE --dataset GB1 --seeds 50 --cluster ibex --dry-run
```

Per-method resource defaults (GPUs, walltime, memory) live in `scripts/hpc/method_resources.yaml`.

Resource tracking: wrap any run with `scripts/hpc/log_resource_use.py` to emit a `resource.json` next to results. `scripts/generate_tables.py --include_resources` aggregates these into wall-hours / GPU-hours columns.

### Workstation (multi-GPU)

For a 2-GPU local box, partition the seed list across GPUs with `--gpu-id` + `--seed-start`:

```bash
# GPU 0 runs seeds [0:15], GPU 1 runs seeds [15:30] (in parallel)
python scripts/hpc/launch.py --method ALDE --dataset GB1 --seeds 15 \
    --cluster local --gpu-id 0 --seed-start 0 &
python scripts/hpc/launch.py --method ALDE --dataset GB1 --seeds 15 \
    --cluster local --gpu-id 1 --seed-start 15 &
wait
```

`--use-method-env` (default) selects the per-method conda env from `method_resources.yaml`, so you don't need to activate envs by hand.

### Profiling

Estimate per-method walltimes before committing to a sweep:

```bash
# Profile every method on GB1, single seed; CSV at results/_profiles/per_method_walltime.csv
python scripts/profile_methods.py --dataset GB1 --seed 42

# Pin to one GPU
python scripts/profile_methods.py --dataset GB1 --seed 42 --gpu-id 0

# Cap any single method at 1 hour
python scripts/profile_methods.py --dataset GB1 --seed 42 --timeout 3600
```

Methods with missing envs (no `<method>/env/bin/python`) are reported as `SKIP_NO_ENV` rather than silently using the host python.

## Statistical reporting

```bash
# Bonferroni-corrected pairwise Wilcoxon test
python scripts/generate_tables.py --stat_test wilcoxon --bonferroni --alpha 0.05
```

## Asana Project

- **Workspace:** kaust.edu.sa (GID: `944030100265405`)
- **Project:** 0.AlphaVariant-benchmark (GID: `1213479076753155`)
- **Sections:** Backlog (`1213479076753156`), Todo (`1213479076753158`), Done (`1213479076753159`), Problem (`1213479076753160`)

## Key Constants

- `rand_seeds.txt`: 500 pre-generated seeds for reproducibility
- GB1 global max: sequence `VWHS`, fitness ~8.76 (verify from data)
- Metric distance functions: `levenshtein` (default, variable-length) or `hamming` (equal-length only)
