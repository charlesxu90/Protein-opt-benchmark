# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Protein Optimization Benchmark Suite comparing **10 methods** (Random, GreedyWalk, ALDE, AdaLead/FLEXS, ftMLDE, CLADE, MULTIevolve, EVOLVEpro, AiCE, AlphaVariant) across **two benchmark panels**:

- **Four-site, true landscape** (`4site_GB1`, `4site_PhoQ`, `4site_TRPB`) — combinatorially-complete 20⁴ libraries, pool selection against measured fitness.
- **Multi-site, learned oracle** (`ms_AAV`, `ms_CreiLOV`, `ms_PAB1`) — a `BaseCNN` oracle as the fitness function, generative proposal over varying positions.

Every (method, dataset) cell uses the **same 30 seeds** (first 30 lines of `rand_seeds.txt`), so all comparisons are paired. See `README.md` for the full benchmark description.

## Working Principles

Behavioral guidelines to reduce common LLM coding mistakes. These bias toward caution over speed; for trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

### 5. Decompose Before Long Runs

**Verify cheaply before committing expensive compute. Decompose to the smallest verifiable units.**

Before running any long-execution task (multi-hour sweep, 30-seed benchmark, full dataset training, large download):
- Break the goal into the smallest tasks that can be verified quickly. A 30-seed sweep decomposes into: data-loads-OK → one-seed-runs-OK → 5-seed-completes-OK → 30-seed-sweep.
- Run the cheap checks first. A 5-second smoke test catches 90% of bugs that would otherwise waste a 10-hour sweep.
- Only commit to the long run after every cheap check passes.
- Prefer staged decomposition: verify the data pipeline → verify one method × one seed → verify N methods × one seed → verify one method × N seeds → full sweep.

The test: before kicking off anything that runs longer than a few minutes, ask "what's the cheapest test that would have caught a failure here?" — and run that first.

This applies recursively. If a subtask itself is long, decompose it further.

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, clarifying questions come before implementation rather than after mistakes, and long runs no longer fail at hour 9 of 10 because of a bug a one-seed smoke test would have caught.

## Running Benchmarks

### Per-method run scripts (the canonical path)

Each method exposes `run_generic.py` plus thin per-dataset wrappers, living **only** in `scripts/<method>/`. Invoke them there with that method's env python; the working directory does not matter.

```bash
ALDE/env/bin/python   scripts/ALDE/run_generic.py   --dataset 4site_GB1  --seed 621
ftMLDE/env/bin/python scripts/ftMLDE/run_generic.py --dataset 4site_PhoQ --seed 621
AiCE/env/bin/python   scripts/AiCE/run_generic.py   --dataset 4site_TRPB --seed 621

# Multiple seeds
ALDE/env/bin/python scripts/ALDE/run_generic.py --dataset 4site_GB1 --seeds 621 100 383
ALDE/env/bin/python scripts/ALDE/run_generic.py --dataset 4site_GB1 --seed_file rand_seeds.txt --num_seeds 30

# Per-dataset wrapper (just injects --dataset)
ALDE/env/bin/python scripts/ALDE/run_4site_GB1.py --seed 621

# Skip metrics (faster iteration)
ALDE/env/bin/python scripts/ALDE/run_generic.py --dataset 4site_GB1 --seed 621 --skip_metrics
```

Shared flags: `--dataset`, `--seed` / `--seeds` / `--seed_file` + `--num_seeds`, `--device`, `--output_path` (default `<Method>/results/<dataset>_<method>/`, resolved absolutely from `BENCHMARK_ROOT`), `--data_dir`, `--skip_metrics`.

Results: `<method>/results/<dataset>_<method>/<dataset>/<variant>/metrics_seed<S>.json`, where `<variant>` is method-specific (`onehot` for ALDE, `aice`, `ftmlde`, `clade`, `random`, `greedy`, `topn` for EVOLVEpro; FLEXS has none).

Each script discovers `BENCHMARK_ROOT` by walking up from its own path until it finds `utils/`, then resolves `data/`, `utils/`, the method's upstream package and the `<Method>/results/` output root from it. So invocation is cwd-independent — no symlink, no `cd` required.

### Unified panel drivers

Two drivers implement the baselines in one process with **identical selection rules across both panels**, so `ALDE`/`CLADE`/`ftMLDE` mean the same thing in both. They emit the compact headline schema (`max_fitness_norm`, `top128_mean_norm`, `diversity_top128`, `novelty_top128_vs_wt`, `fitness_trajectory`).

```bash
# Multi-site oracle panel — methods: Random GreedyWalk ftMLDE CLADE ALDE
#                                   AdaLead MULTIevolve EVOLVEpro AiCE
ALDE/env/bin/python scripts/run_oracle_benchmark.py \
    --method ALDE --dataset ms_CreiLOV --seeds 621 100 383 --device cuda:0
# -> results_oracle/<dataset>/<method>/seed<S>.json

# Four-site panel, CPU-only — methods: Random GreedyWalk ftMLDE CLADE ALDE
python scripts/run_4site_benchmark.py --method ftMLDE --dataset 4site_GB1 --seeds 621 100
```

`run_oracle_benchmark.py` needs only `ALDE/env` (sklearn + torch), except `--method EVOLVEpro`, which needs the transformers/ESM stack from `alphavariant-env`.

### AlphaVariant

One default configuration is used throughout, and it is what every AlphaVariant curve in the figures comes from: GPT-REINFORCE + surrogate ensemble, with **MutCompute** reward shaping and **SHAP** alphabet pruning. Only the flags below differ per panel. The script lives in `scripts/alphavariant/`, but `--prior_model_path` / `--data_dir` are cwd-relative, so `cd alphavariant` first. Export the env's `libstdc++` or matplotlib/torch fail with a `CXXABI` error:

```bash
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:$LD_LIBRARY_PATH

cd alphavariant

# Four-site (lookup landscape)
/home/xux/miniforge3/envs/alphavariant-env/bin/python ../scripts/alphavariant/run_generic.py --dataset 4site_PhoQ --seed 621 \
    --use_mutcompute --plm_reward_lambda 0.5 --shap_prune_alphabet

# Multi-site (CNN oracle, generative proposal)
/home/xux/miniforge3/envs/alphavariant-env/bin/python ../scripts/alphavariant/run_generic.py --dataset ms_CreiLOV --seed 621 \
    --oracle --level uniform \
    --prior_model_path priors/ms_CreiLOV/prior_model.pt \
    --features ev_onehot --use_mutcompute --shap_prune_alphabet \
    --max_n_mut 2 --n_rounds 5 --n_steps_per_round 500 \
    --device cuda:0 --data_dir ../data
```

Per-seed results the figures read: `alphavariant/results/<dataset>_AlphaVariant_mc_shap_winner/seed_*/metrics.json` (four-site) and `results_oracle/<dataset>/AlphaVariant/seed*.json` (multi-site). Sweep launchers: `scripts/alphavariant/run_ev_onehot.sh`, `_run_30seed_evonehot.sh`, `_rerun_4site_newcode.sh`. See `README.md` §3 for the per-flag table.

### Training the pieces

```bash
python scripts/train_oracle.py --dataset ms_CreiLOV --smoke          # cheap pipeline check first
python scripts/train_oracle.py --dataset ms_CreiLOV --device cuda:0  # -> oracles/<ds>/oracle.pt
python scripts/alphavariant/train_ms_prior.py --dataset ms_CreiLOV --device cuda:0
CUDA_VISIBLE_DEVICES=0 AiCE/env/bin/python scripts/compute_mpnn_freqs.py --dataset ms_CreiLOV
python scripts/EVOLVEpro/embed_dataset.py --dataset 4site_GB1
```

### Per-method conda env locations

| Method | Env path | Python |
|---|---|---|
| ALDE | `ALDE/env` | 3.11 |
| AiCE | `AiCE/env` | 3.11 |
| CLADE | `CLADE/env` (symlink to `ALDE/env`) | 3.11 |
| FLEXS | `FLEXS/env` | 3.7 |
| ftMLDE | `ftMLDE/env` | 3.7 |
| EVOLVEpro | `EVOLVEpro/env` | 3.11 |
| MULTIevolve | `MULTIevolve/env` | 3.11 |
| AlphaVariant | `/home/xux/miniforge3/envs/alphavariant-env` (absolute) | 3.10 |
| Random / GreedyWalk | reuse `ALDE/env` (pure-Python, no GPU) | 3.11 |

`scripts/setup_baseline_envs.sh` builds the three later additions (EVOLVEpro, ftMLDE, MULTIevolve) from their upstream `environment.yml`. The others were built by hand — see each `scripts/<method>/Environment.md`.

Prefer `<method>/env/bin/python` over `conda activate`; it is unambiguous and works inside scripts.

## Where Harness Code Lives (Critical)

All benchmark code lives in `scripts/` and is run from there. **Method directories contain only upstream code** — no run scripts, no symlinks. There is nothing to sync: editing `scripts/<method>/run_generic.py` *is* editing the thing that runs.

Script naming: `scripts/<method>/run_<dataset>.py` (e.g. `scripts/ALDE/run_4site_GB1.py`), each a thin wrapper that injects `--dataset` and calls `run_generic.main()`.

Every `run_generic.py` starts with the same bootstrap, which is what makes the method dir unnecessary:

```python
_p = os.path.dirname(os.path.realpath(__file__))
while os.path.dirname(_p) != _p and not os.path.isdir(os.path.join(_p, 'utils')):
    _p = os.path.dirname(_p)
BENCHMARK_ROOT = _p
sys.path.insert(0, BENCHMARK_ROOT)
sys.path.insert(0, os.path.join(BENCHMARK_ROOT, '<Method>'))  # upstream package
```

Copy that block when adding a method. Never reintroduce `dirname(__file__)/'..'` as the benchmark root, and never default `--output_path` to a cwd-relative path — anchor it at `os.path.join(BENCHMARK_ROOT, '<Method>', 'results', ...)` so results stay where the figure pipeline expects them.

Earlier revisions symlinked these scripts into each method dir and synced them with `scripts/add_script_link.sh`; both the symlinks and that script are gone.

## Git Tracking

- **Tracked:** `scripts/`, `utils/`, `oracles/` (including the `oracle.pt` checkpoints), `figures/`, `README.md`, `CLAUDE.md`, `rand_seeds.txt`, and the two documentation files inside the otherwise-ignored `data/`: `data/README.md` and `data/CHECKSUMS.txt`
- **Tracked directly (non-submodule method dirs):** `ALDE/`, `AiCE/`, `FLEXS/`, `Random/`, `GreedyWalk/` — upstream code only (`Random/` and `GreedyWalk/` have no upstream code at all and now exist purely to hold `results/`)
- **Submodules** (`.gitmodules`): `alphavariant`, `CLADE`, `ftMLDE`, `EVOLVEpro`, `MULTIevolve`. Commits to these show up as one-line pointer bumps; commit inside the submodule first, then bump here.
- **Ignored:** `data/`, `results/`, `results_*/` (so `results_oracle/`, `results_ablation/`, `results_backups/`), `sweep_logs/`, `env/`, `output/`, `_logs/`

Note `data/` is **git-ignored** — datasets are not distributed with the repo. Regenerate via `scripts/prepare_combingym.py` / `scripts/prepare_proteingym.py` and verify with `sha256sum -c data/CHECKSUMS.txt` from the benchmark root. `.gitignore` carries explicit `!/data/README.md` / `!/data/CHECKSUMS.txt` negations, so those two files (and only those) are tracked.

## Architecture

### `utils/` — Unified benchmark library

All run scripts import from this package via `sys.path.insert(0, '<benchmark_root>')`.

| Module | Role |
|--------|------|
| `metrics.py` | Metric suite — the two scored metrics plus diagnostic fields (distance, calibration, correlation) |
| `data.py` | Dataset loading, `FitnessLandscape` class for O(1) lookup, encoding utils |
| `compat.py` | Drop-in replacement for ALDE-style `from src.metrics import ...` |
| `evaluator.py` | `BenchmarkEvaluator` class for standardized evaluation |
| `gb1.py` | GB1-specific constants (`GB1_WILD_TYPE_4SITE = "VDGV"`) and helpers |
| `io.py` | Results save/load/aggregate/export |
| `oracle_model.py` | `BaseCNN` architecture + `ALPHABET` / integer encoding |
| `oracle_landscape.py` | `OracleLandscape` — the multi-site fitness function; `get_fitness()` / `get_fitness_normalized()`, `n_calls` budget counter |
| `candidate_generator.py` | `CandidateGenerator` — design-space proposal for the oracle panel |
| `proteingym_oracle.py` | Generic ProteinGym/CombinGym oracle wrapper over `FitnessLandscape` |
| `sequence_plausibility.py` | ESM-2 pseudo-perplexity / PLL plausibility scoring |
| `seed_values.py` | Single source of truth for where each method's per-seed results live and how raw values are normalized — shared by figures and Wilcoxon tests |
| `plot_style_utils.py` | Figure style (`apply_nature_rcparams`, `save_figure`) |

When adding a figure or statistical test, get per-seed values from `utils/seed_values.py` rather than re-deriving result paths.

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

All in `data/<name>/data.csv` with a **`seq`** column (not `sequence`) plus `fitness`; most also carry `AACombo` and `n_muts`.

| Dataset | n | Seq len | Sites / varying positions | Fitness range |
|---|---:|---:|---|---|
| `4site_GB1` | 149,361 | 56 | V39, D40, G41, V54 | 0 – 8.762 |
| `4site_PhoQ` | 140,517 | 486 | A284, V285, S288, T289 | 0 – 133.6 |
| `4site_TRPB` | 159,129 | 397 | V183, F184, S227, S228 | 0 – 1 |
| `ms_AAV` | 44,128 | 28 | 28 positions | 0 – 1 |
| `ms_CreiLOV` | 165,428 | 119 | 15 positions | 620 – 15,690 |
| `ms_PAB1` | 36,522 | 75 | 75 positions | 0.0024 – 2.628 |

Every dataset dir also holds `wt.fasta`, a `*.pdb` structure and `mutcompute.csv`. The rest is panel-specific:

- **four-site:** `embeddings_evolvepro.pt` + `labels_evolvepro.csv` (EVOLVEpro)
- **multi-site:** `varying_positions.txt`, `target_seqs.fasta`, `plmc/` (EVmutation couplings), `prior_aligned.csv` (AlphaVariant homolog prior), `aice_mpnn_freq.npz` (AiCE)

There is no `align/` any more — only the derived `prior_aligned.csv`. Per-file documentation: `data/README.md`.

Standard benchmark budget: 96 sequences/round × 5 rounds = 480 queries.

### Scored metrics

Only **two** metrics are scored, both over the 480 queried sequences of a run and normalized to [0,1]:

| Metric | Field | Definition |
|---|---|---|
| Max fitness | `max_fitness_norm` | best fitness the run discovered |
| Top-128 mean fitness | `top128_mean_norm` | mean fitness of the run's 128 best sequences |

Across the 30 seeds each is reported as median + IQR (Q1/Q3), and the ranking panels rank on that per-dataset median. Everything else `utils/metrics.py` emits (diversity, novelty, calibration, surrogate correlation) is diagnostic and is **not** used for ranking — don't introduce it into headline comparisons.

## Adding a New Run Script

1. Create `scripts/<method>/run_<dataset>.py` as a thin wrapper that injects `--dataset` and calls `run_generic.main()`
2. Import utils via compat or direct pattern (see above)
3. Run it: `<method>/env/bin/python scripts/<method>/run_<dataset>.py --seed 621`

Only live datasets get wrappers — see the dataset table above. Nothing to symlink or register.

## Analysis and Reporting

```bash
# --- aggregate per-seed JSON -> tidy median/IQR CSVs -------------------------
python scripts/aggregate_oracle_results.py                 # results_oracle/ -> figures/ms_oracles/
python scripts/build_oracle_median_iqr_csv.py              # multisite_oracle_median_iqr.csv
python scripts/build_median_iqr_csv.py --plans C           # four-site medians/IQRs
python scripts/aggregate_metrics.py --dataset 4site_GB1 --seed 621   # one dataset/seed, every field

# --- significance (paired Wilcoxon, Bonferroni, n=30) -----------------------
python scripts/compute_oracle_wilcoxon.py --task multisite  # -> figures/ms_oracles/wilcoxon_*
python scripts/compute_oracle_wilcoxon.py --task 4site      # -> figures/wilcoxon_*
python scripts/compute_planC_wilcoxon.py                    # AlphaVariant vs its ablations

# --- figures ----------------------------------------------------------------
python scripts/draw_ranking_panel.py                       # mean-rank lollipop panels
python scripts/draw_raincloud_figures.py                   # per-seed dot + IQR rows
python scripts/draw_trajectory_figures.py --task multisite  # per-round improvement bands
python scripts/plot_oracle_diagnostics.py                  # oracle calibration (supplementary)
python scripts/plot_4site_density.py                       # landscape-difficulty diagnostic
python scripts/draw_supplementary_ablation.py              # AlphaVariant ablation bars

# --- ablation ---------------------------------------------------------------
python scripts/summarize_ablation.py                       # -> results_ablation/ablation_summary.csv
```

AlphaVariant leave-one-out ablations live in `results_ablation/<prefix>_<config>/`, where `full` is the default configuration used in the headline figures and the others switch one component off. Four-site: `no_mcreward`, `no_shap`, `bare`. Multi-site: `no_ev`, `no_shap`, `no_cap`, `no_prior`. They feed the supplementary ablation figure only.

## Broken / Legacy — do not use

Verify before invoking anything below; the tree still contains material from earlier benchmark revisions.

- **`scripts/hpc/` no longer exists.** There is no iBex/Shaheen job-array launcher, no `launch.py`, no `method_resources.yaml`, no `log_resource_use.py`. Consequently:
  - `scripts/profile_methods.py` fails at import (`ModuleNotFoundError: No module named 'launch'`).
  - `scripts/run_sweep_parallel.py` and `scripts/run_30seed_gb1_sweep.sh` shell out to the missing launcher.
  - Live sweeps are driven by the per-method shell scripts under `scripts/alphavariant/` and direct driver invocations.
- **`scripts/generate_tables.py` and `scripts/plotting/`** are hard-coded to the old dataset names (`GB1`, `AAV_med`, `GFP_med`, `GFP_hard`) and will not find current results. Use the `build_*_median_iqr_csv.py` → `draw_*.py` chain above.
- **Dropped datasets:** `4site_TEV` and `ms_GFP` are out of the headline panels but survive in `figures/*median_iqr.csv`, `results_oracle/ms_GFP/` and `oracles/ms_GFP/`. Ranking panels and Wilcoxon tables cover 3 datasets per panel.
- **Removed methods:** `EvoPlay`, `LatProtRL`, `delta_cs`, `Mu-Protein` are gone from the tree; references remain in `scripts/aggregate_metrics.py` and `figures/alphavariant_comparison_median_iqr.csv`.
- **`docs/`, `tables/`, `logs/` and `AGENTS.md` no longer exist.** `scripts/compute_planC_wilcoxon.py`, `scripts/compute_wilcoxon_table.py` and `scripts/compute_landscape_descriptors.py` still default their `--out` into `docs/`, so they recreate that directory unless you pass `--out` explicitly.

## Asana Project

- **Workspace:** kaust.edu.sa (GID: `944030100265405`)
- **Project:** 0.AlphaVariant-benchmark (GID: `1213479076753155`)
- **Sections:** Backlog (`1213479076753156`), Todo (`1213479076753158`), Done (`1213479076753159`), Problem (`1213479076753160`)

## Key Constants

- `rand_seeds.txt`: 500 pre-generated seeds; the benchmark uses the **first 30** (621, 100, 383, 492, 987, … 511)
- Global maxima (used to normalize raw `max_fitness`): `4site_GB1` 8.761966 (combo `FWAA`), `4site_PhoQ` 133.5943 (combo `TEMH`), `4site_TRPB` 1.0 (already normalized, combo `AIKG`)
- Multi-site oracle outputs are normalized to [0,1] at training time; `fit_min`/`fit_max` for de-normalization are stored in `oracles/<dataset>/oracle_meta.json`
- Metric distance functions: `levenshtein` (default, variable-length) or `hamming` (equal-length only)
