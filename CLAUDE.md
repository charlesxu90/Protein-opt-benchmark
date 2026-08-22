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

Each method exposes a single `run_generic.py`, living **only** in `scripts/<method>/`, with the dataset selected by `--dataset`. Invoke it with that method's env python; the working directory does not matter.

```bash
ALDE/env/bin/python   scripts/ALDE/run_generic.py   --dataset 4site_GB1  --seed 621
ftMLDE/env/bin/python scripts/ftMLDE/run_generic.py --dataset 4site_PhoQ --seed 621
AiCE/env/bin/python   scripts/AiCE/run_generic.py   --dataset 4site_TRPB --seed 621

# Multiple seeds
ALDE/env/bin/python scripts/ALDE/run_generic.py --dataset 4site_GB1 --seeds 621 100 383
ALDE/env/bin/python scripts/ALDE/run_generic.py --dataset 4site_GB1 --seed_file rand_seeds.txt --num_seeds 30

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

There is one entry point per method: `scripts/<method>/run_generic.py`, driven by `--dataset`. The old thin `run_<dataset>.py` wrappers are gone — do not reintroduce them.

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

## Adding a New Method

1. Create `scripts/<method>/run_generic.py` with the `BENCHMARK_ROOT` bootstrap above; take `--dataset` and the shared flags, never hardcode a dataset
2. Import utils via the compat or direct pattern (see above)
3. Anchor `--output_path` at `<Method>/results/<dataset>_<method>/` so `utils/seed_values.py` can find the per-seed JSONs
4. Run it: `<method>/env/bin/python scripts/<method>/run_generic.py --dataset 4site_GB1 --seed 621`
5. Add the method's result-path pattern to `utils/seed_values.py` — otherwise it is invisible to the figures

Do **not** add per-dataset wrapper scripts. Nothing to symlink or register.

## What Lives in `scripts/`

After the cleanup, everything here is live — if a script exists, it is either a producer of current results or part of the figure chain. Do not add one-off sweep or tuning scripts to the tree.

| Group | Files |
|---|---|
| Per-method runners | `<method>/run_generic.py` × 9 (+ `alphavariant/run_generic.py`, `train_ms_prior.py`, `align_homologs.py`) |
| Panel drivers | `run_oracle_benchmark.py` (Panel B), `run_4site_benchmark.py` (see caveat below) |
| Data / model prep | `prepare_combingym.py`, `prepare_proteingym.py`, `train_oracle.py`, `compute_mpnn_freqs.py`, `EVOLVEpro/embed_dataset.py`, `compute_dataset_checksums.py`, `setup_baseline_envs.sh` |
| Aggregation | `aggregate_oracle_results.py`, `build_oracle_median_iqr_csv.py`, `build_median_iqr_csv.py`, `build_trajectory_dataset.py`, `summarize_ablation.py`, `aggregate_metrics.py` |
| Significance | `compute_oracle_wilcoxon.py` (both panels), `compute_planC_wilcoxon.py` (ablations) |
| Figures | `draw_ranking_panel.py`, `draw_raincloud_figures.py`, `draw_trajectory_figures.py`, `draw_supplementary_ablation.py`, `plot_oracle_diagnostics.py`, `plot_4site_density.py` |
| Sweep provenance | `alphavariant/run_ev_onehot.sh`, `_run_30seed_evonehot.sh`, `_rerun_4site_newcode.sh`, `_ablation_*.sh` × 4, `_ms_finetune_*.sh` × 2 |

The `_ablation_*.sh` and `_ms_finetune_*.sh` shells are the provenance of `results_ablation/`, which feeds `ablation_summary.csv` — keep them even though they look like one-offs.

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

## Reproducibility Status (audited 2026-08-22)

Every tracked artifact was recomputed from raw per-seed files. **Panel B and the ablations are fully reproducible; Panel A has a known provenance gap.** Read this before regenerating any figure.

Reproduces exactly:
- `figures/ms_oracles/multisite_oracle_median_iqr.csv` — 40/40 rows from `results_oracle/<ds>/<method>/seed*.json`
- `results_ablation/ablation_summary.csv` — 36/36 rows; re-running `summarize_ablation.py` is byte-identical
- All `ranking_panel_*_values.csv` and `wilcoxon_pairwise_*` — byte-identical on re-run
- `alphavariant_trajectory_per_seed.csv` — 3 datasets × 30 seeds × 5 rounds
- Panel A **competitor** rows (9 methods) in `figures/alphavariant_comparison_median_iqr.csv`

Does **not** reproduce — do not silently regenerate:
- **The AlphaVariant rows of `figures/alphavariant_comparison_median_iqr.csv` match no directory on disk.** They sit between `_archive_tier1B_canonical`, `_mc_shap_winner` and `_shap_late`, consistent with the CSV predating `_rerun_4site_newcode.sh`. Regenerating this CSV **will change published Panel A AlphaVariant numbers** (e.g. PhoQ median 0.4635 → 0.5256).
- **Panel A disagrees with itself about which config "AlphaVariant" is.** The ranking panel reads that CSV; the raincloud, 4-site Wilcoxon and 4-site trajectory all read `_archive_tier1B_canonical` (hardcoded in `utils/seed_values.py` and `scripts/draw_trajectory_figures.py`) — the *base* config, not the documented `mc_shap_winner` default. They differ materially on TrpB (0.8842 vs 0.8326).
- `delta_cs` rows (4) — source directory removed with the method.

Also note Panel A per-seed data lives in `<Method>/results/` and `alphavariant/results/`, **not** under `results_*/`. Only Panel B and the ablations satisfy "all results reproducible from `results_*`".

Regenerating figures is non-deterministic at the byte level (PDF metadata), so `git checkout -- figures/` after a verification run rather than committing the churn.

## Broken / Legacy — do not use

Verify before invoking anything below; the tree still contains material from earlier benchmark revisions.

- **`scripts/hpc/` no longer exists** — no job-array launcher, no `launch.py`, no `method_resources.yaml`. The scripts that depended on it (`profile_methods.py`, `run_sweep_parallel.py`, `run_30seed_gb1_sweep.sh`) have been deleted. Live sweeps are driven by the shell scripts under `scripts/alphavariant/` and direct driver invocations.
- **Dropped datasets:** `4site_TEV` and `ms_GFP` are out of the headline panels but survive in `figures/*median_iqr.csv`, `results_oracle/ms_GFP/` and `oracles/ms_GFP/`. Ranking panels and Wilcoxon tables cover 3 datasets per panel.
- **Removed methods:** `EvoPlay`, `LatProtRL`, `delta_cs`, `Mu-Protein` are gone from the tree; references remain in `scripts/aggregate_metrics.py` and `figures/alphavariant_comparison_median_iqr.csv`.
- **`docs/`, `tables/`, `logs/` and `AGENTS.md` no longer exist.** `scripts/compute_planC_wilcoxon.py` still defaults its `--out` into `docs/plan_C/`, so it recreates that directory unless you pass `--out` explicitly.

## Asana Project

- **Workspace:** kaust.edu.sa (GID: `944030100265405`)
- **Project:** 0.AlphaVariant-benchmark (GID: `1213479076753155`)
- **Sections:** Backlog (`1213479076753156`), Todo (`1213479076753158`), Done (`1213479076753159`), Problem (`1213479076753160`)

## Key Constants

- `rand_seeds.txt`: 500 pre-generated seeds; the benchmark uses the **first 30** (621, 100, 383, 492, 987, … 511)
- Global maxima (used to normalize raw `max_fitness`): `4site_GB1` 8.761966 (combo `FWAA`), `4site_PhoQ` 133.5943 (combo `TEMH`), `4site_TRPB` 1.0 (already normalized, combo `AIKG`)
- Multi-site oracle outputs are normalized to [0,1] at training time; `fit_min`/`fit_max` for de-normalization are stored in `oracles/<dataset>/oracle_meta.json`
- Metric distance functions: `levenshtein` (default, variable-length) or `hamming` (equal-length only)
