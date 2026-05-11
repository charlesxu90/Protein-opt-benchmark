# Implementation Plan — AlphaVariant Benchmark, *Nature Methods* Track

## Context

`docs/Refined_benchmark_plan.md` defines a 3-task benchmark (Task 1: combinatorial epistasis; Task 2: sample-efficient DE on ≥10 diverse landscapes; Task 3: multi-objective Pareto), 10 baselines, 4 ablations, and 50-seed statistical evaluation, targeted at *Nature Methods*. The current repo (`/home/xux/Desktop/AlphaVariant/Benchmark/`) has solid foundations but several gaps: missing baselines (EVOLVEpro, ftMLDE, MULTI-evolve, CLADE), missing datasets (CombinGym + most ProteinGym assays), no HPC infrastructure, no AlphaVariant ablation flags, no multi-objective metrics, no ESM-2 plausibility scoring.

This plan delivers the benchmark in **four phases**, each shipping a self-contained, runnable artifact. Phase 1 is required before any other phase. Phases 2–4 are roughly independent and can be parallelized once Phase 1 lands. HPC scaffolding (iBex + Shaheen) is built into Phase 1 so every method/dataset added downstream is immediately submittable.

## What *Nature Methods* requires (and how this plan delivers it)

Reviewer expectations distilled from the journal's computational-methods guidance and from the refined plan's references:

| Requirement | Plan deliverable |
|---|---|
| Method advance is clearly framed against state-of-the-art | Direct head-to-head with EVOLVEpro, MULTI-evolve, ALDE, EvoPlay, AdaLead, AiCE on identical query budgets |
| Generalizability across ≥10 protein systems | Task 2 sweeps ≥10 ProteinGym assays + CombinGym landscapes |
| Statistical rigor: ≥30 (we use 50) replicates, Wilcoxon + Bonferroni | `utils/io.py` aggregation extended; `scripts/generate_tables.py` already does Wilcoxon — add Bonferroni correction |
| Ablations isolating each contribution | 4 AV ablation flags (`--ablation no-gpt|no-space|static-reward|no-rl`) |
| Reproducibility: public code, fixed seeds, env pinning, GPU-hour reporting | Per-method conda envs already exist; add `scripts/hpc/log_resource_use.py`; CHECKSUMS file for datasets |
| Comprehensive metrics beyond fitness | Add ESM-2 PPL, hypervolume, Pareto coverage to `utils/` |

**Phasing rationale.** A *Nature Methods* submission needs Tasks 1+2+ablations at minimum to be defensible; Task 3 (multi-objective) strengthens the case but is not strictly required if the wet-lab campaign in the main paper covers multi-objective use. We treat Task 3 as Phase 4 — high value, lower priority than ablations.

---

## Phase 1 — Core infrastructure (foundation)

Goal: every component needed before any method/dataset runs at scale.

### 1.1 utils/ additions (no changes to existing exports)

| New module | Purpose | Notes |
|---|---|---|
| `utils/sequence_plausibility.py` | ESM-2 perplexity scoring (independent reward model) | Use ESM-2 t33_650M; cache embeddings to disk; reusable as PDFBench-style metric and as AlphaVariant's secondary reward |
| `utils/multi_objective.py` | Hypervolume + Pareto front coverage | Use `pymoo` (mature, BSD-3) for `HV` indicator; reference reuses `utils/metrics.py:max_fitness` |
| `utils/proteingym_oracle.py` | Wrapper turning any prepared dataset into a `FitnessLandscape` for any sequence length | Generalizes `utils/gb1.py` pattern; uses existing `utils/data.py:FitnessLandscape` |

`utils/metrics.py` already implements: AUOC (`area_under_optimization_curve`), hit rate (`hit_rate`), top-k fitness (`normalized_fitness_median_topk`), diversity (`batch_diversity`), novelty (`novelty`), global-max recovery (`global_max_hit_count`/`global_max_hit_rate`). Reuse, do not duplicate.

### 1.2 Datasets — extend acquisition pipeline

- **Extend `scripts/prepare_proteingym.py`**: add to `ASSAYS` dict — PABP_YEAST (PAB1), CAS9 assays, HIV envelope, Zika, AAV/Bryant, ParD-ParE, PafA, TEV. Target ≥12 ProteinGym assays available.
- **New `scripts/prepare_combingym.py`**: clone `https://github.com/chenz16/CombinGym` and convert `GB1`, `PhoQ`, `CR9114`, `CreiLOV`, `eqFP611` to `data/<name>/data.csv` (`seq,fitness` schema). For multi-property datasets (eqFP611 red+blue), emit two columns `fitness_blue`, `fitness_red` and a `--objective` selector in the loader.
- **New `data/CHECKSUMS.txt`**: SHA-256 of every `data.csv` for reproducibility.
- **`utils/data.py:load_landscape_data`** already auto-detects `seq`/`fitness` columns — verify it handles new datasets without modification (likely yes; smoke-test).

### 1.3 HPC orchestration (iBex + Shaheen)

New directory `scripts/hpc/`:

```
scripts/hpc/
├── ibex_array.sbatch          # iBex GPU job-array template (typical: 1 GPU, 16-32 GB RAM, 24h)
├── shaheen_array.sbatch       # Shaheen CPU/GPU template (Cray modules, longer queue)
├── launch.py                  # Generates and submits arrays: --method --dataset --seeds --cluster ibex|shaheen
├── env_setup.sh               # Module loads + conda activate per cluster
└── log_resource_use.py        # Wraps a run_*.py call with /usr/bin/time + nvidia-smi sampling → resource.json
```

`launch.py` design:
- Reads `rand_seeds.txt` (already exists, 500 seeds) and slices first 50 by default.
- For `--method ALDE --dataset GB1 --seeds 50 --cluster ibex`: writes one `sbatch` file with `--array=0-49`, each task runs `python ALDE/run_GB1.py --seed $(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" rand_seeds.txt)`.
- Per-method GPU/CPU/time defaults configured in `scripts/hpc/method_resources.yaml` (e.g., AlphaVariant: 1×A100 24h; Random: 1 CPU 1h).
- Outputs go to `results/<method>/<dataset>/seed<S>.json` matching existing `generate_tables.py` glob.
- Works locally too: `--cluster local` runs serially (smoke tests).

KAUST specifics:
- iBex uses SLURM with partitions like `batch`, `gpu`, `gpu-rtx`. Template uses `#SBATCH --partition=batch --gres=gpu:1`.
- Shaheen uses `#SBATCH --account=<proj>` with Cray PrgEnv modules. Template includes `module load python/3.11`. Confirm account ID with the user before first submission.

### 1.4 Per-dataset scripts for Random and GreedyWalk

`Random/` and `GreedyWalk/` currently only have `run_generic.py`. Generate per-dataset wrappers (one-liner `from run_generic import main; main(dataset_name="GB1")`-style) for GB1, AAV_med, AAV_hard, GFP_med, GFP_hard, and every new dataset added in 1.2. Add to `scripts/add_script_link.sh` if needed (already covers Random/GreedyWalk).

### 1.5 Statistical reporting upgrades

- `scripts/generate_tables.py` already runs Wilcoxon. Add `--bonferroni` flag that divides α by the number of pairwise comparisons.
- Add per-method GPU-hour and wall-clock columns sourced from `resource.json` files written by `log_resource_use.py`.

**Phase 1 exit criterion:** `python scripts/hpc/launch.py --method ALDE --dataset GB1 --seeds 5 --cluster local` runs end-to-end and produces aggregated metrics including AUOC, hit rate, PPL, and diversity. iBex sbatch can be submitted (smoke test with 2 seeds).

---

## Phase 2 — Task 1 (combinatorial epistasis) + missing baselines + ablation refactor

### 2.1 Integrate missing baselines

User-selected: **EVOLVEpro, ftMLDE, MULTI-evolve**. Each follows the established integration pattern documented in `INTEGRATION.md` (compat-layer imports + per-dataset run script).

| Method | Source | Integration steps |
|---|---|---|
| EVOLVEpro | `https://github.com/goolab-community/EVOLVEpro` | Clone into `EVOLVEpro/`, add `EVOLVEpro/env`, write `scripts/EVOLVEpro/run_generic.py` + per-dataset wrappers. Wrap their few-shot active-learning loop to emit per-round selections in our schema. |
| ftMLDE | `https://github.com/bhattacharyya-lab/mlde` (Wittmann et al. *Cell Systems* 2021) | Same pattern. Their CLI takes a CSV + budget; map to our `FitnessLandscape` oracle. |
| MULTI-evolve | `https://github.com/ArcInstitute/MULTI-evolve` | Same pattern. PLM ensemble + epistasis modelling — wrap their inference loop to emit ranked variant proposals per round into our schema. |

`scripts/add_script_link.sh` needs three new `create_links` calls (one per method).

### 2.2 AlphaVariant ablation refactor

Current state: `scripts/alphavariant/run_GB1.py` is 1993 lines, monolithic. Refactor target:

```
alphavariant/
├── pipeline.py          # AlphaVariantPipeline class with pluggable components
├── components/
│   ├── prior.py         # VariantGPTPrior + RandomMutationPrior (for AV-NoGPT)
│   ├── space.py         # DynamicSpace + UnconstrainedSpace (for AV-NoSpace)
│   ├── reward.py        # IterativeLowNReward + StaticZeroShotReward (for AV-StaticReward)
│   └── policy.py        # RLPolicy + GreedyPolicy (for AV-NoRL)
└── config.py            # AVConfig dataclass driving --ablation flag
```

`scripts/alphavariant/run_<dataset>.py` becomes a thin wrapper: parses `--ablation {none,no-gpt,no-space,static-reward,no-rl}`, builds `AVConfig`, calls `AlphaVariantPipeline(config).run()`. Existing dataset-specific tweaks live in dataset-specific config presets, not duplicated code.

This refactor is invasive but unavoidable: the monolithic script makes ablations impossible without copy-paste. Strategy:
1. Pull existing 1993-line `run_GB1.py` into `pipeline.py` as a single `_run_default_gb1` function (no behavior change).
2. Identify the four ablation seams (GPT sampling call, space-definition step, reward-model fitting, RL acquisition).
3. Extract each seam into a component interface; default impl matches current behavior bit-for-bit.
4. Add ablation variants. Verify each ablation's GB1 trajectory differs from the full model in expected directions (AV-NoGPT < AV; AV-StaticReward plateaus earlier; etc.).
5. Apply same refactor to all 6 dataset run scripts (they're 90% duplicated).

### 2.3 Task 1 execution

Datasets: GB1 (already in repo), PhoQ, CR9114, CreiLOV, eqFP611 (added in 1.2). Plus 5 ProteinGym substitutions.

Splits per CombinGym protocol: 1-vs-rest, 2-vs-rest, 3-vs-rest. Implement in `utils/data.py` as `make_hierarchical_split(landscape, train_order, test_order)`.

50 seeds × ~10 datasets × (3 splits) × ~10 methods (incl. 4 ablations) ≈ 15,000 runs. Definitively HPC scale.

**Phase 2 exit criterion:** Task 1 figure (top-k fitness vs queries, mean ± std across 50 seeds) generated via `scripts/plotting/plot_optimization_curves.py` with all baselines + ablations on GB1 and at least 2 CombinGym datasets.

---

## Phase 3 — Task 2 (sample efficiency on diverse landscapes)

### 3.1 Datasets

≥10 ProteinGym assays from 1.2. The dual budget protocol (10×16 EVOLVEpro-style and 5×96 ALDE-style) is parameterized via existing `--rounds N --batch_size K` flags in run scripts. Audit each method's run script to confirm both budgets work; add flags where missing.

### 3.2 Metrics emphasis

Primary reporting metric: hit rate @ round N (top-10% fitness threshold) per the EVOLVEpro definition. Already implemented in `utils/metrics.py:hit_rate`. Add to standard reporting set in `generate_tables.py`.

### 3.3 Execution

50 seeds × 10 datasets × 2 budgets × ~10 methods ≈ 10,000 runs.

**Phase 3 exit criterion:** Heatmap (method × dataset, cell = hit-rate @ round 5) and the standard top-k-vs-queries plot, both for both budgets.

---

## Phase 4 — Task 3 (multi-objective) + paper artifacts

### 4.1 Multi-objective infrastructure

- Multi-objective reward in `alphavariant/components/reward.py`: `R = α·R_fit + β·log P(seq|ESM-2)` for plausibility-shaped tasks; `R = (R_obj1, R_obj2)` with Pareto dominance for true bi-objective.
- New oracle wrappers for ParPgb (from ALDE repo, `https://github.com/jsunn-y/ALDE`), eqFP611 dual-channel, and synthetic combos (GFP+stability, TEM-1+ProteinMPNN).
- `scripts/run_multi_objective.py` orchestrator that calls `utils/multi_objective.py:hypervolume`/`pareto_coverage` against the oracle's true Pareto front (computed once per dataset).

### 4.2 Paper-ready artifacts

- `scripts/plotting/plot_pareto_front.py` (new).
- `scripts/generate_tables.py` extended with multi-objective tables.
- `docs/methods_section.md` — methods text with dataset checksums, env hashes, GPU-hour totals.
- `docs/reproducibility_appendix.md` — exact command lines for every figure.

**Phase 4 exit criterion:** End-to-end paper figure pack regenerable from `make figures` on a fresh checkout (assuming data downloaded and HPC results synced back).

---

## Critical files to modify or create

**Modify:**
- `scripts/prepare_proteingym.py` — extend `ASSAYS` dict
- `scripts/generate_tables.py` — Bonferroni, GPU-hour columns
- `scripts/add_script_link.sh` — register new methods + per-dataset wrappers
- `utils/data.py` — add `make_hierarchical_split`
- `INTEGRATION.md` — document HPC and ablation workflows
- `CLAUDE.md` — add HPC section

**Create:**
- `utils/sequence_plausibility.py`, `utils/multi_objective.py`, `utils/proteingym_oracle.py`
- `scripts/prepare_combingym.py`
- `scripts/hpc/{ibex_array.sbatch, shaheen_array.sbatch, launch.py, env_setup.sh, log_resource_use.py, method_resources.yaml}`
- `alphavariant/{pipeline.py, config.py, components/{prior,space,reward,policy}.py}` + refactored thin run scripts
- `scripts/{EVOLVEpro,ftMLDE,MULTIevolve}/run_*.py`
- `data/CHECKSUMS.txt`
- `Random/run_<dataset>.py`, `GreedyWalk/run_<dataset>.py` per-dataset wrappers

---

## Reused existing components (do not duplicate)

- `utils/metrics.py` — AUOC (line 1080), hit_rate (line 1133), normalized_fitness_median_topk (line 402), batch_diversity (line 353), novelty (line 306), global_max_hit_count (line 674)
- `utils/data.py:FitnessLandscape` — O(1) oracle; works for any `(seq,fitness)` dataset
- `utils/io.py` — multi-seed aggregation, JSON/CSV/LaTeX export, NumpyEncoder
- `utils/evaluator.py:BenchmarkEvaluator` — round-by-round metric loop
- `utils/compat.py` — keeps existing run scripts working unchanged
- `scripts/plotting/{plot_optimization_curves.py, plot_radar.py}` — extend, don't replace
- `rand_seeds.txt` — 500 seeds, slice first 50
- `INTEGRATION.md` — established integration pattern for new methods

---

## Verification plan

**Phase 1:**
1. `python scripts/prepare_proteingym.py --datasets BLAT_ECOLX --max_variants 1000` — succeeds, produces `data/BLAT_ECOLX/data.csv`.
2. `python scripts/prepare_combingym.py --datasets PhoQ` — produces `data/PhoQ/data.csv`.
3. `python -c "from utils.sequence_plausibility import esm2_ppl; print(esm2_ppl(['MKVLW']))"` — returns float.
4. `python -c "from utils.multi_objective import hypervolume; print(hypervolume([[1,2],[2,1]], ref=[3,3]))"` — returns float.
5. `python scripts/hpc/launch.py --method ALDE --dataset GB1 --seeds 2 --cluster local` — produces 2 result files.
6. Submit a 5-seed iBex array; confirm completion.

**Phase 2:**
1. Each new baseline (EVOLVEpro, ftMLDE, MULTI-evolve) passes its `run_GB1.py --seed 42` smoke test.
2. AlphaVariant ablation refactor: `--ablation none` produces metrics within ±1% of pre-refactor numbers (regression check on cached GB1 baseline).
3. `--ablation no-gpt` runs without error and produces results that differ from `--ablation none`.

**Phase 3:** Sweep launches, hit-rate heatmap regenerates without missing cells.

**Phase 4:** `python scripts/run_multi_objective.py --dataset eqFP611 --method alphavariant --seed 42` produces hypervolume + Pareto coverage in result JSON.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| AlphaVariant refactor introduces subtle behavior changes | Snapshot pre-refactor results on GB1 (50 seeds) as a regression baseline before touching code. |
| iBex/Shaheen quotas insufficient for 30k+ runs | Profile method runtime in Phase 1; if AlphaVariant takes >2h/seed, reduce to 30 seeds (still >statistical threshold) or stagger campaigns. |
| ESM-2 PPL on long sequences (>500aa) is slow | Cache embeddings per sequence; batch within a method run; use ESM-2 t12 (35M) if t33 (650M) is too slow for filtering use cases. |
| Dataset license issues (CombinGym, ProteinGym redistribution) | Don't redistribute raw data; keep `prepare_*.py` download scripts only. CHECKSUMS verify integrity. |

---

## Estimated effort (engineering weeks)

| Phase | Effort | Critical path |
|---|---|---|
| 1 — Infra | 2 weeks | HPC launcher + ablation refactor scaffolding |
| 2 — Task 1 + baselines + ablations | 3 weeks | AlphaVariant refactor (highest risk) |
| 3 — Task 2 sweep | 2 weeks (mostly compute time) | HPC throughput |
| 4 — Task 3 + paper artifacts | 2 weeks | Multi-objective integration |

Total: ~9 weeks of engineering + concurrent HPC compute. Phase 1 must complete before any other phase starts; Phases 2–4 can overlap once infra is stable.
