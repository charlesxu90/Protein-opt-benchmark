# Refined Computational Benchmark Plan v2 — Implementation Status

> Companion to `docs/Refined_benchmark_plan.md` (the original scientific design).
> This v2 records what has been built, what is currently running, and what remains.
> Cross-reference: `docs/refine_benchmark_implement_plan.md` (the original 4-phase implementation plan).

## 0. TL;DR — Where we are

| Layer | Status | Notes |
|---|---|---|
| Core infrastructure (Phase 1) | ✅ done | utils/, HPC scaffolding, datasets, stats, profiler all in place |
| AlphaVariant ablations (Phase 2.2) | ✅ done for GB1 | 4 ablation flags wired (`--ablation no-gpt|no-space|static-reward|no-rl`). Non-GB1 datasets accept the flag but raise `NotImplementedError` for non-`none` — extension TODO |
| Missing baselines (Phase 2.1) | 🟡 scaffolding only | EVOLVEpro / ftMLDE / MULTI-evolve repos cloned with READMEs documenting open work. Wrappers exit code 2 with a clear "not yet integrated" message |
| Task 1 datasets (CombinGym) | ✅ data ready | GB1 / CR9114 / CreiLOV / eqFP611 prepared into `data/<name>/data.csv` |
| Task 2 datasets (ProteinGym) | 🟡 acquisition pipeline ready | 16 assays configured in `scripts/prepare_proteingym.py`; downloads deferred to A100 box |
| Task 3 datasets (multi-objective) | ✅ infra ready | `scripts/run_multi_objective.py` verified on eqFP611. Reward integration into AlphaVariant TODO |
| GB1 30-seed sweep | 🟡 **in progress on 2× A100** | Fast methods first (Random / GreedyWalk / AiCE / ALDE / FLEXS); slow methods queued |
| Statistical reporting | ✅ done | `generate_tables.py --bonferroni --stat_test wilcoxon` |
| Reproducibility docs | ✅ done | `docs/methods_section.md` + `docs/reproducibility_appendix.md` with TODO markers for final numbers |

## 1. Original plan recap

From `Refined_benchmark_plan.md` (unchanged):

- **Task 1** Combinatorial epistasis (CombinGym hierarchical splits)
- **Task 2** Sample-efficient DE across ≥10 ProteinGym landscapes (dual budget: 10×16 EVOLVEpro-style and 5×96 ALDE-style)
- **Task 3** Multi-objective Pareto on eqFP611 + synthetic combinations

Baselines: **Random, GreedyWalk, CLADE, ftMLDE, ALDE, EVOLVEpro, AdaLead (FLEXS), EvoPlay, AiCE, Zero-Shot Consensus, MULTI-evolve**. Plus AlphaVariant + 4 ablations.

Metrics: max fitness, AUOC, hit rate @ round N, global-max recovery, sequence plausibility (ESM-2 PPL), diversity, novelty, hypervolume, Pareto coverage. 50 seeds with Wilcoxon + Bonferroni.

## 2. What's implemented

### 2.1 utils/ (new modules — Phase 1.1)

| Module | Purpose |
|---|---|
| `utils/multi_objective.py` | Hypervolume, Pareto front, Pareto coverage (pymoo fallback, exact 2D sweep) |
| `utils/sequence_plausibility.py` | ESM-2 pseudo-PPL with disk caching (lazy torch import) |
| `utils/proteingym_oracle.py` | Generic oracle wrapper, top-percent thresholds, hierarchical splits, mutation-order histograms |
| `utils/metrics.py` (extended) | `epistatic_score_correlation` (sequence-aware), `recall_high_order_mutants_from_seqs` (sequence-aware) |
| `utils/compat.py` (extended) | kwarg aliases for `normalized_fitness_topk`; scalar-tolerant `global_max_hit_count` |
| `utils/data.py` (fixed) | `load_landscape_data` default `sequence_col=None` so auto-detection works for `seq`/`AACombo` columns |

### 2.2 HPC + workstation orchestration (Phase 1.3)

```
scripts/hpc/
├── method_resources.yaml   # per-method GPUs/CPUs/mem/walltime/conda_env, with run_subdir support
├── env_setup.sh            # cluster-aware module loads (iBex/Shaheen/local)
├── ibex_array.sbatch       # KAUST iBex job-array template
├── shaheen_array.sbatch    # KAUST Shaheen template
├── launch.py               # main entrypoint: --method --dataset --seeds --cluster {local,ibex,shaheen}
│                           # supports --gpu-id, --seed-start, --use-method-env, --extra-args
└── log_resource_use.py     # wraps any run with wall/RSS/GPU-mem tracking → resource.json
```

`launch.py:resolve_python_for_method` accepts paths that are absolute, `~`-expanded, or relative to BENCHMARK_ROOT. Configured per-method envs (verified on this workstation):

| Method | conda_env path |
|---|---|
| ALDE / EvoPlay / FLEXS / AiCE | `<method>/env` |
| LatProtRL | `LatProtRL/env/latprotrl_env` (nested) |
| delta_cs | `delta_cs/env/delta_cs_env` (nested) |
| alphavariant | `/home/xux/miniforge3/envs/alphavariant-env` (absolute) |
| Random / GreedyWalk | reuse `ALDE/env` |
| EVOLVEpro / ftMLDE / MULTIevolve | not yet built — `scripts/setup_baseline_envs.sh` builds all three |

### 2.3 Datasets (Phase 1.2)

- `scripts/prepare_proteingym.py` — 16 assays configured (BLAT, CALM1, GFP, DYR, AMIE, KKA2, MK01, HIS7, PABP, HSP82, POLG_HCVJF, DLG4, TPK1, UBE4B, Q2N0S5 HIV Env, SPG1_STRSG_Wu)
- `scripts/prepare_combingym.py` — clones `sitonglab/CombinGym` and converts GB1 / CR9114 / CreiLOV / eqFP611 (+ eqFP611_blue / eqFP611_red split for multi-objective). PhoQ from the original plan is **not in CombinGym v1** — dropped or re-source needed.
- `scripts/compute_dataset_checksums.py` — produces `data/CHECKSUMS.txt` for the reproducibility appendix
- **Known artifact:** `data/AAV_med/data.csv` and `data/AAV_hard/data.csv` are byte-identical pre-existing in the repo. Flagged in `docs/reproducibility_appendix.md`

### 2.4 Method run scripts

- Random / GreedyWalk: full per-dataset wrappers (`run_GB1.py`, `run_AAV_med.py`, `run_AAV_hard.py`, `run_GFP_med.py`, `run_GFP_hard.py`) generated from `run_generic.py`
- FLEXS: `run_GB1.py` shim that delegates to `run_GB1_adalead.py` (added)
- alphavariant: full `--ablation` flag support on GB1; non-GB1 dataset scripts accept the flag and raise `NotImplementedError` for non-`none`
- EVOLVEpro / ftMLDE / MULTIevolve: `run_generic.py` scaffolding (exits code 2 with integration TODO message)

### 2.5 Smoke + profiling + sweep tooling

| Tool | Purpose |
|---|---|
| `scripts/smoke_test_methods.py` | Per-method "does it start?" check on a dataset; watches for success/error markers in real time, kills slow-but-progressing runs after 30s past first success line |
| `scripts/profile_methods.py` | Single-seed wall/RSS/GPU-mem profiler; writes `results/_profiles/per_method_walltime.csv` |
| `scripts/aggregate_metrics.py` | Cross-method aggregator; handles 3 schema flavors and auto-normalizes raw vs normalized max_fitness; outputs `results/<dataset>_summary.json` |
| `scripts/run_30seed_gb1_sweep.sh` | **NEW** — full 30-seed sweep orchestrator splitting seeds across 2 GPUs |
| `scripts/setup_baseline_envs.sh` | Builds EVOLVEpro/ftMLDE/MULTIevolve conda envs (run once before integrating those baselines) |

### 2.6 Multi-objective infrastructure (Phase 4.1)

- `scripts/run_multi_objective.py` — orchestrator for weighted-sum and independent-objective sweeps
- `scripts/plotting/plot_pareto_front.py` — HV-vs-α curves + scatter
- Verified end-to-end on eqFP611 with Random method: 17-point true Pareto front, 5.88% coverage from 480 random queries

### 2.7 Statistical + reproducibility

- `scripts/generate_tables.py`: extended with `--bonferroni`, `--include_resources`; CORE_METRICS now includes hit_rate_value, AUOC, global_max_hit_count; ALL_METHODS=12, ALL_DATASETS=24
- `docs/methods_section.md`: Nature Methods–style methods skeleton with `<!-- TODO -->` markers
- `docs/reproducibility_appendix.md`: per-figure command index, env hash table, known-artifact list

## 3. Bugs fixed during execution (10 total)

| # | Component | Bug | Resolution |
|---|---|---|---|
| 1 | `utils/data.py` | `load_landscape_data` default `sequence_col='sequence'` broke GB1's `seq` column | Default → `None`, auto-detect from candidates |
| 2 | `scripts/AiCE/run_GB1.py` | `spearman_correlation` not imported | Added to `from utils.compat import …` |
| 3 | `scripts/AiCE/run_GB1.py` | `miscalibration_area`/`expected_calibration_error` not imported | Added |
| 4 | `scripts/AiCE/run_GB1.py`, `scripts/FLEXS/run_GB1_adalead.py` | `epistatic_score_correlation` (4-arg sequence-aware) not in unified utils | Added to `utils.metrics`, re-exported |
| 5 | same | `recall_high_order_mutants` had 3-arg signature; scripts called 6-arg sequence-aware variant | Added `recall_high_order_mutants_from_seqs`, aliased imports |
| 6 | `utils/compat.py:normalized_fitness_topk` | FLEXS called with `global_min/global_max` kwargs | Added kwarg aliases |
| 7 | `utils/compat.py:global_max_hit_count` | FLEXS called with a scalar; expected list | Made scalar-tolerant |
| 8 | `scripts/hpc/launch.py`, `scripts/profile_methods.py`, sbatch templates | Symlinked scripts couldn't import sibling `src/` (ALDE etc.) | Prepend `BENCHMARK_ROOT/<method>` to `PYTHONPATH` |
| 9 | `scripts/LatProtRL/run_GB1.py` | `UnboundLocalError: final_seqs` when `env.discovered_sequences` empty | Initialise `final_seqs=[]`, fall back to buffer for metric block, guarded post-block save |
| 10 | `scripts/hpc/method_resources.yaml` + `launch.py` | delta_cs lives under `delta_cs/BioSeq-GFN-AL/` not directly in `delta_cs/`; `from lib.X` imports broke | Added `run_subdir` field; launcher sets cwd + PYTHONPATH to that subdir |

## 4. Current execution status (GB1 sweep)

- **Started:** 2026-05-11 ~10:15 KAUST time, on 2× A100-40GB workstation
- **Running:** `bash scripts/run_30seed_gb1_sweep.sh Random GreedyWalk AiCE ALDE FLEXS`
- **Per-method strategy:** 15 seeds on GPU 0 (seed-start 0) and 15 seeds on GPU 1 (seed-start 15) in parallel; orchestrator advances to the next method after both halves finish.
- **Single-seed pilot complete (n=1, seed=621):**

| Method | max(N) | regret | global_max? | wall |
|---|---|---|---|---|
| alphavariant | 1.0000 | 0.0000 | ✓ | 6 min |
| GreedyWalk | 0.8346 | 0.1654 | ✗ | 3 s |
| AiCE | 0.8311 | 0.1689 | ✗ | 4 s |
| Random | 0.7353 | 0.2647 | ✗ | 3 s |
| delta_cs | 0.7240 | 0.2760 | ✗ | 5 min |
| ALDE | 0.6271 | 0.3729 | ✗ | 21 s |
| FLEXS | 0.6073 | 0.3927 | ✗ | 10 s |
| EvoPlay | 0.5120 | 0.4880 | ✗ | 82 min (CPU) |
| LatProtRL | 0.0005 | 0.9995 | ✗ | 44 min (broken — see §5) |

n=1 means rankings are seed-noise dominated; the 30-seed sweep replaces this with mean±std + Wilcoxon.

## 5. Known limitations

- **LatProtRL doesn't converge at default config.** Log shows `ep_rew_mean=-1` across all 5 rounds — the RL agent gets no reward signal. The init warns `No trained VED found, will use ESM-2 fallback`. **Action:** train the VED (`scripts/LatProtRL/train_GB1_VED.py`) before including LatProtRL in the final sweep. Excluded from `run_30seed_gb1_sweep.sh` by default (`INCLUDE_BROKEN=1` to override).
- **EvoPlay is slow without GPU.** 82 min/seed on CPU. On A100 expect ~8 min/seed → 30 seeds × 8 min ÷ 2 GPUs ≈ 2 hours for the GB1 cell.
- **Method scripts disagree on raw vs normalized max_fitness reporting.** `aggregate_metrics.py` auto-rescales for the comparison table, but for clean reporting the unified `compute_all_metrics` should report both side-by-side. **TODO** (low priority): standardize in compat.
- **AAV_med / AAV_hard data files are byte-identical.** Pre-existing. Need to recover the "hard" split's actual oracle from LatProtRL provenance.
- **PhoQ missing from CombinGym v1.** The original plan listed it; the current `sitonglab/CombinGym` repo does not include it. Either drop PhoQ from Task 1 or sourced separately (Podgornaia & Laub 2015 original data).

## 6. Remaining TODOs

### High-priority (Phase 2 completion)

1. **Extend AlphaVariant ablations to non-GB1 datasets.** Currently `--ablation` raises `NotImplementedError` for AAV / GFP. The per-dataset `IterativeAAVTrainer` / `IterativeGFPTrainer` need analogous seam refactoring (~half-day work each).
2. **Full integration of EVOLVEpro.**
   - Build env via `bash scripts/setup_baseline_envs.sh EVOLVEpro`
   - Write `scripts/EVOLVEpro/embed_dataset.py` for one-time per-dataset ESM-2 embedding pre-computation
   - Implement `grid_search` output → metrics adapter
   - Per-dataset wrappers
3. **Full integration of ftMLDE.** Build env, write design-space builder for combinatorial datasets only (CombinGym), adapter for `run_mlde` output.
4. **Full integration of MULTI-evolve.** Build env (`WANDB_MODE=offline` required), adapt 3-step pipeline to single-seed iterative loop.
5. **Train LatProtRL VED** so it produces non-degenerate results, or drop LatProtRL from the comparison set.

### Medium-priority (Phase 3 execution)

6. **Download ProteinGym substitutions.** `python scripts/prepare_proteingym.py` will pull ~1 GB; takes 10–30 min. Run once on the workstation.
7. **Smoke + 30-seed sweep on remaining 4 CombinGym + 5+ ProteinGym datasets.** Same `run_30seed_gb1_sweep.sh` structure, but parameterize for any dataset.
8. **Generate Task 1 figures.** `plot_optimization_curves.py --datasets GB1 CR9114 CreiLOV` once per-dataset sweeps complete.
9. **Task 2 dual-budget protocol.** Run each method × each ProteinGym dataset at both `--rounds 5 --batch_size 96` and `--rounds 10 --batch_size 16`.

### Lower-priority (Phase 4)

10. **Multi-objective reward in AlphaVariant.** Component refactor under `alphavariant/components/reward.py` (planned but not implemented).
11. **Synthetic Task 3 oracles:** GFP+stability and TEM-1+ProteinMPNN combinations need precomputed inverse-folding scores per variant.
12. **AGGREGATION:** standardize the `compute_all_metrics` interface so all methods write the same schema (currently 3 flavors in the wild).

### Documentation

13. Fill in TODO markers in `docs/methods_section.md` and `docs/reproducibility_appendix.md` with actual citations, env hashes, and final numbers once sweeps complete.
14. Update `data/CHECKSUMS.txt` whenever new datasets are prepared (`python scripts/compute_dataset_checksums.py`).
15. Asana / project tracking for the remaining items in this list — current sections (Backlog / Todo / Done / Problem) per `CLAUDE.md`.

## 7. Updated estimated effort to publication

| Workstream | Effort | Bottleneck |
|---|---|---|
| GB1 30-seed sweep (in progress) | ~4 hours wall (2× A100) | EvoPlay 30 seeds |
| Extend AlphaVariant ablations (5 datasets) | 2-3 days | dataset-specific trainers |
| Integrate EVOLVEpro / ftMLDE / MULTI-evolve | 1 week each | upstream API adaptation |
| Train LatProtRL VED (or drop) | 1-2 days | pre-training cost |
| Task 2 sweep (10 datasets × 8 methods × 30 seeds × 2 budgets) | 1-2 weeks compute | ~50,000 GPU-hours worst case |
| Task 3 sweep + paper figures | 1 week | multi-obj reward, plots |

Realistic publication-ready dataset on this 2× A100 workstation: **6-8 weeks** with disciplined scope (30 seeds, 8 datasets covering Task 1 + most of Task 2, ablations on GB1 only).
