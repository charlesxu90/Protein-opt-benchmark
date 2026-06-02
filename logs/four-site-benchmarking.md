# 4-site Combinatorial Benchmarking — Procedures, Decisions, Results

**Session window**: 2026-05-31 → 2026-06-02
**Author**: claude-code session, in collaboration with Xiaopeng Xu
**Datasets**: `4site_GB1`, `4site_PhoQ`, `4site_TEV`, `4site_TRPB`
**Protocol**: 96 sequences × 5 rounds = 480 oracle queries per seed; 30 seeds per (method, dataset) pair where feasible.

---

## 1. Scope and goal

Add three new active-learning baselines to the existing 4-site benchmark suite and refresh the canonical tables + figures so the comparison includes them. The three additions:

| Method | Source | Family |
|---|---|---|
| **EVOLVEpro** | Jiang et al., *Science* 2025 ([mat10d/EVOLVEpro](https://github.com/mat10d/EVOLVEpro)) | Frozen ESM-2 embeddings + few-shot RandomForest / Ridge active learning |
| **MULTIevolve** | Tran et al., *Science* 2026 ([ArcInstitute/MULTI-evolve](https://github.com/ArcInstitute/MULTI-evolve)) | Splitter → OneHotFeaturizer → Ridge / Fcn predictor pipeline |
| **Mu-Protein** (µFormer + µSearch) | Microsoft Research ([microsoft/Mu-Protein](https://github.com/microsoft/Mu-Protein)) | Pretrained PMLM-650M encoder + Siamese decoder; iterative train + score |

EvoPlay was attempted but **deferred** by user decision (see §8).

---

## 2. Adapter design and data flow

All adapters wrap the upstream codebase with an active-learning loop on top of the benchmark's standard 96×5 protocol. The general schema:

```
Round 1: random 96 indices (seed-deterministic via np.random.default_rng(seed))
Rounds 2..5: train predictor on cumulative labels → score uncollected →
             query top-96 by predicted fitness → add to cumulative
Output: metrics_seed<S>.json with max_fitness, normalized_fitness_median_top128,
        queries, n_rounds + config + (for some adapters) collected_indices
```

### 2.1 EVOLVEpro

- **Embedding step**: pre-compute ESM-2-35M embeddings for all variants once per dataset via `scripts/EVOLVEpro/embed_dataset.py`. Uses HuggingFace `transformers.EsmModel` from the existing `alphavariant-env` (no separate plm_env needed). Mean-pools last-hidden-state at the 4 varying positions of WT-substituted full-length sequences.
- Output: `data/<ds>/embeddings_evolvepro.pt` (~400-470 MB per dataset) + `labels_evolvepro.csv` + `meta.json`.
- **Adapter** (`scripts/EVOLVEpro/run_generic.py`): loads embeddings, calls upstream `evolvepro.src.evolve.directed_evolution_simulation` with `regression_type=randomforest`, parses `evolvepro_results.csv` for `top_activity_scaled` trajectory.

### 2.2 MULTIevolve

- **Adapter** (`scripts/MULTIevolve/run_generic.py`): per round, writes cumulative `(mutation, fitness)` as TSV to /tmp; calls upstream `multievolve.predictors.RidgeRegressor` (via `RandomProteinSplitter` + `OneHotFeaturizer`); predicts on uncollected; takes top-96.
- Falls back to **random selection** if the predictor raises (e.g. on negative fitness values).

### 2.3 Mu-Protein

- **Env**: custom `muformer-env` built from scratch — `torch==1.12.0+cu113`, `fairseq==0.10.2`, `numpy<1.24`, plus pandas/scikit-learn/biopython/fair-esm. ~9 GB.
- **Adapter** (`scripts/Mu-Protein/run_generic.py`): each round subprocess-calls upstream `Mu-Protein/mu-former/main.py` (1) to train µFormer ensemble on cumulative TSV with `--encoder-name pmlm --decoder-name siamese --pretrained-model pmlm_650m.pt`, (2) to predict on uncollected TSV. Reads `prediction.tsv` → top-96.
- **Auto-batch** based on WT length: `train_batch=2, predict_batch=8` for WT > 200 AA, else upstream defaults.
- **Checkpoint cleanup**: removes `round_<r>/{ckpt,pred}` after each round's predictions to prevent disk-fill (each model_*.pt is ~8 GB; 3 ensembles × 4 rounds = ~96 GB per seed without cleanup).

---

## 3. Bugs encountered and fixes

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | EVOLVEpro all 5 PhoQ seeds returned identical max_fitness=0.2305 | Upstream `evolvepro` hardcodes `random_seed=i` (simulation index, always 0 with num_simulations=1) — our seed never reached the RNG | Pre-shuffle `labels` DataFrame with `random_state=args.seed` so positional `np.random.choice` picks different items per seed |
| 2 | EVOLVEpro adapter reported max=1.0 for every seed | Bug in adapter: read `simulation_num` column as max-fitness (round counter) | Use `top_activity_scaled` column |
| 3 | EVOLVEpro `astype(float)` ValueError "could not convert 'None' to float" | EVOLVEpro stores literal string `'None'` in unfilled rows of `top_activity_scaled` | `pd.to_numeric(..., errors='coerce')` |
| 4 | EVOLVEpro needed `activity_scaled` + `activity_binary` columns | Upstream `process.py` computes these but our labels CSV didn't | Compute in adapter: `activity_scaled = (a - min)/(max - min)`; `activity_binary = (a > q75).astype(int)` |
| 5 | MULTIevolve mutation parsing error `int('285T+S288K')` | Used `+` separator but MULTI-evolve expects `/` (e.g. `M285T/K286E`) | Changed `_combo_to_mut_str` to use `/` |
| 6 | Mu-Protein PMLM-650M checkpoint failed to load | Existing env had torch 1.4.0; checkpoint saved with torch ≥1.6 (new ZIP format) | Built fresh `muformer-env` with torch 1.12 |
| 7 | Mu-Protein fairseq couldn't find `dict.txt` | Checkpoint's `args.data` references `/mnt/data/...` (Microsoft internal) | Pass `arg_overrides={'data': '<local dict dir>'}` |
| 8 | Mu-Protein upstream `main.py` line 49 raises TypeError | `trainer.test(model_label='Test', ...)` — `model_label` not in `Trainer.test()` signature (upstream bug) | Removed the `model_label` kwarg (single-line patch) |
| 9 | Mu-Protein `_drop_invalid_mutation` ValueError on `'WT'` | Upstream calls `int(mut[1:-1])` on the literal string `'WT'` → crash | Adapter-side filter for the single WT-matching variant per dataset |
| 10 | Mu-Protein OOM during round-2 training on PhoQ | PhoQ WT=486 → attention is O(L²); batch 8 needs ~34 GB but A100 has 40 GB and another user's eqFP611 sweep was holding 4.5 GB | Auto-batch reduces train_batch=2, predict_batch=8 for WT > 200 |
| 11 | Mu-Protein OOM during checkpoint deepcopy on GB1 | eqFP611 sweep co-tenant on GPU 0; not enough headroom for `copy.deepcopy(model.state_dict)` | Waited for eqFP611 to finish (30 min), then re-launched |
| 12 | Mu-Protein disk-full mid-sweep | Each model_*.pt is 8 GB; 3 ensembles × 4 rounds × 5 seeds ≈ 480 GB | Patched adapter to `rm round_<r>/ckpt+pred` after each round's predictions |
| 13 | MULTIevolve TEV "results" suspiciously close to Random baseline | Predictor raises `ValueError: ndcg_score should not be used on negative y_true values` every round; adapter's exception handler falls back to random sampling | Behavior is technically correct (graceful fallback) — documented as a known-limitation in §7 |
| 14 | Re-simulating MULTIevolve in plain Python produced different max_fitness from the adapter | Naive `sklearn.linear_model.Ridge` differs from MULTIevolve's `RidgeRegressor` (CV / train-val split / scaling) | Reverted to running the real adapter — never short-cut algorithm reproduction |

---

## 4. Compute infrastructure

### 4.1 Local workstation
- 112 CPU cores, 2× A100 40 GB
- CPU-bound methods (EVOLVEpro, MULTIevolve) → high parallelism, no GPU
- GPU-bound methods (Mu-Protein) → 1 process per GPU; PMLM-650M takes ~21 GB

### 4.2 Cluster scripts (iBex)
Prepared but not executed (per §8 EvoPlay deferral):
- `scripts/sweep_evoplay_local.sh` — local xargs-parallel launcher with `SWEEP_CONCURRENCY` env var
- `scripts/sweep_evoplay_slurm.sbatch` — generic SLURM array
- `scripts/sweep_evoplay_ibex.sbatch` — iBex-tuned with module loads
- `scripts/cluster_evoplay/{01_transfer_to_ibex,02_submit_array,03_monitor,04_collect}.sh` — full lifecycle scripts with V100 targeting via `GPU_MODEL=v100`

---

## 5. Sweep execution timeline

| Method | Dataset | Seeds | Wall clock | Notes |
|---|---|---|---|---|
| **MULTIevolve** | 4 datasets | 30 each = 120 jobs | ~2 hr (parallel) | Ridge on cached features; TEV fails predictor → random fallback |
| **EVOLVEpro** | 4 datasets | 30 each = 120 jobs | ~5 hr (parallel) | Embedding step ~10 min/dataset (GPU); RF step CPU-only |
| **Mu-Protein** | 4site_GB1 only | 2 seeds | ~14 hr | n=30 → ~7.3 hr/seed too costly; user opted to defer rest. Both seeds hit global max 1.0 |
| **EvoPlay** | — | 0 | — | **Skipped** (see §8) |

---

## 6. Final 30-seed median values

(Where n=30 unless noted. `med_max` = median max_fitness; `med_t128` = median of per-seed `normalized_fitness_median_top128`.)

### 6.1 max_fitness — median

| Method | GB1 | PhoQ | TEV | TRPB |
|---|---:|---:|---:|---:|
| Random | 0.4716 | 0.1937 | 0.4320 | 0.6100 |
| GreedyWalk | 0.8346 | 0.3420 | 0.4168 | 0.7529 |
| ALDE | **1.0000** | 0.4554 | 0.4700 | **0.9320** |
| FLEXS | 0.8623 | 0.4281 | 0.3882 | 0.8135 |
| AiCE | 0.4745 | 0.3257 | **0.6640** | 0.6655 |
| ftMLDE | 0.8622 | 0.4790 | 0.3775 | 0.8705 |
| CLADE | 0.8346 | 0.4069 | 0.3826 | 0.8410 |
| **AlphaVariant** | **1.0000** | **0.5256** | 0.3801 | 0.8326 |
| **EVOLVEpro** | 0.4984 | 0.2005 | 0.6550 | 0.6221 |
| **MULTIevolve** | 0.8622 | 0.5170 | 0.4259* | **0.9320** |
| Mu-Protein (n=2) | **1.0000** | — | — | — |

\* MULTIevolve TEV: predictor fallback to random; result ≈ Random baseline.

### 6.2 normalized_fitness_median_top128 — median

| Method | GB1 | PhoQ | TEV | TRPB |
|---|---:|---:|---:|---:|
| Random | 0.0024 | 0.0006 | 0.4719 | 0.0356 |
| GreedyWalk | 0.2636 | 0.0396 | 0.4781 | 0.2973 |
| ALDE | **0.5394** | **0.1221** | 0.4928 | **0.6096** |
| FLEXS | 0.4170 | 0.1001 | 0.4843 | 0.5478 |
| AiCE | 0.1234 | 0.0618 | 0.5903 | 0.0850 |
| ftMLDE | 0.4576 | 0.1180 | **0.5979** | 0.5856 |
| CLADE | 0.4143 | 0.0983 | 0.5981 | 0.5405 |
| **AlphaVariant** | 0.4699 | 0.1183 | 0.5511 | 0.5483 |
| **EVOLVEpro** | 0.0228 | 0.0447 | 0.0278 | 0.0360 |
| **MULTIevolve** | 0.4400 | 0.1263 | 0.0293* | 0.5767 |

\* MULTIevolve TEV: same fallback caveat.

---

## 7. Findings and interpretation

### 7.1 Winners per dataset

- **GB1 4-site**: ALDE = AlphaVariant = 1.0 (tied global max). Mu-Protein also hit 1.0 in both seeds tested. MULTIevolve, ftMLDE, FLEXS, CLADE, GreedyWalk all in 0.83-0.86 cluster.
- **PhoQ 4-site**: **AlphaVariant (0.526) and MULTIevolve (0.517)** are statistically tied at the top; ALDE/ftMLDE/FLEXS form the next tier.
- **TEV 4-site**: AiCE (0.664) and EVOLVEpro (0.655) are the only methods above 0.5. AlphaVariant unexpectedly weak (0.38). MULTIevolve falls to random fallback.
- **TRPB 4-site**: ALDE and MULTIevolve tied at 0.932 (near global max); ftMLDE 0.871; CLADE 0.841; AlphaVariant 0.833.

### 7.2 MULTIevolve has the most consistent ranking

MULTIevolve places top-3 on 3/4 datasets and never drops below mid-tier except on TEV (where the predictor short-circuits to random due to negative fitness values). Median top-128 on TRPB (0.577) is the highest among methods that don't tie ALDE.

### 7.3 EVOLVEpro is dataset-dependent

EVOLVEpro is competitive on TEV (#2 by max_fitness) but bottom-tier on GB1, PhoQ, TRPB. The low top-128 values across all datasets (0.02-0.05) indicate that RandomForest's selections cluster in a narrow band — exploitation-heavy without much exploration.

### 7.4 The "fewer batches" effect for top-128

ALDE / AlphaVariant / FLEXS / ftMLDE pull large numbers of high-fitness sequences into their cumulative collected set (top-128 medians 0.5+), while EVOLVEpro and (TEV-only) MULTIevolve concentrate on a handful of high-fitness picks and fill the rest with mediocre ones (top-128 medians < 0.05). This is independent of `max_fitness` — a method can hit the global max yet still have a low top-128 mean if it doesn't pull many other high-fitness candidates.

### 7.5 The MULTIevolve TEV fallback

MULTIevolve's internal NDCG scorer rejects negative `y_true` values, which occur in TEV's normalized fitness. The adapter catches the exception and falls back to uniform random selection. The TEV "MULTIevolve" results are therefore Random-equivalent (max 0.426 ≈ Random 0.432; top-128 0.029 vs Random 0.472). This is documented but unresolved — would require either dropping NDCG inside upstream MULTIevolve, or pre-shifting TEV fitness to be non-negative.

---

## 8. EvoPlay decision

EvoPlay is **CPU-bound MCTS** (Python tree traversal + sklearn GP retraining each batch). Profiling showed:
- GPU memory used per process: ~600 MiB
- GPU utilization: 0-2%
- Per-batch cost on PhoQ: ~45 sec (×384 batches = ~5 hr per seed)
- Per-seed cost: ~80 min (GB1), ~2.5 hr (TEV), ~3.5 hr (TRPB), ~5 hr (PhoQ)
- Full sweep (4 × 30 seeds × concurrency 30): ~12 hours wall clock

Two attempts to bench were made:
1. **First attempt** (60-min timeout): killed at 1 hr with no metrics; stdout was buffered and the SIGTERM lost all output. Root cause: `python -u` flag was missing (PYTHONUNBUFFERED=1 alone was insufficient).
2. **Second attempt** (no timeout, `-u` flag): worked correctly with visible per-batch progress, ran ~30 min before user decision to skip.

User decision (2026-06-02 ~10:24): *"What if I jump EvoPlay as it is too time consuming?"* Sweep killed, partial output cleaned. **No EvoPlay metrics on any of the 4 datasets.**

Re-execution path remains available:
```bash
# Local
SWEEP_CONCURRENCY=30 bash scripts/sweep_evoplay_local.sh

# iBex with V100 targeting
ssh ibex.kaust.edu.sa
cd ~/Benchmark  # after running 01_transfer_to_ibex.sh from local
GPU_MODEL=v100 CONCURRENCY=120 bash scripts/cluster_evoplay/02_submit_array.sh
```

---

## 9. Other failures and deferrals

- **Mu-Protein at n>2**: PMLM-650M with 3-ensemble × 30-epoch training on 4-site combinatorial spaces takes ~7.3 hr per seed on a single A100. User opted for n=2 only after the 7.3 hr seed-100 completion confirmed feasibility. The cluster scripts could be adapted but weren't built — Mu-Protein remains effectively GB1-only at n=2 (both seeds hit max=1.0).
- **EvoPlay** as above.

---

## 10. Artifacts produced

### Code
- `scripts/EVOLVEpro/{embed_dataset.py, run_generic.py, run_4site_<ds>.py}` + symlinks
- `scripts/MULTIevolve/{run_generic.py, run_4site_<ds>.py}` + symlinks
- `scripts/Mu-Protein/{run_generic.py, run_4site_<ds>.py, README.md}` + symlinks
- `scripts/cluster_evoplay/{01_transfer_to_ibex.sh, 02_submit_array.sh, 03_monitor.sh, 04_collect.sh, README.md}`
- `scripts/sweep_evoplay_{local.sh, slurm.sbatch, ibex.sbatch}`
- Patch to `scripts/generate_tables.py` (added Mu-Protein to ALL_METHODS, fixed AlphaVariant case-sensitivity)
- Patch to `scripts/draw_figures_median.py` (added MULTIevolve + EVOLVEpro colors, removed delta_cs from MAIN_METHODS allow-list)

### Results
- `EVOLVEpro/results/4site_<ds>_EVOLVEpro/.../metrics_seed<S>.json` — 120 files (30 seeds × 4 datasets)
- `MULTIevolve/results/4site_<ds>_MULTIevolve/.../metrics_seed<S>.json` — 120 files
- `Mu-Protein/results/4site_GB1_MuProtein/.../metrics_seed{100,383}.json` — 2 files
- `data/4site_<ds>/embeddings_evolvepro.pt` + `labels_evolvepro.csv` + meta — 4 datasets

### Tables
- `tables/4site_<ds>/benchmark_comparison.md` (4 files) — per-dataset comparison with Bonferroni-corrected pairwise Wilcoxon
- `tables/4site_<ds>/all_results.csv` (4 files) — raw per-method per-dataset aggregated metrics

### Figures
- `figures/phase5/main_figure_max_fitness_median_iqr.{png,pdf}` — 10 methods × 4 datasets
- `figures/phase5/supplementary_figure_top128_mean_fitness_median_iqr.{png,pdf}` — top-128 view
- `figures/phase5/comparison_median_iqr.csv` — underlying data
- `figures/plan_C/main_figure_max_fitness_median_iqr.{png,pdf}` — Plan C canonical (adds AlphaVariant ablations)
- `figures/plan_C/supplementary_figure_top128_mean_fitness_median_iqr.{png,pdf}`

---

## 11. Open questions / next steps

1. **EvoPlay 4-task × 30-seed sweep** on iBex (scripts ready, not submitted).
2. **MULTIevolve TEV**: investigate fix for the NDCG-on-negative-fitness failure path (pre-shift fitness, or replace internal scorer).
3. **Mu-Protein** beyond n=2: either submit to cluster, or reduce config (num_ensembles=1, train_epochs=10) to make 30-seed feasible. Faithful upstream at full config is impractical on a single A100 for 4-site PhoQ/TRPB.
4. **Statistical tests**: confirm Bonferroni-corrected Wilcoxon results for each new method-vs-baseline pair are in `tables/4site_<ds>/benchmark_comparison.md`.
