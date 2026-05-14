# Implementation Plan — Three-Dataset Benchmark (GB1, CreiLOV, CR9114-H1)

> Companion to `docs/Benchmark_plan_three_v1.md`. To be saved as
> `docs/Implementation_plan_three_v1.md` after approval.

## Context

`docs/Benchmark_plan_three_v1.md` refocuses the paper on three complementary
landscapes (GB1 / CreiLOV / CR9114-H1) and a small, methodologically diverse
comparator set:

- **Main baselines** (every dataset): Random, Greedy DE, ALDE, AdaLead/FLEXS,
  one classical-MLDE method (CLADE or ftMLDE), one RL/adaptive method
  (EvoPlay or µProtein).
- **Dataset-specific additions**: CLADE & EvoPlay for GB1; ftMLDE/µProtein for
  CreiLOV; EVOLVEpro for CR9114-H1.
- **Extended Data / diagnostic only**: AiCE, delta_cs, MULTI-evolve,
  ProteinMPNN, zero-shot PLM ranking, manufacturing-aware generative models.

The plan also introduces protocol changes vs the current 96×5 GB1 sweep:

- Multi-budget protocol: initial-N ∈ {24, 48, 96, 192}, batch ∈ {24, 48},
  rounds ∈ {3, 4, 5}.
- ≥20 seeds per condition (we keep 30 for parity with GB1 sweep).
- Median + IQR reporting (not mean ± std).
- Paired seed-level statistical comparisons.

**Current state** after the GB1 30-seed sweep:

- GB1 sweep complete for 8 methods (Random, GreedyWalk, AiCE, ALDE, FLEXS,
  delta_cs, alphavariant, EvoPlay) at 96×5; results in `tables/`.
- `data/CR9114/data.csv` (48,841 variants, seq_len=121, max=9.83) and
  `data/CreiLOV/data.csv` (165,428 variants, seq_len=120, max=15,686) ready.
- No per-method `run_CR9114.py` / `run_CreiLOV.py` exist; all 8 methods have
  `run_generic.py` but several have GB1-hardcoded wildtype handling.
- CLADE not in the repo; ftMLDE and EVOLVEpro are scaffolds that exit code 2.
- `scripts/aggregate_metrics.py` GLOBAL_MAX dict has only GB1; needs CR9114
  + CreiLOV entries before aggregation works.
- `scripts/generate_tables.py` reports mean ± std; no IQR option.

## Phasing

Five phases, each shipping a runnable artifact. Phase 1 is required before
Phases 2–4. Phase 5 depends on Phases 2–3 producing results.

---

## Phase 1 — Infrastructure prerequisites

### 1.1 Dataset constants

Edit `scripts/aggregate_metrics.py`:

```python
GLOBAL_MAX = {
    "GB1":     8.76196565571,
    "CreiLOV": 15686.30,       # from data/CreiLOV/data.csv
    "CR9114":  9.83,           # from data/CR9114/data.csv
    "AAV_med": 1.0, "AAV_hard": 1.0, "GFP_med": 1.0, "GFP_hard": 1.561,
}
```

Run `python scripts/compute_dataset_checksums.py` to refresh `data/CHECKSUMS.txt`.

### 1.2 Wildtype auto-detection in method scripts

The Phase-1 audit flagged hardcoded `"VDGV"` (GB1 wildtype) in:

- `scripts/AiCE/run_generic.py` (line ~132, `self.wildtype_combo = "VDGV"`)
- `scripts/EvoPlay/run_generic.py` (`get_wildtype_sequence()` returns "VDGV")
- `scripts/ALDE/run_generic.py` (line ~80, `wildtype_map = {'GB1': 'VDGV'}`)

For each, replace with a dataset-aware lookup that prefers (in order):
1. `--wildtype` CLI override.
2. Sequence at row of max fitness in `data/<dataset>/data.csv`
   (works for GB1: max-fitness row is `MQYK...FWAA...TE` with `AACombo=FWAA`;
   we want the WT *reference* — typically the FIRST sequence in the file,
   which is the wild-type for CombinGym format).
3. First sequence in the landscape (fallback).

Concretely use the existing helper `utils/proteingym_oracle.py:_detect_wildtype`
which already returns `sequences[0]`. Plumb it through each method's run script.

### 1.3 Per-method per-dataset wrappers

For each of 8 methods × 2 new datasets (CR9114, CreiLOV), generate:

```python
# scripts/<method>/run_<dataset>.py
import os, sys
sys.argv.insert(1, "<dataset>")
sys.argv.insert(1, "--dataset")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_generic import main
if __name__ == "__main__":
    main()
```

16 files total, generated with a small bash loop. Refresh symlinks via
`./scripts/add_script_link.sh`.

### 1.4 Median + IQR reporting

Add `--report {mean_std,median_iqr}` to `scripts/generate_tables.py` (default
`mean_std`). The `format_cell()` helper currently prints `mean ± std`; add a
sibling that prints `median [Q1, Q3]`. Both should use the same
`np.median` / `np.percentile` from numpy already in scope.

### 1.5 Extra metrics in compat

Three metrics in the new plan aren't in `utils/metrics.py` yet:

- **Top-1 hit rate**: fraction of seeds whose `max_fitness ≥ top-1% threshold`
  (`top_percent_threshold(fitness, 1.0)` from `utils.proteingym_oracle`).
- **Enrichment over random**: `max_fitness_method / max_fitness_random` per
  seed (computed at aggregation time, not per-run).
- **AUC of best-so-far**: already implemented as `area_under_optimization_curve`
  in `utils/metrics.py` and `compute_all_metrics` emits it as `auoc`. ✓

Add the first two to the aggregator (`scripts/aggregate_metrics.py`) and
the comparison table (`scripts/generate_tables.py:CORE_METRICS`).

### 1.6 Smoke-test loop

Run a 1-seed smoke for every (method, dataset) combination:

```bash
for ds in CR9114 CreiLOV; do
  python scripts/smoke_test_methods.py --dataset $ds --seed 100 \
      --per-method-timeout 600 --gpu-id 0 \
      --methods Random GreedyWalk ALDE AiCE FLEXS EvoPlay alphavariant delta_cs
done
```

Each method must reach a SUCCESS marker within 10 minutes. Bugs surface
here are wildtype / sequence-length mismatches that need patching.

**Phase 1 exit:** smoke-test report shows OK for all 8 methods on both CR9114
and CreiLOV.

---

## Phase 2 — Main-figure baseline integration

The new plan requires CLADE (or ftMLDE) and prefers AdaLead (= FLEXS, already
done). Adding CLADE and finishing ftMLDE/EVOLVEpro:

### 2.1 CLADE

Source: `https://github.com/guo-wei-wei/CLADE` (per refined plan; verify URL
since it 404'd earlier in the project — fallback search needed). Clone into
`CLADE/`, build env via a new entry in `scripts/setup_baseline_envs.sh`.

Write `scripts/CLADE/run_generic.py` modeled on existing baselines. CLADE
takes a CSV of single-mutant data + budget; we feed the first batch (24-96)
as singles, then iterate. Hyperparameters from the published GB1 setup
(K=4-12 clusters depending on dataset size).

Per-dataset wrappers for GB1, CreiLOV, CR9114 only.

### 2.2 ftMLDE

Already cloned to `ftMLDE/` and scaffolded at `scripts/ftMLDE/`. Finish:

- Build env: `bash scripts/setup_baseline_envs.sh ftMLDE` (Python 3.7 + TF1).
- Add `scripts/ftMLDE/build_design_space.py` that converts
  `data/<combinatorial>/data.csv` to ftMLDE's one-hot design space.
- Adapter from `run_mlde` output to our `metrics_seed{seed}.json` schema.
- Per-dataset wrappers for the 3 datasets.

ftMLDE assumes a finite enumerable design space — fits GB1 (4 sites),
CreiLOV (15 sites × 20 mutations = 1.84e19 — too big to enumerate; need
restricted variant set), CR9114 (16 sites × ~2-3 alleles = ~65k — fits).
For CreiLOV, restrict to the 165k measured variants as the candidate pool.

### 2.3 EVOLVEpro

Already cloned to `EVOLVEpro/` and scaffolded. Finish:

- Build env: `bash scripts/setup_baseline_envs.sh EVOLVEpro`.
- Write `scripts/EVOLVEpro/embed_dataset.py` for one-time ESM-2 embedding
  precomputation per dataset (the heavy part: ~30 min/dataset on A100).
- Adapter from `evolvepro.src.evolve.grid_search` output to our schema.
- Per-dataset wrappers for the 3 datasets, but priority is **CR9114**
  (the antibody dataset where EVOLVEpro is most relevant).

### 2.4 µProtein (optional, conditional)

Clone from `https://github.com/<TBD>` (search needed; paper Nat Mach Intell
2025). Same scaffolding pattern. **Lower priority** — only if CreiLOV main
comparison needs strengthening after Phase 5 results.

---

## Phase 3 — Protocol parameterization

Add `--n_init`, `--batch_size`, `--n_rounds` CLI args to each per-method
`run_generic.py`. Default values match the current 96×5; users override for
diagnostic budget sweeps:

```bash
# Main protocol
python scripts/hpc/launch.py --method ALDE --dataset CR9114 --seeds 30 \
    --cluster local --gpu-id 0 \
    --extra-args="--n_init 96 --batch_size 96 --n_rounds 5"

# Diagnostic: low-N budget
... --extra-args="--n_init 24 --batch_size 24 --n_rounds 5"
```

Touched files (one edit each, mechanical):

- `scripts/Random/run_generic.py`
- `scripts/GreedyWalk/run_generic.py`
- `scripts/ALDE/run_generic.py`
- `scripts/AiCE/run_generic.py`
- `scripts/FLEXS/run_generic.py`
- `scripts/EvoPlay/run_generic.py`
- `scripts/alphavariant/run_generic.py`
- `scripts/delta_cs/BioSeq-GFN-AL/run_generic.py`

For methods whose underlying algorithm cannot honor arbitrary budgets
(e.g., delta_cs's GFlowNet training is sized to total queries), document
the binding constraint.

---

## Phase 4 — Compute campaigns

Reuse `scripts/run_sweep_parallel.py` — already parameterized over
`--dataset`. Two campaigns:

### 4.1 Main campaign (publication table)

```bash
for ds in GB1 CreiLOV CR9114; do
  python scripts/run_sweep_parallel.py --dataset $ds --seeds 30 \
      --methods Random GreedyWalk ALDE AdaLead CLADE EvoPlay alphavariant \
      --gpus 0 1
done
```

Where `AdaLead` is the existing FLEXS method. For GB1, we already have 30
seeds at 96×5; skip and reuse. For CreiLOV and CR9114, fresh sweep.

Estimated wall on 2× A100:
- GB1: skip (already done)
- CreiLOV (165k variants, slightly bigger landscapes): ~6 hours
  per method × ~7 methods ≈ 1.5-2 days (EvoPlay dominates)
- CR9114 (48k variants, smaller): ~3 hours × 7 ≈ 1 day

### 4.2 Diagnostic budget sweep (Extended Data)

One method (alphavariant) × all 3 datasets × {24, 48, 96, 192} init × 5
rounds × 20 seeds. ~80 conditions × 20 seeds = 1600 runs. Selected
methods (Random, GreedyWalk, ALDE, alphavariant) for the budget curve;
others held at 96×5.

```bash
for ds in GB1 CreiLOV CR9114; do
  for init in 24 48 96 192; do
    python scripts/run_sweep_parallel.py --dataset $ds --seeds 20 \
        --methods Random GreedyWalk ALDE alphavariant \
        --extra-args="--n_init $init --batch_size 24 --n_rounds 5"
  done
done
```

### 4.3 Extended Data: assumption-mismatched comparators

AiCE / delta_cs / EVOLVEpro / MULTI-evolve / ProteinMPNN / zero-shot —
all 30 seeds × 3 datasets × main protocol only (no budget sweep).

---

## Phase 5 — Aggregation, figures, manuscript

### 5.1 Per-dataset comparison tables

```bash
for ds in GB1 CreiLOV CR9114; do
  python scripts/generate_tables.py --datasets $ds --first_n_seeds 30 \
      --methods Random GreedyWalk ALDE AdaLead CLADE EvoPlay alphavariant \
      --stat_test wilcoxon --bonferroni --report median_iqr \
      --format markdown,latex \
      --output_dir tables/$ds
done
```

Output: median + IQR per metric per method; pairwise paired-Wilcoxon p-values
with Bonferroni correction; LaTeX-ready table fragments.

### 5.2 Extended Data tables

Same script, different method list (AiCE + delta_cs + EVOLVEpro +
MULTI-evolve, etc.). Output to `tables/extended_data/`.

### 5.3 Figures

- **Main Fig** (per dataset): bar chart of normalized max_fitness with
  IQR error bars, methods sorted by median. Star annotations for
  Wilcoxon-significant differences.
- **Budget curve** (Extended Data): best-so-far vs cumulative queries,
  one line per method, faceted by dataset.
- **Pareto** (per dataset if multi-objective applies): use existing
  `scripts/plotting/plot_pareto_front.py`.

### 5.4 Methods-section text

Generate `docs/methods_text_v1.md` with:
- Fairness paragraph (verbatim from `docs/Benchmark_plan_three_v1.md`).
- Dataset descriptions + checksums from `data/CHECKSUMS.txt`.
- Per-method env hashes (run `conda env export | sha256sum` once per env).
- Computational protocol table (init / batch / rounds / seeds / metrics).

---

## Critical files to modify or create

**Modify (existing):**

- `scripts/aggregate_metrics.py` — add CR9114, CreiLOV to GLOBAL_MAX dict;
  add top-1/top-10 hit-rate and enrichment-over-random.
- `scripts/generate_tables.py` — `--report median_iqr` flag; `CORE_METRICS`
  with new fields.
- `scripts/ALDE/run_generic.py` — wildtype lookup; CLI args for budget.
- `scripts/AiCE/run_generic.py` — same.
- `scripts/EvoPlay/run_generic.py` — same.
- `scripts/{Random,GreedyWalk,FLEXS,alphavariant,delta_cs}/run_generic.py`
  — budget CLI args.
- `scripts/add_script_link.sh` — register CLADE and any new per-dataset
  wrappers.
- `scripts/hpc/method_resources.yaml` — add CLADE entry with env path.
- `scripts/setup_baseline_envs.sh` — add CLADE branch.

**Create:**

- `scripts/<method>/run_CR9114.py` and `run_CreiLOV.py` (16 files; one
  liner shims, identical pattern to `run_GB1.py` Random/GreedyWalk shims).
- `scripts/CLADE/run_generic.py` and `run_{GB1,CR9114,CreiLOV}.py`.
- `scripts/CLADE/README_INTEGRATION.md`.
- `scripts/EVOLVEpro/embed_dataset.py` (ESM-2 embedding precomputation).
- `scripts/ftMLDE/build_design_space.py`.
- `scripts/run_three_datasets_sweep.sh` (orchestrator wrapper around
  `run_sweep_parallel.py` for the full main-campaign run).
- `docs/Implementation_plan_three_v1.md` (this plan, copied after approval).
- `docs/methods_text_v1.md` (Phase 5.4 artifact).

---

## Reused existing components (do not duplicate)

- `scripts/run_sweep_parallel.py` — fully parameterized over `--dataset`,
  handles GPU partitioning and per-method concurrency.
- `scripts/hpc/launch.py` — `--gpu-id`, `--seed-start`, `--use-method-env`
  all work as-is.
- `scripts/aggregate_metrics.py` — auto-rescale via per-dataset GLOBAL_MAX
  (already in place; just needs the two new dataset entries).
- `scripts/generate_tables.py` — `--first_n_seeds N` dedup-by-seed
  (added in the GB1 final pass).
- `utils/multi_objective.py` — Pareto / hypervolume for any future
  multi-objective Task 3 work.
- `utils/proteingym_oracle.py:_detect_wildtype` — fallback wildtype.
- `utils/metrics.py:area_under_optimization_curve`, `hit_rate` — already
  emitted as `auoc` and `hit_rate_value`.

---

## Verification plan

**Phase 1 verification:**

1. `python scripts/aggregate_metrics.py --dataset CreiLOV --seed 100` runs
   without "global_max not found" warnings.
2. `python scripts/smoke_test_methods.py --dataset CR9114 --seed 100
   --per-method-timeout 600` shows 8/8 OK.
3. `python scripts/generate_tables.py --datasets GB1 --report median_iqr
   --methods Random ALDE --first_n_seeds 30` produces a "median [Q1, Q3]"
   table (re-using existing GB1 results).

**Phase 2 verification:**

1. `bash scripts/setup_baseline_envs.sh CLADE` builds `CLADE/env/`.
2. `python scripts/CLADE/run_GB1.py --seed 100 --skip_metrics` reaches
   "Round 1" within 5 min.
3. ftMLDE single-seed run produces a valid `metrics_seed*.json`.
4. EVOLVEpro single-seed run on CR9114 with precomputed embeddings produces
   a valid `metrics_seed*.json`.

**Phase 3 verification:**

1. `python <method>/run_GB1.py --seed 100 --n_init 24 --batch_size 24
   --n_rounds 5 --skip_metrics` runs for each method (8 commands).
2. Resulting JSONs report queries == 24 + 24×4 = 120 (not 480).

**Phase 4 verification:**

1. After each campaign run, `python scripts/aggregate_metrics.py
   --dataset $ds --seed <first_seed>` produces a complete row per method.
2. `results/<dataset>_summary.json` exists with all expected method keys.

**Phase 5 verification:**

1. `tables/<dataset>/benchmark_comparison.md` exists and renders
   correctly in GitHub preview.
2. `tables/<dataset>/benchmark_comparison.tex` compiles in a LaTeX test
   document.
3. `docs/methods_text_v1.md` contains: fairness paragraph, dataset
   checksums table, env-hash table, per-figure command index.

---

## Estimated effort

| Phase | Engineer days | Compute time | Notes |
|---|---|---|---|
| 1 — Infrastructure | 1-2 | minutes (smoke tests) | Mechanical; mostly editing wildtype handling and adding CLI args |
| 2 — Baselines (CLADE + finish ftMLDE/EVOLVEpro) | 1 week | hours (env builds, embeddings) | CLADE is the only fully-new method; the others are mostly stub-finishing |
| 3 — Protocol parameterization | 0.5 | none | Add 3 args to 8 scripts |
| 4 — Main + diagnostic sweeps | 0.5 (kick off) | 2-4 days wall | Dominated by EvoPlay; uses existing parallel orchestrator |
| 5 — Aggregation + manuscript artifacts | 1-2 | minutes | Tables + figures + methods text |

**Total: ~2 weeks engineering + 3-5 days compute** to a publication-ready
3-dataset table. CLADE is the gating dependency for Phase 2; if its repo URL
is dead, fall back to ftMLDE-only as the classical-MLDE comparator.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| CLADE upstream code unavailable / 404 | Fall back to ftMLDE only as classical-MLDE comparator. The plan's recommendation tier explicitly allows this. |
| ftMLDE TF 1.x env fails to build on modern A100 | Use the mlde2.yml variant (Phase 2.1 setup script default); if still fails, drop ftMLDE in favor of CLADE only. |
| CR9114 binary-alphabet handling breaks methods | Most methods use one-hot encoding internally; CR9114's 2-3-allele structure should pad to a 20-AA alphabet with most entries unused. If broken: write a CR9114-specific allele encoding adapter in `utils/data.py`. |
| CreiLOV's 15-site space is too big for ftMLDE enumeration | Restrict ftMLDE design space to the 165k measured variants (it's the "candidate pool" constraint anyway). |
| µProtein code unavailable | Plan explicitly marks µProtein as conditional; skip if not available. |
| EvoPlay slow on larger landscapes | 4 concurrent workers per GPU (as for GB1); may take 1-2 days per dataset. Acceptable. |
| Different methods report metrics in different schemas | `scripts/aggregate_metrics.py` already handles 3 schema flavors (dict, list, final_metrics); no new work needed. |
