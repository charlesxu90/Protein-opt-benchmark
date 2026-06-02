# eqFP611_joint MOO Benchmark — Concrete Implementation Plan

**Companion to:** [`eqFP611_moo_plan.md`](./eqFP611_moo_plan.md)
**Author:** Implementation survey (Claude Code)
**Date:** 2026-05-31

## 1. What this document is

The companion plan (`eqFP611_moo_plan.md`) describes the **scientific goals** of benchmarking AlphaVariant on the joint blue/red eqFP611 landscape. This document maps each scientific goal onto **concrete code in this repo**, identifying what already exists, what needs to be written, and the order in which to execute. Every file path is verified against the current working tree.

## 2. Reconciled facts (corrections to the companion plan)

| Item | Companion plan | Actual (verified) |
|---|---|---|
| Pareto-optimal count | "10 globally Pareto-optimal sequences" | **17** (computed by `pareto_front_mask` on `data/eqFP611_joint/data.csv`) |
| Landscape size | unspecified | 8192 unique sequences, 233 aa each |
| Objective ranges | unspecified | blue ∈ [0.091, 1.608], red ∈ [0.025, 1.692] |
| Reference point for HV | "dataset-wise minimum or slightly worse" | **(0, 0)** — already used by `scripts/aggregate_moo.py` |
| Reference HV | unspecified | **0.8821** (vs `(0, 0)`) |
| Wild-type | unspecified | `data/eqFP611_joint/wt.fasta` (233 aa, full-length) |
| Scalarization (existing) | "weighted sum / product" suggested | **`sqrt(blue · red)`** — already hard-coded in `scripts/alphavariant/run_generic.py::_scalarized_fitness` and matched in `utils.data.load_landscape_data` |
| Replicates | "20 independent runs" | Existing 7-method comparison uses **30 seeds**. Plan is to match this — start with 10 seeds, expand to 30 once gated. |

## 3. What already exists (reuse, don't rewrite)

| Asset | Path | What it does |
|---|---|---|
| Joint dataset | `data/eqFP611_joint/data.csv` | 8192×3 (`seq, blue, red`) |
| Wild-type | `data/eqFP611_joint/wt.fasta` | Reference sequence, 233 aa |
| MOO primitives | `utils/multi_objective.py` | `pareto_front_mask`, `hypervolume`, `pareto_front_coverage`, `auto_reference_point` |
| Joint loader | `utils/data.py::load_joint_objectives` | Returns aligned `(sequences, blue, red)` arrays |
| Aggregator | `scripts/aggregate_moo.py` | Per-seed → median[Q1,Q3] per method. Currently knows 7 methods. |
| Renderer | `scripts/render_moo_table.py` | Emits `tables/<dataset>/moo_comparison.md` |
| AlphaVariant wrapper | `scripts/alphavariant/run_eqFP611_joint.py` | Already exists (thin wrapper → `run_generic.py`) |
| AlphaVariant trainer | `scripts/alphavariant/run_generic.py` | Auto-scalarizes blue+red; emits `seed_<N>/metrics.json` with `queried_indices` |
| HPC launcher | `scripts/hpc/launch.py` + `scripts/hpc/method_resources.yaml` | Picks per-method conda env automatically |
| Existing 30-seed comparison | `tables/eqFP611_joint/{moo_comparison.md, moo_summary.json}` | 7 methods × 30 seeds, current state of art for this dataset |
| Single-objective runs | `<Method>/results/eqFP611_{blue,red}_<Method>/eqFP611_{blue,red}/*/metrics_seed*.json` for Method ∈ {Random, ALDE, AiCE, ftMLDE, CLADE, GreedyWalk} | Per-seed `queried_indices` for the SO-only optimizers, ready to project into joint space |

## 4. Method roster (final scope)

### A. Already in `moo_comparison.md` (7 methods × 30 seeds)
Random, GreedyWalk, ALDE, ftMLDE, CLADE, AiCE, FLEXS — no new work, but their metrics rows will be **recomputed** after the aggregator is extended (section 5).

### B. Runnable now, need aggregation wiring (1 method)
- **AlphaVariant** — `scripts/alphavariant/run_eqFP611_joint.py` already exists. One smoke seed already lives at `alphavariant/results/eqFP611_joint_AlphaVariant/`. Just needs (a) full 10→30 seed sweep, (b) new entry in `METHOD_PATTERNS`, (c) seed-id extraction fix in `aggregate_moo.py` (its results use `seed_<N>/metrics.json`, not `metrics_seedNN.json`).

### C. Non-trivial adapter work (1 method, optional / stretch)
- **EvoPlay** — `scripts/EvoPlay/run_GB1.py` is a complete 1500-line implementation (NOT a thin wrapper). To run on eqFP611_joint we must:
  1. Write `scripts/EvoPlay/run_eqFP611_joint.py` modeled on `run_GB1.py`, but:
     - Replace hard-coded `protein='GB1'`, wildtype `'VDGV'`, and `load_gb1_data` with eqFP611 equivalents.
     - Use `seq` column (233-aa strings) instead of `AACombo`.
     - Compute one-hot feature matrix 8192 × (233 × 20) ≈ 38M floats (~150 MB) — feasible.
  2. Verify that MCTS over 233 × 20 = 4660 actions/move converges on a finite design space. EvoPlay was designed for short combinatorial sites (4–6 aa). For 233-aa it may be very slow or behave pathologically.
  3. **Status: stretch goal.** Skip from initial run; flag as future work if Section 7 figures look thin.

### D. Free SO baselines (project existing runs into MOO space)
For each `Method ∈ {Random, ALDE, AiCE, ftMLDE, CLADE, GreedyWalk}` and each `objective ∈ {blue, red}`, results already exist under `<Method>/results/eqFP611_<objective>_<Method>/eqFP611_<objective>/*/metrics_seed*.json`. The aggregator will read `queried_indices`, look up `(blue, red)` from the joint landscape, and compute the full MOO metric suite. **Reported as `Random_so_blue`, `ALDE_so_red`, etc.** — 12 new rows, zero new compute.

### E. Explicitly skipped (per scope decisions)
| Method | Reason |
|---|---|
| LatProtRL | env not built |
| delta_cs | per user request |
| MULTIevolve | `scripts/MULTIevolve/run_generic.py` is **scaffolding only** — refuses to run, requires upstream completion (WandB-free training mode + predictor adapter). Multi-week task. README confirms it's multi-*mutant*, not multi-*objective*. |
| EVOLVEpro | `scripts/EVOLVEpro/run_generic.py` is **scaffolding only** — refuses to run, requires ESM-2 embedding pipeline + `grid_search` parser. Despite its README claiming multi-property optimization, the adapter doesn't currently expose it. Defer to a separate integration project. |
| qNEHVI / NSGA-II native baselines | not in scope; left as future work |

**Final table size after build-out:** 7 existing scalarized + AlphaVariant + 12 SO-projection rows = **20 rows**. (+ EvoPlay if Section C is attempted.)

## 5. Concrete code changes

### 5.1 `scripts/aggregate_moo.py` — surgical edits

**Edit A: extend `METHOD_PATTERNS`**

```python
METHOD_PATTERNS = {
    "Random":       "Random/results/{ds}_Random/{ds}/random/metrics_seed*.json",
    "GreedyWalk":   "GreedyWalk/results/{ds}_GreedyWalk/{ds}/greedy/metrics_seed*.json",
    "AiCE":         "AiCE/results/{ds}_AiCE/{ds}/aice/metrics_seed*.json",
    "ALDE":         "ALDE/results/{ds}_ALDE/{ds}/onehot/metrics_seed*.json",
    "FLEXS":        "FLEXS/results/{ds}_AdaLead/{ds}/metrics_seed*.json",
    "ftMLDE":       "ftMLDE/results/{ds}_ftMLDE/{ds}/ftmlde/metrics_seed*.json",
    "CLADE":        "CLADE/results/{ds}_CLADE/{ds}/clade/metrics_seed*.json",
    # NEW
    "AlphaVariant": "alphavariant/results/{ds}_AlphaVariant/seed_*/metrics.json",
}
```

**Edit B: handle both seed-id conventions in `seed_of()`**

The existing helper extracts seed from a filename like `metrics_seed42.json`. AlphaVariant writes `seed_42/metrics.json`, where the seed is in the parent dir. Update:

```python
def seed_of(p: Path) -> int:
    # filename convention: metrics_seedNN.json
    stem = p.stem
    if stem.startswith("metrics_seed"):
        try: return int(stem.split("seed")[-1])
        except ValueError: pass
    # dir convention: seed_NN/metrics.json
    parent = p.parent.name
    if parent.startswith("seed_") or parent.startswith("seed"):
        try: return int(parent.split("_")[-1] if "_" in parent else parent[4:])
        except ValueError: pass
    return 0
```

**Edit C: extend `per_seed_moo_metrics()`**

Add these fields to the returned dict:

```python
# Normalization constants — compute ONCE at module level when landscape is loaded
# B_min, B_max, R_min, R_max from full landscape
B_tilde = (blue - B_min) / (B_max - B_min)
R_tilde = (red  - R_min) / (R_max - R_min)
b_t = B_tilde[qi]; r_t = R_tilde[qi]

# Candidate-selection metrics
product_score_max     = float(np.max(qb * qr))                         # raw product, not normalized first
max_min_norm_max      = float(np.max(np.minimum(b_t, r_t)))
distance_to_ideal_min = float(np.min(np.sqrt((1.0 - b_t)**2 + (1.0 - r_t)**2)))

# Threshold hits
wt_b, wt_r = blue[wt_idx], red[wt_idx]                                  # wt_idx looked up via wt.fasta
p75_b, p75_r = np.percentile(blue, 75), np.percentile(red, 75)
n_hits_wt  = int(np.sum((qb >= wt_b) & (qr >= wt_r)))
n_hits_p75 = int(np.sum((qb >= p75_b) & (qr >= p75_r)))
frac_hits_wt  = n_hits_wt  / len(qi)
frac_hits_p75 = n_hits_p75 / len(qi)

# HV regret
hv_regret = float(ref_hv - hv)
```

**Edit D: trajectory checkpoints**

```python
TRAJECTORY_CHECKPOINTS = [96, 192, 288, 384, 480]

def trajectory_metrics(qi_ordered, blue, red, ref_front, ref_hv):
    """Replay queried_indices in order, evaluating metrics at each checkpoint."""
    out = {}
    for k in TRAJECTORY_CHECKPOINTS:
        if k > len(qi_ordered): break
        prefix = qi_ordered[:k]
        pts = np.column_stack([blue[prefix], red[prefix]])
        out[k] = {
            "hv_norm":         hypervolume(pts, np.array([0.0, 0.0])) / ref_hv,
            "pareto_coverage": pareto_front_coverage(pts, ref_front),
            "product_score":   float(np.max(blue[prefix] * red[prefix])),
        }
    return out

# HV-AUC over the checkpoint grid (trapezoidal)
hv_auc = float(np.trapz([d["hv_norm"] for d in traj.values()], list(traj.keys())))
```

`queried_indices` is already the ordered list per round in all method outputs.

**Edit E: SO projection mode**

Add a `--so_projection` CLI flag. When set, expand `METHOD_PATTERNS` at runtime with entries like:

```python
"Random_so_blue":  "Random/results/eqFP611_blue_Random/eqFP611_blue/random/metrics_seed*.json",
"Random_so_red":   "Random/results/eqFP611_red_Random/eqFP611_red/random/metrics_seed*.json",
# ... and for ALDE, AiCE, ftMLDE, CLADE, GreedyWalk
```

The same `per_seed_moo_metrics()` then evaluates them — they were optimizing a single objective (blue or red), but we evaluate their `queried_indices` against the JOINT landscape to expose the trade-off they implicitly make.

### 5.2 `scripts/render_moo_table.py`

Extend to include new columns in median[Q1,Q3] format:
- `product_score`, `max_min`, `dist_to_ideal`, `n_hits_p75`, `hv_auc`

Emit a separate `tables/eqFP611_joint/moo_trajectories.md` for the HV-vs-budget table (keeps the main comparison readable).

### 5.3 No source changes needed for these
- `utils/multi_objective.py` — already complete
- `utils/data.py::load_joint_objectives` — already complete
- `scripts/alphavariant/run_eqFP611_joint.py` — already correct (thin wrapper)
- `scripts/alphavariant/run_generic.py` — scalarization already correct

## 6. Metric definitions (formulas)

Using `B̃, R̃` for min–max-normalized values over the full landscape.

| Metric | Formula | Where |
|---|---|---|
| Max scalarized | `max sqrt(b · r)` over queried | already in aggregator |
| Product score max | `max (b · r)` over queried | NEW |
| Max-min (norm) | `max min(B̃, R̃)` over queried | NEW |
| Distance to ideal | `min sqrt((1-B̃)² + (1-R̃)²)` over queried | NEW |
| Dual-threshold hit count (WT) | `#{i : b_i ≥ b_wt ∧ r_i ≥ r_wt}` | NEW |
| Dual-threshold hit count (P75) | `#{i : b_i ≥ P75(blue) ∧ r_i ≥ P75(red)}` | NEW |
| Hypervolume (norm) | `HV(queried, ref=(0,0)) / HV_ref` | already in aggregator |
| HV regret | `HV_ref − HV(queried)` | NEW |
| HV-AUC | `∫ HV_norm(t) dt` over budget grid (trapezoidal) | NEW |
| Pareto coverage | `|{r ∈ P*: ∃d ∈ queried, d ⪰ r}| / |P*|` | already in aggregator |
| Trajectory | per-checkpoint `{hv_norm, pareto_coverage, product_score}` for k ∈ {96, 192, 288, 384, 480} | NEW |

## 7. Staged execution gates

Per CLAUDE.md "decompose before long runs." **Each gate must pass before proceeding.**

### G0 — Data sanity (<5 s)
```bash
python -c "
from utils.data import load_joint_objectives
s, b, r = load_joint_objectives('eqFP611_joint')
print(f'n={len(s)}  blue=[{b.min():.3f}, {b.max():.3f}]  red=[{r.min():.3f}, {r.max():.3f}]')
"
# Expect: n=8192  blue=[0.091, 1.608]  red=[0.025, 1.692]
```

### G1 — Re-aggregate existing 7 methods with extended metrics (<2 min)
After Edit C/D land:
```bash
python scripts/aggregate_moo.py --dataset eqFP611_joint --first_n 1
# Verify: "Pareto-optimal: 17", "reference HV ≈ 0.8821"
# Verify: new metric fields appear in the printed table and moo_summary.json
```

### G2 — AlphaVariant 1-seed smoke
```bash
cd alphavariant && python run_eqFP611_joint.py --seed 42
# Expect: alphavariant/results/eqFP611_joint_AlphaVariant/seed_42/metrics.json
# Expect: metrics.json contains "queried_indices" of length ≈ 480
```
**If this seed takes >2 hours, stop and revisit budget — 30 seeds × N hours is the wall-clock you sign up for.**

### G3 — Aggregator picks up AlphaVariant
After Edit A/B land:
```bash
python scripts/aggregate_moo.py --dataset eqFP611_joint --methods AlphaVariant --first_n 1
# Expect: 1 populated row, no "<no files>" warning
```

### G4 — SO projection rows
After Edit E lands:
```bash
python scripts/aggregate_moo.py --dataset eqFP611_joint --so_projection
# Expect: 12 extra rows (Random_so_blue, Random_so_red, ..., GreedyWalk_so_red)
```

### G5 — AlphaVariant 10-seed sweep
```bash
# Two-GPU workstation pattern (per CLAUDE.md)
python scripts/hpc/launch.py --method alphavariant --dataset eqFP611_joint \
    --seeds 5 --cluster local --gpu-id 0 --seed-start 0 &
python scripts/hpc/launch.py --method alphavariant --dataset eqFP611_joint \
    --seeds 5 --cluster local --gpu-id 1 --seed-start 5 &
wait
# Expect: 10 metrics.json files
```

### G6 — Full extended table at 10 seeds
```bash
python scripts/aggregate_moo.py --dataset eqFP611_joint
python scripts/aggregate_moo.py --dataset eqFP611_joint --so_projection
python scripts/render_moo_table.py --dataset eqFP611_joint
# Inspect: tables/eqFP611_joint/moo_comparison.md should show 8 scalarized rows + 12 SO rows with all new metrics
# Inspect: tables/eqFP611_joint/moo_trajectories.md
```
**Sanity check:** AlphaVariant's max_scalarized should be in a plausible band relative to AiCE (~0.728) and CLADE (~0.645). If it's nonsensically high or low, debug before scaling.

### G7 — Expand to 30 seeds
Only after G6 passes. Add seeds 10–29:
```bash
python scripts/hpc/launch.py --method alphavariant --dataset eqFP611_joint \
    --seeds 10 --cluster local --gpu-id 0 --seed-start 10 &
python scripts/hpc/launch.py --method alphavariant --dataset eqFP611_joint \
    --seeds 10 --cluster local --gpu-id 1 --seed-start 20 &
wait
# Re-run aggregator + renderer
```

### G8 — Optional: EvoPlay adapter (stretch)
Only if Section C is approved as in-scope:
1. Copy `scripts/EvoPlay/run_GB1.py` → `scripts/EvoPlay/run_eqFP611_joint.py`
2. Replace dataset loading with eqFP611-specific code (column `seq` instead of `AACombo`, WT from `wt.fasta`, scalarize blue+red via `sqrt(b·r)`).
3. Run with `--seed 42` smoke first; verify MCTS converges on the enumerated 8192-sequence design space (NOT free amino-acid space — must constrain to landscape lookup).
4. If smoke passes and per-seed runtime is reasonable, sweep 10 seeds → 30 seeds with same launcher pattern.

## 8. Reporting layout

Mapping to the companion plan's Figures 1–5:

| Companion plan | Implementation |
|---|---|
| Figure 1: blue vs. red scatter w/ Pareto highlighted | `scripts/plotting/plot_pareto_front.py` (already exists) — pass `--dataset eqFP611_joint --highlight-pareto` |
| Figure 2: HV vs. query budget | NEW small script `scripts/plotting/plot_moo_trajectory.py` that reads `moo_summary.json`'s `trajectory` field |
| Figure 3: Pareto coverage vs. budget | same as Figure 2, different y-axis |
| Figure 4: best product score / max-min vs. budget | same script, different y-axis |
| Figure 5: discovered candidates in objective space | extend `plot_pareto_front.py` with per-method overlays |
| Table 1: final metric summary | `tables/eqFP611_joint/moo_comparison.md` (extended) |
| Table 2: top AlphaVariant candidates | NEW small script `scripts/dump_top_candidates.py --dataset eqFP611_joint --method AlphaVariant --rank-by product_score --top 20` → emits `tables/eqFP611_joint/alphavariant_top_candidates.md` |

## 9. Risks and open questions

| Risk | Mitigation |
|---|---|
| **AlphaVariant per-seed runtime unknown for 233-aa eqFP611.** The trainer in `run_generic.py` is 3134 lines with GPT fine-tuning per round. GB1 (4-aa) per-seed runtime is known; 233-aa scaling is not. | G2 single-seed smoke is the canary. Stop and revisit if >2 h/seed. |
| **EvoPlay MCTS may not handle 233-aa cleanly.** Action space of 4660 per move; may diverge or time out. | Stretch-goal only; skip if G2 reveals AlphaVariant alone gives a complete enough story. |
| **MULTIevolve & EVOLVEpro scaffolding is multi-week work.** Not feasible to include in this sprint. | Explicitly out of scope; document as future work. |
| **SO projection comparability.** SO-blue and SO-red runs were performed with method-specific defaults (batch size, n_rounds) that may differ from the joint runs. Their `queried_indices` are still well-defined, but the budgets may not align at exactly 480. | Aggregator should report the actual budget used per SO run alongside its metrics. Truncate or pad consistently — preferable: report metrics at the seed's actual final budget AND at a common 480 cutoff if reached. |
| **Wild-type fitness for dual-threshold hit rate.** Need `wt_idx` in the joint landscape. The WT 233-aa sequence is in `wt.fasta`; look up its row in `data.csv` to get `wt_b`, `wt_r`. If the WT is absent from the enumerated landscape, fall back to the 50th-percentile threshold and document the choice. | Document in the aggregator output. |
| **Stochastic tie-breaking in median[Q1,Q3] reports.** 10 seeds is small for tight IQR estimates. | Headline numbers wait for G7 (30 seeds); G6 (10 seeds) is for sanity, not publication. |

## 10. Out of scope

- Native multi-objective Bayesian optimization (qNEHVI / NSGA-II) — left as future work.
- LatProtRL on eqFP611_joint (env not built).
- delta_cs on eqFP611_joint (per user request).
- Finishing MULTIevolve and EVOLVEpro scaffolding — separate integration projects.
- Visual regression / dashboard reporting beyond the markdown tables and matplotlib figures above.

## 11. Estimated effort

| Work item | Effort |
|---|---|
| Aggregator edits A–E + renderer extension | 3–4 h |
| G0–G4 verification | 1 h |
| AlphaVariant 10-seed sweep (G5) | depends on per-seed runtime — TBD by G2 |
| Full extended table render (G6) | 30 min |
| AlphaVariant 30-seed sweep (G7) | 2× G5 wall-clock |
| Figures 1–5 | 2–3 h |
| EvoPlay adapter (G8, stretch) | 6–10 h (likely fails first attempt due to 233-aa scaling) |
| **Critical path total (without EvoPlay)** | **1–2 days of human time + AlphaVariant sweep wall-clock** |
