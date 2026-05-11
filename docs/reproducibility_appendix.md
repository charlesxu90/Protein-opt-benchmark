# Reproducibility appendix

This appendix lists the exact commands required to reproduce every numeric
result in the paper, the dataset checksums, and the per-method environment
hashes. It is mechanical: a colleague with the repo and our checksums
should be able to regenerate every figure end-to-end on iBex with one
`make figures` invocation. <!-- TODO: build the Makefile -->

---

## 1. Bootstrap

```bash
git clone <!-- TODO --> AlphaVariant-Benchmark
cd AlphaVariant-Benchmark

# Per-method conda envs (only build the ones you need). Paths are matched
# by scripts/hpc/method_resources.yaml; if you build elsewhere, edit that
# file's `conda_env:` entry (absolute paths and ~ are supported).
conda env create -f ALDE/environment.yml          -p ALDE/env
conda env create -f EvoPlay/environment.yml       -p EvoPlay/env
conda env create -f FLEXS/environment.yml         -p FLEXS/env
conda env create -f AiCE/environment.yml          -p AiCE/env
# LatProtRL and delta_cs use nested-env layouts (latprotrl_env / delta_cs_env)
# matching their upstream scripts:
conda env create -f LatProtRL/environment.yml     -p LatProtRL/env/latprotrl_env
conda env create -f delta_cs/environment.yml      -p delta_cs/env/delta_cs_env
# AlphaVariant — example uses an absolute path outside the repo:
conda env create -n alphavariant-env -f alphavariant/environment.yml
# Phase 2.1 baselines: build with the helper script
bash scripts/setup_baseline_envs.sh   # builds EVOLVEpro/env, ftMLDE/env, MULTIevolve/env

./scripts/add_script_link.sh

# Sanity-check every env is wired correctly
python -c "
import sys; sys.path.insert(0, 'scripts/hpc')
from launch import load_resources, resolve_method_resources, resolve_python_for_method
from pathlib import Path
doc = load_resources()
for m in ['ALDE','EvoPlay','LatProtRL','FLEXS','AiCE','delta_cs','alphavariant',
          'Random','GreedyWalk','EVOLVEpro','ftMLDE','MULTIevolve']:
    res = resolve_method_resources(m, doc)
    py = resolve_python_for_method(m, res.get('conda_env',''))
    ok = Path(py).exists() and 'env' in py
    print(f'  {m:<14} {res.get(\"conda_env\",\"\"):<55} {\"OK\" if ok else \"MISSING\"}')
"
```

## 2. Dataset preparation

```bash
# CombinGym (clones github.com/sitonglab/CombinGym, ~1GB clone, ~5 min)
python scripts/prepare_combingym.py

# ProteinGym (~1GB download, ~10 min)
python scripts/prepare_proteingym.py

# Verify byte-for-byte match with paper datasets
python scripts/compute_dataset_checksums.py --verify
```

Expected `data/CHECKSUMS.txt` entries (regenerated 2026-05-06; current snapshot):

```
<!-- Paste current CHECKSUMS.txt contents here at submission time -->
```

**Known artifact:** `data/AAV_med/data.csv` and `data/AAV_hard/data.csv` are
byte-identical (same MD5). The "med" / "hard" distinction in the
LatProtRL-derived setup was never reflected in the data file, only in the
oracle protocol. We use both rows so downstream tooling has consistent paths.
<!-- TODO: clarify this provenance with the LatProtRL authors before submission. -->

## 3. Local smoke test (no HPC)

```bash
# 5 seeds of Random + GreedyWalk on GB1 (~30 s)
python scripts/hpc/launch.py --method Random     --dataset GB1 --seeds 5 --cluster local
python scripts/hpc/launch.py --method GreedyWalk --dataset GB1 --seeds 5 --cluster local
```

Expected:
- 5 `metrics_seed*.json` per method under `Random/results/GB1_Random/...`
- Random max fitness ≈ 6.5–7.6 across seeds
- GreedyWalk max fitness ≈ 8.7–8.76 (hits the global maximum on most seeds)

## 4a. Workstation sweeps (2× A100)

For a 2-GPU local workstation, drive both GPUs from one shell by partitioning
the seed range:

```bash
# 30-seed sweep on GB1, split 15/15 across two A100s
python scripts/hpc/launch.py --method ALDE --dataset GB1 --seeds 15 \
    --cluster local --gpu-id 0 --seed-start 0 &
python scripts/hpc/launch.py --method ALDE --dataset GB1 --seeds 15 \
    --cluster local --gpu-id 1 --seed-start 15 &
wait
```

Calibrate the budget first:

```bash
# One-seed pass on every method, CSV at results/_profiles/per_method_walltime.csv
python scripts/profile_methods.py --dataset GB1 --seed 42 --gpu-id 0
```

Use the resulting CSV to estimate sweep cost:

| Methods × seeds × datasets | mean wall(min) | total GPU-hours (single GPU) | wall-time on 2× A100 |
|---|---|---|---|
| <!-- TODO from per_method_walltime.csv --> | | | |

## 4b. iBex sweeps (50 seeds each)

```bash
# Task 1 — Combinatorial epistasis (GB1, CR9114, CreiLOV)
for ds in GB1 CR9114 CreiLOV; do
  for method in Random GreedyWalk ALDE EvoPlay alphavariant FLEXS AiCE; do
    python scripts/hpc/launch.py --method "$method" --dataset "$ds" \
        --seeds 50 --cluster ibex
  done
done

# Task 2 — Sample efficiency (ProteinGym sweep, dual budget)
for ds in BLAT_ECOLX GFP_AEQVI DYR_ECOLI MK01_HUMAN PABP_YEAST POLG_HCVJF \
          DLG4_HUMAN HSP82_YEAST TPK1_HUMAN UBE4B_MOUSE; do
  # Budget A: 5 × 96 (ALDE-style)
  python scripts/hpc/launch.py --method alphavariant --dataset "$ds" \
      --seeds 50 --cluster ibex \
      --extra-args "--n_rounds 5 --batch_size 96"
  # Budget B: 10 × 16 (EVOLVEpro-style)
  python scripts/hpc/launch.py --method alphavariant --dataset "$ds" \
      --seeds 50 --cluster ibex \
      --extra-args "--n_rounds 10 --batch_size 16"
done

# Ablation suite (GB1 only; per Phase 2 plan)
for ablation in none no-gpt no-space static-reward no-rl; do
  python scripts/hpc/launch.py --method alphavariant --dataset GB1 \
      --seeds 50 --cluster ibex \
      --extra-args "--ablation $ablation"
done

# Task 3 — Multi-objective on eqFP611
python scripts/hpc/launch.py --method alphavariant --dataset eqFP611 \
    --seeds 50 --cluster ibex \
    --extra-args "<!-- TODO multi-objective driver flags -->"
```

## 5. Tables and figures

```bash
# Comparison tables with Bonferroni-corrected pairwise Wilcoxon
python scripts/generate_tables.py --stat_test wilcoxon --bonferroni \
    --include_resources

# Optimization curves (Task 1, Task 2)
python scripts/plotting/plot_optimization_curves.py
# Radar chart (multi-metric summary)
python scripts/plotting/plot_radar.py
# Pareto front (Task 3)
python scripts/plotting/plot_pareto_front.py --dataset eqFP611 \
    --methods Random GreedyWalk ALDE alphavariant
```

## 6. Per-figure command index

| Figure | Reproducer | Inputs |
|---|---|---|
| Fig. 1 (overview)         | hand-drawn / vector | — |
| Fig. 2 (Task 1 curves)    | `plot_optimization_curves.py --datasets GB1 CR9114 CreiLOV` | results from §4 Task 1 |
| Fig. 3 (Task 2 heatmap)   | `<!-- TODO heatmap script -->` | results from §4 Task 2 |
| Fig. 4 (ablations)        | `plot_optimization_curves.py --datasets GB1 --methods alphavariant_none alphavariant_no-gpt ...` | §4 ablation suite |
| Fig. 5 (Pareto front)     | `plot_pareto_front.py --dataset eqFP611` | §4 Task 3 |
| Supp. Fig. S1 (compute)   | `generate_tables.py --include_resources` | resource.json from every run |

## 7. Random seeds

We use the first 50 entries of `rand_seeds.txt` (500 pre-generated seeds, fixed
at repo creation time). The seed file is byte-stable; verify with:

```bash
sha256sum rand_seeds.txt
# <!-- TODO paste hash -->
```

## 8. Environment hashes

| Method | Python | Conda env hash | Key deps |
|---|---|---|---|
| ALDE         | 3.11 | <!-- TODO --> | torch, scipy, scikit-learn |
| EvoPlay      | 3.8  | <!-- TODO --> | torch, gym, numpy |
| LatProtRL    | 3.9  | <!-- TODO --> | torch, fair-esm, transformers |
| FLEXS        | 3.7  | <!-- TODO --> | torch, sklearn |
| AiCE         | 3.8  | <!-- TODO --> | torch, ProteinMPNN |
| delta_cs     | 3.7  | <!-- TODO --> | torch, gflownet |
| alphavariant | TBD  | <!-- TODO --> | torch, popgen, popscorer |
| EVOLVEpro    | TBD  | <!-- TODO --> | torch, transformers, ESM-2 |
| ftMLDE       | TBD  | <!-- TODO --> | sklearn, scipy, deep_sequence (opt.) |
| MULTIevolve  | TBD  | <!-- TODO --> | torch, wandb (offline) |

`<!-- TODO -->`: regenerate with `conda env export --no-builds | sha256sum`
inside each env at submission time.

## 9. Notes for reviewers

- **Determinism:** Optimisations seeded by `--seed`; PyTorch operations use
  `torch.use_deterministic_algorithms(True)` where supported. cuDNN nondeterminism
  in conv ops is the dominant remaining source of seed-to-seed variance; we
  report mean ± std across 50 seeds rather than single trajectories.
- **Hardware sensitivity:** The same seed on iBex A100 vs Shaheen MI250x can
  produce different round-by-round trajectories (FP32 reduction order). Final
  metrics agree within reported std.
- **License:** Code released under MIT. ProteinGym data is redistributed under
  its <!-- TODO --> license; we do not redistribute, only download via script.
  CombinGym data is under <!-- TODO --> license.
