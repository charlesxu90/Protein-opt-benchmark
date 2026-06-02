# Running EvoPlay 4-task × 30-seed sweep on iBex (or any SLURM cluster)

Self-contained scripts to push the benchmark to iBex, submit a 120-task
SLURM array job, monitor it, and pull results back.

## Why use the cluster?

EvoPlay is **CPU-bound** (we profiled — MCTS in Python; GPU sits at 0%).
On a single 112-core workstation, 30 concurrent seeds saturate the
machine and total wall clock is ~10-12 hours for 4 datasets × 30 seeds.
On iBex you can run all 120 tasks **truly in parallel** (one per node),
limited only by partition capacity, finishing in ~5 hours (longest = PhoQ).

## One-time setup (from local workstation)

```bash
cd /home/xux/Desktop/AlphaVariant/Benchmark
bash scripts/cluster_evoplay/01_transfer_to_ibex.sh
# Default: ibex.kaust.edu.sa, target dir ~/Benchmark
# Or: bash scripts/cluster_evoplay/01_transfer_to_ibex.sh user@ibex.kaust.edu.sa /scratch/user/Benchmark
```

This rsyncs:
- benchmark code (small)
- `data/` (~5 MB per dataset)
- `rand_seeds.txt`
- `EvoPlay/env/` (3-5 GB conda env)

Then runs `add_script_link.sh` on the remote to create symlinks.

## Submit the array job (on iBex login node)

```bash
ssh ibex.kaust.edu.sa
cd ~/Benchmark

# Standard: 4 datasets × 30 seeds, GPU per task, max parallel
bash scripts/cluster_evoplay/02_submit_array.sh

# With your PI's account flag (if required):
ACCOUNT=k01234 bash scripts/cluster_evoplay/02_submit_array.sh

# CPU-only (recommended — EvoPlay barely uses GPU, faster queue):
CPU_ONLY=1 bash scripts/cluster_evoplay/02_submit_array.sh

# Smaller scope (10 seeds per dataset = 40 tasks):
N_SEEDS=10 bash scripts/cluster_evoplay/02_submit_array.sh
```

Prints the submitted JOB_ID. Save it.

## Monitor progress (on iBex)

```bash
bash scripts/cluster_evoplay/03_monitor.sh <JOB_ID>

# Auto-refresh every 30 sec:
watch -n 30 'bash scripts/cluster_evoplay/03_monitor.sh <JOB_ID>'
```

Shows:
- Running/Pending/Failed task counts (`squeue`)
- Metrics produced per dataset
- Latest progress per seed log
- Recent failures

## Collect results back (on local workstation)

When all tasks finish (`squeue -j JOB_ID` is empty):

```bash
# from local workstation:
bash scripts/cluster_evoplay/04_collect.sh
```

This rsyncs:
- `EvoPlay/results/4site_*_EvoPlay/...` (all metrics)
- `sweep_logs/4site_extra/evoplay_cluster/...` (run logs)

## Aggregate into tables + figures

After results land locally:

```bash
cd /home/xux/Desktop/AlphaVariant/Benchmark

# Per-dataset tables
for ds in 4site_GB1 4site_PhoQ 4site_TEV; do
  python3 scripts/generate_tables.py --datasets $ds \
    --format markdown --output_dir tables/$ds \
    --stat_test wilcoxon --bonferroni --report median_iqr
done
python3 scripts/generate_tables.py --datasets TRPB \
  --format markdown --output_dir tables/4site_TRPB \
  --stat_test wilcoxon --bonferroni --report median_iqr

# Phase 5 figure (add EvoPlay color to draw script first if needed)
# Rebuild CSV including EvoPlay
python3 <<'PY'
import csv, glob, json, numpy as np
# ... (see scripts/build_median_iqr_csv.py for the pattern)
PY

python3 scripts/draw_figures_median.py \
  --csv figures/phase5/comparison_median_iqr.csv \
  --outdir figures/phase5
```

## Resource recommendations

| Setting | Default | When to change |
|---|---|---|
| `N_SEEDS` | 30 | Smaller for faster test runs (5, 10) |
| `CONCURRENCY` | TOTAL | iBex partitions usually have no per-user cap; full parallel is fine |
| `CPUS` | 4 | EvoPlay is single-process, 4 is plenty |
| `MEM` | 24G | Comfortable for EvoPlay |
| `TIME_LIMIT` | 06:00:00 | PhoQ is the slowest (~5 hr); 6 hr is safe |
| `PARTITION` | batch | Check `sinfo` on iBex for actual partition names |
| `CPU_ONLY` | 0 | Set to 1 for CPU-only (recommended — GPU is wasted) |

## Common iBex gotchas

- **Conda init missing**: the script uses `module load anaconda3` but some iBex shells need explicit `source ~/.bashrc`. Adjust at top of submitted sbatch.
- **`env/bin/python` not found on iBex**: the env was built on local workstation; if iBex Python paths differ, rebuild in-place: `conda create -p EvoPlay/env --clone $(which python)` won't work; just `conda env create -f EvoPlay/environment.yml` if you keep one.
- **Account flag**: many KAUST projects require `--account=ku-xxx`. Ask your PI.
- **GPU partition vs CPU partition**: GPU jobs may queue longer. Set `CPU_ONLY=1`.
- **`results/` race**: array tasks all write to `EvoPlay/results/...`. The per-seed file naming is unique, so no conflict.

## What runs where

| File | Run on | Purpose |
|---|---|---|
| `01_transfer_to_ibex.sh` | local | rsync code + env to iBex |
| `02_submit_array.sh` | iBex | generate + submit SLURM array |
| `03_monitor.sh` | iBex | check progress |
| `04_collect.sh` | local | rsync results back |
