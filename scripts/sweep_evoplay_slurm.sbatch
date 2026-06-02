#!/usr/bin/env bash
#SBATCH --job-name=evoplay_4site
#SBATCH --output=sweep_logs/4site_extra/evoplay_slurm/job_%A_%a.out
#SBATCH --error=sweep_logs/4site_extra/evoplay_slurm/job_%A_%a.err
#SBATCH --array=0-119%30
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --partition=batch
# EvoPlay 4site sweep: 4 datasets × 30 seeds = 120 array tasks
# - %30 limits to 30 concurrent jobs
# - Each task = single (dataset, seed) pair
# - GPU usage per task ~600 MiB; can share but easier to request 1 per task
#
# Submission:
#   sbatch scripts/sweep_evoplay_slurm.sbatch
#   sbatch --account=k01234 scripts/sweep_evoplay_slurm.sbatch          # iBex/Shaheen
#   sbatch --array=0-119%50 scripts/sweep_evoplay_slurm.sbatch          # bump concurrency
#   sbatch --array=0-29 --export=DS=4site_PhoQ scripts/...              # single dataset
#
# Resume after partial completion: array tasks whose metrics already exist
# are skipped (idempotent).

set -euo pipefail

BENCH_ROOT="/home/xux/Desktop/AlphaVariant/Benchmark"
cd "$BENCH_ROOT"
mkdir -p sweep_logs/4site_extra/evoplay_slurm

DATASETS=(4site_GB1 4site_PhoQ 4site_TEV 4site_TRPB)
SEEDS=($(head -30 rand_seeds.txt))

# If $DS env var is set, restrict to that one dataset (30 seeds).
# Otherwise iterate all 4 datasets in array index 0..119.
TID=${SLURM_ARRAY_TASK_ID}
if [ -n "${DS:-}" ]; then
  ds="$DS"
  seed="${SEEDS[$TID]}"
else
  ds_idx=$(( TID / 30 ))
  seed_idx=$(( TID % 30 ))
  ds="${DATASETS[$ds_idx]}"
  seed="${SEEDS[$seed_idx]}"
fi

OUT_DIR="EvoPlay/results/${ds}_EvoPlay/${ds}/topn_pvn"
LOG_FILE="sweep_logs/4site_extra/evoplay_slurm/${ds}_seed${seed}.log"

# Idempotent: skip if metrics already produced
if [ -f "$OUT_DIR/metrics_seed${seed}.json" ] || [ -f "$OUT_DIR/seed_${seed}/metrics.json" ]; then
  echo "[skip] ${ds} seed=${seed} (metrics already present)"
  exit 0
fi

echo "[evoplay_slurm] task=${TID} ds=${ds} seed=${seed} host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-0}"
echo "[evoplay_slurm] start $(date)"

cd "$BENCH_ROOT/EvoPlay"
PYTHONUNBUFFERED=1 ./env/bin/python -u run_${ds}.py \
  --seed "$seed" --batch_size 96 --n_rounds 5 \
  --use_gpu --verbose 1 \
  --output_path "results/${ds}_EvoPlay/${ds}/topn_pvn" \
  > "$BENCH_ROOT/$LOG_FILE" 2>&1
rc=$?
cd "$BENCH_ROOT"
echo "[evoplay_slurm] end   $(date)  rc=$rc"
exit $rc
