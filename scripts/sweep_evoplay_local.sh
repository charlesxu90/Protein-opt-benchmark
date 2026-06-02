#!/usr/bin/env bash
# Parallel EvoPlay sweep on local multi-GPU machine.
# 4 datasets × 30 seeds = 120 jobs. CPU-bound (single thread per job),
# ~600 MiB GPU per job (negligible). Split across 2 GPUs.
#
# Concurrency: SWEEP_CONCURRENCY env var (default 60). With 112 CPUs and
# 2×40 GB A100s, 60 parallel jobs = ~30/GPU = ~18 GB/GPU (safe).
#
# Usage:
#   bash scripts/sweep_evoplay_local.sh                    # all 4 datasets × 30 seeds
#   SWEEP_CONCURRENCY=40 bash scripts/sweep_evoplay_local.sh
#   DATASETS="4site_GB1 4site_TEV" bash scripts/sweep_evoplay_local.sh  # subset
#
# Logs:    sweep_logs/4site_extra/evoplay_sweep/<ds>_seed<s>.log
# Metrics: EvoPlay/results/4site_<ds>_EvoPlay/4site_<ds>/topn_pvn/metrics_seed<s>.json
#          OR EvoPlay/results/4site_<ds>_EvoPlay/4site_<ds>/topn_pvn/seed_<s>/metrics.json
set -euo pipefail

BENCH_ROOT="/home/xux/Desktop/AlphaVariant/Benchmark"
cd "$BENCH_ROOT"

DATASETS=${DATASETS:-"4site_GB1 4site_PhoQ 4site_TEV 4site_TRPB"}
SWEEP_CONCURRENCY=${SWEEP_CONCURRENCY:-60}
LOG_DIR="$BENCH_ROOT/sweep_logs/4site_extra/evoplay_sweep"
mkdir -p "$LOG_DIR"

# 30 seeds from rand_seeds.txt
SEEDS=($(head -30 rand_seeds.txt))
echo "[evoplay_sweep] datasets=$DATASETS  seeds=${#SEEDS[@]}  concurrency=$SWEEP_CONCURRENCY"

# Generate the job list (dataset seed gpu_id)
JOB_FILE="$LOG_DIR/_joblist.txt"
> "$JOB_FILE"
# Interleave datasets so all 4 run concurrently (round-robin by seed)
# This way concurrency N parallelizes ACROSS datasets, not within
idx=0
for s in "${SEEDS[@]}"; do
  for ds in $DATASETS; do
    gpu=$(( idx % 2 ))
    echo "$ds $s $gpu" >> "$JOB_FILE"
    idx=$((idx+1))
  done
done
NJOBS=$(wc -l < "$JOB_FILE")
echo "[evoplay_sweep] total jobs: $NJOBS  (split alternately GPU 0 / GPU 1)"

# Worker function — runs one (dataset, seed) on the assigned GPU
run_one() {
  local ds=$1 seed=$2 gpu=$3
  local log="$LOG_DIR/${ds}_seed${seed}.log"
  local out="EvoPlay/results/${ds}_EvoPlay/${ds}/topn_pvn"
  # Skip if metrics already present
  if [ -f "$out/metrics_seed${seed}.json" ] || [ -f "$out/seed_${seed}/metrics.json" ]; then
    echo "[skip] ${ds} seed ${seed} (already done)"
    return
  fi
  echo "[start] ${ds} seed ${seed} gpu=${gpu}  $(date +%T)"
  cd "$BENCH_ROOT/EvoPlay"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 \
    ./env/bin/python -u run_${ds}.py \
      --seed "$seed" --batch_size 96 --n_rounds 5 \
      --use_gpu --verbose 1 \
      --output_path "results/${ds}_EvoPlay/${ds}/topn_pvn" \
      > "$log" 2>&1
  local rc=$?
  cd "$BENCH_ROOT"
  if [ $rc -eq 0 ]; then
    echo "[done ] ${ds} seed ${seed} gpu=${gpu}  $(date +%T)"
  else
    echo "[FAIL] ${ds} seed ${seed} gpu=${gpu} rc=$rc  see $log"
  fi
}

export -f run_one
export BENCH_ROOT LOG_DIR

# Parallel execution: xargs -n 3 reads 3 fields per line into $1 $2 $3 of bash -c
< "$JOB_FILE" xargs -n 3 -P "$SWEEP_CONCURRENCY" bash -c 'run_one "$1" "$2" "$3"' _

echo "[evoplay_sweep] ALL DONE  $(date)"
