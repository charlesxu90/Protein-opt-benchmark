#!/bin/bash
#SBATCH --job-name=av_ALDE_GB1
#SBATCH --output=/home/xux/Desktop/AlphaVariant/Benchmark/scripts/hpc/_logs/%x_%A_%a.out
#SBATCH --error=/home/xux/Desktop/AlphaVariant/Benchmark/scripts/hpc/_logs/%x_%A_%a.err
#SBATCH --array=0-2%32
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# KAUST iBex job-array template. Placeholders are filled by scripts/hpc/launch.py.
# Each array task runs ONE seed of ONE method on ONE dataset.

set -euo pipefail

BENCHMARK_ROOT="/home/xux/Desktop/AlphaVariant/Benchmark"
METHOD="ALDE"
DATASET="GB1"
CONDA_ENV="ALDE/env"
SEED_FILE="/home/xux/Desktop/AlphaVariant/Benchmark/scripts/hpc/_jobs/av_ALDE_GB1_seeds.txt"

cd "$BENCHMARK_ROOT"
source scripts/hpc/env_setup.sh ibex "$CONDA_ENV"

# SLURM_ARRAY_TASK_ID is 0-indexed; seed file is 1-line-per-seed (1-indexed)
SEED=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SEED_FILE")
if [[ -z "$SEED" ]]; then
  echo "No seed for task $SLURM_ARRAY_TASK_ID in $SEED_FILE" >&2
  exit 1
fi

echo "[$(date)] method=$METHOD dataset=$DATASET seed=$SEED task=$SLURM_ARRAY_TASK_ID"

# Each method's run_<dataset>.py is symlinked into <METHOD>/ by add_script_link.sh
RUN_SCRIPT="$BENCHMARK_ROOT/$METHOD/run_${DATASET}.py"
if [[ ! -e "$RUN_SCRIPT" ]]; then
  echo "Run script not found: $RUN_SCRIPT" >&2
  exit 1
fi

cd "$BENCHMARK_ROOT/$METHOD"
python "$RUN_SCRIPT" --seed "$SEED" 
