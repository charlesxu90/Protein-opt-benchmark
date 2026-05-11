#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH --output=__LOG_DIR__/%x_%A_%a.out
#SBATCH --error=__LOG_DIR__/%x_%A_%a.err
#SBATCH --array=__ARRAY_SPEC__
#SBATCH --partition=__PARTITION__
#SBATCH --time=__TIME__
#SBATCH --cpus-per-task=__CPUS__
#SBATCH --mem=__MEM_GB__G
#SBATCH --gres=gpu:__GPUS__
# KAUST iBex job-array template. Placeholders are filled by scripts/hpc/launch.py.
# Each array task runs ONE seed of ONE method on ONE dataset.

set -euo pipefail

BENCHMARK_ROOT="__BENCHMARK_ROOT__"
METHOD="__METHOD__"
DATASET="__DATASET__"
CONDA_ENV="__CONDA_ENV__"
SEED_FILE="__SEED_FILE__"

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
# Prepend method dir so `from src.foo` etc. resolve through the symlinked script.
export PYTHONPATH="$BENCHMARK_ROOT/$METHOD${PYTHONPATH:+:$PYTHONPATH}"
python "$RUN_SCRIPT" --seed "$SEED" __EXTRA_ARGS__
