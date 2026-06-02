#!/usr/bin/env bash
#SBATCH --job-name=evoplay_4site
#SBATCH --output=sweep_logs/4site_extra/evoplay_slurm/job_%A_%a.out
#SBATCH --error=sweep_logs/4site_extra/evoplay_slurm/job_%A_%a.err
#SBATCH --array=0-119%30
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
# iBex-tuned EvoPlay 4site sweep: 4 datasets × 30 seeds = 120 array tasks
#
# iBex-specific notes:
#   - Default partition is usually `batch`; for GPU jobs you may need `--partition=batch`
#     AND `--gres=gpu:1` (any), or `--gres=gpu:a100:1` to force A100.
#   - On iBex CPU nodes (no GPU), EvoPlay still works (GPU usage is minimal,
#     model falls back to CPU). To run CPU-only: remove `--gres=gpu:1` line and
#     drop `--use_gpu` from the python command below. Saves GPU resources for
#     other jobs in your group.
#   - Module load may be needed: see env_setup.sh
#
# Submission:
#   sbatch scripts/sweep_evoplay_ibex.sbatch                              # default
#   sbatch --account=k01234 scripts/sweep_evoplay_ibex.sbatch             # if account flag needed
#   sbatch --array=0-119%50 scripts/sweep_evoplay_ibex.sbatch             # bump to 50 concurrent
#   sbatch --array=0-29 --export=DS=4site_PhoQ scripts/sweep_evoplay_ibex.sbatch
#                                                                         # single dataset
#
# Resume after partial completion: idempotent — array tasks whose metrics
# already exist are skipped (saves wall time on requeued jobs).

set -euo pipefail

BENCHMARK_ROOT="${HOME}/Benchmark"
cd "$BENCHMARK_ROOT"
mkdir -p sweep_logs/4site_extra/evoplay_slurm

# Load conda/python module — adjust if your iBex setup differs
module load anaconda3 2>/dev/null || true
# Or for newer iBex installs: module load python/3.8 anaconda3

DATASETS=(4site_GB1 4site_PhoQ 4site_TEV 4site_TRPB)
SEEDS=($(head -30 rand_seeds.txt))

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

if [ -f "$OUT_DIR/metrics_seed${seed}.json" ] || [ -f "$OUT_DIR/seed_${seed}/metrics.json" ]; then
  echo "[skip] ${ds} seed=${seed} (metrics already present)"
  exit 0
fi

echo "[ibex] task=${TID} ds=${ds} seed=${seed} host=$(hostname)"
echo "[ibex] start $(date)"

cd "$BENCHMARK_ROOT/EvoPlay"
PYTHONUNBUFFERED=1 ./env/bin/python -u run_${ds}.py \
  --seed "$seed" --batch_size 96 --n_rounds 5 \
  --use_gpu --verbose 1 \
  --output_path "results/${ds}_EvoPlay/${ds}/topn_pvn" \
  > "$BENCHMARK_ROOT/$LOG_FILE" 2>&1
rc=$?
cd "$BENCHMARK_ROOT"
echo "[ibex] end   $(date)  rc=$rc"
exit $rc
