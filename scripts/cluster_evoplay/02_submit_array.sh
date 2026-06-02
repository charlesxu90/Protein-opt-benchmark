#!/usr/bin/env bash
# 02_submit_array.sh — submit EvoPlay 4-task × 30-seed array job on iBex.
# Run on iBex login node, NOT local workstation.
#
# Usage:
#   bash scripts/cluster_evoplay/02_submit_array.sh
#   ACCOUNT=k01234 bash scripts/cluster_evoplay/02_submit_array.sh        # if account flag needed
#   PARTITION=gpu bash scripts/cluster_evoplay/02_submit_array.sh         # override partition
#   N_SEEDS=10 bash scripts/cluster_evoplay/02_submit_array.sh            # fewer seeds (40 tasks)
#   CONCURRENCY=120 bash scripts/cluster_evoplay/02_submit_array.sh       # all parallel (no cap)
#   CPU_ONLY=1 bash scripts/cluster_evoplay/02_submit_array.sh            # skip GPU request
#
# Submits 4*N_SEEDS array tasks. With iBex's many GPU nodes, all jobs can
# run truly in parallel — limited only by partition capacity, not %CAP.

set -euo pipefail

cd "${HOME}/Benchmark"
mkdir -p sweep_logs/4site_extra/evoplay_cluster

N_SEEDS=${N_SEEDS:-30}
TOTAL=$(( 4 * N_SEEDS ))
CONCURRENCY=${CONCURRENCY:-${TOTAL}}  # default = no cap, all parallel
ACCOUNT_FLAG=""
[ -n "${ACCOUNT:-}" ] && ACCOUNT_FLAG="--account=${ACCOUNT}"
PARTITION=${PARTITION:-batch}
TIME_LIMIT=${TIME_LIMIT:-06:00:00}
MEM=${MEM:-24G}
CPUS=${CPUS:-4}
CPU_ONLY=${CPU_ONLY:-0}

GPU_FLAG="--gres=gpu:1"
GPU_MODEL=${GPU_MODEL:-}   # e.g., GPU_MODEL=v100  → --gres=gpu:v100:1
if [ -n "$GPU_MODEL" ]; then
  GPU_FLAG="--gres=gpu:${GPU_MODEL}:1"
  echo "[submit] GPU model constraint: ${GPU_MODEL}"
fi
if [ "$CPU_ONLY" = "1" ]; then
  GPU_FLAG=""
  echo "[submit] CPU-only mode — dropping --gres flag (EvoPlay barely uses GPU)"
fi

echo "[submit] tasks=${TOTAL} (4 datasets × ${N_SEEDS} seeds)  concurrency=${CONCURRENCY}"
echo "[submit] account='${ACCOUNT:-<none>}' partition=${PARTITION} time=${TIME_LIMIT} mem=${MEM} cpus=${CPUS}"
echo "[submit] gpu_flag='${GPU_FLAG}'"

# Generate the sbatch script with the runtime parameters baked in
TMP_SBATCH=$(mktemp /tmp/evoplay_array.XXXXXX.sbatch)
cat > "$TMP_SBATCH" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=evoplay_4site
#SBATCH --output=sweep_logs/4site_extra/evoplay_cluster/job_%A_%a.out
#SBATCH --error=sweep_logs/4site_extra/evoplay_cluster/job_%A_%a.err
#SBATCH --array=0-$(( TOTAL - 1 ))%${CONCURRENCY}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}
${GPU_FLAG:+#SBATCH }${GPU_FLAG}

set -euo pipefail
BENCHMARK_ROOT="\${HOME}/Benchmark"
cd "\$BENCHMARK_ROOT"
mkdir -p sweep_logs/4site_extra/evoplay_cluster

# iBex module load — adjust if needed
module load anaconda3 2>/dev/null || module load conda 2>/dev/null || true

DATASETS=(4site_GB1 4site_PhoQ 4site_TEV 4site_TRPB)
SEEDS=(\$(head -${N_SEEDS} rand_seeds.txt))

TID=\${SLURM_ARRAY_TASK_ID}
ds_idx=\$(( TID / ${N_SEEDS} ))
seed_idx=\$(( TID % ${N_SEEDS} ))
ds="\${DATASETS[\$ds_idx]}"
seed="\${SEEDS[\$seed_idx]}"

OUT_DIR="EvoPlay/results/\${ds}_EvoPlay/\${ds}/topn_pvn"
LOG_FILE="sweep_logs/4site_extra/evoplay_cluster/\${ds}_seed\${seed}.log"

# Idempotent skip
if [ -f "\$OUT_DIR/metrics_seed\${seed}.json" ] || [ -f "\$OUT_DIR/seed_\${seed}/metrics.json" ]; then
  echo "[skip] \${ds} seed=\${seed} (already done)"
  exit 0
fi

echo "[cluster] task=\${TID} ds=\${ds} seed=\${seed} host=\$(hostname)"
echo "[cluster] start \$(date)"

cd "\$BENCHMARK_ROOT/EvoPlay"
PYTHONUNBUFFERED=1 ./env/bin/python -u run_\${ds}.py \\
  --seed "\$seed" --batch_size 96 --n_rounds 5 \\
  ${CPU_ONLY:+# CPU-only}${CPU_ONLY:--} \\
  --verbose 1 \\
  --output_path "results/\${ds}_EvoPlay/\${ds}/topn_pvn" \\
  > "\$BENCHMARK_ROOT/\$LOG_FILE" 2>&1
rc=\$?
cd "\$BENCHMARK_ROOT"
echo "[cluster] end   \$(date)  rc=\$rc"
exit \$rc
SBATCH

# Fix the --use_gpu insertion (the heredoc trick above is messy)
if [ "$CPU_ONLY" = "1" ]; then
  sed -i 's|--verbose 1|--verbose 1|' "$TMP_SBATCH"
else
  sed -i 's|--verbose 1|--use_gpu --verbose 1|' "$TMP_SBATCH"
fi

echo ""
echo "=== Generated sbatch script: $TMP_SBATCH ==="
echo ""

JOB_ID=$(sbatch $ACCOUNT_FLAG --parsable "$TMP_SBATCH")
echo "[submit] submitted job_id=$JOB_ID"
echo ""
echo "Monitor with:"
echo "  squeue -j $JOB_ID"
echo "  bash scripts/cluster_evoplay/03_monitor.sh $JOB_ID"
echo ""
echo "When all tasks complete, collect results back to local:"
echo "  bash scripts/cluster_evoplay/04_collect.sh"
