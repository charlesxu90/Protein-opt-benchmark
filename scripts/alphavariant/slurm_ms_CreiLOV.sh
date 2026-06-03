#!/bin/bash
#SBATCH --job-name=av_creilov
#SBATCH --array=0-5%6                      # 30 seeds / SEEDS_PER_TASK(5) = 6 packed tasks
#SBATCH --gres=gpu:1                        # 1x A100-80G per task (packs 5 seeds, see below)
#SBATCH --cpus-per-task=20                  # ~4 cpus x 5 concurrent seeds (sklearn surrogate + MutCompute)
#SBATCH --mem=64G
#SBATCH --time=02:30:00                     # 5 packed CreiLOV seeds (119aa, ~33 min/seed) on one A100
#SBATCH --output=results_oracle/_logs/slurm/av_creilov_%A_%a.out
# ---- EDIT for your cluster ----
###SBATCH --partition=batch
###SBATCH --account=YOUR_ACCOUNT
###SBATCH --constraint=a100
#
# AlphaVariant (Plan C) on ms_CreiLOV, multi-site ORACLE benchmark, packed for 80GB A100.
# Plan C = GPT-REINFORCE (MSA prior) + ensemble surrogate + CLADE-2 selection +
#          MutCompute reward + SHAP pruning. Pure-oracle, 96x5=480, sigma=60 (NaN-guarded).
#
# PACKING: one AlphaVariant seed uses only ~1.5GB / ~20% of an A100-80G (sampling-latency
# bound). We run SEEDS_PER_TASK seeds CONCURRENTLY on the single allocated GPU -> ~5x
# throughput. 30 seeds => 6 array tasks of 5 concurrent seeds each.
#
# Prereqs staged on the cluster:
#   oracles/ms_CreiLOV/oracle.pt
#   alphavariant/priors/ms_CreiLOV/prior_model.{pt,json}
#   data/ms_CreiLOV/{data.csv,wt.fasta,mutcompute.csv}
#   conda env `alphavariant-env` (torch+transformers+sklearn+loguru+lightning+popgen)
#
# Submit:  sbatch scripts/alphavariant/slurm_ms_CreiLOV.sh
# Resume:  just re-submit — each seed is skipped if its JSON already exists.
set -euo pipefail

# ============================ CONFIG (edit) ============================
BENCH="${BENCH:-/home/xux/Desktop/AlphaVariant/Benchmark}"
CONDA_SH="${CONDA_SH:-/home/xux/miniforge3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-alphavariant-env}"
DATASET="ms_CreiLOV"
SEEDS_PER_TASK="${SEEDS_PER_TASK:-5}"       # concurrent seeds per GPU
N_ROUNDS="${N_ROUNDS:-5}"
N_STEPS="${N_STEPS:-500}"                    # internal RL steps/round (Plan C default; NOT oracle calls)
SIGMA="${SIGMA:-60}"
SEEDS=(621 100 383 492 987 167 926 446 390 477 137 531 919 3 194 \
       77 303 331 76 433 652 772 527 563 340 998 171 590 548 511)
# =======================================================================

TASK="${SLURM_ARRAY_TASK_ID:?submit via sbatch (or set SLURM_ARRAY_TASK_ID)}"
START=$(( TASK * SEEDS_PER_TASK ))
SLICE=( "${SEEDS[@]:$START:$SEEDS_PER_TASK}" )
echo "[$DATASET task $TASK] seeds: ${SLICE[*]}  host=$(hostname) $(date)"

# ---- environment ----
# shellcheck disable=SC1090
source "$CONDA_SH"; conda activate "$ENV_NAME"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"   # env libstdc++ (CXXABI) first
export OMP_NUM_THREADS=4                                          # avoid CPU oversubscription across packed seeds
cd "$BENCH/alphavariant"                                          # popgen/popscorer import root
mkdir -p "$BENCH/results_oracle/_logs/slurm"

PRIOR=""
[ -f "priors/$DATASET/prior_model.pt" ] && PRIOR="--prior_model_path priors/$DATASET/prior_model.pt"

# ---- launch SEEDS_PER_TASK seeds concurrently on the one allocated GPU ----
pids=()
for SEED in "${SLICE[@]}"; do
    if [ -f "$BENCH/results_oracle/$DATASET/AlphaVariant/seed${SEED}.json" ]; then
        echo "[$DATASET task $TASK] seed $SEED already done -> skip"; continue
    fi
    ( python run_generic.py --dataset "$DATASET" --seed "$SEED" \
        --oracle --level uniform $PRIOR \
        --use_mutcompute --shap_prune_alphabet \
        --n_rounds "$N_ROUNDS" --n_steps_per_round "$N_STEPS" --sigma "$SIGMA" \
        --device cuda:0 \
        --output_path "$BENCH/results_oracle" --data_dir "$BENCH/data" \
        > "$BENCH/results_oracle/_logs/slurm/av_${DATASET}_seed${SEED}.log" 2>&1 ) &
    pids+=($!)
done
echo "[$DATASET task $TASK] launched ${#pids[@]} concurrent seeds; waiting..."
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[$DATASET task $TASK] done (rc=$rc) $(date)"
exit $rc
