#!/bin/bash
#SBATCH --job-name=av_multisite
#SBATCH --array=0-119%32                 # 4 datasets x 30 seeds = 120 tasks; %32 = max concurrent
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00                   # GFP (237aa) is the long pole (~2h @500 steps); others faster
#SBATCH --output=results_oracle/_logs/slurm/av_%A_%a.out
# ---- EDIT for your cluster (iBex/Shaheen): ----
###SBATCH --partition=batch
###SBATCH --account=YOUR_ACCOUNT
###SBATCH --constraint=a100            # or v100/gpu type
#
# AlphaVariant (Plan C) on the four multi-site ORACLE benchmarks, as a SLURM job
# array: ONE array task = one (dataset, seed). Maximally parallel — wall time of the
# whole sweep ≈ the longest single task (GFP ~2h) given enough concurrent GPUs.
#
# Plan C = GPT-REINFORCE (MSA prior) + ensemble surrogate + CLADE-2 batch selection
#          + MutCompute zero-shot reward + SHAP alphabet pruning. Pure-oracle, 96x5=480.
#
# Prereqs (run once before submitting):
#   - oracles/<dataset>/oracle.pt              (scripts/train_oracle.py)
#   - data/<dataset>/prior_aligned.csv         (scripts/alphavariant/align_homologs.py)
#   - alphavariant/priors/<dataset>/prior_model.pt   (scripts/alphavariant/train_ms_prior.py)
#   - data/<dataset>/mutcompute.csv            (provided)
#
# Submit:   sbatch scripts/alphavariant/run_multisite_benchmark.sh
# Local test (no SLURM): SLURM_ARRAY_TASK_ID=0 bash scripts/alphavariant/run_multisite_benchmark.sh
# Subset:   sbatch --array=0-29 ... (only ms_AAV);  --array=90-119 ... (only ms_GFP)
set -euo pipefail

# ============================ CONFIG (edit) ============================
BENCH="${BENCH:-/home/xux/Desktop/AlphaVariant/Benchmark}"
CONDA_SH="${CONDA_SH:-/home/xux/miniforge3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-alphavariant-env}"
N_ROUNDS="${N_ROUNDS:-5}"
N_STEPS="${N_STEPS:-500}"          # internal RL steps/round (NOT oracle calls). Lower => faster.
SIGMA="${SIGMA:-60}"               # REINFORCE reward scale. Lower (e.g. 20) if a run diverges to NaN.
DATASETS=(ms_AAV ms_PAB1 ms_CreiLOV ms_GFP)
# 30 canonical seeds (rand_seeds.txt first 30). Keep in sync with the other 9 methods.
SEEDS=(621 100 383 492 987 167 926 446 390 477 137 531 919 3 194 \
       77 303 331 76 433 652 772 527 563 340 998 171 590 548 511)
# =======================================================================

N_SEEDS=${#SEEDS[@]}
TASK="${SLURM_ARRAY_TASK_ID:?set SLURM_ARRAY_TASK_ID (or submit via sbatch)}"
DATASET="${DATASETS[$(( TASK / N_SEEDS ))]}"
SEED="${SEEDS[$(( TASK % N_SEEDS ))]}"

echo "[task $TASK] dataset=$DATASET seed=$SEED steps=$N_STEPS sigma=$SIGMA host=$(hostname) $(date)"

# ---- environment ----
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"
# env libstdc++ shadows the system one (matplotlib CXXABI); put env lib first.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$BENCH/alphavariant"   # run from method dir so popgen/popscorer import

PRIOR=""
if [ -f "priors/$DATASET/prior_model.pt" ]; then
    PRIOR="--prior_model_path priors/$DATASET/prior_model.pt"
else
    echo "[task $TASK] WARN: no prior for $DATASET -> random-init GPT"
fi

mkdir -p "$BENCH/results_oracle/_logs/slurm"

# Idempotent: skip if this seed already produced a result (safe re-submission).
if [ -f "$BENCH/results_oracle/$DATASET/AlphaVariant/seed${SEED}.json" ]; then
    echo "[task $TASK] already done -> skip"
    exit 0
fi

srun python run_generic.py \
    --dataset "$DATASET" --seed "$SEED" \
    --oracle --level uniform $PRIOR \
    --use_mutcompute --shap_prune_alphabet \
    --n_rounds "$N_ROUNDS" --n_steps_per_round "$N_STEPS" --sigma "$SIGMA" \
    --device cuda:0 \
    --output_path "$BENCH/results_oracle" --data_dir "$BENCH/data"

echo "[task $TASK] done $(date)"
