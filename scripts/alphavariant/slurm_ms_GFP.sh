#!/bin/bash
#SBATCH --job-name=av_gfp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=64G
#SBATCH --constraint="a100"
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --output=log-%x-%j.out
#SBATCH --error=log-%x-%j.out

# AlphaVariant (Plan C) on ms_GFP — multi-site ORACLE benchmark, 30 seeds.
# Single-node job: distributes the 30 seeds across the allocated GPUs, packing
# CONCURRENCY seeds per GPU (each seed uses only ~2GB/~20% of an A100-80G).
# Plan C = GPT-REINFORCE (MSA prior) + ensemble surrogate + CLADE-2 selection +
#          MutCompute reward + SHAP pruning. Pure-oracle, 96x5=480, sigma=60 (NaN-guarded).
# Re-submittable: any seed whose JSON already exists is skipped.

module purge
module load gcc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/xux/miniconda3/envs/prot-gen-env

# ============================ CONFIG (edit) ============================
BENCH="${BENCH:-/home/xux/Desktop/AlphaVariant/Benchmark}"   # benchmark root on the cluster
DATASET="ms_GFP"
CONCURRENCY="${CONCURRENCY:-4}"          # seeds run concurrently PER GPU
N_ROUNDS="${N_ROUNDS:-5}"
N_STEPS="${N_STEPS:-500}"                # internal RL steps/round (Plan C default; NOT oracle calls)
SIGMA="${SIGMA:-60}"
SEEDS=(621 100 383 492 987 167 926 446 390 477 137 531 919 3 194 \
       77 303 331 76 433 652 772 527 563 340 998 171 590 548 511)
# =======================================================================

export OMP_NUM_THREADS=2                 # limit BLAS threads across packed seeds
cd "$BENCH/alphavariant"                 # popgen/popscorer import root
RES="$BENCH/results_oracle"
LOGD="$RES/_logs/slurm"; mkdir -p "$LOGD"

PRIOR=""
[ -f "priors/$DATASET/prior_model.pt" ] && PRIOR="--prior_model_path priors/$DATASET/prior_model.pt"

# GPUs allocated to this job (auto-detected so it adapts to --gpus-per-node)
mapfile -t GPUS < <(nvidia-smi -L | sed -n 's/^GPU \([0-9]*\):.*/\1/p')
NGPU=${#GPUS[@]}; [ "$NGPU" -lt 1 ] && NGPU=1 && GPUS=(0)
echo "[$DATASET] $NGPU GPU(s): ${GPUS[*]}  | CONCURRENCY=$CONCURRENCY/GPU | $(date)"

run_one() {  # $1=gpu_id  $2=seed
    [ -f "$RES/$DATASET/AlphaVariant/seed${2}.json" ] && { echo "  skip seed $2 (done)"; return 0; }
    CUDA_VISIBLE_DEVICES="$1" python run_generic.py --dataset "$DATASET" --seed "$2" \
        --oracle --level uniform $PRIOR \
        --use_mutcompute --shap_prune_alphabet \
        --n_rounds "$N_ROUNDS" --n_steps_per_round "$N_STEPS" --sigma "$SIGMA" \
        --device cuda:0 \
        --output_path "$RES" --data_dir "$BENCH/data" \
        > "$LOGD/av_${DATASET}_seed${2}.log" 2>&1
}

# Each GPU processes its contiguous chunk of seeds in waves of CONCURRENCY.
worker() {  # $1=gpu_id  $2..=seeds
    local gpu="$1"; shift; local list=("$@") i p pids
    for (( i=0; i<${#list[@]}; i+=CONCURRENCY )); do
        pids=()
        for s in "${list[@]:i:CONCURRENCY}"; do run_one "$gpu" "$s" & pids+=($!); done
        for p in "${pids[@]}"; do wait "$p" || true; done
    done
}

chunk=$(( (${#SEEDS[@]} + NGPU - 1) / NGPU ))
for (( g=0; g<NGPU; g++ )); do
    worker "${GPUS[$g]}" "${SEEDS[@]:$((g*chunk)):$chunk}" &
done
wait
echo "[$DATASET] AV_ORACLE_DONE $(date)  ($(ls "$RES/$DATASET/AlphaVariant/"seed*.json 2>/dev/null | wc -l)/30 seeds)"
