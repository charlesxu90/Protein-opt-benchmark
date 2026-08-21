#!/bin/bash
# AlphaVariant (Plan C, faithful) on the four multi-site oracle benchmarks.
# Plan C = GPT-REINFORCE (MSA prior) + ensemble surrogate + CLADE-2 batch select
#          + MutCompute zero-shot reward + SHAP alphabet pruning. Pure-oracle, 96x5=480.
#
# Parameterized for 2-GPU partitioning:
#   $1 = physical GPU id (CUDA_VISIBLE_DEVICES)
#   SEEDS env = seed subset to run (default = all 30)
# Launch two instances (15 seeds each) for ~balanced wall time.
#
# Env: alphavariant-env + LD_LIBRARY_PATH fix; run from alphavariant/ (popgen import).
set -u
cd "$(dirname "$0")/../../alphavariant"
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES="${1:?usage: _sweep_av_oracle.sh <gpu_id>}"
PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python
SEEDS="${SEEDS:?set SEEDS env}"
OUT=../results_oracle
TAG="gpu$1"
mkdir -p ../results_oracle/_logs

for d in ms_AAV ms_PAB1 ms_CreiLOV ms_GFP; do
  PRIOR=""
  [ -f "priors/$d/prior_model.pt" ] && PRIOR="--prior_model_path priors/$d/prior_model.pt"
  echo "[$TAG] $d  $(date +%H:%M:%S)"
  $PY ../scripts/alphavariant/run_generic.py --dataset "$d" --seeds $SEEDS \
      --oracle --level uniform $PRIOR \
      --use_mutcompute --shap_prune_alphabet \
      --n_rounds 5 --n_steps_per_round 500 --device cuda:0 \
      --output_path "$OUT" --data_dir ../data \
      >> "../results_oracle/_logs/av_${d}_${TAG}.log" 2>&1
  echo "[$TAG] DONE $d ($(ls ../results_oracle/$d/AlphaVariant/seed*.json 2>/dev/null|wc -l) total seeds)"
done
echo "[$TAG] AV_ORACLE_SWEEP_DONE $(date +%H:%M:%S)"
