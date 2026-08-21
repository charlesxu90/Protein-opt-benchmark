#!/bin/bash
# Re-run GFP +finetune at 1 process/GPU (MAXCON=2) to avoid the CUDA OOM seen at 2/GPU.
set -u
cd "$(dirname "$0")/../../alphavariant"
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=6
PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python
BENCH=/home/xux/Desktop/AlphaVariant/Benchmark
OUT="$BENCH/results_ablation/gfp_ms_finetune"
COMMON="--oracle --level uniform --features ev_onehot --use_mutcompute --shap_prune_alphabet \
        --max_n_mut 2 --sigma 60 --n_rounds 5 --n_steps_per_round 500 --device cuda:0 --data_dir ../data"
SEEDS=(621 100 383 492 987)
run_job(){ local seed="$1" gpu="$2"
  local res="$OUT/ms_GFP/AlphaVariant/seed${seed}.json"
  if [ -f "$res" ]; then echo "[skip] GFP seed$seed"; return; fi
  mkdir -p "$OUT/_logs"
  CUDA_VISIBLE_DEVICES="$gpu" nice -n 10 $PY ../scripts/alphavariant/run_generic.py --dataset ms_GFP --seed "$seed" \
    --prior_model_path priors/ms_GFP/prior_model.pt --finetune_prior \
    --output_path "$OUT" $COMMON > "$OUT/_logs/seed${seed}.log" 2>&1; }
echo "[gfp-ft] start $(date) (MAXCON=2)"
running=0; gpu=0
for seed in "${SEEDS[@]}"; do
  run_job "$seed" "$gpu" &
  gpu=$(( 1 - gpu )); running=$(( running + 1 ))
  if [ "$running" -ge 2 ]; then wait -n; running=$(( running - 1 )); fi
done
wait
echo "GFP_FT_RERUN_DONE $(date)"
