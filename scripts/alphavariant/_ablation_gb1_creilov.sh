#!/bin/bash
# Leave-one-out ablation of AlphaVariant on GB1 (four-site) and CreiLOV (multi-site).
# Each config removes ONE component from the shipped Plan C configuration.
# 30 canonical seeds (rand_seeds.txt first 30), 2-GPU pool, resumable (skip-if-exists).
set -u
cd "$(dirname "$0")/../../alphavariant"
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4
PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python
BENCH=/home/xux/Desktop/AlphaVariant/Benchmark
OUT="$BENCH/results_ablation"
PRIOR="priors/ms_CreiLOV/prior_model.pt"
MAXCON=4
COMMON="--n_rounds 5 --n_steps_per_round 500 --sigma 60 --data_dir ../data"

SEEDS=(621 100 383 492 987 167 926 446 390 477 137 531 919 3 194 \
       77 303 331 76 433 652 772 527 563 340 998 171 590 548 511)

# name|dataset|flags  (leave-one-out from Plan C)
JOBS=(
  "gb1_full|4site_GB1|--use_mutcompute --plm_reward_lambda 0.5 --shap_prune_alphabet"
  "gb1_no_mcreward|4site_GB1|--shap_prune_alphabet"
  "gb1_no_shap|4site_GB1|--use_mutcompute --plm_reward_lambda 0.5"
  "gb1_bare|4site_GB1|"
  "cre_full|ms_CreiLOV|--oracle --level uniform --prior_model_path $PRIOR --features ev_onehot --use_mutcompute --shap_prune_alphabet --max_n_mut 2"
  "cre_no_ev|ms_CreiLOV|--oracle --level uniform --prior_model_path $PRIOR --features onehot --use_mutcompute --shap_prune_alphabet --max_n_mut 2"
  "cre_no_shap|ms_CreiLOV|--oracle --level uniform --prior_model_path $PRIOR --features ev_onehot --use_mutcompute --max_n_mut 2"
  "cre_no_cap|ms_CreiLOV|--oracle --level uniform --prior_model_path $PRIOR --features ev_onehot --use_mutcompute --shap_prune_alphabet"
  "cre_no_prior|ms_CreiLOV|--oracle --level uniform --features ev_onehot --use_mutcompute --shap_prune_alphabet --max_n_mut 2"
)

result_path() {  # name dataset seed -> expected metrics file
    local name="$1" ds="$2" seed="$3"
    if [ "$ds" = "ms_CreiLOV" ]; then echo "$OUT/$name/ms_CreiLOV/AlphaVariant/seed${seed}.json"
    else echo "$OUT/$name/seed_${seed}/metrics.json"; fi
}

run_job() {
    local name="$1" ds="$2" flags="$3" seed="$4" gpu="$5"
    local res; res=$(result_path "$name" "$ds" "$seed")
    if [ -f "$res" ]; then echo "[skip] $name seed$seed"; return; fi
    mkdir -p "$OUT/$name/_logs"
    CUDA_VISIBLE_DEVICES="$gpu" nice -n 10 $PY run_generic.py \
        --dataset "$ds" --seed "$seed" --output_path "$OUT/$name" \
        $COMMON $flags > "$OUT/$name/_logs/seed${seed}.log" 2>&1
}

echo "[ablation] start $(date)  9 configs x 30 seeds (GB1 + CreiLOV)"
running=0; gpu=0
for spec in "${JOBS[@]}"; do
    IFS='|' read -r name ds flags <<< "$spec"
    for seed in "${SEEDS[@]}"; do
        run_job "$name" "$ds" "$flags" "$seed" "$gpu" &
        gpu=$(( 1 - gpu ))
        running=$(( running + 1 ))
        if [ "$running" -ge "$MAXCON" ]; then wait -n; running=$(( running - 1 )); fi
    done
done
wait
echo "[ablation] all runs done $(date)"

$PY "$BENCH/scripts/summarize_ablation.py"
echo "ABLATION_DONE $(date)"
