#!/bin/bash
# Leave-one-out ablation of AlphaVariant on the remaining 6 datasets.
# Seed budget (chosen for cost): PhoQ/TEV/TrpB + AAV = 30, PAB1 = 10.
# Cheap datasets first. Resumable (skip-if-exists), 2-GPU pool.
set -u
cd "$(dirname "$0")/../../alphavariant"
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4
PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python
BENCH=/home/xux/Desktop/AlphaVariant/Benchmark
OUT="$BENCH/results_ablation"
MAXCON=4
COMMON="--n_rounds 5 --n_steps_per_round 500 --sigma 60 --data_dir ../data"

SEEDS=(621 100 383 492 987 167 926 446 390 477 137 531 919 3 194 \
       77 303 331 76 433 652 772 527 563 340 998 171 590 548 511)

declare -A NSEEDS=( [4site_PhoQ]=30 [4site_TEV]=30 [4site_TRPB]=30 [ms_AAV]=30 [ms_PAB1]=10 )
declare -A PFX=( [4site_PhoQ]=phoq [4site_TEV]=tev [4site_TRPB]=trpb [ms_AAV]=aav [ms_PAB1]=pab1 )

JOBS=()
for ds in 4site_PhoQ 4site_TEV 4site_TRPB; do
  p=${PFX[$ds]}
  JOBS+=("${p}_full|$ds|--use_mutcompute --plm_reward_lambda 0.5 --shap_prune_alphabet")
  JOBS+=("${p}_no_mcreward|$ds|--shap_prune_alphabet")
  JOBS+=("${p}_no_shap|$ds|--use_mutcompute --plm_reward_lambda 0.5")
  JOBS+=("${p}_bare|$ds|")
done
for ds in ms_AAV ms_PAB1; do
  p=${PFX[$ds]}; P="priors/$ds/prior_model.pt"
  B="--oracle --level uniform"
  JOBS+=("${p}_full|$ds|$B --prior_model_path $P --features ev_onehot --use_mutcompute --shap_prune_alphabet --max_n_mut 2")
  JOBS+=("${p}_no_ev|$ds|$B --prior_model_path $P --features onehot --use_mutcompute --shap_prune_alphabet --max_n_mut 2")
  JOBS+=("${p}_no_shap|$ds|$B --prior_model_path $P --features ev_onehot --use_mutcompute --max_n_mut 2")
  JOBS+=("${p}_no_cap|$ds|$B --prior_model_path $P --features ev_onehot --use_mutcompute --shap_prune_alphabet")
  JOBS+=("${p}_no_prior|$ds|$B --features ev_onehot --use_mutcompute --shap_prune_alphabet --max_n_mut 2")
done

result_path() {
    local name="$1" ds="$2" seed="$3"
    if [[ "$ds" == ms_* ]]; then echo "$OUT/$name/$ds/AlphaVariant/seed${seed}.json"
    else echo "$OUT/$name/seed_${seed}/metrics.json"; fi
}

run_job() {
    local name="$1" ds="$2" flags="$3" seed="$4" gpu="$5"
    local res; res=$(result_path "$name" "$ds" "$seed")
    if [ -f "$res" ]; then echo "[skip] $name seed$seed"; return; fi
    mkdir -p "$OUT/$name/_logs"
    CUDA_VISIBLE_DEVICES="$gpu" nice -n 10 $PY ../scripts/alphavariant/run_generic.py \
        --dataset "$ds" --seed "$seed" --output_path "$OUT/$name" \
        $COMMON $flags > "$OUT/$name/_logs/seed${seed}.log" 2>&1
}

echo "[ablation-rest] start $(date)"
running=0; gpu=0
for spec in "${JOBS[@]}"; do
    IFS='|' read -r name ds flags <<< "$spec"
    n=${NSEEDS[$ds]}
    for seed in "${SEEDS[@]:0:$n}"; do
        run_job "$name" "$ds" "$flags" "$seed" "$gpu" &
        gpu=$(( 1 - gpu )); running=$(( running + 1 ))
        if [ "$running" -ge "$MAXCON" ]; then wait -n; running=$(( running - 1 )); fi
    done
done
wait
echo "[ablation-rest] all runs done $(date)"
$PY "$BENCH/scripts/summarize_ablation.py"
echo "ABLATION_REST_DONE $(date)"
