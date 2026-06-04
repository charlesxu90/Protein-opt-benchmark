#!/bin/bash
# 30-seed AlphaVariant benchmark on GFP (ms_GFP) LOCALLY, with the locked config:
#   ev+onehot surrogate (retrained on accumulated collected variants each round),
#   SHAP-prune w/ hotspot-config sync (now valid + makes GFP rounds actually run),
#   max_n_mut=2, 5 rounds x 96 = 480 oracle calls, 500 RL steps/round, sigma 60.
# GFP is the heavy 237aa dataset: GPU-bound (~5GB/run, saturates ~2-3/GPU). We pack
# 4/GPU (MAXCON=8) -> ~20GB/40GB per card (safe), ~48/112 CPU threads, jobs 'nice'd so
# the box stays responsive (no freeze). Idempotent: skips seeds already done.
set -u
cd "$(dirname "$0")/../../alphavariant"
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=6
PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python
BENCH=/home/xux/Desktop/AlphaVariant/Benchmark
RES="$BENCH/results_oracle"
LOGD="$RES/_logs/evonehot_30seed"; mkdir -p "$LOGD"
MAXCON=8                       # 4/GPU; ~20GB/40GB GPU, leaves CPU/RAM headroom

SEEDS=(621 100 383 492 987 167 926 446 390 477 137 531 919 3 194 \
       77 303 331 76 433 652 772 527 563 340 998 171 590 548 511)

run_job() {  # $1=seed $2=gpu
    local seed="$1" gpu="$2"
    local out="$RES/ms_GFP/AlphaVariant/seed${seed}.json"
    if [ -f "$out" ]; then echo "[skip] GFP seed$seed (done)"; return; fi
    CUDA_VISIBLE_DEVICES="$gpu" nice -n 10 $PY run_generic.py \
        --dataset ms_GFP --seed "$seed" --oracle --level uniform \
        --prior_model_path priors/ms_GFP/prior_model.pt \
        --features ev_onehot --use_mutcompute --shap_prune_alphabet \
        --max_n_mut 2 --sigma 60 \
        --n_rounds 5 --n_steps_per_round 500 --device cuda:0 \
        --output_path "$RES" --data_dir ../data \
        > "$LOGD/ms_GFP_seed${seed}.log" 2>&1
}

echo "[gfp-30seed] start $(date)  30 seeds, MAXCON=$MAXCON (4/GPU, nice'd)"
running=0; gpu=0
for seed in "${SEEDS[@]}"; do
    run_job "$seed" "$gpu" &
    gpu=$(( 1 - gpu ))
    running=$(( running + 1 ))
    if [ "$running" -ge "$MAXCON" ]; then wait -n; running=$(( running - 1 )); fi
done
wait
echo "[gfp-30seed] all runs done $(date)"

$PY - <<'PY'
import json, glob, numpy as np
RES="/home/xux/Desktop/AlphaVariant/Benchmark/results_oracle"
t=[];mx=[]
for f in glob.glob(f"{RES}/ms_GFP/AlphaVariant/seed*.json"):
    m=json.load(open(f))['metrics']; t.append(m['top128_mean_norm']); mx.append(m['max_fitness_norm'])
if t:
    t=np.array(t);mx=np.array(mx)
    print(f"\n=== GFP 30-seed AlphaVariant (ev+onehot, max_n_mut=2) ===")
    print(f"n={len(t)}  top128 median={np.median(t):.3f} [{np.percentile(t,25):.3f},{np.percentile(t,75):.3f}]"
          f"  max median={np.median(mx):.3f} [{np.percentile(mx,25):.3f},{np.percentile(mx,75):.3f}]")
    print("(broken/stalled GFP was top128=0.356, flat, for every cap)")
PY
echo "GFP_30SEED_DONE $(date)"
