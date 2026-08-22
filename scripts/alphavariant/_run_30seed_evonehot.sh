#!/bin/bash
# 30-seed AlphaVariant benchmark on the three combinatorial multi-site ORACLE datasets
# (AAV, PAB1, CreiLOV) with the LOCKED config:
#   ev+onehot surrogate (one-hot ++ EV statistical energy, retrained on accumulated
#   collected variants each round), SHAP-prune w/ hotspot-config sync, max_n_mut=2,
#   5 rounds x 96 = 480 oracle calls, 500 RL steps/round, sigma 60.
# Writes to the real results_oracle/<d>/AlphaVariant/seedN.json (the 10-method benchmark
# dir) and is idempotent (skips seeds already done), so it is safe to re-launch.
set -u
cd "$(dirname "$0")/../../alphavariant"
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4
PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python
BENCH=/home/xux/Desktop/AlphaVariant/Benchmark
RES="$BENCH/results_oracle"
LOGD="$RES/_logs/evonehot_30seed"; mkdir -p "$LOGD"
MAXCON=8                       # ~4/GPU; AAV/PAB1/CreiLOV are light (15-75aa, ~2-3GB/run)

DATASETS=(ms_AAV ms_PAB1 ms_CreiLOV)
SEEDS=(621 100 383 492 987 167 926 446 390 477 137 531 919 3 194 \
       77 303 331 76 433 652 772 527 563 340 998 171 590 548 511)

run_job() {  # $1=dataset $2=seed $3=gpu
    local d="$1" seed="$2" gpu="$3"
    local out="$RES/$d/AlphaVariant/seed${seed}.json"
    if [ -f "$out" ]; then echo "[skip] $d seed$seed (done)"; return; fi
    CUDA_VISIBLE_DEVICES="$gpu" $PY ../scripts/alphavariant/run_generic.py \
        --dataset "$d" --seed "$seed" --oracle --level uniform \
        --prior_model_path "priors/$d/prior_model.pt" \
        --features ev_onehot --use_mutcompute --shap_prune_alphabet \
        --max_n_mut 2 --sigma 60 \
        --n_rounds 5 --n_steps_per_round 500 --device cuda:0 \
        --output_path "$RES" --data_dir ../data \
        > "$LOGD/${d}_seed${seed}.log" 2>&1
}

echo "[30seed] start $(date)  3 datasets x 30 seeds = 90 runs, MAXCON=$MAXCON"
running=0; gpu=0
for d in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_job "$d" "$seed" "$gpu" &
    gpu=$(( 1 - gpu ))
    running=$(( running + 1 ))
    if [ "$running" -ge "$MAXCON" ]; then wait -n; running=$(( running - 1 )); fi
  done
done
wait
echo "[30seed] all runs done $(date)"

$PY - <<'PY'
import json, glob, numpy as np
RES="/home/xux/Desktop/AlphaVariant/Benchmark/results_oracle"
print("\n=== 30-seed AlphaVariant (ev+onehot, max_n_mut=2) — median [IQR] over seeds ===")
print(f"{'dataset':12} {'n':>3} {'top128 median':>16} {'max median':>14}")
for d in ["ms_AAV","ms_PAB1","ms_CreiLOV"]:
    t=[]; mx=[]
    for f in glob.glob(f"{RES}/{d}/AlphaVariant/seed*.json"):
        m=json.load(open(f))['metrics']; t.append(m['top128_mean_norm']); mx.append(m['max_fitness_norm'])
    if t:
        t=np.array(t); mx=np.array(mx)
        print(f"{d:12} {len(t):>3} {np.median(t):>7.3f} [{np.percentile(t,25):.3f},{np.percentile(t,75):.3f}] "
              f"{np.median(mx):>7.3f}")
    else:
        print(f"{d:12}   0   (none)")
PY
echo "EVONEHOT_30SEED_DONE $(date)"
