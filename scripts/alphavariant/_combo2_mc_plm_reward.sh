#!/bin/bash
# Combo #2: bare + MutCompute + PLM-reward λ0.5 (no SHAP, no cap) on 4site_PhoQ
# Hypothesis: does reward shaping alone lift bare 60.8 → ~70 without SHAP?
# 30 seeds, matches the bare baseline seed set.
set -u
cd "$(dirname "$0")/../../alphavariant"
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4
PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python
BENCH=/home/xux/Desktop/AlphaVariant/Benchmark
OUTBASE="$BENCH/results_phoq_sweep/combo2_mc_plm0.5"
LOGD="$OUTBASE/_logs"; mkdir -p "$LOGD"
MAXCON=4

SEEDS=(621 100 383 492 987 167 926 446 390 477 137 531 919 3 194 \
       77 303 331 76 433 652 772 527 563 340 998 171 590 548 511)

run_job() {
    local seed="$1" gpu="$2"
    local out="$OUTBASE/4site_PhoQ_AlphaVariant/seed_${seed}/metrics.json"
    if [ -f "$out" ]; then echo "[skip] PhoQ seed$seed"; return; fi
    CUDA_VISIBLE_DEVICES="$gpu" nice -n 10 $PY ../scripts/alphavariant/run_generic.py \
        --dataset 4site_PhoQ --seed "$seed" \
        --output_path "$OUTBASE" \
        --use_mutcompute --plm_reward_lambda 0.5 \
        > "$LOGD/seed${seed}.log" 2>&1
}

echo "[combo2] start $(date)  4site_PhoQ x 30 seeds: MC + PLM-reward λ0.5, no SHAP, no cap"
running=0; gpu=0
for seed in "${SEEDS[@]}"; do
    run_job "$seed" "$gpu" &
    gpu=$(( 1 - gpu ))
    running=$(( running + 1 ))
    if [ "$running" -ge "$MAXCON" ]; then wait -n; running=$(( running - 1 )); fi
done
wait
echo "[combo2] all runs done $(date)"

$PY - <<'PY'
import json, glob, numpy as np
BASE="/home/xux/Desktop/AlphaVariant/Benchmark/results_phoq_sweep/combo2_mc_plm0.5"
BARE="/home/xux/Desktop/AlphaVariant/Benchmark/results_4site_newcode"
def stats(base, pat):
    top=[]; mx=[]
    for f in glob.glob(f"{base}/{pat}/metrics.json"):
        m=json.load(open(f))['metrics']
        top.append(m.get('normalized_fitness_median_top128'))
        mx.append(m.get('max_fitness'))
    top=[x for x in top if x is not None]; mx=[x for x in mx if x is not None]
    return (len(top), np.median(top) if top else float('nan'), np.median(mx) if mx else float('nan'))
n2,t2,m2=stats(BASE,"4site_PhoQ_AlphaVariant/seed_*")
nb,tb,mb=stats(BARE,"4site_PhoQ_AlphaVariant/seed_*")
print(f"\n=== 4site_PhoQ: combo2 (MC+PLM-reward λ0.5) vs bare ===")
print(f"{'config':30} {'n':>4} {'top128':>8} {'max':>8}")
print(f"{'bare (current code)':30} {nb:>4} {tb:>8.3f} {mb:>8.3f}")
print(f"{'combo2: MC+PLM λ0.5':30} {n2:>4} {t2:>8.3f} {m2:>8.3f}")
PY
echo "COMBO2_DONE $(date)"
