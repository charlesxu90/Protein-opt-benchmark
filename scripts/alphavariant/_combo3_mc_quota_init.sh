#!/bin/bash
# Combo #3: bare + MC-quota init on 4site_PhoQ (no SHAP, no cap)
# --use_mutcompute --plm_sampling_frac 0.25 --plm_sampling_until_round 1
# Hypothesis: isolates MC-included init idea on the good baseline.
# 30 seeds, matches the bare baseline seed set.
set -u
cd "$(dirname "$0")/../../alphavariant"
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4
PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python
BENCH=/home/xux/Desktop/AlphaVariant/Benchmark
OUTBASE="$BENCH/results_phoq_sweep/combo3_mc_quota_init"
LOGD="$OUTBASE/_logs"; mkdir -p "$LOGD"
MAXCON=4

SEEDS=(621 100 383 492 987 167 926 446 390 477 137 531 919 3 194 \
       77 303 331 76 433 652 772 527 563 340 998 171 590 548 511)

run_job() {
    local seed="$1" gpu="$2"
    local out="$OUTBASE/4site_PhoQ_AlphaVariant/seed_${seed}/metrics.json"
    if [ -f "$out" ]; then echo "[skip] PhoQ seed$seed"; return; fi
    CUDA_VISIBLE_DEVICES="$gpu" nice -n 10 $PY run_generic.py \
        --dataset 4site_PhoQ --seed "$seed" \
        --output_path "$OUTBASE" \
        --use_mutcompute --plm_sampling_frac 0.25 --plm_sampling_until_round 1 \
        > "$LOGD/seed${seed}.log" 2>&1
}

echo "[combo3] start $(date)  4site_PhoQ x 30 seeds: MC quota-init, no SHAP, no cap"
running=0; gpu=0
for seed in "${SEEDS[@]}"; do
    run_job "$seed" "$gpu" &
    gpu=$(( 1 - gpu ))
    running=$(( running + 1 ))
    if [ "$running" -ge "$MAXCON" ]; then wait -n; running=$(( running - 1 )); fi
done
wait
echo "[combo3] all runs done $(date)"

$PY - <<'PY'
import json, glob, numpy as np
BASE="/home/xux/Desktop/AlphaVariant/Benchmark/results_phoq_sweep/combo3_mc_quota_init"
BARE="/home/xux/Desktop/AlphaVariant/Benchmark/results_4site_newcode"
def stats(base, pat):
    top=[]; mx=[]
    for f in glob.glob(f"{base}/{pat}/metrics.json"):
        m=json.load(open(f))['metrics']
        top.append(m.get('normalized_fitness_median_top128'))
        mx.append(m.get('max_fitness'))
    top=[x for x in top if x is not None]; mx=[x for x in mx if x is not None]
    return (len(top), np.median(top) if top else float('nan'), np.median(mx) if mx else float('nan'))
n3,t3,m3=stats(BASE,"4site_PhoQ_AlphaVariant/seed_*")
nb,tb,mb=stats(BARE,"4site_PhoQ_AlphaVariant/seed_*")
print(f"\n=== 4site_PhoQ: combo3 (MC quota-init) vs bare ===")
print(f"{'config':30} {'n':>4} {'top128':>8} {'max':>8}")
print(f"{'bare (current code)':30} {nb:>4} {tb:>8.3f} {mb:>8.3f}")
print(f"{'combo3: MC quota-init':30} {n3:>4} {t3:>8.3f} {m3:>8.3f}")
PY
echo "COMBO3_DONE $(date)"
