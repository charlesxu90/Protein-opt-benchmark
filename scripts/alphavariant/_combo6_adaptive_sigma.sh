#!/bin/bash
# Combo #6: bare + adaptive sigma (12→24→36→48→60 across rounds 1-5), no PLM, no MC
# Hypothesis: fixed sigma=60 over-exploits the noisy early-round surrogate; ramping from
#             sigma/n_rounds to sigma keeps the GPT diverse early, exploitative late.
# 30 seeds, matches the bare baseline seed set.
set -u
cd "$(dirname "$0")/../../alphavariant"
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4
PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python
BENCH=/home/xux/Desktop/AlphaVariant/Benchmark
OUTBASE="$BENCH/results_phoq_sweep/combo6_adaptive_sigma"
LOGD="$OUTBASE/_logs"; mkdir -p "$LOGD"
MAXCON=4

SEEDS=(621 100 383 492 987 167 926 446 390 477 137 531 919 3 194 \
       77 303 331 76 433 652 772 527 563 340 998 171 590 548 511)

run_job() {
    local seed="$1" gpu="$2"
    local out="$OUTBASE/seed_${seed}/metrics.json"
    if [ -f "$out" ]; then echo "[skip] PhoQ seed$seed"; return; fi
    CUDA_VISIBLE_DEVICES="$gpu" nice -n 10 $PY run_generic.py \
        --dataset 4site_PhoQ --seed "$seed" \
        --output_path "$OUTBASE" \
        > "$LOGD/seed${seed}.log" 2>&1
}

echo "[combo6] start $(date)  4site_PhoQ x 30 seeds: bare + adaptive sigma (12→60)"
running=0; gpu=0
for seed in "${SEEDS[@]}"; do
    run_job "$seed" "$gpu" &
    gpu=$(( 1 - gpu ))
    running=$(( running + 1 ))
    if [ "$running" -ge "$MAXCON" ]; then wait -n; running=$(( running - 1 )); fi
done
wait
echo "[combo6] all runs done $(date)"

$PY - <<'PY'
import json, glob, numpy as np
BASE="/home/xux/Desktop/AlphaVariant/Benchmark/results_phoq_sweep/combo6_adaptive_sigma"
BARE="/home/xux/Desktop/AlphaVariant/Benchmark/results_4site_newcode"
def stats(base, pat):
    top=[]; mx=[]
    for f in glob.glob(f"{base}/{pat}/metrics.json"):
        d=json.load(open(f))
        m=d.get('metrics', d)
        t=m.get('normalized_fitness_top128') or m.get('normalized_fitness_median_top128')
        x=m.get('max_fitness') or d.get('max_fitness')
        if t is not None: top.append(t)
        if x is not None: mx.append(x)
    n=max(len(top),len(mx)) if (top or mx) else 0
    p=lambda q: np.percentile(mx,q) if mx else float('nan')
    return (n, np.median(top) if top else float('nan'), np.median(mx) if mx else float('nan'), p(25), p(75))
n6,t6,m6,p25_6,p75_6=stats(BASE,"seed_*")
nb,tb,mb,p25b,p75b=stats(BARE,"4site_PhoQ_AlphaVariant/seed_*")
print(f"\n=== 4site_PhoQ: combo6 (adaptive sigma) vs bare ===")
print(f"{'config':35} {'n':>4} {'top128':>8} {'p25':>7} {'median':>7} {'p75':>7}")
print(f"{'bare (current code)':35} {nb:>4} {tb:>8.3f} {p25b:>7.1f} {mb:>7.1f} {p75b:>7.1f}")
print(f"{'combo6: adaptive sigma':35} {n6:>4} {t6:>8.3f} {p25_6:>7.1f} {m6:>7.1f} {p75_6:>7.1f}")
PY
echo "COMBO6_DONE $(date)"
