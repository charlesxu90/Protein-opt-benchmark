#!/bin/bash
# Combo #4: bare + n_gpt_ensemble=2, no PLM, no MC — variance-reduction via bootstrap GPT diversity
# Hypothesis: two bootstrap GPTs per round prevents seeds from getting stuck in low-fitness attractors
# 30 seeds, matches the bare baseline seed set.
set -u
cd "$(dirname "$0")/../../alphavariant"
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4
PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python
BENCH=/home/xux/Desktop/AlphaVariant/Benchmark
OUTBASE="$BENCH/results_phoq_sweep/combo4_ensemble2"
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
        --n_gpt_ensemble 2 \
        > "$LOGD/seed${seed}.log" 2>&1
}

echo "[combo4] start $(date)  4site_PhoQ x 30 seeds: bare + n_gpt_ensemble=2"
running=0; gpu=0
for seed in "${SEEDS[@]}"; do
    run_job "$seed" "$gpu" &
    gpu=$(( 1 - gpu ))
    running=$(( running + 1 ))
    if [ "$running" -ge "$MAXCON" ]; then wait -n; running=$(( running - 1 )); fi
done
wait
echo "[combo4] all runs done $(date)"

$PY - <<'PY'
import json, glob, numpy as np
BASE="/home/xux/Desktop/AlphaVariant/Benchmark/results_phoq_sweep/combo4_ensemble2"
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
    return (n, np.median(top) if top else float('nan'), np.median(mx) if mx else float('nan'))
n4,t4,m4=stats(BASE,"seed_*")
nb,tb,mb=stats(BARE,"4site_PhoQ_AlphaVariant/seed_*")
print(f"\n=== 4site_PhoQ: combo4 (ensemble=2) vs bare ===")
print(f"{'config':35} {'n':>4} {'top128':>8} {'max':>8}")
print(f"{'bare (current code)':35} {nb:>4} {tb:>8.3f} {mb:>8.3f}")
print(f"{'combo4: ensemble=2':35} {n4:>4} {t4:>8.3f} {m4:>8.3f}")
PY
echo "COMBO4_DONE $(date)"
