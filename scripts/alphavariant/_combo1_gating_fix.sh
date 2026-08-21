#!/bin/bash
# Combo #1: orig config + SHAP-sync gated OFF for 4-site + sign-flip gated to
# multi-site only (both now implemented in run_generic.py).
# Config: --use_mutcompute --plm_reward_lambda 0.5 --shap_prune_alphabet
# Hypothesis: recovers the committed ~70 on current code.
# Default: 1 seed smoke test; pass --full for all 30 seeds.
set -u
cd "$(dirname "$0")/../../alphavariant"
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4
PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python
BENCH=/home/xux/Desktop/AlphaVariant/Benchmark
OUTBASE="$BENCH/results_phoq_sweep/combo1_gating_fix"
LOGD="$OUTBASE/_logs"; mkdir -p "$LOGD"

FULL_SEEDS=(621 100 383 492 987 167 926 446 390 477 137 531 919 3 194 \
            77 303 331 76 433 652 772 527 563 340 998 171 590 548 511)
SMOKE_SEEDS=(621)

if [ "${1:-}" = "--full" ]; then
    SEEDS=("${FULL_SEEDS[@]}"); MAXCON=4
    echo "[combo1] FULL run: 30 seeds, MAXCON=$MAXCON"
else
    SEEDS=("${SMOKE_SEEDS[@]}"); MAXCON=1
    echo "[combo1] SMOKE: seed=621 only (pass --full for 30-seed run)"
fi

run_job() {
    local seed="$1" gpu="$2"
    local out="$OUTBASE/4site_PhoQ_AlphaVariant/seed_${seed}/metrics.json"
    if [ -f "$out" ]; then echo "[skip] PhoQ seed$seed"; return; fi
    CUDA_VISIBLE_DEVICES="$gpu" nice -n 10 $PY ../scripts/alphavariant/run_generic.py \
        --dataset 4site_PhoQ --seed "$seed" \
        --output_path "$OUTBASE" \
        --use_mutcompute --plm_reward_lambda 0.5 --shap_prune_alphabet \
        > "$LOGD/seed${seed}.log" 2>&1
}

echo "[combo1] start $(date)"
running=0; gpu=0
for seed in "${SEEDS[@]}"; do
    run_job "$seed" "$gpu" &
    gpu=$(( 1 - gpu ))
    running=$(( running + 1 ))
    if [ "$running" -ge "$MAXCON" ]; then wait -n; running=$(( running - 1 )); fi
done
wait
echo "[combo1] runs done $(date)"

$PY - <<'PY'
import json, glob, numpy as np
BASE="/home/xux/Desktop/AlphaVariant/Benchmark/results_phoq_sweep/combo1_gating_fix"
BARE="/home/xux/Desktop/AlphaVariant/Benchmark/results_4site_newcode"
def stats(base, pat):
    top=[]; mx=[]
    for f in glob.glob(f"{base}/{pat}/metrics.json"):
        m=json.load(open(f))['metrics']
        top.append(m.get('normalized_fitness_median_top128'))
        mx.append(m.get('max_fitness'))
    top=[x for x in top if x is not None]; mx=[x for x in mx if x is not None]
    return (len(top), np.median(top) if top else float('nan'), np.median(mx) if mx else float('nan'))
n1,t1,m1=stats(BASE,"4site_PhoQ_AlphaVariant/seed_*")
nb,tb,mb=stats(BARE,"4site_PhoQ_AlphaVariant/seed_*")
print(f"\n=== 4site_PhoQ: combo1 (gating fix) vs bare ===")
print(f"{'config':35} {'n':>4} {'top128':>8} {'max':>8}")
print(f"{'bare (current code)':35} {nb:>4} {tb:>8.3f} {mb:>8.3f}")
print(f"{'combo1: MC+PLM+SHAP gated':35} {n1:>4} {t1:>8.3f} {m1:>8.3f}")
print(f"  target: max ~70, top128 ~0.120  (old code reference)")
PY
echo "COMBO1_DONE $(date)"
