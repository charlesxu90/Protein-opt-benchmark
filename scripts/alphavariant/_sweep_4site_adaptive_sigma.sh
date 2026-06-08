#!/bin/bash
# Adaptive-sigma 4-site sweep: GB1, TEV, TRPB x 30 seeds each.
# Adaptive sigma (sigma_eff = sigma*(round+1)/n_rounds) is now the default code path
# in run_generic.py — no special flag needed. PhoQ already done (combo6_adaptive_sigma),
# copied into the same results tree separately.
set -u
cd "$(dirname "$0")/../../alphavariant"
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4
PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python
BENCH=/home/xux/Desktop/AlphaVariant/Benchmark
OUTROOT="$BENCH/results_4site_adaptive_sigma"
MAXCON=4

DATASETS=(4site_GB1 4site_TEV 4site_TRPB)
SEEDS=(621 100 383 492 987 167 926 446 390 477 137 531 919 3 194 \
       77 303 331 76 433 652 772 527 563 340 998 171 590 548 511)

run_job() {
    local ds="$1" seed="$2" gpu="$3"
    local outbase="$OUTROOT/${ds}_AlphaVariant"
    local out="$outbase/seed_${seed}/metrics.json"
    mkdir -p "$outbase/_logs"
    if [ -f "$out" ]; then echo "[skip] $ds seed$seed"; return; fi
    CUDA_VISIBLE_DEVICES="$gpu" nice -n 10 $PY run_generic.py \
        --dataset "$ds" --seed "$seed" \
        --output_path "$outbase" \
        > "$outbase/_logs/seed${seed}.log" 2>&1
}

echo "[sweep] start $(date)  GB1/TEV/TRPB x 30 seeds: adaptive sigma (default code)"
running=0; gpu=0
for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_job "$ds" "$seed" "$gpu" &
        gpu=$(( 1 - gpu ))
        running=$(( running + 1 ))
        if [ "$running" -ge "$MAXCON" ]; then wait -n; running=$(( running - 1 )); fi
    done
done
wait
echo "[sweep] all runs done $(date)"
echo "SWEEP_4SITE_ADAPTIVE_DONE $(date)"
