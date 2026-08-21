#!/bin/bash
# Re-run AlphaVariant on the four 4-site datasets (GB1/PhoQ/TEV/TRPB) x 30 seeds with the
# CURRENT code, to refresh the 4-site numbers against the same code version used for the
# multi-site oracle benchmark. Uses the BARE canonical invocation (current defaults:
# cluster-init, onehot, lookup mode -- NO --oracle/--ev_onehot/--shap/--max_n_mut), exactly
# the config whose drift the GB1 regression test exposed.
#
# Old results in alphavariant/results/4site_<ds>_AlphaVariant/ are NOT touched; new results
# go to results_4site_newcode/4site_<ds>_AlphaVariant/seed_<seed>/ for side-by-side compare.
# Idempotent (skips seeds already done). 4-site GPT is tiny -> light/fast (~11 min/run).
set -u
cd "$(dirname "$0")/../../alphavariant"
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4
PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python
BENCH=/home/xux/Desktop/AlphaVariant/Benchmark
OUTBASE="$BENCH/results_4site_newcode"
LOGD="$OUTBASE/_logs"; mkdir -p "$LOGD"
MAXCON=8                       # 4-site is light; nice'd to keep the box responsive

DATASETS=(4site_GB1 4site_PhoQ 4site_TEV 4site_TRPB)
SEEDS=(621 100 383 492 987 167 926 446 390 477 137 531 919 3 194 \
       77 303 331 76 433 652 772 527 563 340 998 171 590 548 511)

run_job() {  # $1=dataset $2=seed $3=gpu
    local d="$1" seed="$2" gpu="$3"
    local out="$OUTBASE/${d}_AlphaVariant/seed_${seed}/metrics.json"
    if [ -f "$out" ]; then echo "[skip] $d seed$seed"; return; fi
    CUDA_VISIBLE_DEVICES="$gpu" nice -n 10 $PY ../scripts/alphavariant/run_generic.py \
        --dataset "$d" --seed "$seed" \
        --output_path "$OUTBASE/${d}_AlphaVariant" \
        > "$LOGD/${d}_seed${seed}.log" 2>&1
}

echo "[4site-rerun] start $(date)  4 datasets x 30 seeds = 120 runs, MAXCON=$MAXCON (current code, bare config)"
# Interleave: seed-outer, dataset-inner -> all 4 datasets appear in wave 1 (catch crashes early)
running=0; gpu=0
for seed in "${SEEDS[@]}"; do
  for d in "${DATASETS[@]}"; do
    run_job "$d" "$seed" "$gpu" &
    gpu=$(( 1 - gpu ))
    running=$(( running + 1 ))
    if [ "$running" -ge "$MAXCON" ]; then wait -n; running=$(( running - 1 )); fi
  done
done
wait
echo "[4site-rerun] all runs done $(date)"

$PY - <<'PY'
import json, glob, numpy as np
OLD="/home/xux/Desktop/AlphaVariant/Benchmark/alphavariant/results"
NEW="/home/xux/Desktop/AlphaVariant/Benchmark/results_4site_newcode"
def stats(base,d):
    t=[];mx=[]
    for f in glob.glob(f"{base}/{d}_AlphaVariant/seed_*/metrics.json"):
        m=json.load(open(f))['metrics']; t.append(m.get('normalized_fitness_median_top128')); mx.append(m.get('max_fitness'))
    return (len(t), np.median(t) if t else float('nan'), np.median(mx) if mx else float('nan'))
print("\n=== 4-site AlphaVariant: OLD code vs CURRENT code (median) ===")
print(f"{'dataset':12} {'OLD n/top128/max':>28} {'NEW n/top128/max':>28}")
for d in ["4site_GB1","4site_PhoQ","4site_TEV","4site_TRPB"]:
    on,ot,om=stats(OLD,d); nn,nt,nm=stats(NEW,d)
    print(f"{d:12} {f'{on}  {ot:.3f}  {om:.3f}':>28} {f'{nn}  {nt:.3f}  {nm:.3f}':>28}")
PY
echo "RERUN_4SITE_DONE $(date)"
