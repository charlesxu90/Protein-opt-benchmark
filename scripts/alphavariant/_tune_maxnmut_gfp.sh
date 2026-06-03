#!/bin/bash
# Clean GFP max_n_mut sweep (caps 1..10, seed 621, final 500-step config). Replaces the
# 9-duplicate-driver mess. ONE driver, MAXCON=4 (2/GPU) -- GFP packs fine (~2.5GB/run).
# GFP AACombo == seq (237aa) so the CreiLOV AACombo fix is a no-op here; this is just a
# clean re-run. Output to its own dir.
set -u
cd "$(dirname "$0")/../../alphavariant"
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=6
PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python
SEED=621
OUT=/tmp/gfp_sweep
LOGD="$OUT/_logs"; mkdir -p "$LOGD"
MAXCON=4

run_job() {  # $1=V $2=gpu
    local V="$1" gpu="$2"
    CUDA_VISIBLE_DEVICES="$gpu" $PY run_generic.py \
        --dataset ms_GFP --seed "$SEED" --oracle --level uniform \
        --prior_model_path priors/ms_GFP/prior_model.pt \
        --use_mutcompute --shap_prune_alphabet --max_n_mut "$V" --sigma 60 \
        --n_rounds 5 --n_steps_per_round 500 --device cuda:0 \
        --output_path "$OUT/m$V" --data_dir ../data \
        > "$LOGD/m${V}.log" 2>&1
}

echo "[gfp] start $(date)  caps 1..10, MAXCON=$MAXCON"
running=0; gpu=0
for V in 1 2 3 4 5 6; do
    run_job "$V" "$gpu" &
    gpu=$(( 1 - gpu ))
    running=$(( running + 1 ))
    if [ "$running" -ge "$MAXCON" ]; then wait -n; running=$(( running - 1 )); fi
done
wait
echo "[gfp] all runs done $(date)"

$PY - <<'PY'
import json, glob, re
OUT="/tmp/gfp_sweep"
print("\n=== GFP max_n_mut sweep, seed 621 ===")
print(f"{'cap':>4} {'top128':>8} {'max':>7} {'best_n_muts':>11}")
best=(None,-1)
for V in range(1,7):
    f=f"{OUT}/m{V}/ms_GFP/AlphaVariant/seed621.json"
    try:
        m=json.load(open(f))['metrics']; t=m['top128_mean_norm']
        print(f"{V:>4} {t:>8.3f} {m['max_fitness_norm']:>7.3f} {str(m.get('best_n_muts')):>11}")
        if t>best[1]: best=(V,t)
    except Exception as e:
        print(f"{V:>4}   --  ({e})")
print(f"\nBEST: max_n_mut={best[0]}  top128={best[1]:.3f}")
PY
echo "GFP_SWEEP_DONE $(date)"
