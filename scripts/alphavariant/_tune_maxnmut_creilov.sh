#!/bin/bash
# Re-run the CreiLOV max_n_mut sweep AFTER the AACombo fix (oracle scoring now
# expands the 15-mer combo onto the 119aa WT; snapping + cap now operate in the
# 15-mer combo space). Caps 1..10, seed 621, final 500-step config. Clean output
# dir so it does not collide with the (broken, pre-fix) /tmp/maxmut_tune results.
# Low concurrency: CreiLOV (15-mer) is light but the GFP sweep is using both GPUs.
set -u
cd "$(dirname "$0")/../../alphavariant"
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=6
PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python
SEED=621
OUT=/tmp/crei_sweep
LOGD="$OUT/_logs"; mkdir -p "$LOGD"
MAXCON=3           # keep light so the in-flight GFP sweep keeps its GPU share

run_job() {  # $1=V $2=gpu
    local V="$1" gpu="$2"
    CUDA_VISIBLE_DEVICES="$gpu" $PY ../scripts/alphavariant/run_generic.py \
        --dataset ms_CreiLOV --seed "$SEED" --oracle --level uniform \
        --prior_model_path priors/ms_CreiLOV/prior_model.pt \
        --use_mutcompute --shap_prune_alphabet --max_n_mut "$V" --sigma 60 \
        --n_rounds 5 --n_steps_per_round 500 --device cuda:0 \
        --output_path "$OUT/m$V" --data_dir ../data \
        > "$LOGD/m${V}.log" 2>&1
}

echo "[crei] start $(date)  caps 1..10, MAXCON=$MAXCON"
running=0; gpu=0
for V in 1 2 3 4 5 6 7 8 9 10; do
    run_job "$V" "$gpu" &
    gpu=$(( 1 - gpu ))
    running=$(( running + 1 ))
    if [ "$running" -ge "$MAXCON" ]; then wait -n; running=$(( running - 1 )); fi
done
wait
echo "[crei] all runs done $(date)"

$PY - <<'PY'
import json
OUT="/tmp/crei_sweep"
print("\n=== CreiLOV max_n_mut sweep (AFTER AACombo fix), seed 621 ===")
print(f"{'max_n_mut':>9} {'top128_norm':>12} {'max_norm':>10} {'best_n_muts':>12} {'div':>7}")
best=(None,-1)
for V in range(1,11):
    f=f"{OUT}/m{V}/ms_CreiLOV/AlphaVariant/seed621.json"
    try:
        m=json.load(open(f))['metrics']; t=m['top128_mean_norm']
        print(f"{V:>9} {t:>12.3f} {m['max_fitness_norm']:>10.3f} "
              f"{str(m.get('best_n_muts')):>12} {m.get('diversity_top128',float('nan')):>7.2f}")
        if t>best[1]: best=(V,t)
    except Exception as e:
        print(f"{V:>9}   --  ({e})")
print(f"\nBEST: max_n_mut={best[0]}  top128={best[1]:.3f}")
print("Reference: broken pre-fix CreiLOV top128 was 0.537 (identical across all caps); Random=0.773")
PY
echo "CREI_SWEEP_DONE $(date)"
