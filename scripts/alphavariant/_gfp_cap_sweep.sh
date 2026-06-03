#!/bin/bash
# GFP max_n_mut sweep (caps 2..5) with the FIXED code: ev+onehot surrogate, SHAP-prune
# that syncs the GPT hotspot config (so rounds 2-5 actually run instead of stalling on the
# alphabet filter), surrogate retrained on accumulated collected variants each round.
# seed 621, final 500-step config. Now that the loop runs, the cap should matter.
# MAXCON=4 (2/GPU) -> all four caps run at once; ~2.5GB+EV/run fits comfortably on 40GB.
set -u
cd "$(dirname "$0")/../../alphavariant"
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=6
PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python
SEED=621
OUT=/tmp/gfp_cap_sweep
LOGD="$OUT/_logs"; mkdir -p "$LOGD"
MAXCON=4

run_job() {  # $1=V $2=gpu
    local V="$1" gpu="$2"
    CUDA_VISIBLE_DEVICES="$gpu" $PY run_generic.py \
        --dataset ms_GFP --seed "$SEED" --oracle --level uniform \
        --prior_model_path priors/ms_GFP/prior_model.pt \
        --features ev_onehot --use_mutcompute --shap_prune_alphabet \
        --max_n_mut "$V" --sigma 60 \
        --n_rounds 5 --n_steps_per_round 500 --device cuda:0 \
        --output_path "$OUT/m$V" --data_dir ../data \
        > "$LOGD/m${V}.log" 2>&1
}

echo "[gfp-cap] start $(date)  caps 2..5, MAXCON=$MAXCON (ev_onehot + fixed SHAP)"
running=0; gpu=0
for V in 2 3 4 5; do
    run_job "$V" "$gpu" &
    gpu=$(( 1 - gpu ))
    running=$(( running + 1 ))
    if [ "$running" -ge "$MAXCON" ]; then wait -n; running=$(( running - 1 )); fi
done
wait
echo "[gfp-cap] all runs done $(date)"

$PY - <<'PY'
import json, glob, re
OUT="/tmp/gfp_cap_sweep"
print("\n=== GFP max_n_mut sweep (ev+onehot, fixed SHAP), seed 621 ===")
print(f"{'cap':>4} {'top128':>8} {'max':>7} {'best_n_muts':>11}  trajectory")
best=(None,-1)
for V in [2,3,4,5]:
    f=f"{OUT}/m{V}/ms_GFP/AlphaVariant/seed621.json"
    try:
        m=json.load(open(f))['metrics']; t=m['top128_mean_norm']
        tr=[round(x,3) for x in m.get('fitness_trajectory',[])]
        print(f"{V:>4} {t:>8.3f} {m['max_fitness_norm']:>7.3f} {str(m.get('best_n_muts')):>11}  {tr}")
        if t>best[1]: best=(V,t)
    except Exception as e:
        print(f"{V:>4}   --  ({e})")
print(f"\nBEST by top128: max_n_mut={best[0]}  top128={best[1]:.3f}")
print("reference: broken/stalled GFP was top128=0.356 (flat traj [0.93]*5) for every cap")
PY
echo "GFP_CAP_SWEEP_DONE $(date)"
