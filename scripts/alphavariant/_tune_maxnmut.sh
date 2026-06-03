#!/bin/bash
# Tune --max_n_mut in [1..10] for AlphaVariant (Plan C, oracle) on all 4 multi-site
# datasets, 1 seed each (621). 40 runs, throttled across 2 GPUs. Outputs to a separate
# tune dir per max_n_mut so nothing collides with the real results_oracle/.
set -u
cd "$(dirname "$0")/../../alphavariant"
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8
PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python
SEED=621
DATASETS=(ms_AAV ms_PAB1 ms_CreiLOV ms_GFP)
TUNE=/tmp/maxmut_tune
LOGD="$TUNE/_logs"; mkdir -p "$LOGD"
MAXCON=8           # concurrent runs (~4 per GPU; each uses ~20% GPU)

run_job() {  # $1=dataset $2=V $3=gpu
    local d="$1" V="$2" gpu="$3"
    CUDA_VISIBLE_DEVICES="$gpu" $PY run_generic.py \
        --dataset "$d" --seed "$SEED" --oracle --level uniform \
        --prior_model_path "priors/$d/prior_model.pt" \
        --use_mutcompute --shap_prune_alphabet --max_n_mut "$V" --sigma 60 \
        --n_rounds 5 --n_steps_per_round 500 --device cuda:0 \
        --output_path "$TUNE/m$V" --data_dir ../data \
        > "$LOGD/${d}_m${V}.log" 2>&1
}

echo "[tune] start $(date)  40 runs, MAXCON=$MAXCON"
running=0; gpu=0
for d in "${DATASETS[@]}"; do
  for V in 1 2 3 4 5 6 7 8 9 10; do
    run_job "$d" "$V" "$gpu" &
    gpu=$(( 1 - gpu ))
    running=$(( running + 1 ))
    if [ "$running" -ge "$MAXCON" ]; then wait -n; running=$(( running - 1 )); fi
  done
done
wait
echo "[tune] all runs done $(date)"

# ---- summary: top128 (and max) per (dataset, max_n_mut); best V per dataset ----
$PY - <<'PY'
import glob, json, numpy as np
TUNE="/tmp/maxmut_tune"
DATA=["ms_AAV","ms_PAB1","ms_CreiLOV","ms_GFP"]
print("\n=== max_n_mut tune: top128_mean_norm (max_fitness_norm in parens) ===")
hdr="max_n_mut " + " ".join(f"{d.replace('ms_',''):>13}" for d in DATA)
print(hdr)
best={d:(None,-1) for d in DATA}
rows={}
for V in range(1,11):
    cells=[]
    for d in DATA:
        f=f"{TUNE}/m{V}/{d}/AlphaVariant/seed621.json"
        try:
            m=json.load(open(f))['metrics']; t=m['top128_mean_norm']; mx=m['max_fitness_norm']
            cells.append(f"{t:.3f}({mx:.2f})")
            if t>best[d][1]: best[d]=(V,t)
        except Exception:
            cells.append("   --   ")
    print(f"{V:>9} " + " ".join(f"{c:>13}" for c in cells))
print("\n=== BEST max_n_mut per dataset (by top128) ===")
for d in DATA:
    print(f"  {d:12}: max_n_mut={best[d][0]}  top128={best[d][1]:.3f}")
print("baseline (no cap) top128: AAV 0.37, others were below/near Random")
PY
echo "TUNE_MAXNMUT_DONE $(date)"
