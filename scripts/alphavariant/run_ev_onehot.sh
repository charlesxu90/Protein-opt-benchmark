#!/bin/bash
# AlphaVariant (Plan C) with the EV+one-hot surrogate on the four multi-site ORACLE
# benchmarks. The surrogate (Ridge x2 + BayesianRidge + RF + GradientBoosting) is fit on
# one-hot ++ EV statistical-energy features and RE-TRAINED each round on the *accumulated*
# collected variants (self.collected_seqs grows 96 -> 480 over the 5 rounds).
#
# SHAP alphabet pruning (--shap_prune_alphabet):
#   ON for ALL four. _update_alphabet_via_shap now (a) indexes residues at the varying
#   positions correctly for full-length sequences (GFP: 237aa, 233 non-contiguous positions),
#   and (b) propagates the pruned alphabet into the GPT hotspot config, so generation in the
#   following rounds samples the pruned subspace. That keeps the per-position filter from
#   rejecting proposals (previously GFP dropped ~100% -> "kept 0/44536" -> rounds 2-5 stalled).
#
# Usage:
#   scripts/alphavariant/run_ev_onehot.sh                 # all 4, seed 621, full 500-step config
#   scripts/alphavariant/run_ev_onehot.sh ms_GFP          # just GFP
#   SEED=621 N_ROUNDS=2 N_STEPS=20 OUT=/tmp/smoke scripts/alphavariant/run_ev_onehot.sh ms_GFP   # smoke
set -u
cd "$(dirname "$0")/../../alphavariant"
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=6
PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python

SEED="${SEED:-621}"
N_ROUNDS="${N_ROUNDS:-5}"
N_STEPS="${N_STEPS:-500}"
SIGMA="${SIGMA:-60}"
MAXNMUT="${MAXNMUT:-2}"          # locked cap (best/near-best on all measured datasets)
OUT="${OUT:-/tmp/ev_onehot}"
GPU="${GPU:-0}"
mkdir -p "$OUT"

DATASETS=("$@")
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(ms_AAV ms_PAB1 ms_CreiLOV ms_GFP)

for d in "${DATASETS[@]}"; do
    SHAP="--shap_prune_alphabet"   # now valid for all 4 (indexing fix + hotspot-config sync)
    log="$OUT/${d}_seed${SEED}.log"
    echo "[ev_onehot] $d  features=ev_onehot  shap='${SHAP}'  cap=$MAXNMUT  gpu=$GPU  $(date)"
    CUDA_VISIBLE_DEVICES="$GPU" $PY ../scripts/alphavariant/run_generic.py \
        --dataset "$d" --seed "$SEED" --oracle --level uniform \
        --prior_model_path "priors/$d/prior_model.pt" \
        --features ev_onehot --use_mutcompute $SHAP \
        --max_n_mut "$MAXNMUT" --sigma "$SIGMA" \
        --n_rounds "$N_ROUNDS" --n_steps_per_round "$N_STEPS" --device cuda:0 \
        --output_path "$OUT/$d" --data_dir ../data \
        > "$log" 2>&1
    rc=$?
    res="$OUT/$d/$d/AlphaVariant/seed${SEED}.json"
    if [ -f "$res" ]; then
        $PY - "$res" <<'PY'
import json, sys
m=json.load(open(sys.argv[1]))['metrics']
print(f"  -> top128={m['top128_mean_norm']:.3f} max={m['max_fitness_norm']:.3f} "
      f"traj={[round(x,2) for x in m.get('fitness_trajectory',[])]}")
PY
    else
        echo "  -> NO RESULT (rc=$rc); tail of log:"; tail -5 "$log"
    fi
done
echo "EV_ONEHOT_RUN_DONE $(date)"
