#!/bin/bash
# Continue the multi-site +finetune check on PAB1 + GFP (the expensive datasets).
# Reduced seeds for cost (PAB1 ~40 min/seed, GFP ~5 h/seed): PAB1 = 10, GFP = 5.
# Compares the +finetune arm against the EXISTING committed full results
# (results_oracle/<ds>) on the same seeds.
set -u
cd "$(dirname "$0")/../../alphavariant"
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4
PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python
BENCH=/home/xux/Desktop/AlphaVariant/Benchmark
OUT="$BENCH/results_ablation"
MAXCON=4
COMMON="--oracle --level uniform --features ev_onehot --use_mutcompute --shap_prune_alphabet \
        --max_n_mut 2 --sigma 60 --n_rounds 5 --n_steps_per_round 500 --device cuda:0 --data_dir ../data"

SEEDS=(621 100 383 492 987 167 926 446 390 477 137 531 919 3 194 \
       77 303 331 76 433 652 772 527 563 340 998 171 590 548 511)

declare -A PFX=( [ms_PAB1]=pab1 [ms_GFP]=gfp )
declare -A NSEED=( [ms_PAB1]=10 [ms_GFP]=5 )

run_job(){ local ds="$1" seed="$2" gpu="$3"
  local name="${PFX[$ds]}_ms_finetune"
  local res="$OUT/$name/$ds/AlphaVariant/seed${seed}.json"
  if [ -f "$res" ]; then echo "[skip] $name seed$seed"; return; fi
  mkdir -p "$OUT/$name/_logs"
  CUDA_VISIBLE_DEVICES="$gpu" nice -n 10 $PY ../scripts/alphavariant/run_generic.py --dataset "$ds" --seed "$seed" \
    --prior_model_path "priors/$ds/prior_model.pt" --finetune_prior \
    --output_path "$OUT/$name" $COMMON > "$OUT/$name/_logs/seed${seed}.log" 2>&1; }

echo "[ms-ft2] start $(date)"
running=0; gpu=0
for ds in ms_PAB1 ms_GFP; do
  n=${NSEED[$ds]}
  for seed in "${SEEDS[@]:0:$n}"; do
    run_job "$ds" "$seed" "$gpu" &
    gpu=$(( 1 - gpu )); running=$(( running + 1 ))
    if [ "$running" -ge "$MAXCON" ]; then wait -n; running=$(( running - 1 )); fi
  done
done
wait
echo "[ms-ft2] all runs done $(date)"

$PY - <<'PY'
import glob, json, numpy as np
BENCH="/home/xux/Desktop/AlphaVariant/Benchmark"
def load(pat):
    d={}
    for f in glob.glob(pat):
        s=f.split("seed")[-1].split(".json")[0]
        m=json.load(open(f)).get("metrics",{})
        d[s]=(m.get("max_fitness_norm"),m.get("top128_mean_norm"))
    return d
print(f"\n{'dataset':9} {'arm':16} {'n':>3} {'max(med)':>9} {'top128(med)':>12}")
print("-"*60)
for ds,pfx in [("ms_PAB1","pab1"),("ms_GFP","gfp")]:
    full=load(f"{BENCH}/results_oracle/{ds}/AlphaVariant/seed*.json")
    ft  =load(f"{BENCH}/results_ablation/{pfx}_ms_finetune/{ds}/AlphaVariant/seed*.json")
    sh=sorted(set(full)&set(ft))
    if not sh: print(f"{ds:9} no shared seeds"); continue
    fmax=np.array([full[s][0] for s in sh]); ftmax=np.array([ft[s][0] for s in sh])
    ftop=np.array([full[s][1] for s in sh]); fttop=np.array([ft[s][1] for s in sh])
    print(f"{ds:9} {'full':16} {len(sh):>3} {np.median(fmax):>9.3f} {np.median(ftop):>12.3f}")
    print(f"{ds:9} {'+ finetune':16} {len(sh):>3} {np.median(ftmax):>9.3f} {np.median(fttop):>12.3f}  "
          f"Δmax={np.median(ftmax)-np.median(fmax):+.3f} Δtop={np.median(fttop)-np.median(ftop):+.3f}")
PY
echo "MS_FT2_DONE $(date)"
