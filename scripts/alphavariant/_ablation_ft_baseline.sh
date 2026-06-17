#!/bin/bash
# Leave-one-out ablation with BARE+FINETUNE as the four-site baseline.
# Baseline = bare + --finetune_prior (already run as <pfx>_finetune).
# Removable components (new arms here):
#   - ensemble : --finetune_prior --single_surrogate   (<pfx>_ft_singlesurr)
#   - RL       : --finetune_prior --no_rl               (<pfx>_ft_norl)
# (Removing finetune itself = the existing <pfx>_bare arm.)
# 4 datasets x 2 arms x 30 seeds. Resumable.
set -u
cd "$(dirname "$0")/../../alphavariant"
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4
PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python
BENCH=/home/xux/Desktop/AlphaVariant/Benchmark
OUT="$BENCH/results_ablation"
MAXCON=4
COMMON="--n_rounds 5 --n_steps_per_round 500 --sigma 60 --data_dir ../data"

SEEDS=(621 100 383 492 987 167 926 446 390 477 137 531 919 3 194 \
       77 303 331 76 433 652 772 527 563 340 998 171 590 548 511)

declare -A PFX=( [4site_GB1]=gb1 [4site_PhoQ]=phoq [4site_TEV]=tev [4site_TRPB]=trpb )

JOBS=()
for ds in 4site_GB1 4site_PhoQ 4site_TEV 4site_TRPB; do
  p=${PFX[$ds]}
  JOBS+=("${p}_ft_singlesurr|$ds|--finetune_prior --single_surrogate")
  JOBS+=("${p}_ft_norl|$ds|--finetune_prior --no_rl")
done

run_job() {
    local name="$1" ds="$2" flags="$3" seed="$4" gpu="$5"
    local res="$OUT/$name/seed_${seed}/metrics.json"
    if [ -f "$res" ]; then echo "[skip] $name seed$seed"; return; fi
    mkdir -p "$OUT/$name/_logs"
    CUDA_VISIBLE_DEVICES="$gpu" nice -n 10 $PY run_generic.py \
        --dataset "$ds" --seed "$seed" --output_path "$OUT/$name" \
        $COMMON $flags > "$OUT/$name/_logs/seed${seed}.log" 2>&1
}

echo "[abl-ft] start $(date)"
running=0; gpu=0
for spec in "${JOBS[@]}"; do
    IFS='|' read -r name ds flags <<< "$spec"
    for seed in "${SEEDS[@]}"; do
        run_job "$name" "$ds" "$flags" "$seed" "$gpu" &
        gpu=$(( 1 - gpu )); running=$(( running + 1 ))
        if [ "$running" -ge "$MAXCON" ]; then wait -n; running=$(( running - 1 )); fi
    done
done
wait
echo "[abl-ft] all runs done $(date)"

$PY - <<'PY'
import glob, json, numpy as np
ABL="/home/xux/Desktop/AlphaVariant/Benchmark/results_ablation"
GMAX={"4site_GB1":8.761966,"4site_PhoQ":133.5943,"4site_TEV":1.0,"4site_TRPB":1.0}
DS=[("gb1","4site_GB1","GB1"),("phoq","4site_PhoQ","PhoQ"),("tev","4site_TEV","TEV"),("trpb","4site_TRPB","TrpB")]
# baseline = finetune; arms = leave-one-out from bare+finetune
ARMS=[("finetune","bare+finetune (baseline)"),
      ("bare","- finetune prior"),
      ("ft_singlesurr","- ensemble (single RF)"),
      ("ft_norl","- RL (generate+prioritize)")]
def load(cfg,ds):
    g=GMAX[ds]; mx=[];t=[]
    for f in glob.glob(f"{ABL}/{cfg}/seed_*/metrics.json"):
        m=json.load(open(f)).get("metrics",{})
        v=m.get("max_fitness"); tt=m.get("normalized_fitness_median_top128")
        if v is not None: mx.append(v/g if (g!=1.0 and v>1.5) else v)
        if tt is not None: t.append(tt)
    return np.array(mx),np.array(t)
print(f"\n{'dataset':6} {'arm':28} {'n':>3} {'max(med)':>9} {'dMax':>7} {'top128':>8} {'dTop':>7}")
print("-"*72)
for pfx,ds,disp in DS:
    bmx,bt=load(f"{pfx}_finetune",ds); bm,btm=np.median(bmx),np.median(bt)
    for suf,label in ARMS:
        mx,t=load(f"{pfx}_{suf}",ds)
        if len(mx)==0: continue
        m=np.median(mx); tt=np.median(t)
        dm="ref" if suf=="finetune" else f"{m-bm:+.3f}"; dt="ref" if suf=="finetune" else f"{tt-btm:+.3f}"
        print(f"{disp:6} {label:28} {len(mx):>3} {m:>9.3f} {dm:>7} {tt:>8.3f} {dt:>7}")
PY
echo "ABLATION_FT_DONE $(date)"
