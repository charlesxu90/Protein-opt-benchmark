#!/bin/bash
# Random sweep: 30 seeds × 10 datasets. Skip datasets where all 30 per-seed JSONs already exist.
cd /home/xux/Desktop/AlphaVariant/Benchmark/Random
PY=/home/xux/Desktop/AlphaVariant/Benchmark/ALDE/env/bin/python
DATASETS=(CreiLOV TRPB 4site_GB1 AAV_hard eqFP611_red 4site_TEV 4site_PhoQ PAB1 mTagBFP2_blue mTagBFP2_red)
for ds in "${DATASETS[@]}"; do
  done_count=$(ls results/${ds}_Random/${ds}/random/metrics_seed*.json 2>/dev/null | wc -l)
  if [ "$done_count" -ge 30 ]; then
    echo "=== Random/$ds SKIP (already $done_count seeds) ==="
    continue
  fi
  echo "=== Random/$ds (30 seeds) starting $(date) ==="
  $PY run_${ds}.py --seed_file ../rand_seeds.txt --num_seeds 30 \
      > ../sweep_logs/Random_${ds}.log 2>&1
  echo "=== Random/$ds done rc=$? $(date) ==="
done
echo "Random sweep complete $(date)"
