#!/bin/bash
cd /home/xux/Desktop/AlphaVariant/Benchmark/ALDE
PY=/home/xux/Desktop/AlphaVariant/Benchmark/ALDE/env/bin/python
export CUDA_VISIBLE_DEVICES=1
DATASETS=(CreiLOV TRPB 4site_GB1 AAV_hard eqFP611_red 4site_TEV 4site_PhoQ PAB1 mTagBFP2_blue mTagBFP2_red)
for ds in "${DATASETS[@]}"; do
  done_count=$(ls results/${ds}_TS_experiments/${ds}/onehot/metrics_seed*.json 2>/dev/null | wc -l)
  if [ "$done_count" -ge 30 ]; then
    echo "=== ALDE/$ds SKIP (already $done_count seeds) ==="
    continue
  fi
  echo "=== ALDE/$ds (30 seeds, GPU1) starting $(date) ==="
  $PY run_${ds}.py --seed_file ../rand_seeds.txt --num_seeds 30 \
      > ../sweep_logs/ALDE_${ds}.log 2>&1
  echo "=== ALDE/$ds done rc=$? $(date) ==="
done
echo "ALDE sweep complete $(date)"
