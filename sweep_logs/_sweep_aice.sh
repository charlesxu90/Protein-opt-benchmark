#!/bin/bash
cd /home/xux/Desktop/AlphaVariant/Benchmark/AiCE
PY=/home/xux/Desktop/AlphaVariant/Benchmark/AiCE/env/bin/python
export CUDA_VISIBLE_DEVICES=0
DATASETS=(CreiLOV TRPB 4site_GB1 AAV_hard eqFP611_red 4site_TEV 4site_PhoQ PAB1 mTagBFP2_blue mTagBFP2_red)
for ds in "${DATASETS[@]}"; do
  done_count=$(ls results/${ds}_AiCE_experiments/${ds}/aice/metrics_seed*.json 2>/dev/null | wc -l)
  if [ "$done_count" -ge 30 ]; then
    echo "=== AiCE/$ds SKIP (already $done_count seeds) ==="
    continue
  fi
  echo "=== AiCE/$ds (30 seeds, GPU0) starting $(date) ==="
  $PY run_${ds}.py --seed_file ../rand_seeds.txt --num_seeds 30 \
      > ../sweep_logs/AiCE_${ds}.log 2>&1
  echo "=== AiCE/$ds done rc=$? $(date) ==="
done
echo "AiCE sweep complete $(date)"
