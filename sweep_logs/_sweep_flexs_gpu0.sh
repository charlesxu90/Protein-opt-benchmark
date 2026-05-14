#!/bin/bash
set -e
cd /home/xux/Desktop/AlphaVariant/Benchmark/FLEXS
export CUDA_VISIBLE_DEVICES=0
for ds in CreiLOV TRPB 4site_GB1 AAV_hard eqFP611_red; do
  echo "=== FLEXS/$ds (30 seeds, GPU0) starting $(date) ==="
  /home/xux/Desktop/AlphaVariant/Benchmark/FLEXS/env/bin/python run_${ds}.py --seed_file ../rand_seeds.txt --num_seeds 30       > ../sweep_logs/FLEXS_${ds}.log 2>&1
  echo "=== FLEXS/$ds done rc=$? $(date) ==="
done
