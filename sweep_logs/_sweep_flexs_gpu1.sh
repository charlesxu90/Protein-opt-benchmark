#!/bin/bash
set -e
cd /home/xux/Desktop/AlphaVariant/Benchmark/FLEXS
export CUDA_VISIBLE_DEVICES=1
for ds in 4site_TEV 4site_PhoQ PAB1 mTagBFP2_blue mTagBFP2_red; do
  echo "=== FLEXS/$ds (30 seeds, GPU1) starting $(date) ==="
  /home/xux/Desktop/AlphaVariant/Benchmark/FLEXS/env/bin/python run_${ds}.py --seed_file ../rand_seeds.txt --num_seeds 30       > ../sweep_logs/FLEXS_${ds}.log 2>&1
  echo "=== FLEXS/$ds done rc=$? $(date) ==="
done
