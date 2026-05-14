#!/bin/bash
set -e
cd /home/xux/Desktop/AlphaVariant/Benchmark/CLADE
for ds in CreiLOV TRPB 4site_GB1 AAV_hard eqFP611_red 4site_TEV 4site_PhoQ PAB1 mTagBFP2_blue mTagBFP2_red; do
  echo "=== CLADE/$ds (30 seeds) starting $(date) ==="
  /home/xux/Desktop/AlphaVariant/Benchmark/ALDE/env/bin/python run_${ds}.py --seeds 621 100 383 492 987 167 926 446 390 477 137 531 919 3 194 77 303 331 76 433 652 772 527 563 340 998 171 590 548 511  > ../sweep_logs/CLADE_${ds}.log 2>&1
  echo "=== CLADE/$ds done rc=$? $(date) ==="
done
