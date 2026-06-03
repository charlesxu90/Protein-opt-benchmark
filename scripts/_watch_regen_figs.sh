#!/bin/bash
set -u
cd /home/xux/Desktop/AlphaVariant/Benchmark
LOG=results_oracle/_logs/_queue_newmethods.out
echo "[watch] waiting for NEWMETHODS_SWEEP_COMPLETE..."
until grep -q "NEWMETHODS_SWEEP_COMPLETE" "$LOG" 2>/dev/null; do sleep 120; done
echo "[watch] sweep complete; regenerating figures at $(date +%H:%M:%S)"
ALDE/env/bin/python scripts/build_oracle_median_iqr_csv.py
ALDE/env/bin/python scripts/draw_figures_median.py --task multisite
echo "[watch] FIGURES_REGENERATED at $(date +%H:%M:%S)"
