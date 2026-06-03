#!/bin/bash
set -u
cd /home/xux/Desktop/AlphaVariant/Benchmark
export LD_LIBRARY_PATH=/home/xux/miniforge3/envs/alphavariant-env/lib:${LD_LIBRARY_PATH:-}
cnt(){ ls results_oracle/$1/AlphaVariant/seed*.json 2>/dev/null | wc -l; }
echo "[av-watch] waiting for 4x30 AlphaVariant results..."
until [ "$(cnt ms_AAV)" -ge 30 ] && [ "$(cnt ms_PAB1)" -ge 30 ] && \
      [ "$(cnt ms_CreiLOV)" -ge 30 ] && [ "$(cnt ms_GFP)" -ge 30 ]; do sleep 600; done
echo "[av-watch] all AlphaVariant seeds in; regenerating 10-method outputs $(date +%H:%M:%S)"
ALDE/env/bin/python scripts/build_oracle_median_iqr_csv.py
ALDE/env/bin/python scripts/draw_figures_median.py --task multisite
ALDE/env/bin/python scripts/compute_oracle_wilcoxon.py
echo "[av-watch] AV_FIGS_REGENERATED $(date +%H:%M:%S)"
