#!/usr/bin/env bash
# 04_collect.sh — sync EvoPlay metrics back from iBex to local workstation.
# Run from LOCAL workstation, NOT iBex.
#
# Usage:
#   bash scripts/cluster_evoplay/04_collect.sh
#   bash scripts/cluster_evoplay/04_collect.sh ibex.kaust.edu.sa
set -euo pipefail

REMOTE="${1:-ibex.kaust.edu.sa}"
REMOTE_DIR="${2:-~/Benchmark}"
LOCAL_DIR="/home/xux/Desktop/AlphaVariant/Benchmark"

echo "[collect] pulling EvoPlay/results/ from ${REMOTE}:${REMOTE_DIR}/ ..."
rsync -avz --progress \
  "${REMOTE}:${REMOTE_DIR}/EvoPlay/results/" \
  "${LOCAL_DIR}/EvoPlay/results/"

echo ""
echo "[collect] sweep logs ..."
rsync -avz --progress \
  "${REMOTE}:${REMOTE_DIR}/sweep_logs/4site_extra/evoplay_cluster/" \
  "${LOCAL_DIR}/sweep_logs/4site_extra/evoplay_cluster/"

echo ""
echo "[collect] Local metric counts now:"
for ds in 4site_GB1 4site_PhoQ 4site_TEV 4site_TRPB; do
  n1=$(find "${LOCAL_DIR}/EvoPlay/results/${ds}_EvoPlay" -name "metrics_seed*.json" 2>/dev/null | wc -l)
  n2=$(find "${LOCAL_DIR}/EvoPlay/results/${ds}_EvoPlay" -path "*seed_*/metrics.json" 2>/dev/null | wc -l)
  printf "  %-14s %s\n" "$ds" "$((n1+n2))"
done

echo ""
echo "Next: regenerate tables + figures with EvoPlay rows:"
echo "  cd ${LOCAL_DIR}"
echo "  python3 scripts/generate_tables.py --datasets 4site_GB1 4site_PhoQ 4site_TEV TRPB ..."
echo "  python3 scripts/draw_figures_median.py --csv figures/phase5/comparison_median_iqr.csv ..."
