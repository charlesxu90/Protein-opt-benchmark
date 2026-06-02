#!/usr/bin/env bash
# 03_monitor.sh — monitor an EvoPlay array job on iBex.
# Run on iBex login node.
#
# Usage:
#   bash scripts/cluster_evoplay/03_monitor.sh <JOB_ID>
#   watch -n 30 'bash scripts/cluster_evoplay/03_monitor.sh <JOB_ID>'   # auto-refresh
set -euo pipefail

JOB_ID="${1:-}"
if [ -z "$JOB_ID" ]; then
  echo "Usage: $0 <JOB_ID>"
  exit 1
fi

cd "${HOME}/Benchmark"

echo "=== Job ${JOB_ID} status ==="
squeue -j "$JOB_ID" -h -o "%T %r" 2>/dev/null | sort | uniq -c | sort -rn || echo "  job not in queue (may be finished)"

echo ""
echo "=== Array task summary ==="
sacct -j "$JOB_ID" --format=JobID%20,State,Elapsed,MaxRSS,ExitCode -n 2>/dev/null | \
  awk 'NR<=5 || /\.batch|\.extern/{next} {print}' | head -10
echo "  ..."

echo ""
echo "=== Metrics produced ==="
for ds in 4site_GB1 4site_PhoQ 4site_TEV 4site_TRPB; do
  n1=$(find EvoPlay/results/${ds}_EvoPlay -name "metrics_seed*.json" 2>/dev/null | wc -l)
  n2=$(find EvoPlay/results/${ds}_EvoPlay -path "*seed_*/metrics.json" 2>/dev/null | wc -l)
  printf "  %-14s %s metrics\n" "$ds" "$((n1+n2))"
done

echo ""
echo "=== Latest seed log progress (most recent 3) ==="
ls -t sweep_logs/4site_extra/evoplay_cluster/*_seed*.log 2>/dev/null | head -3 | while read f; do
  last_seq=$(tac "$f" 2>/dev/null | grep -m1 "Updating predictor" | grep -oE "[0-9]+" | head -1 || echo "?")
  printf "  %s: %s/480 seqs\n" "$(basename $f .log)" "${last_seq}"
done

echo ""
echo "=== Recent failures ==="
sacct -j "$JOB_ID" --format=JobID,State,ExitCode -n -P 2>/dev/null | \
  awk -F'|' '$2 != "COMPLETED" && $2 != "RUNNING" && $2 != "PENDING" && $1 !~ /\./' | head -5
