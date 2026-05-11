#!/bin/bash
# run_30seed_gb1_sweep.sh — 30-seed GB1 sweep, parallelized across two A100s.
#
# For each method, splits seeds 0..29 of rand_seeds.txt into halves and runs
# them in parallel on GPU 0 and GPU 1. Skips methods that are known-broken at
# their default config (pass --include-broken to include LatProtRL).
#
# Usage:
#   bash scripts/run_30seed_gb1_sweep.sh            # default 8 methods
#   bash scripts/run_30seed_gb1_sweep.sh ALDE FLEXS # subset
#   N_SEEDS=15 bash scripts/run_30seed_gb1_sweep.sh # smaller pilot
#
# Env vars:
#   N_SEEDS         total seeds (default 30; must be even)
#   GPU0_ID         GPU index for the first half (default 0)
#   GPU1_ID         GPU index for the second half (default 1)
#   INCLUDE_BROKEN  set to 1 to include LatProtRL
#   LOGS_DIR        where to write logs (default scripts/hpc/_logs/sweep_<DATE>)

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

N_SEEDS="${N_SEEDS:-30}"
HALF=$(( N_SEEDS / 2 ))
GPU0_ID="${GPU0_ID:-0}"
GPU1_ID="${GPU1_ID:-1}"
INCLUDE_BROKEN="${INCLUDE_BROKEN:-0}"
LOGS_DIR="${LOGS_DIR:-$ROOT/scripts/hpc/_logs/sweep_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOGS_DIR"

# Default method order: fast first, slow last. Skips LatProtRL by default.
DEFAULT_METHODS=(
    Random GreedyWalk
    AiCE ALDE FLEXS
    delta_cs alphavariant
    EvoPlay
)
if [[ "$INCLUDE_BROKEN" == "1" ]]; then
    DEFAULT_METHODS+=(LatProtRL)
fi

if [[ $# -gt 0 ]]; then
    METHODS=("$@")
else
    METHODS=("${DEFAULT_METHODS[@]}")
fi

echo "=================================================================="
echo "30-seed GB1 sweep"
echo "  N_SEEDS=$N_SEEDS (split: 0..$((HALF-1)) on GPU $GPU0_ID, $HALF..$((N_SEEDS-1)) on GPU $GPU1_ID)"
echo "  Methods: ${METHODS[*]}"
echo "  Logs:    $LOGS_DIR"
echo "=================================================================="

# Track wall times in a CSV
TIMING_CSV="$LOGS_DIR/sweep_timing.csv"
echo "method,wall_seconds,exit_code,seeds_done,timestamp" > "$TIMING_CSV"

run_method() {
    local method="$1"
    local started=$(date +%s)
    local m_log="$LOGS_DIR/${method}.log"

    echo
    echo "------------------------------------------------------------------"
    echo "[$(date +%H:%M:%S)] $method"
    echo "------------------------------------------------------------------"

    # GPU 0: seeds [0, HALF)
    /usr/bin/python3 scripts/hpc/launch.py --method "$method" --dataset GB1 \
        --seeds "$HALF" --cluster local \
        --gpu-id "$GPU0_ID" --seed-start 0 \
        > "$LOGS_DIR/${method}_gpu0.log" 2>&1 &
    PID0=$!

    # GPU 1: seeds [HALF, N_SEEDS)
    /usr/bin/python3 scripts/hpc/launch.py --method "$method" --dataset GB1 \
        --seeds "$HALF" --cluster local \
        --gpu-id "$GPU1_ID" --seed-start "$HALF" \
        > "$LOGS_DIR/${method}_gpu1.log" 2>&1 &
    PID1=$!

    echo "  GPU0 pid=$PID0 (seeds 0..$((HALF-1)))"
    echo "  GPU1 pid=$PID1 (seeds $HALF..$((N_SEEDS-1)))"

    wait $PID0; rc0=$?
    wait $PID1; rc1=$?

    local wall=$(( $(date +%s) - started ))
    local rc=$rc0
    [[ $rc1 -ne 0 ]] && rc=$rc1

    # Count seeds with results — find all metrics files modified after $started
    local seeds_done
    seeds_done=$(find . -path '*results*' -name 'metrics*seed*.json' -o -name 'metrics.json' 2>/dev/null \
                 | xargs -I{} stat -c '%Y {}' {} 2>/dev/null \
                 | awk -v t="$started" '$1 >= t' | wc -l)

    echo "  [$(date +%H:%M:%S)] $method finished in ${wall}s (rc0=$rc0 rc1=$rc1, seeds_done=$seeds_done)"
    echo "$method,$wall,$rc,$seeds_done,$(date -Iseconds)" >> "$TIMING_CSV"
}

for method in "${METHODS[@]}"; do
    run_method "$method"
done

echo
echo "=================================================================="
echo "Sweep complete. Timing:"
column -t -s, < "$TIMING_CSV" | head -30
echo
echo "Run aggregation:"
echo "  python scripts/generate_tables.py --datasets GB1 --bonferroni --stat_test wilcoxon"
echo "  python scripts/aggregate_metrics.py --dataset GB1 --seed 621"
