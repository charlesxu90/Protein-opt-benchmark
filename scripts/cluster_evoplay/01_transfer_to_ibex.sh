#!/usr/bin/env bash
# 01_transfer_to_ibex.sh — push benchmark + EvoPlay env from local workstation to iBex.
# Run from local workstation, NOT iBex.
#
# Usage:
#   bash scripts/cluster_evoplay/01_transfer_to_ibex.sh ibex.kaust.edu.sa
#   bash scripts/cluster_evoplay/01_transfer_to_ibex.sh user@ibex.kaust.edu.sa ~/Benchmark
#
# Arguments:
#   $1 = ssh target (default: ibex.kaust.edu.sa)
#   $2 = remote benchmark root (default: ~/Benchmark)
set -euo pipefail

REMOTE="${1:-ibex.kaust.edu.sa}"
REMOTE_DIR="${2:-~/Benchmark}"
LOCAL_DIR="/home/xux/Desktop/AlphaVariant/Benchmark"

echo "[transfer] target=${REMOTE}:${REMOTE_DIR}"
ssh "$REMOTE" "mkdir -p ${REMOTE_DIR}"

# (1) Code, data, seeds — small/fast
echo "[transfer] code + data + seeds ..."
rsync -avz --progress \
  --exclude='*/env/' \
  --exclude='*/results/' \
  --exclude='sweep_logs/' \
  --exclude='Mu-Protein/pretrained/' \
  --exclude='__pycache__' --exclude='.git' \
  "${LOCAL_DIR}/" "${REMOTE}:${REMOTE_DIR}/"

# (2) EvoPlay env — heavy (~3-5 GB conda env)
echo "[transfer] EvoPlay/env ..."
rsync -avz --progress "${LOCAL_DIR}/EvoPlay/env/" "${REMOTE}:${REMOTE_DIR}/EvoPlay/env/"

# (3) Run add_script_link.sh on remote to symlink the run scripts
echo "[transfer] running add_script_link.sh on remote ..."
ssh "$REMOTE" "cd ${REMOTE_DIR} && bash scripts/add_script_link.sh"

echo "[transfer] DONE"
echo ""
echo "Next: ssh ${REMOTE} && cd ${REMOTE_DIR} && bash scripts/cluster_evoplay/02_submit_array.sh"
