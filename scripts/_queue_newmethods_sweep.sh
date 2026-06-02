#!/bin/bash
# Queue the 4 new methods' 30-seed sweep to start AFTER the running 5-baseline
# 30-seed sweep finishes. AdaLead/MULTIevolve/AiCE use ALDE/env; EVOLVEpro uses
# alphavariant-env (only env with transformers/ESM).
set -u
cd /home/xux/Desktop/AlphaVariant/Benchmark
mkdir -p results_oracle/_logs

BPID="$1"; shift
SEEDS="$*"
ALDE_PY=ALDE/env/bin/python
AV_PY=/home/xux/miniforge3/envs/alphavariant-env/bin/python

echo "[queue] waiting for baseline sweep (PID $BPID) to finish..."
while kill -0 "$BPID" 2>/dev/null; do sleep 60; done
echo "[queue] baseline finished; starting new-methods sweep at $(date +%H:%M:%S)"

for d in ms_AAV ms_PAB1 ms_CreiLOV ms_GFP; do
  for m in AdaLead MULTIevolve AiCE; do
    $ALDE_PY scripts/run_oracle_benchmark.py --method $m --dataset $d \
      --seeds $SEEDS --device cuda:0 >> results_oracle/_logs/new_${d}_${m}.log 2>&1
    echo "[queue] DONE $d/$m ($(ls results_oracle/$d/$m/seed*.json 2>/dev/null|wc -l) files)"
  done
  $AV_PY scripts/run_oracle_benchmark.py --method EVOLVEpro --dataset $d \
    --seeds $SEEDS --device cuda:0 >> results_oracle/_logs/new_${d}_EVOLVEpro.log 2>&1
  echo "[queue] DONE $d/EVOLVEpro ($(ls results_oracle/$d/EVOLVEpro/seed*.json 2>/dev/null|wc -l) files)"
done
echo "[queue] NEWMETHODS_SWEEP_COMPLETE at $(date +%H:%M:%S)"
