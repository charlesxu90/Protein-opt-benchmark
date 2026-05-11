#!/bin/bash
# env_setup.sh - Cluster-aware environment setup for AlphaVariant Benchmark
#
# Usage (sourced by sbatch templates):
#   source scripts/hpc/env_setup.sh <cluster> <conda_env_relpath>
#
#   cluster: "ibex" | "shaheen" | "local"
#   conda_env_relpath: e.g. "ALDE/env" relative to BENCHMARK_ROOT

set -euo pipefail

CLUSTER="${1:-local}"
ENV_REL="${2:-}"

# Resolve benchmark root (this file lives in scripts/hpc/)
BENCHMARK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export BENCHMARK_ROOT
export PYTHONPATH="$BENCHMARK_ROOT:${PYTHONPATH:-}"

case "$CLUSTER" in
  ibex)
    # KAUST iBex: load standard modules
    module purge 2>/dev/null || true
    module load gcc/12.2.0 2>/dev/null || true
    module load cuda/12.2 2>/dev/null || true
    ;;
  shaheen)
    # KAUST Shaheen III (Cray): swap programming env, load python toolchain
    module purge 2>/dev/null || true
    module load PrgEnv-gnu 2>/dev/null || true
    module load cray-python 2>/dev/null || true
    module load cudatoolkit 2>/dev/null || true
    ;;
  local)
    : # nothing
    ;;
  *)
    echo "Unknown cluster: $CLUSTER" >&2
    exit 1
    ;;
esac

# Activate per-method conda env if provided
if [[ -n "$ENV_REL" ]]; then
  ENV_PATH="$BENCHMARK_ROOT/$ENV_REL"
  if [[ -x "$ENV_PATH/bin/python" ]]; then
    export PATH="$ENV_PATH/bin:$PATH"
    export VIRTUAL_ENV="$ENV_PATH"
    echo "Activated env: $ENV_PATH"
  else
    echo "WARNING: env not found or has no python: $ENV_PATH" >&2
  fi
fi

echo "BENCHMARK_ROOT=$BENCHMARK_ROOT"
echo "PYTHON=$(command -v python)"
