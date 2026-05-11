#!/bin/bash
# setup_baseline_envs.sh — build EVOLVEpro / ftMLDE / MULTIevolve conda envs.
#
# Each upstream repo ships its own environment.yml. This script materialises
# them under <method>/env/ matching the convention used by the existing six
# methods (so scripts/hpc/method_resources.yaml needs no override).
#
# Usage:
#   bash scripts/setup_baseline_envs.sh                # all three
#   bash scripts/setup_baseline_envs.sh EVOLVEpro      # subset
#
# Notes:
#   - Each env build takes 10-40 minutes on a fast box (mlde uses TF1.x and
#     pinned old packages — slowest).
#   - Run from a host that has internet + conda/mamba. Mamba is preferred:
#     export CONDA_BIN=mamba.
#   - If you already have an env elsewhere, point the YAML at it via
#     `conda_env: <absolute path>` in scripts/hpc/method_resources.yaml.

set -euo pipefail

CONDA_BIN="${CONDA_BIN:-conda}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ALL=(EVOLVEpro ftMLDE MULTIevolve)
TARGETS=("${@:-${ALL[@]}}")

build_one() {
    local method="$1"
    local repo="$ROOT/$method"
    local env_dir="$repo/env"
    if [[ ! -d "$repo" ]]; then
        echo "[skip] $repo not found — clone it first." >&2
        return 1
    fi
    if [[ -x "$env_dir/bin/python" ]]; then
        echo "[skip] $env_dir already has bin/python"
        return 0
    fi

    case "$method" in
        EVOLVEpro)
            local yml="$repo/environment.yml"
            ;;
        ftMLDE)
            # mlde.yml has tensorflow-gpu 1.13 (CUDA 10) — known fragile on
            # modern GPUs. mlde2.yml is functionally equivalent for the
            # parts we use; prefer it. Override with FTMLDE_YML=mlde.yml.
            local yml="$repo/${FTMLDE_YML:-mlde2.yml}"
            ;;
        MULTIevolve)
            local yml="$repo/env.yml"
            ;;
        *)
            echo "[fail] unknown method: $method" >&2
            return 1
            ;;
    esac

    if [[ ! -f "$yml" ]]; then
        echo "[fail] env yml missing: $yml" >&2
        return 1
    fi

    echo "=== $method ==="
    echo "    yml:    $yml"
    echo "    target: $env_dir"
    "$CONDA_BIN" env create -p "$env_dir" -f "$yml"

    # MULTIevolve and EVOLVEpro have setup.py — install in editable mode
    if [[ -f "$repo/setup.py" || -f "$repo/pyproject.toml" ]]; then
        echo "    pip install -e $repo"
        "$env_dir/bin/pip" install -e "$repo" || \
            echo "    [warn] editable install failed; falling back to non-editable" >&2
    fi

    echo "=== $method DONE ==="
    "$env_dir/bin/python" --version
}

failures=()
for m in "${TARGETS[@]}"; do
    if ! build_one "$m"; then
        failures+=("$m")
    fi
done

echo
if [[ ${#failures[@]} -gt 0 ]]; then
    echo "FAILED: ${failures[*]}" >&2
    exit 1
fi
echo "All requested envs built. Verify with:"
echo "  python scripts/profile_methods.py --dataset GB1 --seed 42 \\"
echo "      --methods ${TARGETS[*]} --timeout 600"
