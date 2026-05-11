#!/usr/bin/env python
"""
log_resource_use.py — Wrap a benchmark run with resource tracking.

Records wall-clock time, peak RSS, and (if GPU available) max GPU memory and
seconds-of-GPU-use into a JSON next to the run's output.

Usage
-----
    python scripts/hpc/log_resource_use.py \\
        --output-json results/ALDE/GB1/seed42_resource.json \\
        -- python ALDE/run_GB1.py --seed 42

The double-dash separates this script's args from the wrapped command.
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone


def _max_gpu_memory_mib() -> float:
    """Best-effort: query nvidia-smi for max memory used since last reset."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return max(float(x) for x in out.decode().split() if x.strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return 0.0


def _peak_rss_mib() -> float:
    try:
        import resource
        # ru_maxrss is KB on Linux, bytes on macOS
        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        if sys.platform == "darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except Exception:
        return 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-json", required=True,
                   help="Path to write resource.json (parent created if needed)")
    p.add_argument("--label", default=None,
                   help="Free-text label stored in the JSON")
    args, rest = p.parse_known_args()

    if not rest or rest[0] != "--":
        # Allow either explicit '--' or just trailing args
        cmd = rest
    else:
        cmd = rest[1:]

    if not cmd:
        print("No command provided to wrap.", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".",
                exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    proc = subprocess.run(cmd)
    wall = time.time() - t0
    finished = datetime.now(timezone.utc).isoformat()

    record = {
        "label": args.label,
        "command": cmd,
        "started_utc": started,
        "finished_utc": finished,
        "wall_seconds": wall,
        "wall_hours": wall / 3600.0,
        "exit_code": proc.returncode,
        "peak_rss_mib": _peak_rss_mib(),
        "max_gpu_memory_mib": _max_gpu_memory_mib(),
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }

    with open(args.output_json, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[log_resource_use] wrote {args.output_json} (wall={wall:.1f}s)")

    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
