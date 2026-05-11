#!/usr/bin/env python
"""
smoke_test_methods.py — Per-method "does it start?" check on GB1.

Goal: detect import errors, missing files, broken arg paths, and any
configuration-level bug that prevents a method from beginning a run. Stops
each method as soon as it emits a "made progress" line or after `--per-method-timeout`.

A method passes if its run reaches one of these markers within the timeout:
    - any line containing one of the SUCCESS_MARKERS substrings
    - exit code 0 (the method completed)

Failure modes captured:
    - exit code != 0 with traceback
    - timeout with no SUCCESS_MARKER seen (probably hung / waiting for input)

Usage
-----
    python scripts/smoke_test_methods.py --dataset GB1 --seed 42 --gpu-id 0
    python scripts/smoke_test_methods.py --methods ALDE EvoPlay --per-method-timeout 600
"""

from __future__ import annotations
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
HPC_DIR = BENCHMARK_ROOT / "scripts" / "hpc"
sys.path.insert(0, str(HPC_DIR))
from launch import (  # noqa: E402
    load_resources, resolve_method_resources, resolve_python_for_method,
)

DEFAULT_METHODS = [
    "Random", "GreedyWalk",
    "ALDE", "EvoPlay", "FLEXS", "AiCE", "delta_cs",
    "LatProtRL", "alphavariant",
]

# Substrings that indicate the method is past imports and into actual work.
# When any line contains one of these, we declare success and kill the run.
SUCCESS_MARKERS = [
    "Round 1",
    "Round  1",
    "round 1",
    "round=1",
    "round_idx=0",
    "Loading landscape",
    "Loaded landscape",
    "Landscape size",
    "Initial samples",
    "Starting EvoPlay",
    "Starting AlphaVariant",
    "Starting iterative training",
    "Starting AdaLead",
    "Starting AICE",
    "Starting BO",
    "Starting LatProtRL",
    "Starting δ",
    "Starting delta_cs",
    "DNN_ENSEMBLE",
    "Initial training set",
    "training surrogate",
    "Computing evaluation metrics",
    "Saved to:",
    "Done.",
]

ERROR_MARKERS = [
    "Traceback",
    "ModuleNotFoundError",
    "ImportError",
    "FileNotFoundError",
    "AttributeError",
    "TypeError:",
    "ValueError:",
    "AssertionError:",
    "RuntimeError",
    "Killed",
    "FAILED",
]


def smoke_one(method: str, dataset: str, seed: int,
              gpu_id: Optional[str], per_method_timeout: int,
              resources_doc: Dict, log_dir: Path) -> Dict:
    res = resolve_method_resources(method, resources_doc)
    env_rel = res.get("conda_env", "")
    py = resolve_python_for_method(method, env_rel)

    run_subdir = res.get("run_subdir", "")
    base = BENCHMARK_ROOT / method / run_subdir if run_subdir else BENCHMARK_ROOT / method
    run_script = base / f"run_{dataset}.py"
    extra_run_args: List[str] = []
    if not run_script.exists():
        generic = base / "run_generic.py"
        if generic.exists():
            run_script = generic
            extra_run_args = ["--dataset", dataset]
        else:
            return {"method": method, "status": "SKIP_NO_SCRIPT",
                    "reason": f"no run_{dataset}.py or run_generic.py in {base}"}

    # Resolve the env path (absolute or relative)
    if env_rel:
        ep = Path(env_rel).expanduser()
        if not ep.is_absolute():
            ep = BENCHMARK_ROOT / ep
        if not (ep / "bin" / "python").exists():
            return {"method": method, "status": "SKIP_NO_ENV",
                    "reason": f"{ep}/bin/python not found"}

    log_path = log_dir / f"{method}_{dataset}_seed{seed}.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # critical: stream stdout so we can scan
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Honor run_subdir for methods like delta_cs whose code lives nested.
    work_dir = BENCHMARK_ROOT / method
    if run_subdir:
        work_dir = work_dir / run_subdir
    env["PYTHONPATH"] = f"{work_dir}{os.pathsep}{env.get('PYTHONPATH','')}"

    cmd = [py, "-u", str(run_script), *extra_run_args, "--seed", str(seed),
           "--skip_metrics"]
    print(f"\n=== {method} on {dataset} (seed {seed}) ===")
    print("  $", " ".join(shlex.quote(c) for c in cmd))
    print(f"  cwd: {work_dir}")
    print(f"  log: {log_path}")

    proc = subprocess.Popen(
        cmd, cwd=str(work_dir), env=env,
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
    )

    start = time.time()
    success_seen = False
    error_seen = False
    error_text = ""
    last_offset = 0

    try:
        while True:
            elapsed = time.time() - start
            # Drain the log
            with open(log_path) as f:
                f.seek(last_offset)
                new = f.read()
                last_offset = f.tell()
            if new:
                for line in new.splitlines():
                    if any(m in line for m in SUCCESS_MARKERS):
                        if not success_seen:
                            print(f"  ✓ SUCCESS marker @ {elapsed:.1f}s: "
                                  f"{line.strip()[:100]}")
                        success_seen = True
                    if any(m in line for m in ERROR_MARKERS):
                        error_seen = True
                        error_text = line.strip()[:200]
                        print(f"  ✗ ERROR @ {elapsed:.1f}s: {error_text}")

            rc = proc.poll()
            if rc is not None:
                wall = time.time() - start
                if rc == 0:
                    return {
                        "method": method, "status": "OK_COMPLETED",
                        "exit_code": 0, "wall_seconds": wall,
                        "log": str(log_path),
                    }
                # Re-scan to capture last error chunk
                with open(log_path) as f:
                    tail = f.read().splitlines()[-30:]
                err_summary = next(
                    (ln for ln in tail if any(m in ln for m in ERROR_MARKERS)),
                    f"exit code {rc}",
                )
                return {
                    "method": method, "status": f"FAILED_EXIT_{rc}",
                    "exit_code": rc, "wall_seconds": wall,
                    "error": err_summary[:200], "log": str(log_path),
                }

            # If we saw a success marker, we can stop the slow methods early
            if success_seen and elapsed > 30:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return {
                    "method": method, "status": "OK_PROGRESSING",
                    "exit_code": None, "wall_seconds": elapsed,
                    "note": "stopped early after seeing progress markers",
                    "log": str(log_path),
                }

            if elapsed > per_method_timeout:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return {
                    "method": method,
                    "status": "TIMEOUT" if not error_seen else "FAILED_ERROR",
                    "wall_seconds": elapsed,
                    "error": error_text,
                    "saw_success_marker": success_seen,
                    "log": str(log_path),
                }

            time.sleep(0.5)
    finally:
        if proc.poll() is None:
            proc.terminate()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="GB1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--gpu-id", default=None)
    p.add_argument("--per-method-timeout", type=int, default=180,
                   help="Wall-clock cap per method (default 180s)")
    p.add_argument("--log-dir",
                   default=str(BENCHMARK_ROOT / "results" / "_smoke_logs"))
    p.add_argument("--summary",
                   default=str(BENCHMARK_ROOT / "results" / "_smoke_summary.json"))
    args = p.parse_args()

    methods = args.methods or DEFAULT_METHODS
    log_dir = Path(args.log_dir)
    resources_doc = load_resources()

    results: List[Dict] = []
    for method in methods:
        rec = smoke_one(
            method, args.dataset, args.seed,
            args.gpu_id, args.per_method_timeout,
            resources_doc, log_dir,
        )
        results.append(rec)
        print(f"  -> {rec['status']}")

    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary written to {args.summary}\n")

    print(f"  {'method':<14}{'status':<22}{'wall(s)':>10}  notes")
    print("  " + "-" * 70)
    for r in results:
        wall = r.get("wall_seconds")
        wall_s = f"{wall:.1f}" if wall is not None else "—"
        note = r.get("error") or r.get("reason") or r.get("note") or ""
        print(f"  {r['method']:<14}{r['status']:<22}{wall_s:>10}  {note[:60]}")

    n_ok = sum(1 for r in results
               if r["status"] in ("OK_COMPLETED", "OK_PROGRESSING"))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
