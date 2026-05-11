#!/usr/bin/env python
"""
profile_methods.py — Measure each method's per-seed wall-clock and GPU footprint.

Runs `<method>/run_<dataset>.py --seed <N>` once per method, wraps with
`scripts/hpc/log_resource_use.py`, and aggregates the resulting `resource.json`
files into `per_method_walltime.csv`. Use the CSV to size HPC budgets.

Each method is invoked with **its own conda env's python**, resolved from
`scripts/hpc/method_resources.yaml`. Methods missing their env are skipped
with a clear note (no silent fallback to host python — that picks up the
wrong torch/scipy).

Usage
-----
    # Profile every configured method on GB1 with seed 42
    python scripts/profile_methods.py --dataset GB1 --seed 42

    # Specific subset
    python scripts/profile_methods.py --dataset GB1 --seed 42 \\
        --methods Random GreedyWalk ALDE

    # Pin to GPU 0
    python scripts/profile_methods.py --dataset GB1 --seed 42 --gpu-id 0

    # Skip methods that take too long (e.g., AlphaVariant)
    python scripts/profile_methods.py --dataset GB1 --seed 42 --timeout 1800

    # Re-aggregate existing resource.json files without rerunning
    python scripts/profile_methods.py --aggregate-only
"""

from __future__ import annotations
import argparse
import csv
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
PROFILE_DIR = BENCHMARK_ROOT / "results" / "_profiles"
LOGGER = HPC_DIR / "log_resource_use.py"
LAUNCH_PY = HPC_DIR / "launch.py"

sys.path.insert(0, str(HPC_DIR))
from launch import (  # noqa: E402
    load_resources,
    resolve_method_resources,
    resolve_python_for_method,
)


DEFAULT_METHODS = [
    "Random", "GreedyWalk",
    "ALDE", "EvoPlay", "FLEXS", "AiCE", "delta_cs",
    "LatProtRL", "alphavariant",
    # Phase 2.1 baselines (will exit code 2 — captured as such in CSV):
    "EVOLVEpro", "ftMLDE", "MULTIevolve",
]


def profile_one(method: str, dataset: str, seed: int,
                gpu_id: Optional[str], timeout: Optional[int],
                profile_dir: Path,
                resources_doc: Dict) -> Dict:
    res = resolve_method_resources(method, resources_doc)
    env_rel = res.get("conda_env", "")
    py = resolve_python_for_method(method, env_rel)

    run_script = BENCHMARK_ROOT / method / f"run_{dataset}.py"
    extra_run_args: List[str] = []
    if not run_script.exists():
        # Fall back to run_generic.py (established convention for Phase 2.1
        # baselines: EVOLVEpro, ftMLDE, MULTIevolve).
        generic = BENCHMARK_ROOT / method / "run_generic.py"
        if generic.exists():
            run_script = generic
            extra_run_args = ["--dataset", dataset]
        else:
            return {
                "method": method, "dataset": dataset, "seed": seed,
                "status": "SKIP_NO_SCRIPT",
                "reason": f"no run_{dataset}.py or run_generic.py",
            }

    # Verify the env's python actually exists; otherwise we'd silently use sys.executable
    if env_rel:
        env_path = Path(env_rel).expanduser()
        if not env_path.is_absolute():
            env_path = BENCHMARK_ROOT / env_path
        if not (env_path / "bin" / "python").exists():
            return {
                "method": method, "dataset": dataset, "seed": seed,
                "status": "SKIP_NO_ENV",
                "reason": f"{env_path}/bin/python not found",
            }

    profile_dir.mkdir(parents=True, exist_ok=True)
    resource_json = profile_dir / f"{method}_{dataset}_seed{seed}.json"

    cmd = [
        sys.executable, str(LOGGER),
        "--output-json", str(resource_json),
        "--label", f"{method}/{dataset}/seed{seed}",
        "--",
        py, str(run_script),
        *extra_run_args,
        "--seed", str(seed),
        "--skip_metrics",
    ]

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Symlinked run scripts live in scripts/<method>/, so Python's automatic
    # sys.path[0] doesn't include <method>/. Add it explicitly so each method's
    # `from src.foo import ...` resolves to <method>/src/.
    method_dir = str(BENCHMARK_ROOT / method)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{method_dir}{os.pathsep}{existing_pp}" if existing_pp else method_dir
    )

    print(f"\n=== {method} on {dataset} (seed {seed}) ===")
    print("  $", " ".join(shlex.quote(c) for c in cmd))
    if gpu_id is not None:
        print(f"    CUDA_VISIBLE_DEVICES={gpu_id}")

    t0 = time.time()
    try:
        rc = subprocess.run(
            cmd,
            cwd=str(BENCHMARK_ROOT / method),
            env=env,
            timeout=timeout,
        ).returncode
        wall = time.time() - t0
        timed_out = False
    except subprocess.TimeoutExpired:
        wall = float(timeout)
        rc = 124
        timed_out = True
        print(f"  TIMEOUT after {wall:.1f}s")

    record = {
        "method": method, "dataset": dataset, "seed": seed,
        "status": ("TIMEOUT" if timed_out
                   else "OK" if rc == 0
                   else f"EXIT_{rc}"),
        "wall_seconds": wall,
        "wall_minutes": wall / 60.0,
        "exit_code": rc,
        "conda_env": env_rel,
        "resource_json": str(resource_json),
    }
    # Pull richer fields from the wrapper's JSON if present
    if resource_json.exists():
        try:
            with open(resource_json) as f:
                rj = json.load(f)
            record["peak_rss_mib"] = rj.get("peak_rss_mib")
            record["max_gpu_memory_mib"] = rj.get("max_gpu_memory_mib")
        except Exception:
            pass
    return record


def aggregate(profile_dir: Path, csv_path: Path) -> None:
    rows: List[Dict] = []
    for path in sorted(profile_dir.glob("*.json")):
        try:
            rj = json.load(open(path))
        except Exception:
            continue
        # Try to recover method/dataset/seed from filename if not in JSON
        name = path.stem  # METHOD_DATASET_seedN
        try:
            method, dataset, seed_part = name.rsplit("_", 2)
            seed = int(seed_part.replace("seed", ""))
        except ValueError:
            method, dataset, seed = name, "?", -1
        rows.append({
            "method": method,
            "dataset": dataset,
            "seed": seed,
            "wall_seconds": rj.get("wall_seconds"),
            "wall_minutes": (rj.get("wall_seconds") or 0) / 60.0,
            "wall_hours": rj.get("wall_hours"),
            "exit_code": rj.get("exit_code"),
            "peak_rss_mib": rj.get("peak_rss_mib"),
            "max_gpu_memory_mib": rj.get("max_gpu_memory_mib"),
            "label": rj.get("label"),
            "command": " ".join(rj.get("command") or []),
            "started_utc": rj.get("started_utc"),
            "finished_utc": rj.get("finished_utc"),
        })

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"No resource.json files in {profile_dir}; CSV not written.")
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows)")
    # Pretty summary
    print(f"\n{'method':<14}{'dataset':<10}{'seed':>6}{'wall(s)':>10}{'wall(min)':>10}"
          f"{'exit':>6}{'GPU MiB':>10}")
    print("-" * 70)
    for r in rows:
        print(f"{r['method']:<14}{r['dataset']:<10}{r['seed']:>6}"
              f"{(r['wall_seconds'] or 0):>10.1f}"
              f"{(r['wall_minutes'] or 0):>10.2f}"
              f"{(r['exit_code'] if r['exit_code'] is not None else '?'):>6}"
              f"{(r['max_gpu_memory_mib'] or 0):>10.0f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="GB1",
                   help="Dataset to profile on (default: GB1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--methods", nargs="+", default=None,
                   help=f"Subset of methods (default: {DEFAULT_METHODS})")
    p.add_argument("--gpu-id", default=None,
                   help="Set CUDA_VISIBLE_DEVICES")
    p.add_argument("--timeout", type=int, default=None,
                   help="Wall-clock timeout per method (seconds)")
    p.add_argument("--profile-dir", default=str(PROFILE_DIR),
                   help="Where to write per-method resource.json files")
    p.add_argument("--csv", default=str(PROFILE_DIR / "per_method_walltime.csv"),
                   help="Aggregated CSV output path")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Skip profiling; just rebuild the CSV from existing JSONs")
    args = p.parse_args()

    profile_dir = Path(args.profile_dir)
    csv_path = Path(args.csv)

    if not args.aggregate_only:
        methods = args.methods or DEFAULT_METHODS
        resources_doc = load_resources()

        for method in methods:
            rec = profile_one(
                method, args.dataset, args.seed,
                args.gpu_id, args.timeout,
                profile_dir, resources_doc,
            )
            print(f"  -> {rec['status']}, "
                  f"wall={rec.get('wall_seconds', 0):.1f}s")

    aggregate(profile_dir, csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
