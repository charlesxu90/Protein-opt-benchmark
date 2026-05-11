#!/usr/bin/env python
"""
run_sweep_parallel.py — Multi-method, multi-seed sweep with per-GPU concurrency.

Issues with the older shell orchestrator (`run_30seed_gb1_sweep.sh`):
    - Each method runs ONLY 2 processes (one per GPU), so a single A100 is
      idle except for one method's seed at a time. Two A100s + 112 CPU cores
      were heavily underutilized.

This orchestrator runs N independent processes per GPU AND lets CPU-only
methods stream alongside the GPU work. The unit of scheduling is a *single
seed*, dispatched via `launch.py --seeds 1`. A `ThreadPoolExecutor` keeps
the configured number of workers busy.

Concurrency policy
------------------
Per `--workers-per-gpu`, this script launches that many simultaneous seed
runs per GPU. CPU-only methods (Random, GreedyWalk) bypass GPU assignment and
use a separate `--cpu-workers` pool that runs alongside.

GPU memory budgets per concurrent seed (A100-40GB, empirical):
    Random / GreedyWalk : 0 GB     (CPU only)
    ALDE                : ~2 GB    → 4-6 per GPU OK
    AiCE                : ~6 GB    → 2-3 per GPU OK (ESM-2 + ProteinMPNN)
    FLEXS               : ~2 GB    → 4 per GPU OK
    delta_cs            : ~4 GB    → 4 per GPU OK
    EvoPlay             : ~2 GB    → 4 per GPU OK
    alphavariant        : ~6 GB    → 3-4 per GPU OK
    LatProtRL           : ~15 GB   → 2 per GPU max (PPO + ESM-2)

Defaults in METHOD_DEFAULTS reflect these.

Usage
-----
    # 30 seeds × all default methods on 2 A100s
    python scripts/run_sweep_parallel.py --dataset GB1 --seeds 30

    # Subset
    python scripts/run_sweep_parallel.py --dataset GB1 --seeds 30 \\
        --methods ALDE AiCE FLEXS

    # Override workers
    python scripts/run_sweep_parallel.py --dataset GB1 --seeds 30 \\
        --methods EvoPlay --workers-per-gpu 6
"""

from __future__ import annotations
import argparse
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_METHODS = [
    "Random", "GreedyWalk",
    "AiCE", "ALDE", "FLEXS",
    "delta_cs", "alphavariant",
    "EvoPlay",
]

# Per-method concurrency. `gpu` is workers per GPU; `cpu_only` runs in the
# CPU pool. `gpu=None` → fall back to --workers-per-gpu default.
METHOD_DEFAULTS: Dict[str, Dict] = {
    "Random":       {"gpu": 0, "cpu_only": True},
    "GreedyWalk":   {"gpu": 0, "cpu_only": True},
    "ALDE":         {"gpu": 4, "cpu_only": False},
    "AiCE":         {"gpu": 2, "cpu_only": False},
    "FLEXS":        {"gpu": 4, "cpu_only": False},
    "delta_cs":     {"gpu": 3, "cpu_only": False},
    "EvoPlay":      {"gpu": 4, "cpu_only": False},
    "alphavariant": {"gpu": 3, "cpu_only": False},
    "LatProtRL":    {"gpu": 2, "cpu_only": False},
}


def load_seeds(seed_file: Path, n: int) -> List[int]:
    seeds = []
    with open(seed_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            seeds.append(int(line))
            if len(seeds) >= n:
                break
    return seeds


def run_one_seed(method: str, dataset: str, seed: int,
                 gpu_id: Optional[str], log_dir: Path,
                 extra_args: List[str]) -> Dict:
    """Run a single seed via launch.py. Returns dict with status + timing."""
    log_path = log_dir / f"{method}_seed{seed}.log"
    cmd = [
        sys.executable, str(BENCHMARK_ROOT / "scripts" / "hpc" / "launch.py"),
        "--method", method,
        "--dataset", dataset,
        "--seeds", "1",
        "--seed-file", str(BENCHMARK_ROOT / "rand_seeds.txt"),
        "--cluster", "local",
        # --use-method-env is on by default in launch.py
    ]
    # Compute seed-start index in rand_seeds.txt for this specific seed
    # (launch.py expects --seed-start N --seeds 1 to pick the Nth entry)
    seeds_file = BENCHMARK_ROOT / "rand_seeds.txt"
    seed_idx = _index_of_seed(seeds_file, seed)
    if seed_idx is None:
        return {"method": method, "seed": seed, "status": "SEED_NOT_IN_FILE",
                "wall_seconds": 0.0, "exit_code": -1, "log": str(log_path)}
    cmd += ["--seed-start", str(seed_idx)]
    if gpu_id is not None:
        cmd += ["--gpu-id", str(gpu_id)]
    if extra_args:
        # Use --extra-args=VALUE to prevent argparse from treating "--use_gpu"
        # (which starts with --) as a new flag for the orchestrator's argparse.
        cmd += ["--extra-args=" + " ".join(shlex.quote(a) for a in extra_args)]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=str(BENCHMARK_ROOT), env=env,
                              stdout=f, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    return {
        "method": method, "seed": seed,
        "status": "OK" if proc.returncode == 0 else f"FAIL_{proc.returncode}",
        "wall_seconds": wall,
        "exit_code": proc.returncode,
        "gpu_id": gpu_id,
        "log": str(log_path),
    }


def _index_of_seed(seed_file: Path, seed: int) -> Optional[int]:
    """Return the 0-indexed position of `seed` in seed_file (skipping blanks)."""
    idx = 0
    with open(seed_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if int(line) == seed:
                return idx
            idx += 1
    return None


def run_method_parallel(method: str, dataset: str, seeds: List[int],
                        n_workers_gpu: int, gpus: List[str],
                        cpu_only: bool, n_cpu_workers: int,
                        log_dir: Path, extra_args: List[str]) -> List[Dict]:
    """Submit `seeds` for `method` to a thread pool with the configured concurrency.

    If `cpu_only`, no GPU is assigned and the CPU pool size is used. Otherwise
    seeds are round-robin assigned to GPUs with `n_workers_gpu` slots each.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict] = []
    if cpu_only:
        max_workers = n_cpu_workers
        gpu_assign = [None] * len(seeds)
    else:
        n_gpus = len(gpus)
        max_workers = max(1, n_workers_gpu * n_gpus)
        gpu_assign = [gpus[i % n_gpus] for i in range(len(seeds))]

    started = time.time()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {method} → "
          f"{len(seeds)} seeds, {max_workers} concurrent "
          f"({'CPU only' if cpu_only else f'{n_workers_gpu}/GPU × {len(gpus)} GPUs'})")

    n_done_lock = threading.Lock()
    n_done = [0]
    n_total = len(seeds)

    def task(seed, gpu_id):
        rec = run_one_seed(method, dataset, seed, gpu_id, log_dir, extra_args)
        with n_done_lock:
            n_done[0] += 1
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                  f"{method} seed={seed} {rec['status']} "
                  f"wall={rec['wall_seconds']:.1f}s "
                  f"({n_done[0]}/{n_total}) gpu={gpu_id}")
        return rec

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(task, s, g) for s, g in zip(seeds, gpu_assign)]
        for f in as_completed(futures):
            results.append(f.result())

    method_wall = time.time() - started
    n_ok = sum(1 for r in results if r["status"] == "OK")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {method} complete: "
          f"{n_ok}/{n_total} OK, total wall={method_wall:.1f}s")
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seeds", type=int, default=30,
                   help="Number of seeds (taken from the top of rand_seeds.txt)")
    p.add_argument("--methods", nargs="+", default=None,
                   help=f"Subset (default: {DEFAULT_METHODS})")
    p.add_argument("--workers-per-gpu", type=int, default=None,
                   help="Override per-method GPU concurrency (default per METHOD_DEFAULTS)")
    p.add_argument("--cpu-workers", type=int, default=8,
                   help="Concurrency for CPU-only methods (default 8)")
    p.add_argument("--gpus", nargs="+", default=["0", "1"],
                   help="GPU IDs to use (default 0 1)")
    p.add_argument("--log-dir", default=None,
                   help="Default: scripts/hpc/_logs/sweep_par_<DATE>")
    p.add_argument("--extra-args", default="",
                   help="Forwarded to every run_<dataset>.py")
    args = p.parse_args()

    methods = args.methods or DEFAULT_METHODS
    extra_args = shlex.split(args.extra_args) if args.extra_args else []

    log_dir = (Path(args.log_dir) if args.log_dir
               else BENCHMARK_ROOT / "scripts" / "hpc" / "_logs"
                    / f"sweep_par_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    log_dir.mkdir(parents=True, exist_ok=True)

    seeds = load_seeds(BENCHMARK_ROOT / "rand_seeds.txt", args.seeds)
    if len(seeds) < args.seeds:
        print(f"WARNING: only {len(seeds)} seeds available", file=sys.stderr)

    summary_csv = log_dir / "sweep_results.csv"
    with open(summary_csv, "w") as f:
        f.write("method,seed,status,wall_seconds,exit_code,gpu_id\n")

    print(f"\n{'='*68}\nParallel GB1 sweep")
    print(f"  dataset:  {args.dataset}")
    print(f"  seeds:    {len(seeds)} ({seeds[0]}..{seeds[-1]})")
    print(f"  methods:  {methods}")
    print(f"  gpus:     {args.gpus}")
    print(f"  log dir:  {log_dir}")
    print(f"{'='*68}\n")

    all_results: Dict[str, List[Dict]] = {}
    for method in methods:
        cfg = METHOD_DEFAULTS.get(method, {})
        cpu_only = cfg.get("cpu_only", False)
        n_workers_gpu = args.workers_per_gpu or cfg.get("gpu", 2)
        results = run_method_parallel(
            method, args.dataset, seeds, n_workers_gpu, args.gpus,
            cpu_only, args.cpu_workers, log_dir, extra_args,
        )
        all_results[method] = results
        with open(summary_csv, "a") as f:
            for r in results:
                f.write(f"{r['method']},{r['seed']},{r['status']},"
                        f"{r['wall_seconds']:.1f},{r['exit_code']},"
                        f"{r.get('gpu_id','')}\n")

    # Final summary
    print(f"\n{'='*68}\nDone. Per-method success counts:")
    for method, results in all_results.items():
        n_ok = sum(1 for r in results if r["status"] == "OK")
        total_wall = sum(r["wall_seconds"] for r in results)
        max_wall = max((r["wall_seconds"] for r in results), default=0.0)
        print(f"  {method:<14} {n_ok}/{len(results)} OK, "
              f"total wall (sum)={total_wall:.0f}s, max-seed={max_wall:.0f}s")
    print(f"\nResults CSV: {summary_csv}")
    print(f"Re-aggregate metrics: python scripts/aggregate_metrics.py --dataset "
          f"{args.dataset} --seed {seeds[0]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
