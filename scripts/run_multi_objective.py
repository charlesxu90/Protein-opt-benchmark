#!/usr/bin/env python
"""
run_multi_objective.py — Task 3 (multi-objective Pareto) orchestrator.

Wraps a single-objective optimization run with bi-objective evaluation. Given
a multi-property dataset (eqFP611 with `blue` + `red`, or any custom 2-column
landscape), this script:

    1. Loads both objective columns from data/<dataset>/data.csv.
    2. Computes the *true* Pareto front from the full landscape (one-time).
    3. Runs the requested method on a chosen scalarization (weighted sum,
       Chebyshev, or per-objective independent runs).
    4. Reads back the queried indices and computes Hypervolume + Pareto-front
       coverage of the discovered set.

Two scalarization modes are supported:
    weighted_sum  — R = α*obj1 + (1-α)*obj2; runs once per α value.
    independent   — Run each objective separately, then combine queried sets
                    for joint Pareto evaluation. Matches "single-objective per
                    objective" baseline used in ALDE.

Usage
-----
    # Weighted-sum sweep on eqFP611 (alpha grid)
    python scripts/run_multi_objective.py --dataset eqFP611 --method Random \\
        --seed 42 --mode weighted_sum --alphas 0.0 0.25 0.5 0.75 1.0

    # Independent per-objective baseline
    python scripts/run_multi_objective.py --dataset eqFP611 --method Random \\
        --seed 42 --mode independent

Outputs
-------
results/<method>/<dataset>_multi_objective/seed{S}/{
    indices_alpha{a}.npy,           # queried indices per alpha (weighted_sum)
    pareto_front_true.npy,          # the reference Pareto front
    pareto_metrics.json,            # hypervolume + coverage per alpha
}
"""

from __future__ import annotations
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCHMARK_ROOT))

from utils.multi_objective import (
    pareto_front,
    hypervolume,
    pareto_front_coverage,
    auto_reference_point,
)


def load_multi_property(dataset: str, data_dir: Path,
                        objectives: List[str]) -> Tuple[List[str], np.ndarray]:
    """Load (sequences, objective matrix shape (N, len(objectives)))."""
    csv = data_dir / dataset / "data.csv"
    if not csv.exists():
        raise FileNotFoundError(f"No multi-property file at {csv}")
    df = pd.read_csv(csv)
    seq_col = next(c for c in ("seq", "sequence", "genotype")
                   if c in df.columns)
    missing = [o for o in objectives if o not in df.columns]
    if missing:
        raise ValueError(
            f"Objectives {missing} not in {csv}. Columns: {list(df.columns)}"
        )
    seqs = df[seq_col].tolist()
    obj = df[objectives].astype(float).values  # (N, M)
    return seqs, obj


def write_scalar_landscape(dataset: str, seqs: List[str], values: np.ndarray,
                           tmp_dir: Path, alpha_label: str) -> Path:
    """Write a temporary single-objective data.csv for this scalarization.

    The downstream run_<dataset>.py expects `data/<dataset>/data.csv`. We write
    a sibling directory `data/<dataset>__alpha<...>/data.csv` and pass it via
    `--data_dir` (which the run scripts honor).
    """
    sub_name = f"{dataset}__alpha{alpha_label}"
    sub = tmp_dir / sub_name
    sub.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"seq": seqs, "fitness": values}).to_csv(
        sub / "data.csv", index=False
    )
    return tmp_dir, sub_name


def run_method_on_scalar(method: str, dataset_alias: str, seed: int,
                         tmp_data_dir: Path, output_path: Path) -> int:
    """Invoke <method>/run_generic.py --dataset <dataset_alias>.

    All methods accept --dataset, --seed, --data_dir, --output_path. We use the
    generic runner so this works for any method without per-dataset wrappers.
    """
    run_script = BENCHMARK_ROOT / method / "run_generic.py"
    if not run_script.exists():
        print(f"No run_generic.py for {method} at {run_script}", file=sys.stderr)
        return 1
    cmd = [
        sys.executable, str(run_script),
        "--dataset", dataset_alias,
        "--seed", str(seed),
        "--data_dir", str(tmp_data_dir),
        "--output_path", str(output_path),
        "--skip_metrics",
    ]
    print("  $", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, cwd=str(BENCHMARK_ROOT / method)).returncode


def collect_queried_indices(method: str, dataset_alias: str,
                            output_path: Path, seq_to_idx: dict) -> List[int]:
    """Find indices.pt or metrics_seed*.json from the run and map back to original landscape indices.

    Each method's run_generic.py writes either Indices.pt (Random / GreedyWalk)
    or metrics_seed{seed}.json with a 'queried_indices' or similar list. We try a
    couple of conventions; if none match the user must wire this themselves.
    """
    import torch
    candidates: List[Path] = []
    base = output_path
    for p in base.rglob("*indices*.pt"):
        candidates.append(p)
    for p in base.rglob("metrics_seed*.json"):
        candidates.append(p)
    if not candidates:
        return []
    # Prefer the most recent file
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    if latest.suffix == ".pt":
        idx = torch.load(latest).tolist()
        return [int(i) for i in idx]
    # JSON fallback: look for queried_indices field
    with open(latest) as f:
        rec = json.load(f)
    if isinstance(rec, dict) and "queried_indices" in rec:
        return [int(i) for i in rec["queried_indices"]]
    return []


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="Multi-property dataset name (e.g., eqFP611)")
    p.add_argument("--objectives", nargs="+", default=["blue", "red"],
                   help="Two objective column names in data.csv (default: blue red)")
    p.add_argument("--method", required=True,
                   help="Method name (must have <method>/run_generic.py)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["weighted_sum", "independent"],
                   default="weighted_sum")
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.0, 0.25, 0.5, 0.75, 1.0],
                   help="Weighted-sum mixing parameters (only used for "
                        "--mode=weighted_sum)")
    p.add_argument("--data-dir", default=str(BENCHMARK_ROOT / "data"))
    p.add_argument("--output-path", default=None)
    p.add_argument("--tmp-dir", default="/tmp/av_multi_objective")
    args = p.parse_args()

    if len(args.objectives) != 2:
        print("Currently only 2-objective optimization is supported.",
              file=sys.stderr)
        return 1

    data_dir = Path(args.data_dir)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    seqs, obj = load_multi_property(args.dataset, data_dir, args.objectives)
    n = len(seqs)
    seq_to_idx = {s: i for i, s in enumerate(seqs)}

    # Reference Pareto front (over the entire landscape)
    true_front = pareto_front(obj)
    ref_point = auto_reference_point(obj)
    print(f"\n[run_multi_objective] dataset={args.dataset} N={n}, "
          f"|true Pareto front|={len(true_front)}, ref_point={ref_point}")

    out_root = (Path(args.output_path) if args.output_path
                else BENCHMARK_ROOT / "results" / args.method
                     / f"{args.dataset}_multi_objective"
                     / f"seed{args.seed}")
    out_root.mkdir(parents=True, exist_ok=True)
    np.save(out_root / "pareto_front_true.npy", true_front)

    pareto_records = {
        "dataset": args.dataset,
        "method": args.method,
        "seed": args.seed,
        "objectives": args.objectives,
        "ref_point": ref_point.tolist(),
        "true_pareto_size": int(len(true_front)),
        "results": [],
    }

    if args.mode == "weighted_sum":
        for a in args.alphas:
            scalar = a * obj[:, 0] + (1.0 - a) * obj[:, 1]
            tmp_data_root, alias = write_scalar_landscape(
                args.dataset, seqs, scalar, tmp_dir, f"{a:.3f}"
            )
            run_out = out_root / f"alpha{a:.3f}"
            run_out.mkdir(parents=True, exist_ok=True)
            rc = run_method_on_scalar(
                args.method, alias, args.seed, tmp_data_root, run_out
            )
            queried_indices = collect_queried_indices(
                args.method, alias, run_out, seq_to_idx
            )
            if not queried_indices:
                pareto_records["results"].append({
                    "alpha": a, "status": "no_indices_found",
                    "queried_n": 0,
                })
                continue
            queried_obj = obj[queried_indices]
            np.save(out_root / f"indices_alpha{a:.3f}.npy",
                    np.asarray(queried_indices))
            hv = hypervolume(queried_obj, ref_point)
            cov = pareto_front_coverage(queried_obj, true_front)
            pareto_records["results"].append({
                "alpha": a,
                "queried_n": len(queried_indices),
                "hypervolume": hv,
                "pareto_coverage": cov,
                "rc": rc,
            })
            print(f"  alpha={a:.3f}: HV={hv:.4f}, coverage={cov:.4f}, "
                  f"queried={len(queried_indices)}")

    elif args.mode == "independent":
        for j, name in enumerate(args.objectives):
            tmp_data_root, alias = write_scalar_landscape(
                args.dataset, seqs, obj[:, j], tmp_dir, f"obj{j}"
            )
            run_out = out_root / f"obj_{name}"
            run_out.mkdir(parents=True, exist_ok=True)
            rc = run_method_on_scalar(
                args.method, alias, args.seed, tmp_data_root, run_out
            )
            queried_indices = collect_queried_indices(
                args.method, alias, run_out, seq_to_idx
            )
            np.save(out_root / f"indices_obj{j}.npy",
                    np.asarray(queried_indices or []))
            pareto_records["results"].append({
                "objective": name,
                "queried_n": len(queried_indices),
                "rc": rc,
            })
        # Joint evaluation: union queried indices across both objectives
        all_idx = set()
        for j in range(len(args.objectives)):
            arr = np.load(out_root / f"indices_obj{j}.npy")
            all_idx.update(int(i) for i in arr.tolist())
        if all_idx:
            joint_obj = obj[list(all_idx)]
            hv = hypervolume(joint_obj, ref_point)
            cov = pareto_front_coverage(joint_obj, true_front)
            pareto_records["joint"] = {
                "queried_n": len(all_idx),
                "hypervolume": hv,
                "pareto_coverage": cov,
            }
            print(f"\n  Joint (independent runs): HV={hv:.4f}, coverage={cov:.4f}")

    out_json = out_root / "pareto_metrics.json"
    with open(out_json, "w") as f:
        json.dump(pareto_records, f, indent=2)
    print(f"\nWrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
