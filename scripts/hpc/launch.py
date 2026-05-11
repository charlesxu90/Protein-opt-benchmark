#!/usr/bin/env python
"""
launch.py — Submit benchmark sweeps to iBex / Shaheen, or run locally.

Generates a SLURM job-array (one task per seed) from a template, with per-method
resource defaults from `method_resources.yaml`. Submits via `sbatch`, or — when
`--cluster local` — runs each seed serially in the foreground.

Examples
--------
    # 5-seed local smoke test
    python scripts/hpc/launch.py --method Random --dataset GB1 --seeds 5 --cluster local

    # 50-seed iBex array
    python scripts/hpc/launch.py --method ALDE --dataset GB1 --seeds 50 --cluster ibex

    # Custom seed file + extra args forwarded to the run script
    python scripts/hpc/launch.py --method alphavariant --dataset PhoQ \\
        --seed-file rand_seeds.txt --num-seeds 30 --cluster shaheen \\
        --account k01234 --extra-args "--ablation no-gpt"

    # Dry run: print sbatch script instead of submitting
    python scripts/hpc/launch.py --method ALDE --dataset GB1 --seeds 50 \\
        --cluster ibex --dry-run

Outputs
-------
- Submitted: sbatch script written to `scripts/hpc/_jobs/<job_name>.sbatch`,
  logs go to `scripts/hpc/_logs/`.
- Local: each seed's stdout/stderr streamed to the terminal; method's run
  script is responsible for placing results under `results/<method>/<dataset>/`.
"""

from __future__ import annotations
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError:
    yaml = None

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent.parent
HPC_DIR = Path(__file__).resolve().parent
TEMPLATES = {
    "ibex": HPC_DIR / "ibex_array.sbatch",
    "shaheen": HPC_DIR / "shaheen_array.sbatch",
}
RESOURCES_YAML = HPC_DIR / "method_resources.yaml"


def load_resources() -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to read method_resources.yaml. "
            "Install with `pip install pyyaml`."
        )
    with open(RESOURCES_YAML) as f:
        return yaml.safe_load(f)


def resolve_method_resources(method: str, resources: Dict[str, Any]) -> Dict[str, Any]:
    base = dict(resources.get("defaults", {}))
    base.update(resources.get("methods", {}).get(method, {}))
    env = base.get("conda_env", "")
    if env:
        # Verify the configured env actually exists. If not, fall back to the
        # heuristic <method>/env. This makes the YAML the source of truth but
        # tolerates user-specific overrides via env vars or local paths.
        env_path = Path(env).expanduser()
        if not env_path.is_absolute():
            env_path = BENCHMARK_ROOT / env_path
        if not (env_path / "bin" / "python").exists():
            heuristic = BENCHMARK_ROOT / method / "env"
            if heuristic.exists():
                base["conda_env"] = f"{method}/env"
    elif not env:
        candidate = BENCHMARK_ROOT / method / "env"
        if candidate.exists():
            base["conda_env"] = f"{method}/env"
    return base


def slice_seeds(seed_file: Path, n: int, start: int = 0) -> List[int]:
    """Read seeds from `seed_file`, skipping the first `start` valid lines.

    Used to partition a fixed seed list across two GPUs by passing different
    `start` offsets:
        GPU 0: --seed-start 0  --num-seeds 15  -> seeds[0:15]
        GPU 1: --seed-start 15 --num-seeds 15  -> seeds[15:30]
    """
    all_seeds: List[int] = []
    with open(seed_file) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            all_seeds.append(int(s))
    if start + n > len(all_seeds):
        raise ValueError(
            f"{seed_file} has {len(all_seeds)} seeds; requested "
            f"start={start} + n={n} = {start + n}."
        )
    return all_seeds[start : start + n]


def write_seed_subset(seeds: List[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in seeds:
            f.write(f"{s}\n")


def resolve_python_for_method(method: str, conda_env: str) -> str:
    """Pick the python interpreter for a method's local run.

    `conda_env` may be:
      - an absolute path (e.g. "/home/xux/miniforge3/envs/alphavariant-env")
      - a path relative to the benchmark root (e.g. "ALDE/env")
      - empty / missing -> falls back to sys.executable

    Methods have heterogeneous Python versions (3.7-3.11) and incompatible deps;
    using the wrong python silently picks up the wrong torch/scipy/sklearn.
    """
    if conda_env:
        env_path = Path(conda_env).expanduser()
        if not env_path.is_absolute():
            env_path = BENCHMARK_ROOT / env_path
        candidate = env_path / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    return sys.executable


def run_local(method: str, dataset: str, seeds: List[int],
              extra_args: List[str], log_dir: Path,
              gpu_id: Optional[str] = None,
              python_path: Optional[str] = None,
              run_subdir: str = "") -> int:
    """Run each seed serially in the foreground. Logs streamed to terminal.

    `gpu_id` may be a single index ("0") or a comma list ("0,1") and is exported
    as CUDA_VISIBLE_DEVICES. None leaves the host environment alone.
    `python_path` overrides sys.executable so the method's conda env is used.
    `run_subdir` (optional) selects the cwd / PYTHONPATH prefix when the method's
    code lives in a nested directory (e.g. delta_cs/BioSeq-GFN-AL).
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Methods with run_subdir keep their run scripts inside that subdir.
    # For others, scripts live directly in <method>/.
    if run_subdir:
        run_script = BENCHMARK_ROOT / method / run_subdir / f"run_{dataset}.py"
    else:
        run_script = BENCHMARK_ROOT / method / f"run_{dataset}.py"
    if not run_script.exists():
        print(f"Run script not found: {run_script}", file=sys.stderr)
        return 1

    py = python_path or sys.executable
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Working directory must match what the script's relative imports expect.
    work_dir = BENCHMARK_ROOT / method
    if run_subdir:
        work_dir = work_dir / run_subdir

    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{work_dir}{os.pathsep}{existing_pp}" if existing_pp else str(work_dir)
    )

    overall_rc = 0
    for i, seed in enumerate(seeds):
        cmd = [py, str(run_script), "--seed", str(seed)] + extra_args
        print(f"[{i+1}/{len(seeds)}] {' '.join(shlex.quote(c) for c in cmd)}")
        if gpu_id is not None:
            print(f"    CUDA_VISIBLE_DEVICES={gpu_id}")
        if run_subdir:
            print(f"    cwd={work_dir}")
        rc = subprocess.run(cmd, cwd=str(work_dir), env=env).returncode
        if rc != 0:
            print(f"  seed {seed} exited with {rc}", file=sys.stderr)
            overall_rc = overall_rc or rc
    return overall_rc


def render_sbatch(template: Path, replacements: Dict[str, str]) -> str:
    text = template.read_text()
    for k, v in replacements.items():
        text = text.replace(f"__{k}__", str(v))
    return text


def submit_array(
    cluster: str,
    method: str,
    dataset: str,
    seeds: List[int],
    extra_args: List[str],
    resources: Dict[str, Any],
    partition: str,
    account: str,
    dry_run: bool,
) -> int:
    template = TEMPLATES[cluster]
    if not template.exists():
        print(f"Template missing: {template}", file=sys.stderr)
        return 1

    job_name = f"av_{method}_{dataset}"
    jobs_dir = HPC_DIR / "_jobs"
    logs_dir = HPC_DIR / "_logs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    seed_file = jobs_dir / f"{job_name}_seeds.txt"
    write_seed_subset(seeds, seed_file)

    array_spec = f"0-{len(seeds) - 1}"
    if len(seeds) > 1:
        # Optional throttle: max 32 concurrent; comment out if not desired
        array_spec += "%32"

    replacements = {
        "JOB_NAME": job_name,
        "LOG_DIR": str(logs_dir),
        "ARRAY_SPEC": array_spec,
        "PARTITION": partition,
        "TIME": resources["time_per_seed"],
        "CPUS": str(resources["cpus_per_task"]),
        "MEM_GB": str(resources["mem_gb"]),
        "GPUS": str(resources["gpus"]),
        "ACCOUNT": account,
        "BENCHMARK_ROOT": str(BENCHMARK_ROOT),
        "METHOD": method,
        "DATASET": dataset,
        "CONDA_ENV": resources.get("conda_env", ""),
        "SEED_FILE": str(seed_file),
        "EXTRA_ARGS": " ".join(shlex.quote(a) for a in extra_args),
    }

    rendered = render_sbatch(template, replacements)
    sbatch_path = jobs_dir / f"{job_name}.sbatch"
    sbatch_path.write_text(rendered)
    sbatch_path.chmod(0o755)

    if resources["gpus"] == 0:
        # Strip the gres line for CPU-only methods
        new_lines = [ln for ln in rendered.splitlines() if "--gres=gpu" not in ln]
        sbatch_path.write_text("\n".join(new_lines) + "\n")

    print(f"Wrote {sbatch_path}")
    if dry_run:
        print("--- sbatch script ---")
        print(sbatch_path.read_text())
        return 0

    rc = subprocess.run(["sbatch", str(sbatch_path)]).returncode
    if rc != 0:
        print("sbatch submission failed", file=sys.stderr)
    return rc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True,
                   help="Method name (matches method directory)")
    p.add_argument("--dataset", required=True,
                   help="Dataset name (e.g., GB1, AAV_med, PhoQ)")
    p.add_argument("--cluster", choices=["ibex", "shaheen", "local"],
                   default="local")
    p.add_argument("--seeds", type=int, default=None,
                   help="Number of seeds (sliced from seed file)")
    p.add_argument("--num-seeds", type=int, default=None,
                   help="Alias for --seeds")
    p.add_argument("--seed-file", default=str(BENCHMARK_ROOT / "rand_seeds.txt"))
    p.add_argument("--partition", default="batch",
                   help="SLURM partition (iBex: batch/gpu/gpu-rtx; Shaheen: workq/gpu)")
    p.add_argument("--account", default="",
                   help="SLURM account (Shaheen requires this)")
    p.add_argument("--extra-args", default="",
                   help="String of extra args forwarded to run_<dataset>.py "
                        "(quoted, e.g. \"--ablation no-gpt --batch_size 96\")")
    p.add_argument("--dry-run", action="store_true",
                   help="Print sbatch script instead of submitting")
    p.add_argument("--seed-start", type=int, default=0,
                   help="Skip the first N entries of --seed-file before "
                        "slicing --seeds. Use to partition a fixed seed list "
                        "across multiple GPUs (e.g. --seed-start 0 / "
                        "--seed-start 15 for two A100s).")
    p.add_argument("--gpu-id", default=None,
                   help="In --cluster local, set CUDA_VISIBLE_DEVICES to this "
                        "(e.g. '0' or '0,1'). Ignored for SLURM clusters.")
    p.add_argument("--use-method-env", action="store_true", default=True,
                   help="In --cluster local, run with the method's conda env "
                        "python (resolved from method_resources.yaml). "
                        "Default: True. Pass --no-use-method-env to disable.")
    p.add_argument("--no-use-method-env", dest="use_method_env",
                   action="store_false")

    args = p.parse_args()
    n_seeds = args.seeds or args.num_seeds or 5

    resources_doc = load_resources()
    resources = resolve_method_resources(args.method, resources_doc)

    seeds = slice_seeds(Path(args.seed_file), n_seeds, start=args.seed_start)
    extra_args = shlex.split(args.extra_args) if args.extra_args else []

    if args.cluster == "local":
        log_dir = HPC_DIR / "_logs" / f"{args.method}_{args.dataset}_local"
        py = (resolve_python_for_method(args.method, resources.get("conda_env", ""))
              if args.use_method_env else sys.executable)
        return run_local(args.method, args.dataset, seeds, extra_args, log_dir,
                         gpu_id=args.gpu_id, python_path=py,
                         run_subdir=resources.get("run_subdir", ""))

    if args.cluster == "shaheen" and not args.account:
        print("WARNING: Shaheen requires --account; submission may be rejected.",
              file=sys.stderr)

    return submit_array(
        cluster=args.cluster,
        method=args.method,
        dataset=args.dataset,
        seeds=seeds,
        extra_args=extra_args,
        resources=resources,
        partition=args.partition,
        account=args.account,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
