# Repository Guidelines

## Project Structure & Module Organization

This repository is a protein optimization benchmark suite. Shared code lives in `utils/`; datasets are under `data/<dataset>/data.csv`; generated outputs are in `results/`, `tables/`, `sweep_logs/`, and `smoke_logs/`. Method implementations live in top-level directories such as `ALDE/`, `AiCE/`, `delta_cs/`, and `alphavariant/`.

Run scripts live only under `scripts/<method>/` and are invoked from there; method directories contain upstream code only. There are no symlinks to refresh.

## Build, Test, and Development Commands

- `python scripts/hpc/launch.py --method ALDE --dataset GB1 --seeds 5 --cluster local`: run a local smoke benchmark.
- `python scripts/hpc/launch.py --method ALDE --dataset GB1 --seeds 50 --cluster ibex --dry-run`: render HPC submission.
- `ALDE/env/bin/python scripts/ALDE/run_generic.py --dataset 4site_GB1 --seed 621`: run one method/dataset seed.
- `python scripts/aggregate_metrics.py --dataset GB1 --seed 621`: summarize seed metrics.
- `python scripts/generate_tables.py --datasets GB1 AAV_hard --format markdown`: rebuild tables.
- `python test_utils_on_alde.py`: run utility compatibility check.

Use the method-specific conda environment before direct runs, for example `conda activate ./ALDE/env`. Launcher resources and env paths are in `scripts/hpc/method_resources.yaml`.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and descriptive snake_case names. Keep run scripts named `run_<dataset>.py`, for example `run_GB1.py`. Prefer `utils/` over duplicated metric or dataset code. New datasets should use `data/<name>/data.csv` with `seq` and one or more numeric labels, usually `fitness`; multi-objective datasets may use labels such as `blue` and `red`.

## Testing Guidelines

Use cheap checks before long sweeps: verify data loading, run one seed, then scale up. Add focused tests when changing `utils/`; use `pytest` where package tests exist, such as `ftMLDE/pytest.ini`. Verify data with `sha256sum -c data/CHECKSUMS.txt` after dataset changes.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries such as `fix a ALDE bug` and `save running scripts`. Keep commits focused. Pull requests should name the method or dataset affected, list commands run, note environment assumptions, and include result table or log paths when benchmark behavior changes.

## Security & Configuration Tips

Do not commit conda environments, checkpoints, generated logs, or large outputs unless requested. Treat `scripts/hpc/method_resources.yaml` as the source of truth for environments and resources. Prefer `--extra-args` for experiment settings.

## Agent-Specific Instructions

Preserve user results and logs unless asked to clean them. Do not edit generated outputs unless the task is aggregation or reporting. Prefer tracked `scripts/`, `utils/`, `data/`, and `docs/` paths, and document skipped verification.
