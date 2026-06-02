#!/usr/bin/env python
"""
EVOLVEpro adapter for the 4-site combinatorial benchmark.

EVOLVEpro (Jiang et al., Science 2025; https://github.com/mat10d/EVOLVEpro)
runs few-shot active learning on top of frozen ESM-2 per-variant embeddings.
This wrapper:

  1. Loads pre-computed embeddings from `data/<dataset>/embeddings_evolvepro.pt`
     (produced by `scripts/EVOLVEpro/embed_dataset.py`).
  2. Loads matching labels from `data/<dataset>/labels_evolvepro.csv`.
  3. Calls `evolvepro.src.evolve.directed_evolution_simulation` with the
     benchmark's 96-per-round, 5-round protocol and the given seed.
  4. Translates EVOLVEpro's per-round results dataframe into our standard
     metrics_seed*.json format (max_fitness, etc.).

Run via the symlinked per-dataset wrappers (run_4site_GB1.py …) under
EVOLVEpro/ so `__file__/..` correctly resolves to BENCHMARK_ROOT.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
# Symlinked location is EVOLVEpro/run_generic.py; canonical is
# scripts/EVOLVEpro/run_generic.py. Walk up until BENCHMARK_ROOT is found.
_p = THIS_DIR
while _p.parent != _p and not (_p / "utils").is_dir():
    _p = _p.parent
BENCHMARK_ROOT = _p
sys.path.insert(0, str(BENCHMARK_ROOT))

# Allow `import evolvepro` from the cloned source tree.
sys.path.insert(0, str(BENCHMARK_ROOT / "EVOLVEpro"))


def _load_embeddings(emb_path: Path) -> pd.DataFrame:
    """
    Load embeddings produced by `embed_dataset.py` (a dict
    {variant: 1D-tensor}) and convert to the (n × D) DataFrame
    EVOLVEpro expects, indexed by variant string.
    """
    import torch
    d = torch.load(emb_path, weights_only=False)
    rows = list(d.keys())
    mat = np.stack([np.asarray(d[k], dtype=np.float32) for k in rows])
    cols = [f"e{i}" for i in range(mat.shape[1])]
    df = pd.DataFrame(mat, index=rows, columns=cols)
    df.index.name = "variant"
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="EVOLVEpro 4-site benchmark adapter")
    parser.add_argument("--dataset", required=True,
                        help="4site_GB1 | 4site_PhoQ | 4site_TEV | 4site_TRPB")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=96,
                        help="Per-round oracle batch (benchmark default 96).")
    parser.add_argument("--n_rounds", type=int, default=5,
                        help="Number of rounds (benchmark default 5).")
    parser.add_argument("--data_dir", default=str(BENCHMARK_ROOT / "data"))
    parser.add_argument("--embeddings_path", default=None,
                        help="Override embeddings .pt path; default <data>/<dataset>/embeddings_evolvepro.pt")
    parser.add_argument("--labels_path", default=None,
                        help="Override labels .csv path; default <data>/<dataset>/labels_evolvepro.csv")
    parser.add_argument("--regression_type", default="randomforest",
                        choices=["ridge", "randomforest", "gradientboosting"],
                        help="EVOLVEpro regressor (paper default randomforest).")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--skip_metrics", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    dataset_dir = Path(args.data_dir) / args.dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    emb_path = Path(args.embeddings_path) if args.embeddings_path else dataset_dir / "embeddings_evolvepro.pt"
    labels_path = Path(args.labels_path) if args.labels_path else dataset_dir / "labels_evolvepro.csv"
    if not emb_path.is_file():
        raise FileNotFoundError(
            f"Embeddings not found at {emb_path}. Run "
            f"`python scripts/EVOLVEpro/embed_dataset.py --dataset {args.dataset}` first."
        )
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels not found at {labels_path}.")

    print(f"[EVOLVEpro] dataset={args.dataset} seed={args.seed} "
          f"batch={args.batch_size} n_rounds={args.n_rounds}", flush=True)

    labels = pd.read_csv(labels_path)
    if "variant" not in labels.columns or "activity" not in labels.columns:
        raise RuntimeError(f"labels.csv must have 'variant' + 'activity' columns, got {list(labels.columns)}")
    # EVOLVEpro internally expects extra columns from their preprocessing step
    # (see EVOLVEpro/evolvepro/src/process.py): activity_scaled in [0,1] and
    # activity_binary (top-q threshold). Add them on the fly.
    a = labels["activity"].astype(float)
    rng_min, rng_max = float(a.min()), float(a.max())
    labels["activity_scaled"] = (a - rng_min) / max(rng_max - rng_min, 1e-9)
    # Binary cutoff: top 25% of measured activities (paper uses similar quantile).
    cutoff = float(a.quantile(0.75))
    labels["activity_binary"] = (a > cutoff).astype(int)
    # EVOLVEpro's first_round uses labels.variant (variant must be a column, not index).
    # Embeddings are indexed by variant so EVOLVEpro can do .loc[variant] lookup.
    embeddings = _load_embeddings(emb_path)
    common = set(labels["variant"]).intersection(embeddings.index)
    labels = labels[labels["variant"].isin(common)].reset_index(drop=True)
    embeddings = embeddings.loc[sorted(common)]
    # EVOLVEpro hardcodes `random_seed=i` (simulation index) when calling
    # first_round; with num_simulations=1 that's always 0, so every CLI --seed
    # would pick the same first-round 96 variants from `labels.variant`. To make
    # the seed actually take effect, we permute the labels DataFrame with a
    # seed-dependent order; `np.random.choice` then picks different positional
    # indices off a different label ordering, giving genuinely seed-dependent
    # starting samples.
    labels = labels.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    print(f"[EVOLVEpro] {len(common)} variants with embeddings + labels "
          f"(activity_scaled range [{labels['activity_scaled'].min():.3f}, "
          f"{labels['activity_scaled'].max():.3f}]; binary top-25%; "
          f"labels permuted with random_state={args.seed})", flush=True)

    # EVOLVEpro's directed_evolution_simulation expects:
    #   labels: DataFrame indexed by variant, with `activity` column
    #   embeddings: DataFrame indexed by variant, with feature columns
    from evolvepro.src.evolve import directed_evolution_simulation

    print(f"[EVOLVEpro] running directed_evolution_simulation ({args.regression_type}) ...", flush=True)
    results = directed_evolution_simulation(
        labels=labels,
        embeddings=embeddings,
        num_simulations=1,
        num_iterations=args.n_rounds,
        num_mutants_per_round=args.batch_size,
        measured_var="activity",
        regression_type=args.regression_type,
        learning_strategy="topn",
        first_round_strategy="random",
    )
    if not isinstance(results, pd.DataFrame) or results.empty:
        raise RuntimeError("EVOLVEpro returned empty results")
    print(f"[EVOLVEpro] returned {len(results)} rows", flush=True)

    # Cumulative oracle-queried variants up to each round to compute max_fitness.
    # EVOLVEpro's results columns include 'variant_id' lists per round; we use
    # 'next_iter_top_variants' (mutants picked next round) to identify selections.
    # Simpler path: re-derive collected variants by intersection at each round.
    # We just need a per-round vector of max fitness achieved so far.

    # EVOLVEpro's directed_evolution_simulation returns one row per round
    # (round_num 0..n_rounds). The column `top_activity_scaled` holds the
    # maximum activity_scaled of all cumulatively queried variants up to that
    # round. The final cumulative max is row n_rounds.
    if "top_activity_scaled" not in results.columns:
        raise RuntimeError(
            f"Expected 'top_activity_scaled' column in EVOLVEpro results; got {list(results.columns)}"
        )
    # EVOLVEpro's in-memory DataFrame puts the literal string 'None' (not NaN)
    # in unfilled rows; use to_numeric(errors='coerce') to coerce 'None' → NaN.
    top_traj = pd.to_numeric(results["top_activity_scaled"], errors="coerce").to_numpy()
    # Skip the round 0 NaN.
    valid = top_traj[~np.isnan(top_traj)]
    if len(valid) == 0:
        raise RuntimeError("EVOLVEpro returned no valid top_activity_scaled values")
    cum_max = float(np.max(valid))
    print(f"[EVOLVEpro] top_activity_scaled trajectory: {top_traj.tolist()}", flush=True)
    print(f"[EVOLVEpro] final cumulative max_fitness (activity_scaled) = {cum_max:.4f}", flush=True)

    # Save results
    if args.output_path is None:
        args.output_path = str(BENCHMARK_ROOT / f"EVOLVEpro/results/{args.dataset}_EVOLVEpro/{args.dataset}/topn")
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_dir = out_dir / f"seed_{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(seed_dir / "evolvepro_results.csv", index=False)

    if args.skip_metrics:
        metrics = {"max_fitness": cum_max}
    else:
        # Match other methods' metrics.json format with at least max_fitness
        metrics = {
            "metrics": {
                "max_fitness": cum_max,
                "queries": args.batch_size * args.n_rounds,
                "n_rounds": args.n_rounds,
            },
            "config": {
                "method": "EVOLVEpro",
                "dataset": args.dataset,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "n_rounds": args.n_rounds,
                "regression_type": args.regression_type,
            },
        }

    # Default output path matches the pattern used by other methods:
    # <method>/results/<dataset>_<method>/<dataset>/<algo>/metrics_seed<SEED>.json
    metrics_path = out_dir / f"metrics_seed{args.seed}.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[EVOLVEpro] wrote {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
