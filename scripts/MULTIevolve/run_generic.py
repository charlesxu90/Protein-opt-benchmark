#!/usr/bin/env python
"""
MULTI-evolve adapter for the 4-site combinatorial benchmark.

MULTI-evolve (Tran et al., Science 2026; https://github.com/ArcInstitute/MULTI-evolve)
is a multi-step pipeline (featurizer → predictor → combinatorial proposer)
that trains NN ensembles on single-mutant data and predicts combinatorial
variants. For our 96 × 5 iterative protocol we use MULTI-evolve's `Fcn`
neural-net predictor in an active-learning loop:

  Round 1: random 96 variants from the landscape (no training data yet).
  Rounds 2-5: fit `Fcn` (FCN ensemble) on the cumulative (combo, fitness) labels;
              score every uncollected variant; query top-96 by predicted fitness.

Output: standard metrics_seedNNN.json with max_fitness across rounds.

Run from anywhere; BENCHMARK_ROOT is discovered by walking up from this file:
    MULTIevolve/env/bin/python scripts/MULTIevolve/run_generic.py --dataset 4site_PhoQ --seed 621
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
_p = THIS_DIR
while _p.parent != _p and not (_p / "utils").is_dir():
    _p = _p.parent
BENCHMARK_ROOT = _p
sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT / "MULTIevolve"))


def _reexec_with_env_libstdcxx() -> None:
    """Re-exec once with this env's lib/ on LD_LIBRARY_PATH, then continue.

    multievolve imports matplotlib transitively (predictors.base_regressors),
    whose C extension needs a newer CXXABI than the system libstdc++ provides.
    Without this the run dies at import with
    `ImportError: ... version 'CXXABI_1.3.15' not found`.

    LD_LIBRARY_PATH has to be set before the process starts -- the loader reads
    it at startup -- so we re-exec ourselves once. (Loading the env's
    libstdc++ via ctypes.CDLL(RTLD_GLOBAL) instead gets past the ImportError
    but then segfaults mid-run, so don't do that.)
    """
    if os.environ.get("_MULTIEVOLVE_LIBSTDCXX_REEXEC"):
        return
    env_lib = Path(sys.prefix) / "lib"
    if not (env_lib / "libstdc++.so.6").exists():
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if str(env_lib) in current.split(os.pathsep):
        return
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(env_lib), current]))
    os.environ["_MULTIEVOLVE_LIBSTDCXX_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_reexec_with_env_libstdcxx()


def _load_wt(dataset_dir: Path) -> str:
    wt = dataset_dir / "wt.fasta"
    seq = []
    for ln in wt.read_text().splitlines():
        if ln.startswith(">") or not ln.strip():
            continue
        seq.append(ln.strip())
    return "".join(seq)


def _detect_varying(df: pd.DataFrame, wt: str, combo_col: str):
    sample = df["seq"].iloc[0]
    pos = [i for i in range(len(wt)) if sample[i] != wt[i]]
    if not pos:
        for k in range(min(2000, len(df))):
            s = df["seq"].iloc[k]
            pos = [i for i in range(len(wt)) if s[i] != wt[i]]
            if pos: break
    return pos[: len(df[combo_col].iloc[0])]


def _combo_to_seq(combo: str, wt: str, positions):
    arr = list(wt)
    for j, p in enumerate(positions[: len(combo)]):
        arr[p] = combo[j]
    return "".join(arr)


def _combo_to_mut_str(combo: str, wt: str, positions):
    """Convert AACombo (e.g. 'TEMH') into MULTI-evolve's mutation string
    'A1B+C2D+...' (1-indexed)."""
    parts = []
    for j, p in enumerate(positions[: len(combo)]):
        wt_aa = wt[p]
        mut_aa = combo[j]
        if mut_aa != wt_aa:
            parts.append(f"{wt_aa}{p + 1}{mut_aa}")
    if not parts:
        return "WT"
    return "/".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="MULTI-evolve 4-site adapter")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--n_rounds", type=int, default=5)
    parser.add_argument("--data_dir", default=str(BENCHMARK_ROOT / "data"))
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--predictor", default="ridge",
                        choices=["ridge", "fcn"],
                        help="MULTI-evolve predictor. Ridge is fast/CPU; Fcn is the paper default but slower.")
    parser.add_argument("--fcn_epochs", type=int, default=50,
                        help="Epochs for Fcn predictor (paper used 300; we use less for time).")
    args = parser.parse_args()

    np.random.seed(args.seed)
    import random; random.seed(args.seed)

    dataset_dir = Path(args.data_dir) / args.dataset
    df = pd.read_csv(dataset_dir / "data.csv")
    combo_col = next((c for c in ("AACombo", "Combo", "combo") if c in df.columns), None)
    if combo_col is None:
        raise RuntimeError(f"No AACombo column in {dataset_dir}/data.csv")
    wt_seq = _load_wt(dataset_dir)
    positions = _detect_varying(df, wt_seq, combo_col)

    fitness = df["fitness"].astype(float).to_numpy()
    gmax = float(np.nanmax(fitness))
    fitness_norm = fitness / gmax if gmax > 1.5 else fitness.copy()
    combos = df[combo_col].astype(str).tolist()
    full_seqs = [_combo_to_seq(c, wt_seq, positions) for c in combos]

    n = len(combos)
    print(f"[MULTI-evolve] {args.dataset} seed={args.seed} n_variants={n}", flush=True)

    rng = np.random.default_rng(args.seed)
    collected_idx: list[int] = []
    collected_fit: list[float] = []

    # Round 1: random 96 from the landscape.
    round1 = rng.choice(n, size=min(args.batch_size, n), replace=False).tolist()
    collected_idx.extend(round1)
    collected_fit.extend([fitness_norm[i] for i in round1])
    print(f"[MULTI-evolve] round 1: random {len(round1)} → max_so_far = "
          f"{max(collected_fit):.4f}", flush=True)

    # Rounds 2..N: fit predictor on cumulative, score uncollected, take top-batch.
    from multievolve.splitters import RandomProteinSplitter
    from multievolve.featurizers import OneHotFeaturizer
    if args.predictor == "ridge":
        from multievolve.predictors import RidgeRegressor as Predictor
    elif args.predictor == "fcn":
        from multievolve.predictors import Fcn as Predictor
    else:
        raise ValueError(f"unknown predictor {args.predictor}")

    for r in range(2, args.n_rounds + 1):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            csv_p = tmp_p / "train.csv"
            wt_p = tmp_p / "wt.fasta"
            wt_p.write_text(f">wt\n{wt_seq}\n")

            mut_strs = [_combo_to_mut_str(combos[i], wt_seq, positions) for i in collected_idx]
            train_df = pd.DataFrame({
                "mutation": mut_strs,
                "property_value": collected_fit,
            })
            train_df.to_csv(csv_p, index=False)

            try:
                protein_name = f"adapter_{args.dataset}_seed{args.seed}_r{r}"
                splitter = RandomProteinSplitter(
                    protein_name, str(csv_p), str(wt_p),
                    csv_has_header=True, use_cache=False, y_scaling=False,
                    val_split=None if args.predictor == "ridge" else 0.15,
                )
                splitter.split_data(test_size=0.15)
                featurizer = OneHotFeaturizer(protein=protein_name, use_cache=False,
                                              flatten_features=(args.predictor != "fcn"))
                if args.predictor == "ridge":
                    predictor = Predictor(splitter, featurizer, use_cache=False, show_plots=False)
                else:
                    config = {
                        "layer_size": 100, "num_layers": 2,
                        "learning_rate": 0.001, "batch_size": 32,
                        "optimizer": "adam", "epochs": args.fcn_epochs,
                    }
                    predictor = Predictor(splitter, featurizer, config=config,
                                          use_cache=False, show_plots=False)
                predictor.run_model()

                uncollected_mask = np.ones(n, dtype=bool)
                uncollected_mask[collected_idx] = False
                uncollected_idx = np.where(uncollected_mask)[0]
                uncollected_seqs = [full_seqs[i] for i in uncollected_idx]
                preds = np.asarray(predictor.predict(uncollected_seqs), dtype=float).ravel()
                top_k = uncollected_idx[np.argsort(-preds)[: args.batch_size]]
            except Exception as e:
                print(f"[MULTI-evolve] round {r} predictor failed "
                      f"({type(e).__name__}: {e}); falling back to random pick", flush=True)
                uncollected_mask = np.ones(n, dtype=bool)
                uncollected_mask[collected_idx] = False
                uncollected_idx = np.where(uncollected_mask)[0]
                top_k = rng.choice(uncollected_idx,
                                   size=min(args.batch_size, len(uncollected_idx)),
                                   replace=False)

        collected_idx.extend(top_k.tolist())
        collected_fit.extend([fitness_norm[i] for i in top_k])
        print(f"[MULTI-evolve] round {r}: queried {len(top_k)}; max_so_far = "
              f"{max(collected_fit):.4f}", flush=True)

    max_f = float(max(collected_fit))
    print(f"[MULTI-evolve] FINAL max_fitness = {max_f:.4f} over "
          f"{len(collected_idx)} queries", flush=True)

    if args.output_path is None:
        args.output_path = str(BENCHMARK_ROOT /
                               f"MULTIevolve/results/{args.dataset}_MULTIevolve/{args.dataset}/{args.predictor}")
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute normalized_fitness_median_top128: median of top-128 fitness values
    # among the 480 queried variants (normalized — fitness_norm is already in [0,1]).
    fit_sorted = sorted(collected_fit, reverse=True)
    top128 = fit_sorted[:128]
    top128_median = float(np.median(top128)) if top128 else 0.0

    metrics = {
        "metrics": {
            "max_fitness": max_f,
            "normalized_fitness_median_top128": top128_median,
            "queries": len(collected_idx),
            "n_rounds": args.n_rounds,
        },
        "config": {
            "method": "MULTI-evolve",
            "dataset": args.dataset,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "n_rounds": args.n_rounds,
            "predictor": args.predictor,
        },
        "collected_indices": [int(i) for i in collected_idx],
    }
    out_path = out_dir / f"metrics_seed{args.seed}.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"[MULTI-evolve] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
