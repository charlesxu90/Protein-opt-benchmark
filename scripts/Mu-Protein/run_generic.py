#!/usr/bin/env python
"""Mu-Protein (µFormer + iterative active learning) adapter.

Faithful upstream wrapper. Each round delegates to `Mu-Protein/mu-former/main.py`
(Microsoft Research, https://github.com/microsoft/Mu-Protein) for both training
and prediction, using the released PMLM-650M as the encoder and the Siamese
decoder. The 96 × 5 protocol is wrapped on top of that:

    Round 1: random 96 → real oracle (landscape CSV).
    Rounds 2-N:
        1. Write cumulative (mutation, fitness) TSV to /tmp.
        2. Call main.py --train  ... → trained µFormer checkpoint ensemble.
        3. Write uncollected mutation TSV (dummy scores) to /tmp.
        4. Call main.py --saved-model-dir ... --test uncollected.tsv
           → ensemble prediction.tsv.
        5. Rank uncollected by predicted score, query oracle on top-96.
        6. Add to cumulative.

The upstream `main.py` is invoked verbatim via subprocess so the adapter cannot
silently diverge from the paper's training pipeline.

Note on µSearch: the paper's µSearch component is a FLEXS RL explorer on top of
a trained µFormer-as-proxy. For a fully-enumerable 4-site combinatorial space
(≤ 160k variants), an explorer like AdaLead/CMAES would converge to the same
top-96 that exhaustive scoring already gives, so we score the entire uncollected
set and pick top-96 directly. The fitness model (µFormer) is unchanged.

Run via the symlinked per-dataset wrappers (e.g. Mu-Protein/run_4site_PhoQ.py).
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
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

MUFORMER_DIR = BENCHMARK_ROOT / "Mu-Protein" / "mu-former"
PMLM_CKPT = BENCHMARK_ROOT / "Mu-Protein" / "pretrained" / "pmlm_650m.pt"
MUFORMER_ENV_PY = Path("/home/xux/miniforge3/envs/muformer-env/bin/python")


def _load_wt(dataset_dir: Path) -> str:
    seq = []
    for ln in (dataset_dir / "wt.fasta").read_text().splitlines():
        if ln.startswith(">") or not ln.strip():
            continue
        seq.append(ln.strip())
    return "".join(seq)


def _detect_positions(df: pd.DataFrame, combo_col: str, wt: str) -> list[int]:
    sample = df["seq"].iloc[0]
    pos = [i for i in range(len(wt)) if sample[i] != wt[i]]
    if not pos:
        for k in range(min(2000, len(df))):
            s = df["seq"].iloc[k]
            pos = [i for i in range(len(wt)) if s[i] != wt[i]]
            if pos:
                break
    return pos[: len(df[combo_col].iloc[0])]


def _combo_to_mutstr(combo: str, wt: str, positions: list[int]) -> str:
    """Convert AACombo (e.g. 'TEMH') → mu-former mutation string 'A1B;C2D'."""
    parts = []
    for j, p in enumerate(positions[: len(combo)]):
        wt_aa = wt[p]
        mt_aa = combo[j]
        if mt_aa != wt_aa:
            parts.append(f"{wt_aa}{p + 1}{mt_aa}")
    return ";".join(parts) if parts else "WT"


def _write_tsv(path: Path, mutations: list[str], scores: list[float]) -> None:
    df = pd.DataFrame({"mutation": mutations, "score": scores})
    df.to_csv(path, sep="\t", index=False)


def _run_main(args_list: list[str], log_path: Path, gpu_id: int) -> None:
    """Invoke mu-former/main.py via subprocess in muformer-env."""
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": str(29500 + (gpu_id * 100) + int(os.getpid()) % 90),
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
    })
    cmd = [str(MUFORMER_ENV_PY), "main.py"] + args_list
    with open(log_path, "a") as logf:
        logf.write(f"\n\n=== CMD {' '.join(cmd)} ===\n")
        logf.flush()
        proc = subprocess.run(
            cmd, cwd=str(MUFORMER_DIR), env=env,
            stdout=logf, stderr=subprocess.STDOUT, check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"mu-former subprocess failed (rc={proc.returncode}). See {log_path}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Mu-Protein iterative adapter")
    parser.add_argument("--dataset", required=True,
                        help="4site_GB1 | 4site_PhoQ | 4site_TEV | 4site_TRPB")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--n_rounds", type=int, default=5)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_ensembles", type=int, default=3,
                        help="Number of µFormer models in the ensemble (paper default 5; we use 3 for tractability).")
    parser.add_argument("--train_epochs", type=int, default=30,
                        help="Training epochs per round (paper default 300; we use 30 for small per-round data).")
    parser.add_argument("--train_batch", type=int, default=8,
                        help="Train batch. Default 8 (upstream default); reduce to 2 for sequences ≥ 400 AA to avoid OOM with PMLM-650M on 40 GB GPU.")
    parser.add_argument("--predict_batch", type=int, default=32,
                        help="Inference batch. Default 32; reduce to 8 for long sequences.")
    parser.add_argument("--auto_batch", action="store_true", default=True,
                        help="Automatically pick train_batch=2 and predict_batch=8 when WT sequence length > 200 (avoids OOM on PhoQ/long-seq datasets).")
    parser.add_argument("--data_dir", default=str(BENCHMARK_ROOT / "data"))
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--workdir", default=None,
                        help="Directory for per-round checkpoints + TSVs. Default: /tmp.")
    parser.add_argument("--skip_metrics", action="store_true")
    args = parser.parse_args()

    if not MUFORMER_ENV_PY.is_file():
        raise FileNotFoundError(f"muformer-env python not found at {MUFORMER_ENV_PY}")
    if not PMLM_CKPT.is_file():
        raise FileNotFoundError(f"PMLM-650M checkpoint not found at {PMLM_CKPT}")
    if not (MUFORMER_DIR / "main.py").is_file():
        raise FileNotFoundError(f"mu-former main.py not at {MUFORMER_DIR}")

    np.random.seed(args.seed)
    import random; random.seed(args.seed)

    dataset_dir = Path(args.data_dir) / args.dataset
    df = pd.read_csv(dataset_dir / "data.csv")
    combo_col = next((c for c in ("AACombo", "Combo", "combo") if c in df.columns), None)
    if combo_col is None:
        raise RuntimeError(f"No AACombo column in {dataset_dir}/data.csv")

    wt_seq = _load_wt(dataset_dir)
    positions = _detect_positions(df, combo_col, wt_seq)
    combos = df[combo_col].astype(str).tolist()
    fitness = df["fitness"].astype(float).to_numpy()
    gmax = float(np.nanmax(fitness))
    # Normalise fitness to [0, 1] for µFormer training (more stable than raw values).
    fitness_norm = fitness / gmax if gmax > 1.5 else fitness.copy()
    mutstrs = [_combo_to_mutstr(c, wt_seq, positions) for c in combos]

    # Auto-reduce batch sizes for long WT sequences. PMLM-650M attention is
    # O(L²); at L=486 (PhoQ) batch 8 OOMs on a 40 GB A100.
    if args.auto_batch and len(wt_seq) > 200:
        args.train_batch = min(args.train_batch, 2)
        args.predict_batch = min(args.predict_batch, 8)
        print(f"[Mu-Protein] auto_batch: WT length {len(wt_seq)} > 200 → "
              f"train_batch={args.train_batch}, predict_batch={args.predict_batch}", flush=True)

    # Drop WT-matching variants (mutation string == 'WT'); upstream
    # `_drop_invalid_mutation` cannot parse 'WT' (it calls `int(mut[1:-1])`).
    # At most one row per dataset; harmless to exclude from the search.
    valid_mask = np.array([m != "WT" for m in mutstrs], dtype=bool)
    n_skip = int((~valid_mask).sum())
    if n_skip > 0:
        keep_idx = np.where(valid_mask)[0]
        combos = [combos[i] for i in keep_idx]
        fitness_norm = fitness_norm[keep_idx]
        mutstrs = [mutstrs[i] for i in keep_idx]
        print(f"[Mu-Protein] dropped {n_skip} WT-matching variant(s) "
              "(upstream parser cannot handle 'WT' label).", flush=True)

    n = len(combos)
    print(f"[Mu-Protein] {args.dataset} seed={args.seed} n_variants={n} wt_len={len(wt_seq)} positions={positions}", flush=True)

    if args.workdir is None:
        args.workdir = tempfile.mkdtemp(prefix=f"muformer_{args.dataset}_seed{args.seed}_")
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "muformer.log"

    # Write a single WT fasta (shared across rounds).
    wt_fasta = workdir / "wt.fasta"
    wt_fasta.write_text(f">wt\n{wt_seq}\n")

    rng = np.random.default_rng(args.seed)
    collected_idx: list[int] = []
    collected_fit: list[float] = []

    # Round 1: random 96 from the landscape.
    round1 = rng.choice(n, size=min(args.batch_size, n), replace=False).tolist()
    collected_idx.extend(round1)
    collected_fit.extend(float(fitness_norm[i]) for i in round1)
    print(f"[Mu-Protein] round 1: random {len(round1)} → max_so_far = {max(collected_fit):.4f}", flush=True)

    # Rounds 2..N.
    for r in range(2, args.n_rounds + 1):
        round_dir = workdir / f"round_{r}"
        round_dir.mkdir(parents=True, exist_ok=True)
        train_tsv = round_dir / "train.tsv"
        test_tsv = round_dir / "uncollected.tsv"
        ckpt_dir = round_dir / "ckpt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 1. Cumulative train TSV.
        train_muts = [mutstrs[i] for i in collected_idx]
        train_scs = [collected_fit[k] for k in range(len(collected_idx))]
        _write_tsv(train_tsv, train_muts, train_scs)

        # 2. Train µFormer ensemble on cumulative labels.
        print(f"[Mu-Protein] round {r}: training µFormer ensemble "
              f"(n_train={len(train_muts)}, ensembles={args.num_ensembles}, "
              f"epochs={args.train_epochs}) ...", flush=True)
        _run_main([
            "--train", str(train_tsv),
            "--fasta", str(wt_fasta),
            "--output-dir", str(ckpt_dir),
            "--pretrained-model", str(PMLM_CKPT),
            "--encoder-name", "pmlm",
            "--decoder-name", "siamese",
            "--num-ensembles", str(args.num_ensembles),
            "--epochs", str(args.train_epochs),
            "--batch-size", str(args.train_batch),
            "--num-workers", "0",
            "--seed", str(args.seed),
            "--local-rank", "0",
        ], log_path, args.gpu_id)

        # 3. Uncollected TSV with dummy scores.
        uncollected_mask = np.ones(n, dtype=bool)
        uncollected_mask[collected_idx] = False
        uncollected_idx = np.where(uncollected_mask)[0]
        uncoll_muts = [mutstrs[i] for i in uncollected_idx]
        _write_tsv(test_tsv, uncoll_muts, [0.0] * len(uncoll_muts))

        # 4. Predict with the trained ensemble.
        # Upstream Trainer always reads train_tsv during init even when running
        # --saved-model-dir + --test, so we pass the same cumulative train.tsv.
        print(f"[Mu-Protein] round {r}: predicting on {len(uncollected_idx)} uncollected ...", flush=True)
        pred_dir = round_dir / "pred"
        pred_dir.mkdir(parents=True, exist_ok=True)
        _run_main([
            "--train", str(train_tsv),
            "--saved-model-dir", str(ckpt_dir),
            "--test", str(test_tsv),
            "--fasta", str(wt_fasta),
            "--output-dir", str(pred_dir),
            "--pretrained-model", str(PMLM_CKPT),
            "--encoder-name", "pmlm",
            "--decoder-name", "siamese",
            "--num-ensembles", str(args.num_ensembles),
            "--batch-size", str(args.predict_batch),
            "--num-workers", "0",
            "--seed", str(args.seed),
            "--local-rank", "0",
        ], log_path, args.gpu_id)

        # 5. Read predictions and rank.
        pred_path = pred_dir / "prediction.tsv"
        if not pred_path.is_file():
            raise RuntimeError(f"No prediction.tsv at {pred_path}; see {log_path}")
        pred_df = pd.read_csv(pred_path, sep="\t")
        if "prediction" not in pred_df.columns:
            raise RuntimeError(f"prediction.tsv lacks 'prediction' column: {list(pred_df.columns)}")
        # Map predictions back to dataset indices.
        mut_to_idx = {mutstrs[i]: i for i in uncollected_idx}
        pred_df["idx"] = pred_df["mutation"].map(mut_to_idx)
        pred_df = pred_df.dropna(subset=["idx"])
        pred_df["idx"] = pred_df["idx"].astype(int)
        pred_df = pred_df.sort_values("prediction", ascending=False)
        top_idx = pred_df["idx"].tolist()[: args.batch_size]

        # 6. Add to cumulative.
        collected_idx.extend(top_idx)
        collected_fit.extend(float(fitness_norm[i]) for i in top_idx)
        print(f"[Mu-Protein] round {r}: queried {len(top_idx)}; "
              f"max_so_far = {max(collected_fit):.4f}", flush=True)

        # 7. Free disk: remove this round's checkpoints + prediction TSVs.
        # Each model_*.pt is ~8 GB; with 3 ensembles × 4 rounds that's 96 GB/seed
        # which fills a 1.8 TB disk after 5 seeds. Keep only the muformer.log.
        import shutil
        for sub in ("ckpt", "pred"):
            try:
                shutil.rmtree(round_dir / sub, ignore_errors=True)
            except Exception:
                pass
        # Also drop the per-round TSVs (small but unneeded once we have top_idx).
        for f in (train_tsv, test_tsv):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass

    max_f = float(max(collected_fit))
    print(f"[Mu-Protein] FINAL max_fitness = {max_f:.4f} over {len(collected_idx)} queries", flush=True)

    if args.output_path is None:
        args.output_path = str(BENCHMARK_ROOT /
                               f"Mu-Protein/results/{args.dataset}_MuProtein/{args.dataset}/iterative")
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "metrics": {
            "max_fitness": max_f,
            "queries": len(collected_idx),
            "n_rounds": args.n_rounds,
        },
        "config": {
            "method": "Mu-Protein",
            "dataset": args.dataset,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "n_rounds": args.n_rounds,
            "num_ensembles": args.num_ensembles,
            "train_epochs": args.train_epochs,
            "train_batch": args.train_batch,
            "encoder": "pmlm",
            "decoder": "siamese",
            "pretrained": "pmlm_650m.pt",
        },
    }
    (out_dir / f"metrics_seed{args.seed}.json").write_text(json.dumps(metrics, indent=2))
    print(f"[Mu-Protein] wrote {out_dir / f'metrics_seed{args.seed}.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
