"""
OracleLandscape - a learned-oracle fitness function for the multi-site benchmark.

Wraps a trained GGS/LatProtRL `BaseCNN` (see `scripts/train_oracle.py`) so that any
sequence in the design space can be scored, replacing the lookup-table landscape
(which returned 0 off-pool and collapsed generative search). This is the
"pure oracle" setting: every queried sequence is scored by the CNN; measured wet-lab
labels were only used to train it.

`get_fitness(sequences)` returns RAW fitness units (de-normalized via the fit_min/
fit_max scaler stored at training time), so oracle scores are directly comparable to
the dataset's measured fitness. `get_fitness_normalized` returns [0,1].

A query counter (`n_calls`) is exposed for budget accounting; budget *enforcement*
is the runner's responsibility (96 queries/round x 5 rounds).
"""

from __future__ import annotations

import os
from typing import List, Sequence

import numpy as np
import torch

from .oracle_model import BaseCNN, encode_int, N_TOKENS

DEFAULT_ORACLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "oracles"))
DEFAULT_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data"))


def _read_wt(dataset: str, data_dir: str) -> str | None:
    path = os.path.join(data_dir, dataset, "wt.fasta")
    if not os.path.exists(path):
        return None
    return "".join(l.strip() for l in open(path)
                   if l.strip() and not l.startswith(">"))


class OracleLandscape:
    def __init__(
        self,
        dataset: str,
        oracle_dir: str = DEFAULT_ORACLE_DIR,
        data_dir: str = DEFAULT_DATA_DIR,
        device: str = "cuda:0",
        batch_size: int = 4096,
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size

        ckpt_path = os.path.join(oracle_dir, dataset, "oracle.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"No oracle for {dataset} at {ckpt_path}. "
                f"Train one with: python scripts/train_oracle.py --dataset {dataset}")
        ckpt = torch.load(ckpt_path, map_location=device)
        arch = ckpt.get("arch", {})
        self.model = BaseCNN(
            n_tokens=arch.get("n_tokens", N_TOKENS),
            kernel_size=arch.get("kernel_size", 5),
            input_size=arch.get("input_size", 256),
            make_one_hot=arch.get("make_one_hot", True),
        ).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.fit_min = float(ckpt["fit_min"])
        self.fit_max = float(ckpt["fit_max"])
        self.scale = self.fit_max - self.fit_min
        self.seq_len = int(ckpt["seq_len"])
        self.test_spearman = float(ckpt.get("test_spearman", float("nan")))
        self.wildtype = _read_wt(dataset, data_dir)
        self.n_calls = 0

    def _predict_norm(self, sequences: Sequence[str]) -> np.ndarray:
        X = encode_int(list(sequences), self.seq_len)
        out = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                out.append(self.model(X[i:i + self.batch_size].to(self.device))
                           .cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    def get_fitness(self, sequences: Sequence[str]) -> np.ndarray:
        """Oracle fitness in RAW units (de-normalized). Increments n_calls."""
        if len(sequences) == 0:
            return np.array([])
        self.n_calls += len(sequences)
        return self._predict_norm(sequences) * self.scale + self.fit_min

    def get_fitness_normalized(self, sequences: Sequence[str]) -> np.ndarray:
        """Oracle fitness in [0,1] (training scale). Increments n_calls."""
        if len(sequences) == 0:
            return np.array([])
        self.n_calls += len(sequences)
        return self._predict_norm(sequences)

    def __repr__(self):
        return (f"OracleLandscape({self.dataset}, L={self.seq_len}, "
                f"fit=[{self.fit_min:.3g},{self.fit_max:.3g}], "
                f"test_spearman={self.test_spearman:.3f})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    ls = OracleLandscape(args.dataset, device=args.device)
    print(ls)
    if ls.wildtype:
        print("WT fitness:", ls.get_fitness([ls.wildtype])[0])
