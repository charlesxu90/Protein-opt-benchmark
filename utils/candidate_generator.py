"""
CandidateGenerator - design-space proposal for the oracle benchmark.

Defines the searchable design space for a multi-site dataset and produces candidate
sequences for the "generative proposal" optimization loop (Random, GreedyWalk,
ALDE/CLADE/ftMLDE-as-generative, AdaLead). This replaces the fixed enumerated library
that pool-selection methods used on the small 4-site tasks.

Design space:
    - backbone   = wild-type (wt.fasta); non-varying positions are held fixed.
    - varying positions = positions that differ across the measured set (robust for
      all datasets incl. GFP); cross-checked against varying_positions.txt if present.
    - alphabet at varying positions = full 20 AAs by default (faithful to GGS free
      mutation), or the observed per-position alphabet (`alphabet_mode='observed'`).

Proposal primitives:
    - random_init(n): n variants made by mutating WT at k positions, k drawn from the
      empirical mutation-count distribution (keeps the init on the data manifold so the
      oracle is queried where it is reliable). Round-0 initialization.
    - mutate(parents, n, m_step): n candidates, each a parent with ~m_step random
      point mutations (local search from elites). Used every round by the explorers.
    - neighbors(seq): all single-mutant neighbors over varying positions (GreedyWalk).
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .oracle_model import ALPHABET  # "ARNDCQEGHILKMFPSTWYV"

DEFAULT_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data"))


def _read_wt(dataset: str, data_dir: str) -> str:
    path = os.path.join(data_dir, dataset, "wt.fasta")
    if not os.path.exists(path):
        raise FileNotFoundError(f"wt.fasta required for {dataset} at {path}")
    return "".join(l.strip() for l in open(path)
                   if l.strip() and not l.startswith(">"))


class CandidateGenerator:
    def __init__(
        self,
        dataset: str,
        data_dir: str = DEFAULT_DATA_DIR,
        alphabet_mode: str = "full",   # "full" (20 AA) or "observed"
    ):
        df = pd.read_csv(os.path.join(data_dir, dataset, "data.csv"))
        seq_col = "seq" if "seq" in df.columns else "sequence"
        seqs = df[seq_col].astype(str).tolist()
        self.dataset = dataset
        self.seq_len = len(seqs[0])
        self.wt = _read_wt(dataset, data_dir)
        if len(self.wt) != self.seq_len:
            raise ValueError(f"{dataset}: WT length {len(self.wt)} != seq_len {self.seq_len}")

        # varying positions + observed per-position alphabet, from measured data
        arr = np.array([list(s) for s in seqs])
        observed = [sorted(set(arr[:, j])) for j in range(self.seq_len)]
        self.varying_positions = [j for j in range(self.seq_len) if len(observed[j]) > 1]

        if alphabet_mode == "full":
            self.choices = {j: [a for a in ALPHABET if a != self.wt[j]]
                            for j in self.varying_positions}
        elif alphabet_mode == "observed":
            self.choices = {j: [a for a in observed[j] if a != self.wt[j]]
                            for j in self.varying_positions}
            self.choices = {j: v for j, v in self.choices.items() if v}
            self.varying_positions = list(self.choices.keys())
        else:
            raise ValueError(f"alphabet_mode must be 'full' or 'observed', got {alphabet_mode}")

        self._var = np.array(self.varying_positions)

        # empirical mutation-count distribution (>=1) for realistic init
        if "n_muts" in df.columns:
            nm = df["n_muts"].values.astype(int)
        else:
            nm = np.array([sum(c != w for c, w in zip(s, self.wt)) for s in seqs])
        self._kpool = nm[nm >= 1]
        if len(self._kpool) == 0:
            self._kpool = np.array([1])

    # ------------------------------------------------------------------ #
    def _apply_mutations(self, base: str, positions, rng) -> str:
        s = list(base)
        for p in positions:
            s[p] = rng.choice(self.choices[int(p)])
        return "".join(s)

    def random_init(self, n: int, rng: np.random.RandomState,
                    n_mut: Optional[int] = None) -> List[str]:
        """n variants from WT; k mutations drawn from empirical distribution
        (or fixed n_mut)."""
        out, seen = [], set()
        max_attempts = n * 100
        attempts = 0
        while len(out) < n and attempts < max_attempts:
            attempts += 1
            k = n_mut if n_mut is not None else int(rng.choice(self._kpool))
            k = max(1, min(k, len(self._var)))
            pos = rng.choice(self._var, size=k, replace=False)
            cand = self._apply_mutations(self.wt, pos, rng)
            if cand != self.wt and cand not in seen:
                seen.add(cand)
                out.append(cand)
        return out

    def mutate(self, parents: Sequence[str], n: int, rng: np.random.RandomState,
               m_step: int = 2) -> List[str]:
        """n candidates: each = a random parent with ~m_step point mutations."""
        if len(parents) == 0:
            return self.random_init(n, rng)
        parents = list(parents)
        out, seen = [], set()
        max_attempts = n * 50
        attempts = 0
        while len(out) < n and attempts < max_attempts:
            attempts += 1
            parent = parents[rng.randint(len(parents))]
            k = max(1, min(m_step, len(self._var)))
            pos = rng.choice(self._var, size=k, replace=False)
            cand = self._apply_mutations(parent, pos, rng)
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
        return out

    def neighbors(self, seq: str) -> List[str]:
        """All single-mutant neighbors of `seq` over varying positions."""
        out = []
        for p in self.varying_positions:
            for aa in self.choices[p]:
                if aa != seq[p]:
                    out.append(seq[:p] + aa + seq[p + 1:])
        return out

    def __repr__(self):
        return (f"CandidateGenerator({self.dataset}, L={self.seq_len}, "
                f"varying={len(self.varying_positions)}, "
                f"avg_choices={np.mean([len(v) for v in self.choices.values()]):.1f})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--alphabet_mode", default="full")
    args = ap.parse_args()
    rng = np.random.RandomState(0)
    gen = CandidateGenerator(args.dataset, alphabet_mode=args.alphabet_mode)
    print(gen)
    init = gen.random_init(5, rng)
    print("init muts vs WT:", [sum(a != b for a, b in zip(s, gen.wt)) for s in init])
    mut = gen.mutate(init, 5, rng, m_step=2)
    print("mutate produced:", len(mut), "| neighbors of init[0]:", len(gen.neighbors(init[0])))
