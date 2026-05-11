#!/usr/bin/env python
"""
run_generic.py — MULTI-evolve adapter (Phase 2.1, partial integration).

MULTI-evolve (Tran et al., Science 2026; https://github.com/ArcInstitute/MULTI-evolve)
is a 3-step pipeline: train fully-connected NN ensembles on single-mutant data,
predict combinatorial variant fitness, design assembly oligos. For our
benchmark we only need steps 1+2; step 3 is wet-lab specific.

Bridging:

    1. Load our prepared landscape; export as a "training-dataset CSV" with
       columns (mutation, property_value) in MULTI-evolve's format.
    2. Run `multievolve.predictors.train_predictor` with `mode='test'` and
       our seed.
    3. Use the trained predictor to score every multi-mutant in the landscape;
       select top-`batch_size` per round in our iterative loop.

STATUS: scaffolding only — see README_INTEGRATION.md.
"""

from __future__ import annotations
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="MULTI-evolve adapter (scaffolding)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--n_rounds", type=int, default=5)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--skip_metrics", action="store_true")
    args = parser.parse_args()

    print(
        "[MULTI-evolve adapter] This wrapper is a Phase 2.1 scaffold. "
        "It cannot yet run end-to-end without:\n"
        "  1) WandB-free training mode (upstream uses WandB by default)\n"
        "  2) A predictor → iterative selection adapter\n"
        "  3) An adapter from predictor output to our metrics schema\n"
        "See scripts/MULTIevolve/README_INTEGRATION.md.\n"
        "Refusing to silently produce wrong results.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
