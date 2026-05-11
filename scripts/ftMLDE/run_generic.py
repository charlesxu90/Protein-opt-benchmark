#!/usr/bin/env python
"""
run_generic.py — ftMLDE adapter (Phase 2.1, partial integration).

ftMLDE / MLDE (Wittmann et al., Cell Systems 2021;
https://github.com/fhalab/MLDE) is an "MLDE" (machine-learning-assisted directed
evolution) framework operating on a fixed combinatorial design space. Their
`simulate_mlde.py` runs N simulations over predetermined CV / training-index
splits and writes results to a save directory.

Bridging from our oracle interface:

    1. Load our prepared landscape (only works for true combinatorial datasets:
       GB1, PhoQ, CR9114, CreiLOV, eqFP611). Variable-length AAV/GFP datasets
       are NOT compatible with ftMLDE.
    2. Build the EncodedSpace (one-hot or a saved encoding) and feed it to
       `run_mlde` with our seed and budget.
    3. Parse the saved per-simulation results into our metrics schema.

STATUS: scaffolding only — see README_INTEGRATION.md.
"""

from __future__ import annotations
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="ftMLDE adapter (scaffolding)")
    parser.add_argument("--dataset", required=True,
                        help="Combinatorial dataset (GB1/PhoQ/CR9114/CreiLOV/eqFP611)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--n_rounds", type=int, default=5)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--skip_metrics", action="store_true")
    args = parser.parse_args()

    print(
        "[ftMLDE adapter] This wrapper is a Phase 2.1 scaffold. "
        "It cannot yet run end-to-end without:\n"
        "  1) An EncodedSpace builder for our combinatorial datasets\n"
        "     (see scripts/ftMLDE/README_INTEGRATION.md)\n"
        "  2) An adapter from run_mlde output to our metrics schema\n"
        "Refusing to silently produce wrong results.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
