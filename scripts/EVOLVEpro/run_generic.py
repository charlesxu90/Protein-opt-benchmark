#!/usr/bin/env python
"""
run_generic.py — EVOLVEpro adapter (Phase 2.1, partial integration).

EVOLVEpro (Jiang et al., Science 2025; https://github.com/mat10d/EVOLVEpro) uses
a frozen PLM (ESM-2 by default) to embed every variant in the design space, then
runs few-shot active learning on top of those frozen embeddings. Their entry
point `evolvepro.src.evolve.grid_search` expects:

    - embeddings_path: a .pt file with shape (N, D) where N = #variants
    - labels_path:     a .csv with one fitness column aligned to embeddings
    - num_iterations, num_mutants_per_round, num_simulations, etc.

This wrapper bridges our `--seed --dataset` interface to EVOLVEpro's API:

    1. Load our prepared landscape via utils.proteingym_oracle.load_oracle.
    2. (One-time per dataset) Compute ESM-2 embeddings for every variant and
       cache to <data_dir>/<dataset>/embeddings_esm2.pt.
    3. Materialise EVOLVEpro-format labels alongside the embeddings.
    4. Invoke grid_search with num_simulations=1, the requested batch sizes,
       and our seed.
    5. Parse the output to produce our standard metrics_seed*.json.

STATUS: scaffolding only — steps 2 and 4 are TODO. ESM-2 embedding requires
GPU and ~1 GB per landscape; do this once per dataset on iBex (see
README_INTEGRATION.md). Step 5 needs a small adapter once we see EVOLVEpro's
output schema on a real run.

Usage (will refuse to run until TODOs are completed):
    python run_generic.py --dataset GB1 --seed 42
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = THIS_DIR.parent.parent if "scripts" in str(THIS_DIR) else THIS_DIR.parent
sys.path.insert(0, str(BENCHMARK_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="EVOLVEpro adapter (scaffolding)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="EVOLVEpro default per-round mutants (paper used 16)")
    parser.add_argument("--n_rounds", type=int, default=10,
                        help="EVOLVEpro default rounds (paper used 10)")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--embeddings_path", default=None,
                        help="Pre-computed ESM-2 embeddings .pt file")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--skip_metrics", action="store_true")
    args = parser.parse_args()

    print(
        "[EVOLVEpro adapter] This wrapper is a Phase 2.1 scaffold. "
        "It cannot yet run end-to-end without:\n"
        "  1) Pre-computed ESM-2 embeddings for the dataset\n"
        "     (see scripts/EVOLVEpro/README_INTEGRATION.md)\n"
        "  2) An adapter from grid_search output to our metrics schema\n"
        "Refusing to silently produce wrong results.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
