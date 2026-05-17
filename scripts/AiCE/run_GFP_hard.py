#!/usr/bin/env python
"""Per-dataset wrapper: AiCE on GFP_hard. Delegates to run_generic.py.

Replaces the previous standalone implementation, which carried a legacy
AiCEScorer that leaked fitness labels into the scoring (top-10%-by-fitness
templates). All wrappers now route through run_generic.py for a single,
leak-free, iterative AiCE implementation.
"""
import os, sys
sys.argv.insert(1, "GFP_hard")
sys.argv.insert(1, "--dataset")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_generic import main
if __name__ == "__main__":
    main()
