#!/usr/bin/env python
"""Per-dataset wrapper: Random on GFP_hard. Delegates to run_generic.py."""
import os, sys
sys.argv.insert(1, "GFP_hard")
sys.argv.insert(1, "--dataset")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_generic import main
if __name__ == "__main__":
    main()
