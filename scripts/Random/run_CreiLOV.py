#!/usr/bin/env python
"""Per-dataset wrapper: Random on CreiLOV. Delegates to run_generic.py."""
import os, sys
sys.argv.insert(1, "CreiLOV")
sys.argv.insert(1, "--dataset")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_generic import main
if __name__ == "__main__":
    main()
