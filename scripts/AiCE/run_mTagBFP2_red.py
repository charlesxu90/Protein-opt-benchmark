#!/usr/bin/env python
"""Per-dataset wrapper: AiCE on mTagBFP2_red. Delegates to run_generic.py."""
import os, sys
sys.argv.insert(1, "mTagBFP2_red")
sys.argv.insert(1, "--dataset")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_generic import main
if __name__ == "__main__":
    main()
