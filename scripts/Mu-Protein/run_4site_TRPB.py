#!/usr/bin/env python
"""Per-dataset wrapper: Mu-Protein iterative on 4site_TRPB."""
import os, sys
sys.argv.insert(1, "4site_TRPB")
sys.argv.insert(1, "--dataset")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_generic import main
if __name__ == "__main__":
    sys.exit(main())
