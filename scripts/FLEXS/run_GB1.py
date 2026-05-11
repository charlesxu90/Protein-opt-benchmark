#!/usr/bin/env python
"""run_GB1.py — FLEXS/AdaLead on GB1.

Thin shim: FLEXS ships run_GB1_adalead.py for the AdaLead variant; the
launcher's `<method>/run_<dataset>.py` convention expects run_GB1.py here,
so this file delegates to that script. Behavior is identical.
"""
import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TARGET = os.path.join(_HERE, "run_GB1_adalead.py")
if not os.path.exists(_TARGET):
    sys.exit(f"FLEXS adapter expects {_TARGET}; not found.")
runpy.run_path(_TARGET, run_name="__main__")
