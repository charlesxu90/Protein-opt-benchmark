#!/usr/bin/env python
"""
compute_dataset_checksums.py — SHA-256 every data/<name>/data.csv.

Writes data/CHECKSUMS.txt in the format expected by `sha256sum -c`:

    <hex>  data/<name>/data.csv

Used in the *Nature Methods* reproducibility appendix so reviewers can verify
their downloaded landscapes byte-for-byte match the ones used in the paper.

Usage
-----
    # Hash everything under data/
    python scripts/compute_dataset_checksums.py

    # Verify existing CHECKSUMS.txt
    python scripts/compute_dataset_checksums.py --verify

    # Hash only specific datasets
    python scripts/compute_dataset_checksums.py --datasets GB1 PhoQ
"""

from __future__ import annotations
import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Iterable, List


BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BENCHMARK_ROOT / "data"
CHECKSUMS_PATH = DATA_DIR / "CHECKSUMS.txt"


def find_dataset_csvs(data_dir: Path,
                      whitelist: List[str] = None) -> List[Path]:
    """Return every <name>/data.csv directly under data/."""
    out: List[Path] = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        if whitelist and child.name not in whitelist:
            continue
        csv = child / "data.csv"
        if csv.exists():
            out.append(csv)
    return out


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def write_checksums(csvs: Iterable[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for p in csvs:
        rel = p.relative_to(BENCHMARK_ROOT)
        digest = sha256_of(p)
        lines.append(f"{digest}  {rel}")
        print(f"  {digest[:12]}…  {rel}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_path}  ({len(lines)} entries)")


def verify_checksums(checksums_path: Path) -> int:
    """Return 0 if all match, 1 if any mismatch, 2 if a referenced file is missing."""
    if not checksums_path.exists():
        print(f"No CHECKSUMS.txt at {checksums_path}", file=sys.stderr)
        return 2

    n_ok = n_bad = n_missing = 0
    for line in checksums_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            expected, rel = line.split(maxsplit=1)
            rel = rel.strip()
        except ValueError:
            print(f"  malformed line: {line!r}", file=sys.stderr)
            continue
        path = BENCHMARK_ROOT / rel
        if not path.exists():
            print(f"  MISSING  {rel}")
            n_missing += 1
            continue
        actual = sha256_of(path)
        if actual == expected:
            print(f"  OK       {rel}")
            n_ok += 1
        else:
            print(f"  MISMATCH {rel}")
            print(f"           expected {expected[:16]}…")
            print(f"           actual   {actual[:16]}…")
            n_bad += 1

    print(f"\n{n_ok} ok, {n_bad} mismatched, {n_missing} missing")
    if n_bad or n_missing:
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--output", default=str(CHECKSUMS_PATH))
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Only hash listed datasets (default: all)")
    p.add_argument("--verify", action="store_true",
                   help="Verify existing CHECKSUMS.txt instead of regenerating")
    args = p.parse_args()

    if args.verify:
        return verify_checksums(Path(args.output))

    csvs = find_dataset_csvs(Path(args.data_dir), args.datasets)
    if not csvs:
        print("No data/<name>/data.csv files found.", file=sys.stderr)
        return 1
    write_checksums(csvs, Path(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
