#!/usr/bin/env python
"""
prepare_combingym.py - Download and prepare CombinGym combinatorial datasets.

CombinGym (https://github.com/chenz16/CombinGym) provides exhaustively measured
combinatorial libraries used in the refined benchmark plan's Task 1
(high-order epistasis navigation). This script clones (or pulls) the repo and
converts the selected datasets to the standardized `data/<name>/data.csv`
schema (`seq, fitness` columns) used everywhere else in this benchmark.

Selected datasets
-----------------
    GB1       — Protein G domain B1, 4 sites, 160k variants
    PhoQ      — PhoQ sensor kinase, 4 sites, 160k variants
    CR9114    — Influenza antibody, 16 sites, 65k variants
    CreiLOV   — Fluorescent protein, 15 sites, 165k variants
    eqFP611   — Red FP, 5 sites, 32k variants. Multi-property (blue + red).

For multi-property datasets, separate single-objective files are emitted
(e.g., `data/eqFP611_red/data.csv`, `data/eqFP611_blue/data.csv`) plus a
joint `data/eqFP611/data.csv` containing both columns
(`fitness_red`, `fitness_blue`) for multi-objective evaluation.

Usage
-----
    # Default: clone (if needed) and prepare all 5
    python prepare_combingym.py

    # Specific datasets
    python prepare_combingym.py --datasets GB1 PhoQ

    # Use an existing local clone instead of cloning
    python prepare_combingym.py --local-repo /path/to/CombinGym

    # List available datasets
    python prepare_combingym.py --list

Notes
-----
CombinGym's exact CSV schema and filenames may evolve. This script auto-detects
sequence and fitness columns from a list of candidate names; if both detection
strategies fail, it prints the column list so the user can specify
`--seq-col` / `--fitness-col` explicitly.
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


COMBINGYM_REPO = "https://github.com/sitonglab/CombinGym"

# Dataset registry. CombinGym layout: Data/DMS/Clean/<name>_clean.xlsx, columns
# include 'genotype' (variant string), 'n_mut', and one or more fitness columns.
# `subpaths` is the list of candidates within the repo (first match wins).
# `seq_col`, `fitness_col`, and `properties` override auto-detection.
#
# Optional `combo_positions` (1-indexed) extracts an AACombo from full-length
# `genotype` for combinatorial datasets; this preserves backwards compatibility
# with run scripts that read df['AACombo'] (delta_cs, alphavariant for GB1).
DATASETS: Dict[str, Dict] = {
    "GB1": {
        "description": "Protein G B1 domain, 4-site combinatorial (~149k)",
        "subpaths": ["Data/DMS/Clean/GB1_clean.xlsx"],
        "seq_col": "genotype",
        "fitness_col": "fitness",
        "properties": None,
        "combo_positions": [39, 40, 41, 54],  # 1-indexed
    },
    "CR9114": {
        "description": "CR9114 influenza antibody (H1 subtype), 16-site combinatorial",
        "subpaths": ["Data/DMS/Clean/bnAbs_CR9114_H1_clean.xlsx"],
        "seq_col": "genotype",
        "fitness_col": "h1_mean",
        "properties": None,
    },
    "CreiLOV": {
        "description": "CreiLOV fluorescent protein, ~165k combinatorial",
        "subpaths": ["Data/DMS/Clean/CreiLOV_clean.xlsx"],
        "seq_col": "genotype",
        "fitness_col": "mean",
        "properties": None,
    },
    "eqFP611": {
        "description": "eqFP611 red FP, 5-site, dual-property (blue + red)",
        "subpaths": ["Data/DMS/Clean/eqFP611_clean.xlsx"],
        "seq_col": "genotype",
        "fitness_col": None,  # multi-property; properties listed below
        "properties": ["blue", "red"],
    },
    "mTagBFP2": {
        # eqFP611(mTagBFP2) per CombinGym Data_summary.xlsx.
        "description": "mTagBFP2 fluorescent protein, 13-site, 8192 variants, multi-property (blue/red/combined)",
        "subpaths": ["Data/DMS/Clean/eqFP611_clean.xlsx"],
        "seq_col": "genotype",
        "fitness_col": None,
        "properties": ["blue", "red", "combined"],
    },
    "SpCas9": {
        "description": "SpCas9 nuclease, combinatorial DMS",
        "subpaths": ["Data/DMS/Clean/SpCas9_clean.xlsx"],
        "seq_col": "genotype",
        "fitness_col": None,  # auto-detect first numeric column
        "properties": None,
    },
    "SaCas9": {
        "description": "SaCas9 nuclease, combinatorial DMS",
        "subpaths": ["Data/DMS/Clean/SaCas9_clean.xlsx"],
        "seq_col": "genotype",
        "fitness_col": None,
        "properties": None,
    },
    "RhlA": {
        "description": "RhlA rhamnosyltransferase, 11-site, 2.3k variants, multi-property (substrate selectivity + specific activity, log-transformed)",
        "subpaths": ["Data/DMS/Clean/RhlA_clean.xlsx"],
        "seq_col": "genotype",
        "fitness_col": None,
        "properties": ["%Rha-(C8-C10)_log", "Rha-(C8-C10)_log"],
        "property_aliases": {
            "%Rha-(C8-C10)_log": "selectivity",
            "Rha-(C8-C10)_log": "activity",
        },
        # CombinGym's RhlA genotype is just the 11-char combo; embed into the
        # 295-aa WT at the user-spec positions so seq is full-length.
        "expand_combo": {
            "wt": "MRRESLLVSVCKGLRVHVERVGQDPGRSTVMLVNGAMATTASFARTCKCLAEHFNVVLFDLPFAGQSRQHNPQRGLITKDDEVEILLALIERFEVNHLVSASWGGISTLLALSRNPRGIRSSVVMAFAPGLNQAMLDYVGRAQALIELDDKSAIGHLLNETVGKYLPQRLKASNHQHMASLATGEYEQARFHIDQVLALNDRGYLACLERIQSHVHFINGSWDEYTTAEDARQFRDYLPHCSFSRVEGTGHFLDLESKLAAVRVHRALLEHLLKQPEPQRAERAAGFHEMAIGYA",
            "positions": [42, 43, 73, 74, 101, 143, 148, 173, 176, 177, 182],
        },
    },
    "CR6261_H1": {
        "description": "CR6261 anti-influenza bnAb, H1 hemagglutinin binding, 11-site (~1.9k variants)",
        "subpaths": ["Data/DMS/Clean/bnAbs_CR6261_H1_clean.xlsx"],
        "seq_col": "genotype",
        "fitness_col": "h1_mean",
        "properties": None,
    },
    "CR6261_H9": {
        "description": "CR6261 anti-influenza bnAb, H9 hemagglutinin binding, 11-site (~1.9k variants)",
        "subpaths": ["Data/DMS/Clean/bnAbs_CR6261_H9_clean.xlsx"],
        "seq_col": "genotype",
        "fitness_col": "h9_mean",
        "properties": None,
    },
    "CR6261": {
        "description": "CR6261 anti-influenza bnAb, joint H1+H9 binding multi-objective (inner-merge on genotype)",
        "merge_subpaths": [
            "Data/DMS/Clean/bnAbs_CR6261_H1_clean.xlsx",
            "Data/DMS/Clean/bnAbs_CR6261_H9_clean.xlsx",
        ],
        "merge_on": "genotype",
        "seq_col": "genotype",
        "fitness_col": None,
        "properties": ["h1_mean", "h9_mean"],
    },
}

SEQ_COL_CANDIDATES = ["genotype", "sequence", "seq", "AACombo", "Combo", "variant", "mutant", "AAseq"]
FITNESS_COL_CANDIDATES = ["fitness", "score", "Fitness", "DMS_score", "mean", "log_fitness", "h1_mean"]


def clone_or_pull(repo_url: str, dest: Path) -> None:
    if (dest / ".git").exists():
        print(f"  Updating existing clone: {dest}")
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=False)
    else:
        print(f"  Cloning {repo_url} -> {dest}")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(dest)],
                       check=True)


def find_csv(repo_root: Path, subpaths: List[str]) -> Optional[Path]:
    for sp in subpaths:
        candidate = repo_root / sp
        if candidate.exists():
            return candidate
    # Last-resort search by basename
    base = subpaths[0].rsplit("/", 1)[-1]
    for found in repo_root.rglob(base):
        return found
    return None


def read_table(path: Path) -> pd.DataFrame:
    """Read .csv / .tsv / .xlsx / .xls."""
    suffix = path.suffix.lower()
    if suffix in (".csv",):
        return pd.read_csv(path)
    if suffix in (".tsv",):
        return pd.read_csv(path, sep="\t")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported extension: {suffix}")


def detect_columns(
    df: pd.DataFrame,
    seq_col: Optional[str],
    fitness_col: Optional[str],
    properties: Optional[List[str]],
) -> Dict[str, object]:
    cols = list(df.columns)
    out: Dict[str, object] = {}

    # Sequence column
    if seq_col and seq_col in cols:
        out["seq"] = seq_col
    else:
        for c in SEQ_COL_CANDIDATES:
            if c in cols:
                out["seq"] = c
                break

    # Fitness column(s)
    if properties:
        present = [p for p in properties if p in cols]
        if not present:
            present = [c for c in cols if any(k in c.lower() for k in ("fit", "score", "blue", "red"))][:2]
        out["properties"] = present
    else:
        if fitness_col and fitness_col in cols:
            out["fitness"] = fitness_col
        else:
            for c in FITNESS_COL_CANDIDATES:
                if c in cols:
                    out["fitness"] = c
                    break
        # Last-resort: first numeric non-seq column
        if "fitness" not in out:
            seq_name = out.get("seq")
            for c in cols:
                if c == seq_name:
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    out["fitness"] = c
                    break

    return out


def write_dataset(
    df: pd.DataFrame,
    detected: Dict,
    output_path: Path,
    properties: Optional[List[str]],
    property_aliases: Optional[Dict[str, str]] = None,
) -> Dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seq_col = detected.get("seq")
    if not seq_col:
        raise ValueError(
            f"No sequence column detected. Available: {list(df.columns)}"
        )

    aliases = property_aliases or {}
    if properties:
        property_cols = detected.get("properties") or []
        if len(property_cols) < 2:
            raise ValueError(
                f"Multi-property dataset needs >=2 property columns; "
                f"found {property_cols}. Available: {list(df.columns)}"
            )
        out = pd.DataFrame({"seq": df[seq_col].values})
        for p in property_cols:
            out[aliases.get(p, p)] = df[p].values
        out = out.dropna().drop_duplicates(subset="seq", keep="first")
        out.to_csv(output_path, index=False)
        return {
            "n_variants": len(out),
            "seq_len": len(str(out["seq"].iloc[0])),
            "properties": [aliases.get(p, p) for p in property_cols],
        }

    fitness_col = detected.get("fitness")
    if not fitness_col:
        raise ValueError(
            f"No fitness column detected. Available: {list(df.columns)}"
        )
    # Strip terminal stop codons ("*") from sequences — they are metadata,
    # not residues, and break methods that use 20-AA alphabets (EvoPlay,
    # alphavariant, delta_cs). CreiLOV's CombinGym data contains a single
    # trailing "*" on every sequence; other datasets are unaffected.
    seqs_clean = [s.rstrip("*") if isinstance(s, str) else s
                  for s in df[seq_col].values]
    cols = {
        "seq": seqs_clean,
        "fitness": df[fitness_col].astype(float).values,
    }
    # Pass-through n_mut (CombinGym writes it directly)
    if "n_mut" in df.columns:
        cols["n_muts"] = df["n_mut"].values

    out = pd.DataFrame(cols).dropna(subset=["seq", "fitness"]).drop_duplicates(
        subset="seq", keep="first")
    out = out.sort_values("fitness", ascending=False).reset_index(drop=True)

    # Derive AACombo from positions if specified (backwards compat for GB1
    # alphavariant/delta_cs run scripts that read df['AACombo']).
    combo_positions = detected.get("combo_positions") or []
    if combo_positions:
        zero_idx = [p - 1 for p in combo_positions]
        out["AACombo"] = [
            "".join(str(s)[i] for i in zero_idx) for s in out["seq"].values
        ]

    out.to_csv(output_path, index=False)
    return {
        "n_variants": len(out),
        "seq_len": len(str(out["seq"].iloc[0])),
        "fitness_min": float(out["fitness"].min()),
        "fitness_max": float(out["fitness"].max()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=None,
                   help=f"Subset to prepare (default: all). Choices: {list(DATASETS)}")
    p.add_argument("--list", action="store_true",
                   help="List available CombinGym datasets and exit")
    p.add_argument("--local-repo", default=None,
                   help="Path to existing CombinGym clone (skip git clone)")
    p.add_argument("--data-dir", default=None,
                   help="Output directory (default: <repo-root>/data)")
    p.add_argument("--cache-dir", default="/tmp/combingym_cache",
                   help="Where to clone CombinGym")
    p.add_argument("--seq-col", default=None, help="Override sequence-column auto-detection")
    p.add_argument("--fitness-col", default=None, help="Override fitness-column auto-detection")
    args = p.parse_args()

    if args.list:
        print("\nAvailable CombinGym datasets:\n")
        for name, info in DATASETS.items():
            multi = " [multi-property]" if info["properties"] else ""
            print(f"  {name:<10} {info['description']}{multi}")
        return 0

    benchmark_root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else benchmark_root / "data"

    if args.local_repo:
        repo_root = Path(args.local_repo)
        if not repo_root.exists():
            print(f"Local repo not found: {repo_root}", file=sys.stderr)
            return 1
    else:
        cache = Path(args.cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        repo_root = cache / "CombinGym"
        clone_or_pull(COMBINGYM_REPO, repo_root)

    selected = args.datasets or list(DATASETS)
    selected = [s for s in selected if s in DATASETS]
    if not selected:
        print(f"No valid datasets. Available: {list(DATASETS)}", file=sys.stderr)
        return 1

    print(f"\nPreparing {len(selected)} CombinGym dataset(s)")
    print(f"Output: {data_dir}")

    summary: Dict[str, Dict] = {}
    for name in selected:
        info = DATASETS[name]
        print(f"\n--- {name} ---")

        # Resolve dataframe: either single source xlsx, or inner-merge across
        # multiple xlsx files (used for CR6261 joint = H1 + H9).
        if info.get("merge_subpaths"):
            merge_on = info.get("merge_on", "genotype")
            dfs = []
            missing = []
            for sp in info["merge_subpaths"]:
                p = repo_root / sp
                if not p.exists():
                    missing.append(sp)
                    continue
                dfs.append(read_table(p))
            if missing:
                summary[name] = {"status": "FAILED", "error": f"merge sources missing: {missing}"}
                print(f"  Merge sources not found: {missing}", file=sys.stderr)
                continue
            df = dfs[0]
            for extra in dfs[1:]:
                shared = [c for c in extra.columns if c in df.columns and c != merge_on]
                df = df.merge(extra.drop(columns=shared), on=merge_on, how="inner")
            print(f"  Merged {len(info['merge_subpaths'])} files on '{merge_on}' "
                  f"-> {len(df)} rows, columns: {list(df.columns)}")
        else:
            csv_path = find_csv(repo_root, info["subpaths"])
            if csv_path is None:
                print(f"  CSV not found in repo. Tried: {info['subpaths']}", file=sys.stderr)
                summary[name] = {"status": "FAILED", "error": "CSV not found"}
                continue
            try:
                df = read_table(csv_path)
            except Exception as e:
                summary[name] = {"status": "FAILED", "error": str(e)}
                print(f"  ERROR reading {csv_path}: {e}", file=sys.stderr)
                continue
            print(f"  Read {csv_path} ({len(df)} rows, columns: {list(df.columns)})")

        seq_override = args.seq_col or info.get("seq_col")
        fitness_override = args.fitness_col or info.get("fitness_col")
        detected = detect_columns(df, seq_override, fitness_override, info["properties"])
        if info.get("combo_positions"):
            detected["combo_positions"] = info["combo_positions"]
        aliases = info.get("property_aliases") or {}

        # If source stores only the combo (not full-length seq), expand it
        # into the WT at the configured positions so output seqs are usable
        # by methods that need full sequences.
        expand = info.get("expand_combo")
        if expand:
            wt = expand["wt"]
            zero_idx = [p - 1 for p in expand["positions"]]
            seq_col = detected.get("seq")
            def _expand(combo, _wt=wt, _idx=zero_idx):
                if not isinstance(combo, str) or len(combo) != len(_idx):
                    return None
                s = list(_wt)
                for i, pos in enumerate(_idx):
                    s[pos] = combo[i]
                return "".join(s)
            df = df.copy()
            df["_AACombo"] = df[seq_col].values
            df[seq_col] = df[seq_col].map(_expand)
            df = df.dropna(subset=[seq_col])
            detected["combo_col"] = "_AACombo"

        try:
            if info["properties"]:
                # Joint multi-property file
                joint_path = data_dir / name / "data.csv"
                joint_stats = write_dataset(df, detected, joint_path,
                                            info["properties"], aliases)
                summary[name] = {"status": "OK", "joint": str(joint_path), **joint_stats}
                # Per-property single-objective files
                for prop in detected["properties"]:
                    suffix = aliases.get(prop, prop.replace("fitness_", ""))
                    sub_path = data_dir / f"{name}_{suffix}" / "data.csv"
                    single = pd.DataFrame({
                        "seq": df[detected["seq"]].values,
                        "fitness": df[prop].astype(float).values,
                    }).dropna().drop_duplicates(subset="seq", keep="first").sort_values(
                        "fitness", ascending=False).reset_index(drop=True)
                    sub_path.parent.mkdir(parents=True, exist_ok=True)
                    single.to_csv(sub_path, index=False)
                    print(f"  Wrote {sub_path} ({len(single)} variants, prop={prop})")
            else:
                out_path = data_dir / name / "data.csv"
                stats = write_dataset(df, detected, out_path, None)
                summary[name] = {"status": "OK", "path": str(out_path), **stats}
                print(f"  Wrote {out_path} ({stats['n_variants']} variants, "
                      f"seq_len={stats['seq_len']})")
        except Exception as e:
            summary[name] = {"status": "FAILED", "error": str(e)}
            print(f"  ERROR: {e}", file=sys.stderr)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for name, res in summary.items():
        if res["status"] == "OK":
            n = res.get("n_variants", "?")
            print(f"  {name:<12} OK    n={n}")
        else:
            print(f"  {name:<12} FAIL  {res.get('error', '?')}")
    ok = sum(1 for r in summary.values() if r["status"] == "OK")
    print(f"\n  {ok}/{len(summary)} datasets prepared.")

    return 0 if ok == len(summary) else 2


if __name__ == "__main__":
    sys.exit(main())
