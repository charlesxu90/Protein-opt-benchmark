#!/usr/bin/env python
"""
prepare_proteingym.py - Download and prepare ProteinGym DMS assays for benchmarking

Downloads selected DMS substitution assays from ProteinGym and converts them
to the standardized data/<name>/data.csv format (seq, fitness columns).

Selected assays (8 datasets covering diverse protein families and sizes):
    - BLAT_ECOLX_Stiffler_2015   (TEM-1 beta-lactamase, 286aa, 4996 mutants)
    - CALM1_HUMAN_Weile_2017     (Calmodulin, 149aa, 1813 mutants)
    - GFP_AEQVI_Sarkisyan_2016   (avGFP, 238aa, 51714 mutants)
    - DYR_ECOLI_Nguyen_2023      (DHFR, 159aa, 2916 mutants)
    - AMIE_PSEAE_Wrenbeck_2017   (Amidase, 346aa, 6227 mutants)
    - KKA2_KLEPN_Melnikov_2014   (Aminoglycoside kinase, 264aa, 4960 mutants)
    - MK01_HUMAN_Brenan_2016     (ERK2/MAPK1, 360aa, 6809 mutants)
    - HIS7_YEAST_Pokusaeva_2019  (Imidazoleglycerol-phosphate synthase, 220aa, 496137 mutants)

Source: https://github.com/OATML-Markslab/ProteinGym
Data URL: https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/

Usage:
    # Download and prepare all datasets
    python prepare_proteingym.py

    # Prepare specific datasets only
    python prepare_proteingym.py --datasets BLAT_ECOLX CALM1_HUMAN

    # List available datasets
    python prepare_proteingym.py --list

    # Use a local ProteinGym zip instead of downloading
    python prepare_proteingym.py --local_zip /path/to/DMS_ProteinGym_substitutions.zip

    # Subsample large datasets to a maximum number of variants
    python prepare_proteingym.py --max_variants 50000
"""

from __future__ import annotations
import argparse
import os
import sys
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ProteinGym v1.3 download URL
PROTEINGYM_URL = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip"

# Selected assays: short_name -> ProteinGym filename (without .csv extension)
ASSAYS = {
    "BLAT_ECOLX": {
        "filename": "BLAT_ECOLX_Stiffler_2015",
        "description": "TEM-1 beta-lactamase (E. coli), 286aa, ampicillin resistance",
        "seq_len": 286,
    },
    "CALM1_HUMAN": {
        "filename": "CALM1_HUMAN_Weile_2017",
        "description": "Calmodulin (human), 149aa, complementation fitness",
        "seq_len": 149,
    },
    "GFP_AEQVI": {
        "filename": "GFP_AEQVI_Sarkisyan_2016",
        "description": "avGFP (Aequorea victoria), 238aa, fluorescence",
        "seq_len": 238,
    },
    "DYR_ECOLI": {
        "filename": "DYR_ECOLI_Nguyen_2023",
        "description": "DHFR (E. coli), 159aa, trimethoprim resistance",
        "seq_len": 159,
    },
    "AMIE_PSEAE": {
        "filename": "AMIE_PSEAE_Wrenbeck_2017",
        "description": "Amidase (P. aeruginosa), 346aa, acetamide growth",
        "seq_len": 346,
    },
    "KKA2_KLEPN": {
        "filename": "KKA2_KLEPN_Melnikov_2014",
        "description": "Aminoglycoside kinase (K. pneumoniae), 264aa, kanamycin resistance",
        "seq_len": 264,
    },
    "MK01_HUMAN": {
        "filename": "MK01_HUMAN_Brenan_2016",
        "description": "ERK2/MAPK1 kinase (human), 360aa, cell viability",
        "seq_len": 360,
    },
    "HIS7_YEAST": {
        "filename": "HIS7_YEAST_Pokusaeva_2019",
        "description": "IGP synthase (S. cerevisiae), 220aa, histidine biosynthesis",
        "seq_len": 220,
    },
    # ---------------------------------------------------------------------
    # Phase 1.2 additions: broaden coverage for refined-plan Tasks 1 + 2
    # All names follow the ProteinGym v1.3 substitutions filename convention
    # (<UNIPROT>_<Author>_<Year>). Verify exact match with `--list` after
    # download if you encounter "File not found in zip".
    # ---------------------------------------------------------------------
    "PABP_YEAST": {
        "filename": "PABP_YEAST_Melamed_2013",
        "description": "Poly(A)-binding protein (S. cerevisiae), RNA-binding fitness",
        "seq_len": 75,
    },
    "HSP82_YEAST": {
        "filename": "HSP82_YEAST_Mishra_2016",
        "description": "Hsp90 chaperone (S. cerevisiae), competitive growth",
        "seq_len": 220,
    },
    "POLG_HCVJF": {
        "filename": "POLG_HCVJF_Qi_2014",
        "description": "HCV NS5A (Hepatitis C virus), viral fitness",
        "seq_len": 466,
    },
    "DLG4_HUMAN": {
        "filename": "DLG4_HUMAN_Faure_2021",
        "description": "PSD95 PDZ3 domain (human), peptide binding",
        "seq_len": 84,
    },
    "TPK1_HUMAN": {
        "filename": "TPK1_HUMAN_Weile_2017",
        "description": "Thiamine pyrophosphokinase (human), complementation",
        "seq_len": 243,
    },
    "UBE4B_MOUSE": {
        "filename": "UBE4B_MOUSE_Starita_2013",
        "description": "U-box E3 ubiquitin ligase (mouse), in vitro activity",
        "seq_len": 102,
    },
    "Q2N0S5_9HIV1": {
        "filename": "Q2N0S5_9HIV1_Haddox_2018",
        "description": "HIV-1 Env (BG505), viral infectivity",
        "seq_len": 836,
    },
    "SPG1_STRSG_Wu": {
        "filename": "SPG1_STRSG_Wu_2016",
        "description": "GB1 (4-site combinatorial, alternative to default GB1)",
        "seq_len": 56,
    },
}


def download_proteingym(dest_dir: str, url: str = PROTEINGYM_URL) -> str:
    """Download ProteinGym substitutions zip file."""
    import urllib.request

    zip_path = os.path.join(dest_dir, "DMS_ProteinGym_substitutions.zip")
    if os.path.exists(zip_path):
        print(f"  Using cached zip: {zip_path}")
        return zip_path

    print(f"  Downloading from {url}")
    print(f"  This may take a few minutes (~1GB)...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"  Downloaded to: {zip_path}")
    return zip_path


def extract_assay(
    zip_path: str,
    assay_filename: str,
    extract_dir: str,
) -> Optional[str]:
    """Extract a single assay CSV from the ProteinGym zip."""
    target = f"{assay_filename}.csv"

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Search for the file (may be in a subdirectory)
        matching = [n for n in zf.namelist() if n.endswith(target)]
        if not matching:
            # Try partial match
            matching = [n for n in zf.namelist() if assay_filename in n and n.endswith('.csv')]

        if not matching:
            print(f"  WARNING: {target} not found in zip")
            return None

        member = matching[0]
        extracted = zf.extract(member, extract_dir)
        return extracted


def convert_assay(
    csv_path: str,
    output_path: str,
    max_variants: Optional[int] = None,
    singles_only: bool = False,
) -> Dict:
    """Convert a ProteinGym CSV to standardized format.

    ProteinGym columns: mutated_sequence, DMS_score, DMS_score_bin, mutant
    Output columns: seq, fitness
    """
    df = pd.read_csv(csv_path)

    # Validate expected columns
    if 'mutated_sequence' not in df.columns or 'DMS_score' not in df.columns:
        raise ValueError(f"Expected columns 'mutated_sequence' and 'DMS_score', "
                         f"got: {list(df.columns)}")

    # Filter to single mutants only if requested
    if singles_only and 'mutant' in df.columns:
        singles = df[~df['mutant'].str.contains(':')]
        print(f"  Filtered to {len(singles)} single mutants (from {len(df)} total)")
        df = singles

    # Drop duplicates by sequence
    n_before = len(df)
    df = df.drop_duplicates(subset='mutated_sequence', keep='first')
    if len(df) < n_before:
        print(f"  Removed {n_before - len(df)} duplicate sequences")

    # Drop rows with NaN fitness
    df = df.dropna(subset=['DMS_score'])

    # Subsample if too large
    if max_variants and len(df) > max_variants:
        print(f"  Subsampling from {len(df)} to {max_variants} variants")
        df = df.sample(n=max_variants, random_state=42)

    # Convert to standard format
    out_df = pd.DataFrame({
        'seq': df['mutated_sequence'].values,
        'fitness': df['DMS_score'].values,
    })

    # Sort by fitness descending for convenience
    out_df = out_df.sort_values('fitness', ascending=False).reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)

    stats = {
        'n_variants': len(out_df),
        'seq_len': len(out_df['seq'].iloc[0]),
        'fitness_min': float(out_df['fitness'].min()),
        'fitness_max': float(out_df['fitness'].max()),
        'fitness_mean': float(out_df['fitness'].mean()),
        'fitness_std': float(out_df['fitness'].std()),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare ProteinGym DMS assays for benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets", type=str, nargs='+', default=None,
        help="Specific datasets to prepare (default: all). Use short names like BLAT_ECOLX",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available datasets and exit",
    )
    parser.add_argument(
        "--local_zip", type=str, default=None,
        help="Path to local ProteinGym zip file (skip download)",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Output data directory (default: data/ relative to Benchmark root)",
    )
    parser.add_argument(
        "--max_variants", type=int, default=None,
        help="Maximum number of variants per dataset (subsample if larger)",
    )
    parser.add_argument(
        "--singles_only", action="store_true",
        help="Keep only single-mutant variants (exclude multi-mutants)",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Directory to cache downloaded zip (default: /tmp/proteingym_cache)",
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print("\nAvailable ProteinGym datasets:\n")
        print(f"  {'Short Name':<16} {'ProteinGym ID':<36} {'Len':>4}  Description")
        print(f"  {'-'*15:<16} {'-'*35:<36} {'-'*4:>4}  {'-'*40}")
        for name, info in ASSAYS.items():
            print(f"  {name:<16} {info['filename']:<36} {info['seq_len']:>4}  {info['description']}")
        print(f"\nTotal: {len(ASSAYS)} datasets")
        return

    # Resolve paths
    benchmark_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if args.data_dir is None:
        args.data_dir = os.path.join(benchmark_root, 'data')
    if args.cache_dir is None:
        args.cache_dir = '/tmp/proteingym_cache'

    os.makedirs(args.cache_dir, exist_ok=True)

    # Determine which datasets to process
    if args.datasets:
        selected = {}
        for name in args.datasets:
            if name in ASSAYS:
                selected[name] = ASSAYS[name]
            else:
                print(f"WARNING: Unknown dataset '{name}'. Available: {list(ASSAYS.keys())}")
        if not selected:
            print("No valid datasets specified.")
            return
    else:
        selected = ASSAYS

    print(f"\nPreparing {len(selected)} ProteinGym dataset(s)")
    print(f"Output directory: {args.data_dir}")

    # Download or locate zip
    if args.local_zip:
        zip_path = args.local_zip
        if not os.path.exists(zip_path):
            print(f"ERROR: Local zip not found: {zip_path}")
            return
    else:
        print("\nStep 1: Download ProteinGym substitutions")
        zip_path = download_proteingym(args.cache_dir)

    # Extract and convert each assay
    print("\nStep 2: Extract and convert assays")
    extract_dir = os.path.join(args.cache_dir, 'extracted')
    os.makedirs(extract_dir, exist_ok=True)

    results = {}
    for short_name, info in selected.items():
        print(f"\n--- {short_name} ({info['description']}) ---")

        csv_path = extract_assay(zip_path, info['filename'], extract_dir)
        if csv_path is None:
            results[short_name] = {'status': 'FAILED', 'error': 'File not found in zip'}
            continue

        output_path = os.path.join(args.data_dir, short_name, 'data.csv')

        try:
            stats = convert_assay(
                csv_path, output_path,
                max_variants=args.max_variants,
                singles_only=args.singles_only,
            )
            results[short_name] = {'status': 'OK', **stats}
            print(f"  Saved: {output_path}")
            print(f"  Variants: {stats['n_variants']}, Seq len: {stats['seq_len']}")
            print(f"  Fitness: [{stats['fitness_min']:.4f}, {stats['fitness_max']:.4f}] "
                  f"(mean={stats['fitness_mean']:.4f}, std={stats['fitness_std']:.4f})")
        except Exception as e:
            results[short_name] = {'status': 'FAILED', 'error': str(e)}
            print(f"  ERROR: {e}")

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"  {'Dataset':<16} {'Status':<8} {'Variants':>10} {'SeqLen':>8} {'Fitness Range':>20}")
    print(f"  {'-'*15:<16} {'-'*7:<8} {'-'*9:>10} {'-'*7:>8} {'-'*19:>20}")
    for name, res in results.items():
        if res['status'] == 'OK':
            frange = f"[{res['fitness_min']:.2f}, {res['fitness_max']:.2f}]"
            print(f"  {name:<16} {'OK':<8} {res['n_variants']:>10,} {res['seq_len']:>8} {frange:>20}")
        else:
            print(f"  {name:<16} {'FAILED':<8} {res.get('error', 'unknown'):>40}")

    ok_count = sum(1 for r in results.values() if r['status'] == 'OK')
    print(f"\n  {ok_count}/{len(results)} datasets prepared successfully.")
    print(f"  Data saved to: {args.data_dir}/")
    print(f"\n  These datasets can be used with any method's run_generic.py script:")
    example = next((n for n, r in results.items() if r["status"] == "OK"), "<dataset>")
    print(f"    <method>/env/bin/python scripts/<method>/run_generic.py --dataset {example} --seed 621")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
