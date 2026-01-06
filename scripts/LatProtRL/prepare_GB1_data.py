#!/usr/bin/env python
"""
prepare_GB1_data.py - Prepare GB1 dataset for LatProtRL

This script converts the GB1 data from the benchmark format to the LatProtRL format
and creates the necessary data files for training.

Usage:
    python prepare_GB1_data.py --data_dir /path/to/data
"""

import os
import argparse
import numpy as np
import pandas as pd


# GB1 constants
GB1_WILDTYPE = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
GB1_MUTABLE_POSITIONS = [38, 39, 40, 53]  # 0-indexed: V39, D40, G41, V54


def prepare_gb1_data(
    data_dir: str = "/home/xux/Desktop/AlphaVariant/Benchmark/data",
    output_dir: str = "data/GB1",
    level: str = "medium",
):
    """
    Prepare GB1 data for LatProtRL.

    Args:
        data_dir: Source data directory
        output_dir: Output directory for LatProtRL data
        level: Difficulty level ('hard' or 'medium')
    """
    print(f"Preparing GB1 data for level: {level}")

    # Load original data
    data_path = os.path.join(data_dir, 'GB1', 'data.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"GB1 data not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} sequences from {data_path}")

    # Normalize fitness to [0, 1]
    fitness_norm = df['fitness'].values / df['fitness'].max()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create training data based on level
    if level == 'hard':
        # Hard: start from low-fitness sequences (bottom 20%)
        threshold = np.percentile(fitness_norm, 20)
        train_mask = fitness_norm <= threshold
        description = "bottom 20% fitness"
    else:  # medium
        # Medium: start from medium-fitness sequences (40-60%)
        threshold_low = np.percentile(fitness_norm, 40)
        threshold_high = np.percentile(fitness_norm, 60)
        train_mask = (fitness_norm >= threshold_low) & (fitness_norm <= threshold_high)
        description = "40-60% fitness percentile"

    train_df = df[train_mask].copy()
    train_df['target'] = fitness_norm[train_mask]
    train_df = train_df.rename(columns={'seq': 'sequence'})

    # Select relevant columns
    train_df = train_df[['sequence', 'target']]

    # Add augmentation column (for compatibility)
    train_df['augmented'] = 0
    train_df['source'] = 'true'

    # Save training data
    train_path = os.path.join(output_dir, f'{level}.csv')
    train_df.to_csv(train_path, index=False)
    print(f"Saved {len(train_df)} training sequences ({description}) to {train_path}")

    # Create 'all.csv' with all sequences for evaluation
    all_df = df.copy()
    all_df['target'] = fitness_norm
    all_df = all_df.rename(columns={'seq': 'sequence'})
    all_df = all_df[['sequence', 'target']]

    all_path = os.path.join(output_dir, 'all.csv')
    all_df.to_csv(all_path, index=False)
    print(f"Saved all {len(all_df)} sequences to {all_path}")

    # Print statistics
    print(f"\nDataset statistics:")
    print(f"  Total sequences: {len(df)}")
    print(f"  Training sequences ({level}): {len(train_df)}")
    print(f"  Sequence length: {len(df['seq'].iloc[0])}")
    print(f"  Fitness range: [{fitness_norm.min():.4f}, {fitness_norm.max():.4f}]")
    print(f"  Training fitness range: [{train_df['target'].min():.4f}, {train_df['target'].max():.4f}]")

    return train_df, all_df


def main():
    parser = argparse.ArgumentParser(description="Prepare GB1 data for LatProtRL")
    parser.add_argument("--data_dir", type=str,
                        default="/home/xux/Desktop/AlphaVariant/Benchmark/data",
                        help="Source data directory")
    parser.add_argument("--output_dir", type=str,
                        default="data/GB1",
                        help="Output directory")

    args = parser.parse_args()

    # Prepare both levels
    for level in ['medium', 'hard']:
        prepare_gb1_data(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            level=level,
        )
        print()


if __name__ == "__main__":
    main()
