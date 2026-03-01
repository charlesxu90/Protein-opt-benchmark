#!/usr/bin/env python
"""
plot_optimization_curves.py - Plot optimization curves comparing methods

Plots the best fitness found at each round for multiple methods on each dataset.
Shows mean ± std across seeds as shaded regions.

Usage:
    python plot_optimization_curves.py
    python plot_optimization_curves.py --datasets GB1 AAV_med --methods ALDE EvoPlay
    python plot_optimization_curves.py --output_dir figures/
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

ALL_METHODS = ['ALDE', 'EvoPlay', 'LatProtRL', 'FLEXS', 'AiCE', 'delta_cs', 'AlphaVariant', 'Random', 'GreedyWalk']
ALL_DATASETS = ['GB1', 'AAV_med', 'AAV_hard', 'GFP_med', 'GFP_hard']

METHOD_COLORS = {
    'ALDE': '#1f77b4',
    'EvoPlay': '#ff7f0e',
    'LatProtRL': '#2ca02c',
    'FLEXS': '#d62728',
    'AiCE': '#9467bd',
    'delta_cs': '#8c564b',
    'AlphaVariant': '#e377c2',
    'Random': '#7f7f7f',
    'GreedyWalk': '#bcbd22',
}

METHOD_MARKERS = {
    'ALDE': 'o', 'EvoPlay': 's', 'LatProtRL': '^', 'FLEXS': 'D',
    'AiCE': 'v', 'delta_cs': 'P', 'AlphaVariant': '*', 'Random': 'x', 'GreedyWalk': '+',
}


def find_trajectories(base_dir: Path, method: str, dataset: str) -> List[List[float]]:
    """Find fitness trajectories from metrics JSON files."""
    trajectories = []
    method_dir = base_dir / method
    if not method_dir.exists():
        return trajectories

    for path in method_dir.rglob('metrics_seed*.json'):
        if dataset.lower() not in str(path).lower():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            traj = data.get('fitness_trajectory', [])
            if not traj and 'metrics' in data:
                traj = data['metrics'].get('fitness_trajectory', [])
            if traj:
                trajectories.append(traj)
        except (json.JSONDecodeError, KeyError):
            continue

    return trajectories


def plot_dataset(
    base_dir: Path,
    dataset: str,
    methods: List[str],
    output_dir: Path,
    figsize: tuple = (8, 5),
):
    """Plot optimization curves for a single dataset."""
    fig, ax = plt.subplots(figsize=figsize)

    has_data = False
    for method in methods:
        trajectories = find_trajectories(base_dir, method, dataset)
        if not trajectories:
            continue

        has_data = True
        # Pad to equal length
        max_len = max(len(t) for t in trajectories)
        padded = []
        for t in trajectories:
            # Extend with last value
            extended = t + [t[-1]] * (max_len - len(t))
            # Cumulative max
            extended = list(np.maximum.accumulate(extended))
            padded.append(extended)

        arr = np.array(padded)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        rounds = np.arange(len(mean))

        color = METHOD_COLORS.get(method, '#333333')
        marker = METHOD_MARKERS.get(method, 'o')
        ax.plot(rounds, mean, color=color, marker=marker, markersize=5,
                label=f"{method} (n={len(trajectories)})", linewidth=1.5)
        if len(trajectories) > 1:
            ax.fill_between(rounds, mean - std, mean + std, alpha=0.15, color=color)

    if not has_data:
        plt.close(fig)
        return False

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Best Fitness', fontsize=12)
    ax.set_title(f'Optimization Curves — {dataset}', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = output_dir / f'optimization_curve_{dataset}.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return True


def plot_all_datasets_grid(
    base_dir: Path,
    datasets: List[str],
    methods: List[str],
    output_dir: Path,
):
    """Plot optimization curves for all datasets in a grid."""
    n_datasets = len(datasets)
    ncols = min(3, n_datasets)
    nrows = (n_datasets + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        has_data = False

        for method in methods:
            trajectories = find_trajectories(base_dir, method, dataset)
            if not trajectories:
                continue
            has_data = True

            max_len = max(len(t) for t in trajectories)
            padded = []
            for t in trajectories:
                extended = t + [t[-1]] * (max_len - len(t))
                extended = list(np.maximum.accumulate(extended))
                padded.append(extended)

            arr = np.array(padded)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            rounds = np.arange(len(mean))

            color = METHOD_COLORS.get(method, '#333333')
            ax.plot(rounds, mean, color=color, label=method, linewidth=1.5)
            if len(trajectories) > 1:
                ax.fill_between(rounds, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_title(dataset, fontsize=12)
        ax.set_xlabel('Round')
        ax.set_ylabel('Best Fitness')
        ax.grid(True, alpha=0.3)
        if not has_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    # Hide unused axes
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)

    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=min(5, len(handles)),
                   bbox_to_anchor=(0.5, 1.02), fontsize=9)

    fig.tight_layout()
    output_path = output_dir / 'optimization_curves_grid.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    if not HAS_MPL:
        print("Error: matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Plot optimization curves")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--methods", type=str, nargs='+', default=None)
    parser.add_argument("--datasets", type=str, nargs='+', default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    benchmark_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    base_dir = Path(args.results_dir) if args.results_dir else benchmark_root
    methods = args.methods or ALL_METHODS
    datasets = args.datasets or ALL_DATASETS
    output_dir = Path(args.output_dir) if args.output_dir else benchmark_root / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating optimization curve plots...")
    print(f"  Results dir: {base_dir}")
    print(f"  Output dir: {output_dir}")

    # Individual plots
    for dataset in datasets:
        plot_dataset(base_dir, dataset, methods, output_dir)

    # Grid plot
    plot_all_datasets_grid(base_dir, datasets, methods, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
