#!/usr/bin/env python
"""
plot_radar.py - Radar/spider charts comparing methods across metrics

Produces radar charts showing each method's performance across multiple metrics
for a given dataset. Useful for visualizing trade-offs between exploration,
exploitation, and model quality.

Usage:
    python plot_radar.py
    python plot_radar.py --datasets GB1 --methods ALDE EvoPlay AlphaVariant
    python plot_radar.py --output_dir figures/
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

ALL_METHODS = ['ALDE', 'EvoPlay', 'LatProtRL', 'FLEXS', 'AiCE', 'delta_cs', 'AlphaVariant']
ALL_DATASETS = ['GB1', 'AAV_med', 'AAV_hard', 'GFP_med', 'GFP_hard']

# Metrics for radar chart — balanced across categories
RADAR_METRICS = [
    'max_fitness',
    'normalized_fitness_median_top128',
    'high_fitness_proximity',
    'novelty',
    'batch_diversity',
    'spearman_correlation',
]

RADAR_LABELS = [
    'Max Fitness',
    'Norm. Fitness\n(Top-128)',
    'High-Fitness\nProximity',
    'Novelty',
    'Batch\nDiversity',
    'Spearman\nCorrelation',
]

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


def find_metrics(base_dir: Path, method: str, dataset: str) -> List[Dict]:
    """Find metric values from result JSONs."""
    metrics_list = []
    method_dir = base_dir / method
    if not method_dir.exists():
        return metrics_list

    for path in method_dir.rglob('metrics_seed*.json'):
        if dataset.lower() not in str(path).lower():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            m = data.get('metrics', data)
            if 'max_fitness' in m:
                metrics_list.append(m)
        except (json.JSONDecodeError, KeyError):
            continue

    return metrics_list


def aggregate_for_radar(metrics_list: List[Dict], radar_metrics: List[str]) -> Optional[List[float]]:
    """Get mean values for radar metrics."""
    if not metrics_list:
        return None

    values = []
    for metric in radar_metrics:
        vals = [m.get(metric, float('nan')) for m in metrics_list]
        vals = [v for v in vals if not np.isnan(v)]
        values.append(np.mean(vals) if vals else 0.0)
    return values


def normalize_radar_values(
    all_values: Dict[str, List[float]],
    radar_metrics: List[str],
) -> Dict[str, List[float]]:
    """Normalize each metric to [0, 1] across all methods for fair comparison."""
    n_metrics = len(radar_metrics)
    # Find min/max per metric
    mins = [float('inf')] * n_metrics
    maxs = [float('-inf')] * n_metrics

    for values in all_values.values():
        for i, v in enumerate(values):
            mins[i] = min(mins[i], v)
            maxs[i] = max(maxs[i], v)

    normalized = {}
    for method, values in all_values.items():
        norm = []
        for i, v in enumerate(values):
            if maxs[i] - mins[i] > 1e-10:
                norm.append((v - mins[i]) / (maxs[i] - mins[i]))
            else:
                norm.append(0.5)
        normalized[method] = norm

    return normalized


def plot_radar_chart(
    values_dict: Dict[str, List[float]],
    labels: List[str],
    title: str,
    output_path: Path,
    figsize: tuple = (8, 8),
):
    """Plot a single radar chart."""
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for method, values in values_dict.items():
        vals = values + values[:1]  # Close
        color = METHOD_COLORS.get(method, '#333333')
        ax.plot(angles, vals, 'o-', linewidth=1.5, label=method, color=color, markersize=4)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7, alpha=0.7)
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    if not HAS_MPL:
        print("Error: matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Generate radar charts for benchmark comparison")
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

    print(f"Generating radar charts...")
    print(f"  Results dir: {base_dir}")
    print(f"  Output dir: {output_dir}")

    for dataset in datasets:
        raw_values = {}
        for method in methods:
            metrics_list = find_metrics(base_dir, method, dataset)
            vals = aggregate_for_radar(metrics_list, RADAR_METRICS)
            if vals is not None:
                raw_values[method] = vals

        if len(raw_values) < 2:
            print(f"  {dataset}: Not enough methods with data (need ≥ 2), skipping.")
            continue

        # Normalize to [0,1] for fair radar comparison
        normalized = normalize_radar_values(raw_values, RADAR_METRICS)

        output_path = output_dir / f'radar_{dataset}.png'
        plot_radar_chart(
            normalized, RADAR_LABELS,
            f'Method Comparison — {dataset}',
            output_path,
        )

    print("Done.")


if __name__ == "__main__":
    main()
