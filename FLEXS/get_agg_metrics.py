#!/usr/bin/env python
"""
get_agg_metrics.py - Aggregate metrics from multiple AdaLead runs

Usage:
    python get_agg_metrics.py
    python get_agg_metrics.py --results_dir results/GB1_AdaLead_experiments/GB1
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from glob import glob


def load_metrics_from_dir(results_dir: str):
    """Load all metrics JSON files from a directory."""
    pattern = os.path.join(results_dir, "metrics_seed*.json")
    metric_files = sorted(glob(pattern))

    if not metric_files:
        print(f"No metrics files found matching: {pattern}")
        return []

    print(f"Found {len(metric_files)} metrics files")

    metrics_list = []
    for f in metric_files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                # Metrics are nested under 'metrics' key
                if 'metrics' in data:
                    metrics_list.append(data['metrics'])
                else:
                    metrics_list.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    return metrics_list


def aggregate_metrics(metrics_list):
    """Aggregate metrics across all runs."""
    if not metrics_list:
        return None

    # ALDE metric order
    metric_names = [
        'high_fitness_proximity',
        'novelty',
        'batch_diversity',
        'normalized_fitness_median_top128',
        'normalized_fitness_median_top256',
        'max_fitness',
        'spearman_correlation',
        'epistatic_correlation',
        'recall_high_order',
        'simple_regret',
        'miscalibration_area',
        'expected_calibration_error',
    ]

    aggregated = {}
    for name in metric_names:
        values = [m.get(name, 0.0) for m in metrics_list]
        aggregated[name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }

    # Global max hit count
    hit_count = sum(m.get('global_max_hit_count', 0) for m in metrics_list)
    aggregated['global_max_hit_rate'] = {
        'count': hit_count,
        'rate': hit_count / len(metrics_list),
        'total': len(metrics_list)
    }

    return aggregated


def print_summary(aggregated, n_runs):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("AGGREGATED METRICS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<45} {'Mean':>10} {'Std':>10}")
    print("-" * 70)

    for metric, stats in aggregated.items():
        if metric == 'global_max_hit_rate':
            rate = stats['rate'] * 100
            count = stats['count']
            total = stats['total']
            print(f"{metric:<45} {rate:>9.1f}% ({count}/{total})")
        else:
            print(f"{metric:<45} {stats['mean']:>10.4f} {stats['std']:>10.4f}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate metrics from AdaLead experiment runs"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/GB1_AdaLead_experiments/GB1",
        help="Directory containing metrics_seed*.json files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (optional)"
    )

    args = parser.parse_args()

    # Load metrics
    metrics_list = load_metrics_from_dir(args.results_dir)

    if not metrics_list:
        print("No metrics to aggregate.")
        return

    # Aggregate
    aggregated = aggregate_metrics(metrics_list)

    # Print summary
    print_summary(aggregated, len(metrics_list))

    # Save to CSV if requested
    output_path = args.output or os.path.join(args.results_dir, "aggregated_metrics.csv")

    rows = []
    for metric, stats in aggregated.items():
        if metric == 'global_max_hit_rate':
            rows.append({
                'metric': metric,
                'mean': stats['rate'],
                'std': 0,
                'min': stats['count'],
                'max': stats['total']
            })
        else:
            rows.append({
                'metric': metric,
                'mean': stats['mean'],
                'std': stats['std'],
                'min': stats['min'],
                'max': stats['max']
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nAggregated metrics saved to: {output_path}")

    # Also save as JSON
    json_path = output_path.replace('.csv', '.json')
    with open(json_path, 'w') as f:
        json.dump({
            'aggregated_metrics': aggregated,
            'n_runs': len(metrics_list)
        }, f, indent=2)
    print(f"Aggregated metrics saved to: {json_path}")


if __name__ == "__main__":
    main()
