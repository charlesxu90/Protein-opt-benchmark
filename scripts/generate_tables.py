#!/usr/bin/env python
"""
generate_tables.py - Generate comparison tables from benchmark results

Scans result directories for aggregated metrics across methods and datasets,
then produces summary tables in LaTeX, Markdown, and CSV formats.

Also runs pairwise statistical tests between methods when multiple seeds are available.

Usage:
    # Generate tables from default results directories
    python generate_tables.py

    # Specify methods and datasets
    python generate_tables.py --methods ALDE EvoPlay FLEXS --datasets GB1 AAV_med

    # Output format and directory
    python generate_tables.py --format latex --output_dir tables/

    # Include statistical tests
    python generate_tables.py --stat_test wilcoxon

    # Scan a specific results base directory
    python generate_tables.py --results_dir /path/to/results
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.metrics import paired_comparison
from utils.io import export_summary_table

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# All methods and datasets in the benchmark
ALL_METHODS = [
    # Established
    'ALDE', 'EvoPlay', 'LatProtRL', 'FLEXS', 'AiCE', 'delta_cs',
    'AlphaVariant', 'Random', 'GreedyWalk',
    # Phase 2.1 baselines (scaffolding; not yet runnable)
    'EVOLVEpro', 'ftMLDE', 'MULTIevolve',
]
ALL_DATASETS = [
    # Original
    'GB1', 'AAV_med', 'AAV_hard', 'GFP_med', 'GFP_hard',
    # Phase 1.2 additions (CombinGym)
    'CR9114', 'CreiLOV', 'eqFP611_blue', 'eqFP611_red',
    # Phase 1.2 additions (ProteinGym; populated by prepare_proteingym.py)
    'BLAT_ECOLX', 'CALM1_HUMAN', 'GFP_AEQVI', 'DYR_ECOLI',
    'AMIE_PSEAE', 'KKA2_KLEPN', 'MK01_HUMAN', 'HIS7_YEAST',
    'PABP_YEAST', 'HSP82_YEAST', 'POLG_HCVJF', 'DLG4_HUMAN',
    'TPK1_HUMAN', 'UBE4B_MOUSE', 'Q2N0S5_9HIV1',
]

# Core metrics to include in the main comparison table.
# Aligned with refined_benchmark_plan.md Section 4.
CORE_METRICS = [
    'max_fitness',
    'simple_regret',
    'normalized_fitness_median_top128',
    'high_fitness_proximity',
    'novelty',
    'batch_diversity',
    'spearman_correlation',
    'auoc',                # Area Under Optimization Curve
    'hit_rate_value',      # EVOLVEpro-style hit rate at last round
    'global_max_hit_count', # CLADE / Wittmann global-max recovery
]


def find_result_files(base_dir: Path, method: str, dataset: str) -> List[Path]:
    """Find all metrics JSON files for a method-dataset pair."""
    candidates = [
        base_dir / method / 'results' / dataset,
        base_dir / method / f'results/{dataset}_*',
        base_dir / f'results/{dataset}_{method}' / dataset,
        base_dir / method / dataset,
    ]

    results = []
    for pattern_dir in candidates:
        parent = pattern_dir.parent
        if not parent.exists():
            continue
        # Search for metrics files recursively
        for path in parent.rglob('metrics_seed*.json'):
            if dataset.lower() in str(path).lower() or method.lower() in str(path).lower():
                results.append(path)

    # Also check method-specific result directories
    method_dir = base_dir / method
    if method_dir.exists():
        for path in method_dir.rglob('metrics_seed*.json'):
            if dataset.lower() in str(path).lower():
                if path not in results:
                    results.append(path)
        # alphavariant writes alphavariant/results/<DS>_AlphaVariant/seed_<S>/metrics.json
        for path in method_dir.rglob('metrics.json'):
            if dataset.lower() in str(path).lower() or (
                path.parent.name.startswith('seed_')
                and (path.parent.parent.name.lower().startswith(dataset.lower())
                     or method.lower() in path.parent.parent.name.lower())
            ):
                if path not in results:
                    results.append(path)

    return sorted(set(results))


def load_metrics_from_files(files: List[Path]) -> List[Dict]:
    """Load metrics from a list of JSON files.

    Handles three schema flavors:
      - data["metrics"] is a dict (most methods)
      - data["metrics"] is a list of per-round dicts (delta_cs); prefer
        data["final_metrics"], otherwise take the last round
      - top-level dict already has metric keys (legacy)
    """
    metrics_list = []
    for fpath in files:
        try:
            with open(fpath) as f:
                data = json.load(f)
            m = None
            if isinstance(data.get('final_metrics'), dict):
                m = data['final_metrics']
            elif isinstance(data.get('metrics'), dict):
                m = data['metrics']
            elif isinstance(data.get('metrics'), list) and data['metrics']:
                last = data['metrics'][-1]
                if isinstance(last, dict):
                    m = last
            elif 'max_fitness' in data:
                m = data
            if m is not None:
                metrics_list.append(m)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Could not load {fpath}: {e}")
    return metrics_list


def aggregate_metrics(metrics_list: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Compute mean/std/median for each metric across runs."""
    if not metrics_list:
        return {}

    all_values = defaultdict(list)
    for m in metrics_list:
        for key, val in m.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if val is not None and not np.isnan(val):
                    all_values[key].append(val)

    agg = {}
    for key, vals in all_values.items():
        if vals:
            agg[key] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'median': float(np.median(vals)),
                'n': len(vals),
            }
    return agg


def collect_all_results(
    base_dir: Path,
    methods: List[str],
    datasets: List[str],
) -> Dict[Tuple[str, str], Dict]:
    """Collect aggregated results for all method-dataset pairs."""
    results = {}
    for method in methods:
        for dataset in datasets:
            files = find_result_files(base_dir, method, dataset)
            if files:
                metrics_list = load_metrics_from_files(files)
                if metrics_list:
                    agg = aggregate_metrics(metrics_list)
                    results[(method, dataset)] = {
                        'aggregated': agg,
                        'n_runs': len(metrics_list),
                        'raw': metrics_list,
                    }
    return results


def format_cell(mean: float, std: float, n: int, bold: bool = False) -> str:
    """Format a table cell as mean ± std."""
    if n <= 1:
        text = f"{mean:.4f}"
    else:
        text = f"{mean:.4f} ± {std:.4f}"
    if bold:
        return f"**{text}**"
    return text


def format_cell_latex(mean: float, std: float, n: int, bold: bool = False) -> str:
    """Format a table cell for LaTeX."""
    if n <= 1:
        text = f"{mean:.4f}"
    else:
        text = f"{mean:.4f} $\\pm$ {std:.4f}"
    if bold:
        return f"\\textbf{{{text}}}"
    return text


def generate_per_dataset_table(
    results: Dict,
    dataset: str,
    methods: List[str],
    metrics: List[str],
    fmt: str = 'markdown',
) -> str:
    """Generate a comparison table for a single dataset."""
    lines = []

    # Determine which metrics have data
    available_metrics = []
    for metric in metrics:
        for method in methods:
            key = (method, dataset)
            if key in results and metric in results[key]['aggregated']:
                available_metrics.append(metric)
                break

    if not available_metrics:
        return f"No results found for {dataset}.\n"

    # Find best method per metric (highest mean)
    best_method = {}
    for metric in available_metrics:
        best_val = -np.inf
        best_m = None
        # For simple_regret, lower is better
        is_lower_better = metric in ('simple_regret', 'miscalibration_area', 'expected_calibration_error')
        for method in methods:
            key = (method, dataset)
            if key in results and metric in results[key]['aggregated']:
                val = results[key]['aggregated'][metric]['mean']
                if is_lower_better:
                    if best_m is None or val < best_val:
                        best_val = val
                        best_m = method
                else:
                    if val > best_val:
                        best_val = val
                        best_m = method
        best_method[metric] = best_m

    if fmt == 'markdown':
        # Header
        header = "| Method | n |" + " | ".join(m.replace('_', ' ') for m in available_metrics) + " |"
        separator = "| --- | --- |" + " | ".join(["---"] * len(available_metrics)) + " |"
        lines.append(f"### {dataset}\n")
        lines.append(header)
        lines.append(separator)

        for method in methods:
            key = (method, dataset)
            if key not in results:
                continue
            n = results[key]['n_runs']
            cells = [f"| {method} | {n} |"]
            for metric in available_metrics:
                agg = results[key]['aggregated']
                if metric in agg:
                    bold = (best_method.get(metric) == method)
                    cells.append(format_cell(agg[metric]['mean'], agg[metric]['std'], n, bold))
                else:
                    cells.append("—")
            lines.append(" | ".join(cells) + " |")
        lines.append("")

    elif fmt == 'latex':
        n_cols = len(available_metrics) + 2
        lines.append(f"% {dataset}")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(f"\\caption{{Benchmark results on {dataset}}}")
        lines.append(f"\\label{{tab:{dataset.lower()}}}")
        col_spec = "l" + "r" * (n_cols - 1)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        header_cells = ["Method", "n"] + [m.replace('_', '\\_') for m in available_metrics]
        lines.append(" & ".join(header_cells) + " \\\\")
        lines.append("\\midrule")

        for method in methods:
            key = (method, dataset)
            if key not in results:
                continue
            n = results[key]['n_runs']
            cells = [method.replace('_', '\\_'), str(n)]
            for metric in available_metrics:
                agg = results[key]['aggregated']
                if metric in agg:
                    bold = (best_method.get(metric) == method)
                    cells.append(format_cell_latex(agg[metric]['mean'], agg[metric]['std'], n, bold))
                else:
                    cells.append("---")
            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

    elif fmt == 'csv':
        header = ["method", "dataset", "n_runs"] + available_metrics
        lines.append(",".join(header))
        for method in methods:
            key = (method, dataset)
            if key not in results:
                continue
            n = results[key]['n_runs']
            cells = [method, dataset, str(n)]
            for metric in available_metrics:
                agg = results[key]['aggregated']
                if metric in agg:
                    cells.append(f"{agg[metric]['mean']:.6f}")
                else:
                    cells.append("")
            lines.append(",".join(cells))
        lines.append("")

    return "\n".join(lines)


def generate_stat_tests(
    results: Dict,
    dataset: str,
    methods: List[str],
    metric: str = 'max_fitness',
    test: str = 'wilcoxon',
    bonferroni: bool = False,
    alpha: float = 0.05,
) -> str:
    """Generate pairwise statistical test table for a single dataset and metric.

    When `bonferroni` is True, the significance threshold α is divided by the
    number of pairwise comparisons (k(k-1)/2 unordered pairs). The corrected
    threshold is annotated in the header so reviewers can verify it.
    """
    lines = []

    active_methods = [m for m in methods if (m, dataset) in results
                      and metric in results[(m, dataset)]['aggregated']
                      and results[(m, dataset)]['n_runs'] >= 2]

    if len(active_methods) < 2:
        lines.append(f"\n#### Statistical tests: {metric} on {dataset} ({test})\n")
        lines.append("Not enough methods with multiple runs for statistical tests.\n")
        return "\n".join(lines)

    n_pairs = len(active_methods) * (len(active_methods) - 1) // 2
    if bonferroni and n_pairs > 0:
        alpha_corrected = alpha / n_pairs
        header_line = (f"\n#### Statistical tests: {metric} on {dataset} "
                       f"({test}, Bonferroni-corrected α={alpha_corrected:.5f} "
                       f"for {n_pairs} pairs)\n")
        # Adjust significance markers to use corrected threshold
        levels = [(alpha_corrected / 5, "***"), (alpha_corrected, "**"),
                  (alpha, "*")]
    else:
        header_line = f"\n#### Statistical tests: {metric} on {dataset} ({test})\n"
        levels = [(0.001, "***"), (0.01, "**"), (0.05, "*")]
    lines.append(header_line)

    # Header
    header = "| |" + " | ".join(active_methods) + " |"
    sep = "| --- |" + " | ".join(["---"] * len(active_methods)) + " |"
    lines.append(header)
    lines.append(sep)

    for i, ma in enumerate(active_methods):
        cells = [f"| {ma} |"]
        raw_a = [m.get(metric, float('nan')) for m in results[(ma, dataset)]['raw']]
        for j, mb in enumerate(active_methods):
            if i == j:
                cells.append("—")
                continue
            raw_b = [m.get(metric, float('nan')) for m in results[(mb, dataset)]['raw']]
            try:
                stat, pval = paired_comparison(raw_a, raw_b, test=test)
                sig = ""
                for thresh, marker in levels:
                    if pval < thresh:
                        sig = marker
                        break
                cells.append(f"p={pval:.4f}{sig}")
            except Exception:
                cells.append("N/A")
        lines.append(" | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def collect_resource_usage(
    base_dir: Path,
    methods: List[str],
    datasets: List[str],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Aggregate resource.json files written by scripts/hpc/log_resource_use.py.

    Returns a dict (method, dataset) -> {wall_hours_mean, wall_hours_total,
    gpu_hours_total, n_runs_with_resource}.
    """
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for method in methods:
        for dataset in datasets:
            patterns = [
                base_dir / method / "results" / dataset / "**" / "*resource*.json",
                base_dir / "results" / method / dataset / "**" / "*resource*.json",
            ]
            wall_hours = []
            gpu_hours = []
            for pattern in patterns:
                for path in Path(str(pattern).split("**")[0]).rglob("*resource*.json"):
                    try:
                        with open(path) as f:
                            rec = json.load(f)
                    except Exception:
                        continue
                    if "wall_hours" in rec:
                        wall_hours.append(float(rec["wall_hours"]))
                    # GPU hours: approximate as wall_hours * gpus_used (1 if gpu memory > 0)
                    gh = (float(rec.get("wall_hours", 0.0))
                          if rec.get("max_gpu_memory_mib", 0) > 0 else 0.0)
                    gpu_hours.append(gh)
            if wall_hours:
                out[(method, dataset)] = {
                    "wall_hours_mean": float(np.mean(wall_hours)),
                    "wall_hours_total": float(np.sum(wall_hours)),
                    "gpu_hours_total": float(np.sum(gpu_hours)),
                    "n_runs_with_resource": len(wall_hours),
                }
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison tables from benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Base directory containing method results (default: Benchmark root)",
    )
    parser.add_argument(
        "--methods", type=str, nargs='+', default=None,
        help=f"Methods to include (default: all). Options: {ALL_METHODS}",
    )
    parser.add_argument(
        "--datasets", type=str, nargs='+', default=None,
        help=f"Datasets to include (default: all). Options: {ALL_DATASETS}",
    )
    parser.add_argument(
        "--metrics", type=str, nargs='+', default=None,
        help=f"Metrics to include (default: core set)",
    )
    parser.add_argument(
        "--format", type=str, default='markdown', choices=['markdown', 'latex', 'csv', 'all'],
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for tables (default: tables/ in Benchmark root)",
    )
    parser.add_argument(
        "--stat_test", type=str, default=None,
        choices=['wilcoxon', 'mannwhitney', 'ttest', 'permutation'],
        help="Run pairwise statistical tests between methods",
    )
    parser.add_argument(
        "--stat_metric", type=str, default='max_fitness',
        help="Metric to use for statistical tests (default: max_fitness)",
    )
    parser.add_argument(
        "--bonferroni", action="store_true",
        help="Apply Bonferroni correction to significance thresholds",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Base significance level (default 0.05; divided by pair count if --bonferroni)",
    )
    parser.add_argument(
        "--include_resources", action="store_true",
        help="Include GPU-hour / wall-clock columns from resource.json files",
    )

    args = parser.parse_args()

    benchmark_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    base_dir = Path(args.results_dir) if args.results_dir else benchmark_root
    methods = args.methods or ALL_METHODS
    datasets = args.datasets or ALL_DATASETS
    metrics = args.metrics or CORE_METRICS
    output_dir = Path(args.output_dir) if args.output_dir else benchmark_root / 'tables'

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning results in: {base_dir}")
    print(f"Methods: {methods}")
    print(f"Datasets: {datasets}")

    # Collect all results
    results = collect_all_results(base_dir, methods, datasets)

    if not results:
        print("\nNo results found. Make sure result files exist as:")
        print("  <method>/results/<dataset>/**/metrics_seed*.json")
        return

    found_pairs = [(m, d) for m, d in results.keys()]
    print(f"\nFound results for {len(found_pairs)} method-dataset pairs:")
    for m, d in sorted(found_pairs):
        print(f"  {m} × {d}: {results[(m, d)]['n_runs']} runs")

    # Generate tables
    formats = ['markdown', 'latex', 'csv'] if args.format == 'all' else [args.format]

    for fmt in formats:
        ext = {'markdown': 'md', 'latex': 'tex', 'csv': 'csv'}[fmt]
        output_path = output_dir / f"benchmark_comparison.{ext}"

        all_tables = []
        if fmt == 'markdown':
            all_tables.append("# Benchmark Comparison Tables\n")
        elif fmt == 'latex':
            all_tables.append("% Auto-generated benchmark comparison tables")
            all_tables.append("% Generated by scripts/generate_tables.py\n")

        for dataset in datasets:
            has_data = any((m, dataset) in results for m in methods)
            if has_data:
                table = generate_per_dataset_table(results, dataset, methods, metrics, fmt)
                all_tables.append(table)

        # Statistical tests (markdown only for readability)
        if args.stat_test and fmt == 'markdown':
            all_tables.append("\n## Statistical Tests\n")
            for dataset in datasets:
                has_data = any((m, dataset) in results for m in methods)
                if has_data:
                    test_table = generate_stat_tests(
                        results, dataset, methods, args.stat_metric, args.stat_test,
                        bonferroni=args.bonferroni, alpha=args.alpha)
                    all_tables.append(test_table)

        content = "\n".join(all_tables)
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"\nSaved {fmt} table to: {output_path}")

    # Optional: collect GPU-hour / wall-clock from resource.json files
    resource_table: Dict[Tuple[str, str], Dict[str, float]] = {}
    if args.include_resources:
        resource_table = collect_resource_usage(base_dir, methods, datasets)
        if resource_table:
            print(f"\nResource usage collected for {len(resource_table)} pairs.")
        else:
            print("\nNo resource.json files found; skipping GPU-hour columns.")

    # Also generate a combined CSV with all data if pandas available
    if HAS_PANDAS:
        rows = []
        for (method, dataset), data in sorted(results.items()):
            row = {'method': method, 'dataset': dataset, 'n_runs': data['n_runs']}
            for metric, stats in data['aggregated'].items():
                row[f"{metric}_mean"] = stats['mean']
                row[f"{metric}_std"] = stats['std']
            res = resource_table.get((method, dataset))
            if res:
                row['wall_hours_mean'] = res['wall_hours_mean']
                row['wall_hours_total'] = res['wall_hours_total']
                row['gpu_hours_total'] = res['gpu_hours_total']
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            csv_path = output_dir / 'all_results.csv'
            df.to_csv(csv_path, index=False)
            print(f"Saved full results CSV to: {csv_path}")

    print(f"\nDone. Tables saved to: {output_dir}/")


if __name__ == "__main__":
    main()
