"""
I/O Utilities for Benchmark Results

This module provides functions for saving, loading, and aggregating
benchmark results across methods and datasets.

Supported formats:
- JSON: Human-readable, standard results format
- CSV: Tabular format for metric summaries
- Pickle: Full Python object serialization
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# =============================================================================
# JSON Encoder for NumPy types
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# =============================================================================
# Save Functions
# =============================================================================

def save_results(
    results: Union[Dict, Any],
    filepath: Union[str, Path],
    format: str = 'json'
) -> None:
    """
    Save benchmark results to file.

    Parameters
    ----------
    results : dict or object with to_dict method
        Results to save
    filepath : str or Path
        Output file path
    format : str, default='json'
        Output format: 'json', 'pickle', or 'csv'
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict if needed
    if hasattr(results, 'to_dict'):
        results_dict = results.to_dict()
    else:
        results_dict = results

    if format == 'json':
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2, cls=NumpyEncoder)
    elif format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    elif format == 'csv':
        _save_csv(results_dict, filepath)
    else:
        raise ValueError(f"Unknown format: {format}")


def _save_csv(results: Dict, filepath: Path) -> None:
    """Save results to CSV format."""
    if not HAS_PANDAS:
        # Fallback without pandas
        with open(filepath, 'w') as f:
            if 'metrics_mean' in results:
                # Aggregate results
                f.write("metric,mean,std,median\n")
                for metric in results.get('metrics_mean', {}).keys():
                    mean = results['metrics_mean'].get(metric, '')
                    std = results['metrics_std'].get(metric, '')
                    median = results['metrics_median'].get(metric, '')
                    f.write(f"{metric},{mean},{std},{median}\n")
            else:
                # Single result
                f.write("key,value\n")
                for key, value in results.items():
                    if isinstance(value, (int, float, str)):
                        f.write(f"{key},{value}\n")
        return

    # Use pandas for better CSV handling
    if 'metrics_mean' in results:
        # Aggregate results format
        df = pd.DataFrame({
            'metric': list(results['metrics_mean'].keys()),
            'mean': list(results['metrics_mean'].values()),
            'std': list(results['metrics_std'].values()),
            'median': list(results['metrics_median'].values()),
        })
    else:
        # Single result - flatten nested structures
        flat = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float, str, type(None))):
                        flat[f"{key}_{subkey}"] = subvalue
            elif isinstance(value, (int, float, str, type(None))):
                flat[key] = value
        df = pd.DataFrame([flat])

    df.to_csv(filepath, index=False)


def save_run_results(
    run_results: Any,
    output_dir: Union[str, Path],
    prefix: str = ""
) -> Path:
    """
    Save a single run's results.

    Parameters
    ----------
    run_results : RunResults
        Run results object
    output_dir : str or Path
        Output directory
    prefix : str, default=""
        Filename prefix

    Returns
    -------
    Path
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(run_results, 'run_id') and hasattr(run_results, 'seed'):
        filename = f"{prefix}run_{run_results.run_id}_seed_{run_results.seed}.json"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}run_{timestamp}.json"

    filepath = output_dir / filename
    save_results(run_results, filepath, format='json')
    return filepath


def save_aggregate_results(
    aggregate: Any,
    output_dir: Union[str, Path],
    prefix: str = ""
) -> Dict[str, Path]:
    """
    Save aggregated results in multiple formats.

    Parameters
    ----------
    aggregate : AggregateResults
        Aggregated results
    output_dir : str or Path
        Output directory
    prefix : str, default=""
        Filename prefix

    Returns
    -------
    Dict[str, Path]
        Mapping of format to filepath
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method = getattr(aggregate, 'method', 'unknown')
    dataset = getattr(aggregate, 'dataset', 'unknown')
    base_name = f"{prefix}{method}_{dataset}_aggregate"

    paths = {}

    # Save JSON
    json_path = output_dir / f"{base_name}.json"
    save_results(aggregate, json_path, format='json')
    paths['json'] = json_path

    # Save CSV summary
    csv_path = output_dir / f"{base_name}_summary.csv"
    save_results(aggregate, csv_path, format='csv')
    paths['csv'] = csv_path

    return paths


# =============================================================================
# Load Functions
# =============================================================================

def load_results(
    filepath: Union[str, Path],
    format: Optional[str] = None
) -> Dict:
    """
    Load benchmark results from file.

    Parameters
    ----------
    filepath : str or Path
        Path to results file
    format : str, optional
        File format. If None, inferred from extension.

    Returns
    -------
    dict
        Loaded results
    """
    filepath = Path(filepath)

    if format is None:
        ext = filepath.suffix.lower()
        format = {'.json': 'json', '.pkl': 'pickle', '.pickle': 'pickle', '.csv': 'csv'}.get(ext, 'json')

    if format == 'json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif format == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif format == 'csv':
        if HAS_PANDAS:
            return pd.read_csv(filepath).to_dict('records')
        else:
            raise ImportError("pandas required for CSV loading")
    else:
        raise ValueError(f"Unknown format: {format}")


def load_all_run_results(
    results_dir: Union[str, Path],
    pattern: str = "*.json"
) -> List[Dict]:
    """
    Load all run results from a directory.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing result files
    pattern : str, default="*.json"
        Glob pattern for result files

    Returns
    -------
    List[dict]
        List of loaded results
    """
    results_dir = Path(results_dir)
    results = []

    for filepath in sorted(results_dir.glob(pattern)):
        try:
            result = load_results(filepath)
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")

    return results


# =============================================================================
# Aggregation Functions
# =============================================================================

def aggregate_results(
    results_list: List[Dict],
    method: Optional[str] = None,
    dataset: Optional[str] = None
) -> Dict:
    """
    Aggregate multiple run results into summary statistics.

    Parameters
    ----------
    results_list : List[dict]
        List of individual run results
    method : str, optional
        Method name (inferred from results if None)
    dataset : str, optional
        Dataset name (inferred from results if None)

    Returns
    -------
    dict
        Aggregated results with mean, std, median for each metric
    """
    if len(results_list) == 0:
        raise ValueError("No results to aggregate")

    # Infer method/dataset from first result
    if method is None:
        method = results_list[0].get('method', 'unknown')
    if dataset is None:
        dataset = results_list[0].get('dataset', 'unknown')

    # Collect all metrics
    all_metrics: Dict[str, List[float]] = {}
    best_sequences = []
    seeds = []

    for result in results_list:
        if 'seed' in result:
            seeds.append(result['seed'])

        if 'best_sequence' in result:
            best_sequences.append(result['best_sequence'])

        # Get final metrics
        final = result.get('final_metrics', result)
        for key, value in final.items():
            if isinstance(value, (int, float)) and key not in ['round_idx', 'n_generated', 'n_unique']:
                if key not in all_metrics:
                    all_metrics[key] = []
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    all_metrics[key].append(value)

    # Compute statistics
    metrics_mean = {}
    metrics_std = {}
    metrics_median = {}

    for metric, values in all_metrics.items():
        if len(values) > 0:
            metrics_mean[metric] = float(np.mean(values))
            metrics_std[metric] = float(np.std(values))
            metrics_median[metric] = float(np.median(values))
        else:
            metrics_mean[metric] = None
            metrics_std[metric] = None
            metrics_median[metric] = None

    return {
        'method': method,
        'dataset': dataset,
        'n_runs': len(results_list),
        'seeds': seeds,
        'metrics_mean': metrics_mean,
        'metrics_std': metrics_std,
        'metrics_median': metrics_median,
        'best_sequences': best_sequences,
    }


def aggregate_from_directory(
    results_dir: Union[str, Path],
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    pattern: str = "run_*.json"
) -> Dict:
    """
    Load and aggregate all results from a directory.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing run result files
    method : str, optional
        Method name
    dataset : str, optional
        Dataset name
    pattern : str, default="run_*.json"
        Glob pattern for result files

    Returns
    -------
    dict
        Aggregated results
    """
    results_list = load_all_run_results(results_dir, pattern)
    return aggregate_results(results_list, method, dataset)


# =============================================================================
# Export Functions
# =============================================================================

def export_summary_table(
    aggregate_results_list: List[Dict],
    output_path: Union[str, Path],
    format: str = 'csv'
) -> None:
    """
    Export a summary table comparing multiple methods/datasets.

    Parameters
    ----------
    aggregate_results_list : List[dict]
        List of aggregated results from different methods
    output_path : str or Path
        Output file path
    format : str, default='csv'
        Output format: 'csv', 'latex', 'markdown'
    """
    output_path = Path(output_path)

    # Build table data
    rows = []
    for agg in aggregate_results_list:
        row = {
            'method': agg.get('method', ''),
            'dataset': agg.get('dataset', ''),
            'n_runs': agg.get('n_runs', 0),
        }

        # Add all metrics
        for metric, value in agg.get('metrics_mean', {}).items():
            std = agg.get('metrics_std', {}).get(metric, 0)
            if value is not None:
                row[metric] = f"{value:.4f}"
                row[f"{metric}_std"] = f"{std:.4f}"
            else:
                row[metric] = "N/A"
                row[f"{metric}_std"] = "N/A"

        # Add global max hit count if present
        if 'global_max_hit_count' in agg:
            row['global_max_hit_count'] = agg['global_max_hit_count']
        if 'global_max_hit_rate' in agg:
            row['global_max_hit_rate'] = f"{agg['global_max_hit_rate']:.4f}"

        rows.append(row)

    if format == 'csv':
        _export_csv_table(rows, output_path)
    elif format == 'latex':
        _export_latex_table(rows, output_path)
    elif format == 'markdown':
        _export_markdown_table(rows, output_path)
    else:
        raise ValueError(f"Unknown format: {format}")


def _export_csv_table(rows: List[Dict], output_path: Path) -> None:
    """Export table as CSV."""
    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
    else:
        if len(rows) == 0:
            return
        with open(output_path, 'w') as f:
            # Header
            columns = list(rows[0].keys())
            f.write(','.join(columns) + '\n')
            # Data
            for row in rows:
                values = [str(row.get(col, '')) for col in columns]
                f.write(','.join(values) + '\n')


def _export_latex_table(rows: List[Dict], output_path: Path) -> None:
    """Export table as LaTeX."""
    if len(rows) == 0:
        return

    columns = list(rows[0].keys())
    n_cols = len(columns)

    with open(output_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Benchmark Results}\n")
        f.write("\\label{tab:benchmark}\n")
        f.write("\\begin{tabular}{" + "l" * n_cols + "}\n")
        f.write("\\toprule\n")

        # Header
        header = " & ".join(col.replace('_', '\\_') for col in columns)
        f.write(f"{header} \\\\\n")
        f.write("\\midrule\n")

        # Data
        for row in rows:
            values = [str(row.get(col, '')).replace('_', '\\_') for col in columns]
            f.write(" & ".join(values) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def _export_markdown_table(rows: List[Dict], output_path: Path) -> None:
    """Export table as Markdown."""
    if len(rows) == 0:
        return

    columns = list(rows[0].keys())

    with open(output_path, 'w') as f:
        # Header
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")

        # Data
        for row in rows:
            values = [str(row.get(col, '')) for col in columns]
            f.write("| " + " | ".join(values) + " |\n")


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_methods(
    results_dir: Union[str, Path],
    methods: List[str],
    dataset: str,
    output_path: Optional[Union[str, Path]] = None
) -> Dict:
    """
    Compare multiple methods on a single dataset.

    Parameters
    ----------
    results_dir : str or Path
        Base directory containing method subdirectories
    methods : List[str]
        List of method names to compare
    dataset : str
        Dataset name
    output_path : str or Path, optional
        Path to save comparison table

    Returns
    -------
    dict
        Comparison results
    """
    results_dir = Path(results_dir)
    aggregate_list = []

    for method in methods:
        method_dir = results_dir / method / dataset
        if method_dir.exists():
            agg = aggregate_from_directory(method_dir, method=method, dataset=dataset)
            aggregate_list.append(agg)

    if output_path:
        export_summary_table(aggregate_list, output_path)

    return {
        'dataset': dataset,
        'methods': methods,
        'results': aggregate_list,
    }


def create_results_directory(
    base_dir: Union[str, Path],
    method: str,
    dataset: str,
    experiment_name: Optional[str] = None
) -> Path:
    """
    Create a standardized results directory structure.

    Parameters
    ----------
    base_dir : str or Path
        Base results directory
    method : str
        Method name
    dataset : str
        Dataset name
    experiment_name : str, optional
        Additional experiment identifier

    Returns
    -------
    Path
        Created directory path
    """
    base_dir = Path(base_dir)

    if experiment_name:
        results_dir = base_dir / method / dataset / experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = base_dir / method / dataset / timestamp

    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir
