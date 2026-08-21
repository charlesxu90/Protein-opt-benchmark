# Integration Guide: Using Unified Benchmark Utils

This guide explains how to update each method's `run_GB1.py` to use the new unified benchmark utilities located in `Benchmark/utils/`.

## New modules (2026-05)

The refined benchmark plan added three modules under `utils/`:

- `utils.multi_objective` — `hypervolume`, `pareto_front`, `pareto_front_coverage` for Task 3 multi-objective evaluation.
- `utils.proteingym_oracle` — `load_oracle(name)`, `top_percent_threshold`, `mutation_order_distribution`, `hierarchical_split` for any prepared dataset.
- `utils.sequence_plausibility` — `esm2_ppl`, `pll`, `perplexity` for ESM-2 plausibility scoring (lazy-imports torch/transformers).

All three are also re-exported from `utils.__init__` (except plausibility, which stays lazy).

## HPC integration

Run scripts must accept `--seed` (single int) for the job-array launcher to drive them. `scripts/hpc/launch.py` invokes `<method>/run_<dataset>.py --seed $SEED [extra args]`. To add a new dataset:

1. Drop the prepared CSV at `data/<name>/data.csv` (`seq, fitness` columns).
2. Add per-dataset wrappers at `scripts/<method>/run_<name>.py` that delegate to `run_generic.py`.
3. Run it from `scripts/<method>/` — there are no symlinks to refresh.
4. (HPC) The launcher picks resource defaults from `scripts/hpc/method_resources.yaml`.

## Per-method environments

The launcher resolves the python interpreter per method from
`scripts/hpc/method_resources.yaml`. Each entry's `conda_env:` accepts:
  - a path relative to the benchmark root (e.g. `ALDE/env`)
  - an absolute path (e.g. `/home/xux/miniforge3/envs/alphavariant-env`)
  - a `~/...` path (expanded at resolution time)

If the configured env is missing, the launcher falls back to `<method>/env`
if that exists; otherwise to `sys.executable` with a warning. Use the snippet
in `docs/reproducibility_appendix.md` §1 to verify every entry resolves on
your host.

## AlphaVariant ablations (Phase 2)

Pass `--ablation {none|no-gpt|no-space|static-reward|no-rl}` to AlphaVariant run scripts to swap a single component:

- `none` (default) — full pipeline.
- `no-gpt` — replace VariantGPT prior with random single-site mutations.
- `no-space` — disable dynamic space definition (search the full sequence space).
- `static-reward` — replace iterative low-N reward with static zero-shot ESM-1v.
- `no-rl` — replace RL acquisition with greedy top-k from the prior.

The flag is forwarded through `scripts/hpc/launch.py --extra-args "--ablation no-gpt"`.



## Quick Start

### Option 1: Drop-in Replacement (Recommended)

Replace your local metrics imports with the compatibility module:

```python
# Before (in each method's run_GB1.py):
from src.metrics import (
    compute_all_metrics,
    aggregate_run_metrics,
    load_landscape_data,
    MetricsResult,
    global_max_hit_count,
    # ... other imports
)

# After:
import sys
sys.path.insert(0, '/home/xux/Desktop/AlphaVariant/Benchmark')  # Add path to utils
from utils.compat import (
    compute_all_metrics,
    aggregate_run_metrics,
    load_landscape_data,
    MetricsResult,
    global_max_hit_count,
    hamming_distance,
    high_fitness_proximity,
    novelty,
    batch_diversity,
    normalized_fitness_topk,
    max_fitness,
    simple_regret,
    spearman_correlation,
    miscalibration_area,
    expected_calibration_error,
)
```

### Option 2: Direct GB1 Utilities (New Interface)

For new implementations or major refactoring:

```python
import sys
sys.path.insert(0, '/home/xux/Desktop/AlphaVariant/Benchmark')
from utils import (
    load_gb1_landscape,
    compute_gb1_metrics,
    aggregate_gb1_metrics,
    save_gb1_results,
    print_gb1_metrics_summary,
    GB1MetricsResult,
    GB1_WILD_TYPE_4SITE,
)
```

---

## Method-Specific Instructions

### 1. ALDE

**File:** `/home/xux/Desktop/AlphaVariant/Benchmark/ALDE/run_GB1.py`

**Changes:**

```python
# Line ~54-60: Replace local imports
# Before:
from src.metrics import (
    compute_all_metrics,
    aggregate_run_metrics,
    load_landscape_data,
    MetricsResult,
    global_max_hit_count
)

# After:
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.compat import (
    compute_all_metrics,
    aggregate_run_metrics,
    load_landscape_data,
    MetricsResult,
    global_max_hit_count
)
```

No other changes needed - the interface is fully compatible.

---

### 2. EvoPlay

**File:** `/home/xux/Desktop/AlphaVariant/Benchmark/EvoPlay/run_GB1.py`

**Changes:**

```python
# Line ~92-250: Remove local metric function definitions

# Add at top (after other imports):
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.compat import (
    compute_all_metrics,
    aggregate_run_metrics,
    load_landscape_data,
    MetricsResult,
    global_max_hit_count,
    hamming_distance,
    high_fitness_proximity,
    novelty,
    batch_diversity,
    normalized_fitness_topk,
    max_fitness,
    simple_regret,
)
```

Remove the locally defined functions:
- `hamming_distance()`
- `high_fitness_proximity()`
- `novelty()`
- `batch_diversity()`
- `normalized_fitness_topk()`
- `max_fitness()`
- `simple_regret()`
- `MetricsResult` class

---

### 3. AdaLead (FLEXS)

**File:** `/home/xux/Desktop/AlphaVariant/Benchmark/FLEXS/run_GB1_adalead.py`

**Changes:**

```python
# Line ~132-250: Remove local metric function definitions

# Add at top (after other imports):
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.compat import (
    compute_all_metrics,
    aggregate_run_metrics,
    load_landscape_data,
    MetricsResult,
    global_max_hit_count,
    hamming_distance,
    high_fitness_proximity,
    novelty,
    batch_diversity,
    normalized_fitness_topk,
    max_fitness,
    simple_regret,
    spearman_correlation,
)
```

---

### 4. LatProtRL

**File:** `/home/xux/Desktop/AlphaVariant/Benchmark/LatProtRL/run_GB1.py`

**Changes:**

```python
# Line ~72-93: Replace ALDE metrics import attempt

# Before:
ALDE_METRICS_PATH = os.path.join(os.path.dirname(__file__), '..', 'ALDE')
sys.path.insert(0, ALDE_METRICS_PATH)
try:
    from src.metrics import (...)
    ALDE_METRICS_AVAILABLE = True
except ImportError:
    ALDE_METRICS_AVAILABLE = False

# After:
UTILS_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, UTILS_PATH)
from utils.compat import (
    compute_all_metrics,
    aggregate_run_metrics,
    load_landscape_data,
    MetricsResult,
    global_max_hit_count,
    hamming_distance,
    high_fitness_proximity,
    novelty,
    batch_diversity,
    normalized_fitness_topk,
    max_fitness as compute_max_fitness,
    simple_regret,
)
ALDE_METRICS_AVAILABLE = True  # Always available now
```

---

### 5. AiCE

**File:** `/home/xux/Desktop/AlphaVariant/Benchmark/AiCE/run_GB1.py`

**Changes:**

```python
# Add at top (after other imports):
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.compat import (
    compute_all_metrics,
    aggregate_run_metrics,
    load_landscape_data,
    MetricsResult,
    global_max_hit_count,
    hamming_distance,
    high_fitness_proximity,
    novelty,
    batch_diversity,
    normalized_fitness_topk,
    max_fitness,
    simple_regret,
)
```

Remove local metric definitions.

---

### 6. δ-Conservative Search (BioSeq-GFN-AL)

**File:** `/home/xux/Desktop/AlphaVariant/Benchmark/delta_cs/BioSeq-GFN-AL/run_GB1.py`

**Changes:**

```python
# Add at top (after other imports):
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.compat import (
    compute_all_metrics,
    aggregate_run_metrics,
    load_landscape_data,
    MetricsResult,
    global_max_hit_count,
    hamming_distance,
    high_fitness_proximity,
    novelty,
    batch_diversity,
    normalized_fitness_topk,
    max_fitness,
    simple_regret,
    miscalibration_area,
    expected_calibration_error,
)
```

---

### 7. AlphaVariant

**File:** `/home/xux/Desktop/AlphaVariant/Benchmark/alphavariant/run_GB1.py`

**Changes:**

```python
# Add at top (after other imports):
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.compat import (
    compute_all_metrics,
    aggregate_run_metrics,
    load_landscape_data,
    MetricsResult,
    global_max_hit_count,
    hamming_distance,
    high_fitness_proximity,
    novelty,
    batch_diversity,
    normalized_fitness_topk,
    max_fitness,
    simple_regret,
    spearman_correlation,
)
```

---

## Metrics Computed

All methods now compute the same 13 metrics in the same order:

| Metric | Description | Reference |
|--------|-------------|-----------|
| `high_fitness_proximity` | Median min distance to top 10% sequences | LatProtRL |
| `novelty` | Median min distance to initial training set | LatProtRL |
| `batch_diversity` | Median pairwise distance within batch | Energy Matching |
| `normalized_fitness_median_top128` | Normalized median of top 128 | GGS |
| `normalized_fitness_median_top256` | Normalized median of top 256 | GGS |
| `max_fitness` | Absolute highest fitness found | δ-CS |
| `spearman_correlation` | Predicted vs true fitness ranking | μProtein |
| `epistatic_correlation` | Correlation of non-additive effects | μProtein |
| `recall_high_order` | % of true top multi-mutants found | μProtein |
| `simple_regret` | Gap from global optimum | VSD |
| `miscalibration_area` | Calibration curve deviation | ALDE |
| `expected_calibration_error` | Weighted calibration error | ALDE |
| `global_max_hit_count` | Runs finding global max (GB1 only) | EvoPlay |

---

## File Structure

```
Benchmark/
├── utils/
│   ├── __init__.py          # Main exports
│   ├── metrics.py            # Core metric implementations
│   ├── data.py               # Data loading utilities
│   ├── evaluator.py          # BenchmarkEvaluator class
│   ├── io.py                 # Results I/O
│   ├── gb1.py                # GB1-specific utilities
│   └── compat.py             # Drop-in compatibility layer
├── ALDE/
├── EvoPlay/
├── FLEXS/
├── LatProtRL/
├── AiCE/
├── delta_cs/
├── alphavariant/
└── data/
    └── GB1/
        └── data.csv
```

---

## Testing Integration

After updating a method, test the integration:

```bash
cd /home/xux/Desktop/AlphaVariant/Benchmark/ALDE  # or other method
python run_GB1.py --seed 42 --skip_metrics  # Test optimization only
python run_GB1.py --seed 42                  # Test with metrics
```

Compare metrics with previous runs to ensure consistency.
