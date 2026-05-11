# EvoPlay ALDE Adaptation - Quick Reference

## What Changed?

### ✅ Unchanged (Original Implementation)
- Core EvoPlay algorithm (MCTS + GP + Policy-Value Network)
- Single continuous run to 384 sequences
- No round breaking
- Same initialization and update mechanisms

### ✨ New Capability
- Compute metrics at ALDE checkpoints: **96, 192, 288, 384**
- Per-checkpoint aggregation across multiple runs
- Checkpoint-by-checkpoint progress tracking

## Result Structure

### Single Run
```
results/GB1_EvoPlay_experiments/GB1/onehot/
├── metrics_seed{seed}.json
│   ├── checkpoint_metrics
│   │   ├── n_96: {max_fitness, novelty, diversity, ...}
│   │   ├── n_192: {...}
│   │   ├── n_288: {...}
│   │   └── n_384: {...}
│   ├── metrics (same as n_384)
│   └── fitness_trajectory
```

### Multiple Runs
```
results/GB1_EvoPlay_experiments/GB1/onehot/
├── aggregated_results.json
│   ├── checkpoint_aggregates
│   │   ├── n_96: {mean, std, min, max for each metric}
│   │   ├── n_192: {...}
│   │   ├── n_288: {...}
│   │   └── n_384: {...}
│   └── aggregated_metrics_final (for backward compatibility)
├── aggregated_metrics.csv (final 384 summary)
└── metrics_seed*.json (per-run data)
```

## Usage Commands

### Single Run
```bash
python run_GB1.py --seed 42
```

### Multiple Runs (Better Statistics)
```bash
python run_GB1.py --seeds 42 123 456 789 1000
```

### With ALDE Data Format
```bash
python run_GB1.py --data_dir ../ALDE/data --seeds 42 123 456
```

### From Seeds File
```bash
python run_GB1.py --seed_file rand_seeds.txt --num_seeds 5
```

## Benchmark Comparison

### Getting Final Metrics (384 sequences)
```python
import json

# Single run
with open('results/GB1_EvoPlay_experiments/GB1/onehot/metrics_seed42.json') as f:
    result = json.load(f)
    final_metrics = result['metrics']  # 384-sequence metrics

# Multiple runs aggregated
with open('results/GB1_EvoPlay_experiments/GB1/onehot/aggregated_results.json') as f:
    agg = json.load(f)
    final_agg = agg['aggregated_metrics_final']  # Aggregated 384 metrics
```

### Comparing with ALDE at Each Checkpoint
```python
# Get checkpoint-specific metrics
with open('aggregated_results.json') as f:
    data = json.load(f)
    
for checkpoint in ['n_96', 'n_192', 'n_288', 'n_384']:
    checkpoint_data = data['checkpoint_aggregates'][checkpoint]
    print(f"\n{checkpoint}:")
    print(f"  Max Fitness: {checkpoint_data['max_fitness']['mean']:.4f} ± {checkpoint_data['max_fitness']['std']:.4f}")
    print(f"  Simple Regret: {checkpoint_data['simple_regret']['mean']:.4f} ± {checkpoint_data['simple_regret']['std']:.4f}")
```

## Key Metrics

Each checkpoint reports:
1. **Exploration Metrics**
   - `high_fitness_proximity`: Distance to top-performing sequences
   - `novelty`: Distance to initial training set
   - `batch_diversity`: Diversity within discovered sequences

2. **Functional Metrics**
   - `max_fitness`: Best fitness found
   - `normalized_fitness_median_top128`: Median of top-128 normalized fitness

3. **Success Metrics**
   - `simple_regret`: Gap from global optimum
   - `global_max_found`: Whether global max was found

## Verification

### Check checkpoint metrics exist
```bash
python -c "
import json
with open('results/GB1_EvoPlay_experiments/GB1/onehot/aggregated_results.json') as f:
    data = json.load(f)
    print('Checkpoints:', list(data['checkpoint_aggregates'].keys()))
    print('Runs:', data['n_runs'])
"
```

### View per-checkpoint progress
```bash
python -c "
import json, pandas as pd
with open('results/GB1_EvoPlay_experiments/GB1/onehot/aggregated_results.json') as f:
    data = json.load(f)
    
for checkpoint in sorted(data['checkpoint_aggregates'].keys()):
    metrics = data['checkpoint_aggregates'][checkpoint]
    print(f'{checkpoint}: max_fitness={metrics[\"max_fitness\"][\"mean\"]:.4f}')
"
```

## Fair Comparison Checklist

✅ **For EvoPlay vs ALDE Comparison**:
- [ ] Both methods run to at least 384 sequences
- [ ] Metrics are reported at the same checkpoints (96, 192, 288, 384)
- [ ] Same random seeds used for fair comparison
- [ ] Aggregated over multiple runs (5-10 recommended)
- [ ] Error bars (std dev) are reported

## Troubleshooting

### If checkpoint metrics are missing
- Ensure `compute_metrics` is True (default)
- Check that the algorithm collected enough sequences
- Verify data path is correct

### If aggregation fails
- Ensure all runs completed successfully
- Check that metrics_seed*.json files exist
- Verify JSON format in individual run files

### For GPU acceleration
```bash
python run_GB1.py --seed 42 --use_gpu
```

## Output Interpretation

### High-Fitness Proximity
- Lower = better (sequences closer to high-fitness region)
- Indicates exploration efficiency toward good regions

### Novelty
- Higher = better (sequences more different from initial set)
- Indicates effective exploration beyond starting point

### Batch Diversity
- Higher = better (more diverse discovered sequences)
- Indicates broad exploration rather than converging to single optimum

### Simple Regret
- Lower = better (closer to global optimum)
- Direct measure of optimization success

### Normalized Fitness
- Higher = better (top sequences are of high quality)
- Normalized to [0,1] range for fair comparison
