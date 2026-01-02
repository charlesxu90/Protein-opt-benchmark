# GB1 benchmark
Run with permission skipping to avoid repeatitive stopping.
```shell
claude  --dangerously-skip-permissions
```
## ALDE

## EvoPlay
Now I need to implement EvoPlay to run on GB1 in the paradigm of ALDE. An implementation is available in /home/xux/Desktop/AlphaVariant/Benchmark/ALDE/run_GB1.py. Can you implement run_GB1.py for EvoPlay. Please reference the original implementation in EvoPlay while adapting.

## LatProtRL
Now I need to implement LatProtRL on GB1 for benchmarking. A reference implementation of ALDE is available in /home/xux/Desktop/AlphaVariant/Benchmark/ALDE/run_GB1.py. Can you implement @run_GB1.py for LatProtRL. Please reference the original implementation in LatProtRL while adapting.

## AdaLead
Now I need to implement AdaLead on GB1 for benchmarking. A reference implementation of ALDE is available in /home/xux/Desktop/AlphaVariant/Benchmark/ALDE/run_GB1.py. Can you implement @run_GB1_adalead.py for AdaLead. Please reference the original implementation in AdaLead while adapting.

## AiCE
Now I need to implement AiCE on GB1 for benchmarking. A reference implementation of ALDE is available in /home/xux/Desktop/AlphaVariant/Benchmark/ALDE/run_GB1.py. Can you implement @run_GB1.py for AiCE. Please reference the original implementation in AiCE while adapting.


## δ-Conservative Search
Now I need to implement δ-Conservative Search on GB1 for benchmarking. A reference implementation of ALDE is available in /home/xux/Desktop/AlphaVariant/Benchmark/ALDE/run_GB1.py. Can you implement @run_GB1.py for δ-Conservative Search. Please reference the original implementation in δ-Conservative Search while adapting.

## AlphaVariant
Now I need to implement AlphaVariant on GB1 for benchmarking. A reference implementation of ALDE is available in /home/xux/Desktop/AlphaVariant/Benchmark/ALDE/run_GB1.py. Can you implement @run_GB1.py for AlphaVariant. Please reference the original implementation in AlphaVariant while adapting.

Can you make the metrics align with ALDE with the following order: 
  high_fitness_proximity
  novelty
  batch_diversity
  normalized_fitness_median_top128
  normalized_fitness_median_top256
  max_fitness
  spearman_correlation
  epistatic_correlation
  recall_high_order
  simple_regret
  miscalibration_area
  expected_calibration_error
  global_max_hit_count