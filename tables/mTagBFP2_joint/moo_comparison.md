# mTagBFP2_joint — Multi-objective benchmark (30 seeds × 7 methods)

**Landscape:** 8192 sequences, 17 Pareto-optimal, max(blue)=1.6077, max(red)=1.6924, reference HV=0.8821

**Scalarized fitness:** sqrt(blue × red), landscape max = 1.6495 (achievable by Pareto points only)


| Method | n | max scalarized | max blue | max red | hypervolume (norm) | Pareto coverage |
|---|---|---|---|---|---|---|
| Random | 30 | 0.575 [0.563, 0.600] | 1.565 [1.558, 1.584] | 1.482 [1.441, 1.500] | 0.696 [0.676, 0.750] | 0.000 [0.000, 0.059] |
| GreedyWalk | 30 | 0.624 [0.595, 0.624] | 1.577 [1.568, 1.584] | 1.521 [1.487, 1.656] | 0.740 [0.707, 0.767] | 0.176 [0.059, 0.294] |
| ftMLDE | 30 | 0.645 [0.645, 0.645] | 1.550 [1.515, 1.573] | 1.692 [1.692, 1.692] | 0.768 [0.732, 0.822] | 0.412 [0.412, 0.412] |
| CLADE | 30 | 0.645 [0.645, 0.707] | 1.583 [1.572, 1.589] | 1.692 [1.692, 1.692] | 0.791 [0.768, 0.870] | 0.441 [0.368, 0.529] |
| AiCE | 30 | 0.728 [0.728, 0.728] | 1.608 [1.608, 1.608] | 1.394 [1.394, 1.434] | 0.924 [0.924, 0.934] | 0.235 [0.235, 0.235] |
| ALDE | 30 | 0.548 [0.522, 0.567] | 1.608 [1.608, 1.608] | 1.389 [1.339, 1.443] | 0.613 [0.570, 0.630] | 0.059 [0.059, 0.059] |
| FLEXS | 30 | 0.394 [0.281, 0.454] | 0.852 [0.830, 1.430] | 0.314 [0.273, 0.745] | 0.231 [0.156, 0.303] | 0.000 [0.000, 0.000] |

Values are median [Q1, Q3] across 30 seeds. max scalarized = max sqrt(blue × red) of queried set. Hypervolume normalized by landscape reference HV vs (0, 0). Pareto coverage = fraction of landscape Pareto front weakly dominated by queries.
