# eqFP611_joint — Multi-objective benchmark (30 seeds × 7 methods)

**Landscape:** 8192 sequences, 17 Pareto-optimal, max(blue)=1.6077, max(red)=1.6924, reference HV=0.8821

**Scalarized fitness:** sqrt(blue × red), landscape max = 1.6495 (achievable by Pareto points only)


| Method | n | max scalarized | max blue | max red | hypervolume (norm) | Pareto coverage |
|---|---|---|---|---|---|---|
| Random | 30 | 0.578 [0.561, 0.602] | 1.577 [1.569, 1.585] | 1.472 [1.441, 1.521] | 0.713 [0.694, 0.729] | 0.059 [0.000, 0.059] |
| GreedyWalk | 30 | 0.612 [0.588, 0.624] | 1.578 [1.559, 1.587] | 1.520 [1.489, 1.525] | 0.730 [0.709, 0.759] | 0.118 [0.000, 0.176] |
| ftMLDE | 30 | 0.645 [0.645, 0.645] | 1.545 [1.520, 1.568] | 1.692 [1.692, 1.692] | 0.770 [0.740, 0.808] | 0.412 [0.412, 0.412] |
| CLADE | 30 | 0.645 [0.645, 0.728] | 1.585 [1.569, 1.589] | 1.692 [1.692, 1.692] | 0.790 [0.770, 0.869] | 0.441 [0.353, 0.471] |
| AiCE | 30 | 0.728 [0.728, 0.728] | 1.608 [1.608, 1.608] | 1.394 [1.394, 1.418] | 0.932 [0.932, 0.939] | 0.353 [0.353, 0.353] |
| ALDE | 30 | 0.553 [0.532, 0.562] | 1.575 [1.575, 1.575] | 1.401 [1.324, 1.475] | 0.586 [0.557, 0.615] | 0.000 [0.000, 0.000] |
| FLEXS | 30 | 0.393 [0.266, 0.457] | 0.834 [0.813, 1.410] | 0.307 [0.260, 0.773] | 0.251 [0.149, 0.322] | 0.000 [0.000, 0.000] |

Values are median [Q1, Q3] across 30 seeds. max scalarized = max sqrt(blue × red) of queried set. Hypervolume normalized by landscape reference HV vs (0, 0). Pareto coverage = fraction of landscape Pareto front weakly dominated by queries.
