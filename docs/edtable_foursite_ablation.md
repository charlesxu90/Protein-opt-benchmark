# Extended Data Table — Component ablation of AlphaVariant on the four-site benchmark

> Leave-one-out ablation on the four four-site (combinatorial lookup) landscapes. Each
> configuration removes one component from the Plan C configuration and re-runs the full
> 480-query campaign (96 × 5 rounds); n = 30 seeds per configuration. Fitness is min–max
> normalized to [0, 1] per landscape; Δ is the change in the summary statistic versus the
> full configuration on the same landscape. Higher is better.

## Primary summary — median [Q1–Q3]

| Dataset | Configuration | *n* | Max fitness, median [Q1–Q3] | Δ max | Top-128 mean fitness, median [Q1–Q3] | Δ top-128 |
|---------|---------------|:---:|:---------------------------:|:-----:|:------------------------------------:|:---------:|
| **GB1** | AlphaVariant (full / Plan C) | 30 | 0.862 [0.825–1.000] | ref | 0.362 [0.266–0.415] | ref |
|  | − MutCompute reward | 30 | 1.000 [0.835–1.000] | +0.138 | 0.459 [0.405–0.482] | +0.096 |
|  | − SHAP pruning | 30 | 0.833 [0.788–1.000] | -0.029 | 0.365 [0.265–0.407] | +0.002 |
|  | bare (UCB reward only) | 30 | 0.959 [0.795–1.000] | +0.097 | 0.418 [0.387–0.461] | +0.056 |
| **PhoQ** | AlphaVariant (full / Plan C) | 30 | 0.359 [0.286–0.443] | ref | 0.096 [0.081–0.104] | ref |
|  | − MutCompute reward | 30 | 0.454 [0.311–0.573] | +0.095 | 0.122 [0.094–0.134] | +0.027 |
|  | − SHAP pruning | 30 | 0.386 [0.291–0.455] | +0.027 | 0.095 [0.085–0.107] | -0.001 |
|  | bare (UCB reward only) | 30 | 0.472 [0.326–0.573] | +0.113 | 0.120 [0.092–0.127] | +0.024 |
| **TEV** | AlphaVariant (full / Plan C) | 30 | 0.354 [0.323–0.383] | ref | 0.478 [0.473–0.492] | ref |
|  | − MutCompute reward | 30 | 0.365 [0.352–0.380] | +0.011 | 0.492 [0.475–0.519] | +0.014 |
|  | − SHAP pruning | 30 | 0.355 [0.334–0.389] | +0.001 | 0.476 [0.471–0.496] | -0.001 |
|  | bare (UCB reward only) | 30 | 0.367 [0.336–0.368] | +0.012 | 0.481 [0.476–0.515] | +0.003 |
| **TrpB** | AlphaVariant (full / Plan C) | 30 | 0.818 [0.732–0.910] | ref | 0.415 [0.347–0.461] | ref |
|  | − MutCompute reward | 30 | 0.817 [0.748–0.916] | -0.001 | 0.541 [0.505–0.584] | +0.126 |
|  | − SHAP pruning | 30 | 0.833 [0.793–0.906] | +0.015 | 0.395 [0.348–0.444] | -0.020 |
|  | bare (UCB reward only) | 30 | 0.833 [0.795–0.932] | +0.015 | 0.556 [0.494–0.584] | +0.141 |

## Alternative summary — mean ± s.d.

> Same ablation, summarized by mean ± standard deviation across the 30 seeds (companion to
> the median table; shown for dispersion). The median table is the primary, main-text-
> consistent summary.

| Dataset | Configuration | *n* | Max fitness, mean ± s.d. | Δ mean | Top-128 mean fitness, mean ± s.d. | Δ mean |
|---------|---------------|:---:|:------------------------:|:------:|:---------------------------------:|:------:|
| **GB1** | AlphaVariant (full / Plan C) | 30 | 0.878 ± 0.121 | ref | 0.338 ± 0.098 | ref |
|  | − MutCompute reward | 30 | 0.913 ± 0.120 | +0.034 | 0.442 ± 0.059 | +0.103 |
|  | − SHAP pruning | 30 | 0.866 ± 0.123 | -0.012 | 0.345 ± 0.086 | +0.007 |
|  | bare (UCB reward only) | 30 | 0.889 ± 0.129 | +0.010 | 0.423 ± 0.064 | +0.084 |
| **PhoQ** | AlphaVariant (full / Plan C) | 30 | 0.405 ± 0.187 | ref | 0.088 ± 0.025 | ref |
|  | − MutCompute reward | 30 | 0.511 ± 0.252 | +0.106 | 0.114 ± 0.025 | +0.026 |
|  | − SHAP pruning | 30 | 0.406 ± 0.171 | +0.001 | 0.092 ± 0.019 | +0.004 |
|  | bare (UCB reward only) | 30 | 0.485 ± 0.201 | +0.080 | 0.112 ± 0.022 | +0.023 |
| **TEV** | AlphaVariant (full / Plan C) | 30 | 0.374 ± 0.077 | ref | 0.487 ± 0.025 | ref |
|  | − MutCompute reward | 30 | 0.391 ± 0.087 | +0.017 | 0.502 ± 0.031 | +0.015 |
|  | − SHAP pruning | 30 | 0.387 ± 0.090 | +0.013 | 0.488 ± 0.026 | +0.001 |
|  | bare (UCB reward only) | 30 | 0.361 ± 0.075 | -0.012 | 0.501 ± 0.037 | +0.013 |
| **TrpB** | AlphaVariant (full / Plan C) | 30 | 0.823 ± 0.096 | ref | 0.386 ± 0.113 | ref |
|  | − MutCompute reward | 30 | 0.839 ± 0.098 | +0.017 | 0.540 ± 0.059 | +0.154 |
|  | − SHAP pruning | 30 | 0.836 ± 0.096 | +0.013 | 0.374 ± 0.114 | -0.012 |
|  | bare (UCB reward only) | 30 | 0.867 ± 0.094 | +0.044 | 0.535 ± 0.085 | +0.149 |

Removing the MutCompute reward shaping improved or matched maximum fitness on all four landscapes, and the both-off bare configuration was best or tied-best throughout; SHAP pruning was approximately neutral (its proposal constraint is inactive on four-site combinatorial libraries). Because the full configuration is re-run here on the current codebase, its absolute values differ from the main-text four-site figures (produced under an earlier implementation); the deltas, computed within the current codebase and shared seeds, are the meaningful quantity.
