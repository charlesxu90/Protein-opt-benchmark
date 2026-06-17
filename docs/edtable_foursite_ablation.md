# Extended Data Table — Component ablation of AlphaVariant on the four-site benchmark

> Leave-one-out ablation with the shipped four-site configuration, **bare + finetune**
> (GPT-REINFORCE + surrogate-UCB reward + 5-model ensemble + cluster sampling + per-round
> prior finetuning), as the baseline. Each row removes one component and re-runs the full
> 480-query campaign (96 × 5 rounds), n = 30 seeds. Fitness is min–max normalized to [0, 1]
> per landscape; Δ is the change versus the baseline on the same landscape. Higher is better.

## Primary summary — median [Q1–Q3]

| Dataset | Configuration | *n* | Max fitness, median [Q1–Q3] | Δ max | Top-128 mean fitness, median [Q1–Q3] | Δ top-128 |
|---------|---------------|:---:|:---------------------------:|:-----:|:------------------------------------:|:---------:|
| **GB1** | AlphaVariant (bare + finetune) | 30 | 1.000 [0.827–1.000] | ref | 0.462 [0.407–0.510] | ref |
|  | − finetune prior | 30 | 0.959 [0.795–1.000] | -0.041 | 0.418 [0.387–0.461] | -0.044 |
|  | − ensemble scoring (single RF) | 30 | 1.000 [0.897–1.000] | +0.000 | 0.466 [0.443–0.500] | +0.003 |
|  | − RL step (generate + prioritize) | 30 | 0.862 [0.785–1.000] | -0.138 | 0.460 [0.433–0.485] | -0.002 |
| **PhoQ** | AlphaVariant (bare + finetune) | 30 | 0.464 [0.313–0.532] | ref | 0.121 [0.106–0.127] | ref |
|  | − finetune prior | 30 | 0.472 [0.326–0.573] | +0.008 | 0.120 [0.092–0.127] | -0.001 |
|  | − ensemble scoring (single RF) | 30 | 0.448 [0.312–0.573] | -0.016 | 0.117 [0.106–0.121] | -0.004 |
|  | − RL step (generate + prioritize) | 30 | 0.464 [0.422–0.617] | +0.000 | 0.115 [0.100–0.122] | -0.006 |
| **TEV** | AlphaVariant (bare + finetune) | 30 | 0.367 [0.351–0.383] | ref | 0.542 [0.498–0.571] | ref |
|  | − finetune prior | 30 | 0.367 [0.336–0.368] | -0.000 | 0.481 [0.476–0.515] | -0.061 |
|  | − ensemble scoring (single RF) | 30 | 0.365 [0.342–0.402] | -0.002 | 0.500 [0.484–0.512] | -0.042 |
|  | − RL step (generate + prioritize) | 30 | 0.368 [0.367–0.378] | +0.002 | 0.585 [0.545–0.602] | +0.043 |
| **TrpB** | AlphaVariant (bare + finetune) | 30 | 0.833 [0.809–0.932] | ref | 0.581 [0.560–0.614] | ref |
|  | − finetune prior | 30 | 0.833 [0.795–0.932] | +0.000 | 0.556 [0.494–0.584] | -0.026 |
|  | − ensemble scoring (single RF) | 30 | 0.909 [0.809–1.000] | +0.076 | 0.527 [0.486–0.569] | -0.054 |
|  | − RL step (generate + prioritize) | 30 | 0.923 [0.801–1.000] | +0.090 | 0.543 [0.492–0.581] | -0.038 |

## Alternative summary — mean ± s.d.

> Same ablation summarized by mean ± standard deviation across the 30 seeds.

| Dataset | Configuration | *n* | Max fitness, mean ± s.d. | Δ mean | Top-128 mean fitness, mean ± s.d. | Δ mean |
|---------|---------------|:---:|:------------------------:|:------:|:---------------------------------:|:------:|
| **GB1** | AlphaVariant (bare + finetune) | 30 | 0.896 ± 0.126 | ref | 0.454 ± 0.063 | ref |
|  | − finetune prior | 30 | 0.889 ± 0.129 | -0.008 | 0.423 ± 0.064 | -0.031 |
|  | − ensemble scoring (single RF) | 30 | 0.943 ± 0.107 | +0.047 | 0.462 ± 0.052 | +0.008 |
|  | − RL step (generate + prioritize) | 30 | 0.876 ± 0.113 | -0.020 | 0.449 ± 0.043 | -0.005 |
| **PhoQ** | AlphaVariant (bare + finetune) | 30 | 0.484 ± 0.220 | ref | 0.116 ± 0.020 | ref |
|  | − finetune prior | 30 | 0.485 ± 0.201 | +0.001 | 0.112 ± 0.022 | -0.004 |
|  | − ensemble scoring (single RF) | 30 | 0.502 ± 0.237 | +0.018 | 0.110 ± 0.018 | -0.006 |
|  | − RL step (generate + prioritize) | 30 | 0.525 ± 0.210 | +0.042 | 0.109 ± 0.020 | -0.007 |
| **TEV** | AlphaVariant (bare + finetune) | 30 | 0.398 ± 0.137 | ref | 0.538 ± 0.044 | ref |
|  | − finetune prior | 30 | 0.361 ± 0.075 | -0.037 | 0.501 ± 0.037 | -0.037 |
|  | − ensemble scoring (single RF) | 30 | 0.411 ± 0.140 | +0.013 | 0.499 ± 0.017 | -0.038 |
|  | − RL step (generate + prioritize) | 30 | 0.387 ± 0.080 | -0.011 | 0.566 ± 0.048 | +0.028 |
| **TrpB** | AlphaVariant (bare + finetune) | 30 | 0.871 ± 0.083 | ref | 0.579 ± 0.046 | ref |
|  | − finetune prior | 30 | 0.867 ± 0.094 | -0.005 | 0.535 ± 0.085 | -0.044 |
|  | − ensemble scoring (single RF) | 30 | 0.889 ± 0.098 | +0.017 | 0.525 ± 0.066 | -0.054 |
|  | − RL step (generate + prioritize) | 30 | 0.891 ± 0.096 | +0.019 | 0.538 ± 0.054 | -0.041 |

Per-round prior finetuning contributes positively, chiefly to top-128 mean fitness
(removing it lowers top-128 by 0.026–0.061 on GB1/TEV/TrpB and GB1 max by 0.041). The RL
step is essential for GB1 maximum fitness (−0.138 when removed) but, once the prior is
finetuned, is largely redundant on the other landscapes (PhoQ max unchanged; TEV/TrpB max
unchanged or higher under generate-and-prioritize) — finetuning and REINFORCE both supply
high-quality candidates, so they partially substitute. The 5-model ensemble mainly aids
top-128 (TEV/TrpB) with a landscape-dependent effect on maximum fitness. Separately, the
MutCompute reward shaping and SHAP alphabet pruning of the earlier Plan C configuration were
found not to improve over bare on four-site and are therefore excluded from the shipped
configuration (see four-site ablation, bare baseline).
