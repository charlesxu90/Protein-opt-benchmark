# Four-site benchmark — paired Wilcoxon (n=30 seeds)

Bonferroni: pairwise α=0.05/45=1.11e-03; vs-best α=0.05/9=5.56e-03


## max fitness — best method vs rest (one-sided, Bonferroni)


**4site_GB1** — best = **ALDE** (median 1.000); significantly beats 3/9 other methods:

| vs | Δmedian | p(>) | sig |
|---|---|---|---|
| AlphaVariant | +0.000 | 1.6e-01 | ns |
| FLEXS | +0.138 | 4.4e-02 | ns |
| ftMLDE | +0.138 | 6.2e-02 | ns |
| MULTIevolve | +0.138 | 1.3e-02 | ns |
| GreedyWalk | +0.165 | 1.1e-02 | ns |
| CLADE | +0.165 | 9.5e-03 | ns |
| EVOLVEpro | +0.502 | 2.8e-09 | **yes** |
| AiCE | +0.525 | 8.4e-07 | **yes** |
| Random | +0.528 | 2.8e-09 | **yes** |

**4site_PhoQ** — best = **MULTIevolve** (median 0.517); significantly beats 6/9 other methods:

| vs | Δmedian | p(>) | sig |
|---|---|---|---|
| ftMLDE | +0.038 | 3.6e-01 | ns |
| AlphaVariant | +0.053 | 8.2e-02 | ns |
| ALDE | +0.062 | 1.4e-01 | ns |
| FLEXS | +0.089 | 4.5e-04 | **yes** |
| CLADE | +0.110 | 2.7e-03 | **yes** |
| GreedyWalk | +0.175 | 5.3e-04 | **yes** |
| AiCE | +0.191 | 6.2e-06 | **yes** |
| EVOLVEpro | +0.316 | 1.9e-09 | **yes** |
| Random | +0.323 | 1.8e-06 | **yes** |

**4site_TRPB** — best = **ALDE** (median 0.932); significantly beats 4/9 other methods:

| vs | Δmedian | p(>) | sig |
|---|---|---|---|
| MULTIevolve | +0.000 | 5.9e-01 | ns |
| AlphaVariant | +0.048 | 2.2e-01 | ns |
| ftMLDE | +0.062 | 1.4e-01 | ns |
| CLADE | +0.091 | 1.0e-01 | ns |
| FLEXS | +0.118 | 2.0e-02 | ns |
| GreedyWalk | +0.179 | 2.7e-03 | **yes** |
| AiCE | +0.267 | 2.8e-06 | **yes** |
| EVOLVEpro | +0.310 | 1.8e-08 | **yes** |
| Random | +0.322 | 1.9e-09 | **yes** |

## top-128 median — best method vs rest (one-sided, Bonferroni)


**4site_GB1** — best = **ALDE** (median 0.539); significantly beats 9/9 other methods:

| vs | Δmedian | p(>) | sig |
|---|---|---|---|
| ftMLDE | +0.082 | 4.4e-05 | **yes** |
| AlphaVariant | +0.097 | 2.2e-05 | **yes** |
| MULTIevolve | +0.099 | 8.4e-07 | **yes** |
| FLEXS | +0.122 | 1.4e-06 | **yes** |
| CLADE | +0.125 | 1.6e-07 | **yes** |
| GreedyWalk | +0.276 | 1.9e-09 | **yes** |
| AiCE | +0.416 | 9.3e-10 | **yes** |
| EVOLVEpro | +0.517 | 9.3e-10 | **yes** |
| Random | +0.537 | 9.3e-10 | **yes** |

**4site_PhoQ** — best = **MULTIevolve** (median 0.126); significantly beats 8/9 other methods:

| vs | Δmedian | p(>) | sig |
|---|---|---|---|
| ALDE | +0.004 | 2.2e-02 | ns |
| ftMLDE | +0.008 | 2.9e-03 | **yes** |
| AlphaVariant | +0.022 | 8.0e-06 | **yes** |
| FLEXS | +0.026 | 2.8e-09 | **yes** |
| CLADE | +0.028 | 9.3e-10 | **yes** |
| AiCE | +0.064 | 9.3e-10 | **yes** |
| EVOLVEpro | +0.082 | 9.3e-10 | **yes** |
| GreedyWalk | +0.087 | 9.3e-10 | **yes** |
| Random | +0.126 | 9.3e-10 | **yes** |

**4site_TRPB** — best = **ALDE** (median 0.610); significantly beats 7/9 other methods:

| vs | Δmedian | p(>) | sig |
|---|---|---|---|
| ftMLDE | +0.024 | 4.0e-02 | ns |
| MULTIevolve | +0.033 | 8.2e-03 | ns |
| AlphaVariant | +0.044 | 2.3e-03 | **yes** |
| FLEXS | +0.062 | 3.0e-06 | **yes** |
| CLADE | +0.069 | 1.4e-04 | **yes** |
| GreedyWalk | +0.312 | 9.3e-10 | **yes** |
| AiCE | +0.525 | 9.3e-10 | **yes** |
| EVOLVEpro | +0.574 | 9.3e-10 | **yes** |
| Random | +0.567 | 1.9e-09 | **yes** |