# Multi-site oracle benchmark — paired Wilcoxon (n=30 seeds)

Bonferroni: pairwise α=0.05/36=1.39e-03; vs-best α=0.05/8=6.25e-03


## max fitness — best method vs rest (one-sided, Bonferroni)


**ms_AAV** — best = **AiCE** (median 0.655); significantly beats 7/8 other methods:

| vs | Δmedian | p(>) | sig |
|---|---|---|---|
| GreedyWalk | +0.002 | 5.5e-02 | ns |
| ftMLDE | +0.032 | 2.8e-05 | **yes** |
| AdaLead | +0.041 | 7.1e-07 | **yes** |
| MULTIevolve | +0.054 | 6.5e-09 | **yes** |
| CLADE | +0.058 | 1.8e-08 | **yes** |
| ALDE | +0.066 | 2.8e-09 | **yes** |
| EVOLVEpro | +0.068 | 9.3e-10 | **yes** |
| Random | +0.083 | 2.8e-09 | **yes** |

**ms_CreiLOV** — best = **AdaLead** (median 0.965); significantly beats 8/8 other methods:

| vs | Δmedian | p(>) | sig |
|---|---|---|---|
| ftMLDE | +0.021 | 1.3e-05 | **yes** |
| CLADE | +0.030 | 1.8e-08 | **yes** |
| GreedyWalk | +0.037 | 4.0e-08 | **yes** |
| MULTIevolve | +0.044 | 2.3e-08 | **yes** |
| AiCE | +0.051 | 9.3e-10 | **yes** |
| EVOLVEpro | +0.052 | 9.3e-10 | **yes** |
| ALDE | +0.054 | 2.8e-09 | **yes** |
| Random | +0.095 | 9.3e-10 | **yes** |

**ms_GFP** — best = **AiCE** (median 0.950); significantly beats 8/8 other methods:

| vs | Δmedian | p(>) | sig |
|---|---|---|---|
| GreedyWalk | +0.096 | 9.3e-10 | **yes** |
| Random | +0.111 | 9.3e-10 | **yes** |
| EVOLVEpro | +0.131 | 9.3e-10 | **yes** |
| ALDE | +0.138 | 9.3e-10 | **yes** |
| CLADE | +0.138 | 9.3e-10 | **yes** |
| ftMLDE | +0.138 | 9.3e-10 | **yes** |
| AdaLead | +0.138 | 9.3e-10 | **yes** |
| MULTIevolve | +0.138 | 9.3e-10 | **yes** |

**ms_PAB1** — best = **AdaLead** (median 0.565); significantly beats 8/8 other methods:

| vs | Δmedian | p(>) | sig |
|---|---|---|---|
| ftMLDE | +0.015 | 8.0e-06 | **yes** |
| GreedyWalk | +0.037 | 9.3e-10 | **yes** |
| MULTIevolve | +0.040 | 1.8e-08 | **yes** |
| EVOLVEpro | +0.063 | 1.9e-09 | **yes** |
| AiCE | +0.072 | 9.3e-10 | **yes** |
| CLADE | +0.078 | 9.3e-10 | **yes** |
| ALDE | +0.097 | 9.3e-10 | **yes** |
| Random | +0.146 | 9.3e-10 | **yes** |

## top-128 mean — best method vs rest (one-sided, Bonferroni)


**ms_AAV** — best = **AiCE** (median 0.597); significantly beats 8/8 other methods:

| vs | Δmedian | p(>) | sig |
|---|---|---|---|
| GreedyWalk | +0.030 | 6.2e-04 | **yes** |
| ftMLDE | +0.041 | 1.8e-08 | **yes** |
| AdaLead | +0.046 | 1.8e-08 | **yes** |
| MULTIevolve | +0.068 | 9.3e-10 | **yes** |
| CLADE | +0.100 | 9.3e-10 | **yes** |
| EVOLVEpro | +0.101 | 9.3e-10 | **yes** |
| ALDE | +0.113 | 9.3e-10 | **yes** |
| Random | +0.196 | 9.3e-10 | **yes** |

**ms_CreiLOV** — best = **AdaLead** (median 0.935); significantly beats 8/8 other methods:

| vs | Δmedian | p(>) | sig |
|---|---|---|---|
| ftMLDE | +0.012 | 8.0e-06 | **yes** |
| GreedyWalk | +0.038 | 2.8e-09 | **yes** |
| MULTIevolve | +0.044 | 4.7e-09 | **yes** |
| CLADE | +0.046 | 9.3e-10 | **yes** |
| ALDE | +0.060 | 9.3e-10 | **yes** |
| EVOLVEpro | +0.061 | 9.3e-10 | **yes** |
| AiCE | +0.066 | 9.3e-10 | **yes** |
| Random | +0.162 | 9.3e-10 | **yes** |

**ms_GFP** — best = **AiCE** (median 0.847); significantly beats 8/8 other methods:

| vs | Δmedian | p(>) | sig |
|---|---|---|---|
| GreedyWalk | +0.074 | 9.3e-10 | **yes** |
| ftMLDE | +0.178 | 9.3e-10 | **yes** |
| Random | +0.209 | 9.3e-10 | **yes** |
| MULTIevolve | +0.213 | 9.3e-10 | **yes** |
| EVOLVEpro | +0.216 | 9.3e-10 | **yes** |
| ALDE | +0.229 | 9.3e-10 | **yes** |
| AdaLead | +0.234 | 9.3e-10 | **yes** |
| CLADE | +0.236 | 9.3e-10 | **yes** |

**ms_PAB1** — best = **AdaLead** (median 0.503); significantly beats 7/8 other methods:

| vs | Δmedian | p(>) | sig |
|---|---|---|---|
| ftMLDE | +0.010 | 1.0e-02 | ns |
| GreedyWalk | +0.033 | 9.3e-10 | **yes** |
| MULTIevolve | +0.041 | 1.9e-09 | **yes** |
| EVOLVEpro | +0.060 | 9.3e-10 | **yes** |
| AiCE | +0.069 | 9.3e-10 | **yes** |
| CLADE | +0.098 | 9.3e-10 | **yes** |
| ALDE | +0.123 | 9.3e-10 | **yes** |
| Random | +0.220 | 9.3e-10 | **yes** |