# Multi-site oracle benchmark — paired Wilcoxon (n=30 seeds)

Bonferroni: pairwise α=0.05/45=1.11e-03; vs-best α=0.05/9=5.56e-03


## max fitness — best method vs rest (one-sided, Bonferroni)


**ms_AAV** — best = **AlphaVariant** (median 0.713); significantly beats 9/9 other methods:

| vs | Δmedian | p(>) | sig |
|---|---|---|---|
| AiCE | +0.058 | 5.1e-08 | **yes** |
| GreedyWalk | +0.060 | 2.4e-07 | **yes** |
| ftMLDE | +0.090 | 9.3e-10 | **yes** |
| AdaLead | +0.099 | 9.3e-10 | **yes** |
| MULTIevolve | +0.112 | 9.3e-10 | **yes** |
| CLADE | +0.116 | 9.3e-10 | **yes** |
| ALDE | +0.124 | 9.3e-10 | **yes** |
| EVOLVEpro | +0.126 | 9.3e-10 | **yes** |
| Random | +0.141 | 9.3e-10 | **yes** |

**ms_CreiLOV** — best = **AlphaVariant** (median 0.986); significantly beats 9/9 other methods:

| vs | Δmedian | p(>) | sig |
|---|---|---|---|
| AdaLead | +0.021 | 1.2e-06 | **yes** |
| ftMLDE | +0.042 | 9.3e-10 | **yes** |
| CLADE | +0.051 | 9.3e-10 | **yes** |
| GreedyWalk | +0.057 | 9.3e-10 | **yes** |
| MULTIevolve | +0.065 | 9.3e-10 | **yes** |
| AiCE | +0.072 | 8.7e-07 | **yes** |
| EVOLVEpro | +0.073 | 9.3e-10 | **yes** |
| ALDE | +0.075 | 9.3e-10 | **yes** |
| Random | +0.116 | 9.3e-10 | **yes** |

**ms_PAB1** — best = **AlphaVariant** (median 0.576); significantly beats 8/9 other methods:

| vs | Δmedian | p(>) | sig |
|---|---|---|---|
| AdaLead | +0.011 | 6.5e-02 | ns |
| ftMLDE | +0.026 | 3.0e-06 | **yes** |
| GreedyWalk | +0.048 | 4.7e-09 | **yes** |
| MULTIevolve | +0.051 | 2.9e-07 | **yes** |
| EVOLVEpro | +0.074 | 2.8e-09 | **yes** |
| AiCE | +0.083 | 9.3e-10 | **yes** |
| CLADE | +0.089 | 1.9e-09 | **yes** |
| ALDE | +0.108 | 9.3e-10 | **yes** |
| Random | +0.158 | 9.3e-10 | **yes** |

## top-128 mean — best method vs rest (one-sided, Bonferroni)


**ms_AAV** — best = **AlphaVariant** (median 0.649); significantly beats 9/9 other methods:

| vs | Δmedian | p(>) | sig |
|---|---|---|---|
| AiCE | +0.051 | 9.3e-09 | **yes** |
| GreedyWalk | +0.081 | 4.7e-09 | **yes** |
| ftMLDE | +0.092 | 9.3e-10 | **yes** |
| AdaLead | +0.098 | 9.3e-10 | **yes** |
| MULTIevolve | +0.119 | 9.3e-10 | **yes** |
| CLADE | +0.151 | 9.3e-10 | **yes** |
| EVOLVEpro | +0.153 | 9.3e-10 | **yes** |
| ALDE | +0.165 | 9.3e-10 | **yes** |
| Random | +0.248 | 9.3e-10 | **yes** |

**ms_CreiLOV** — best = **AdaLead** (median 0.935); significantly beats 9/9 other methods:

| vs | Δmedian | p(>) | sig |
|---|---|---|---|
| ftMLDE | +0.012 | 8.0e-06 | **yes** |
| AlphaVariant | +0.036 | 4.7e-09 | **yes** |
| GreedyWalk | +0.038 | 2.8e-09 | **yes** |
| MULTIevolve | +0.044 | 4.7e-09 | **yes** |
| CLADE | +0.046 | 9.3e-10 | **yes** |
| ALDE | +0.060 | 9.3e-10 | **yes** |
| EVOLVEpro | +0.061 | 9.3e-10 | **yes** |
| AiCE | +0.066 | 9.3e-10 | **yes** |
| Random | +0.162 | 9.3e-10 | **yes** |

**ms_PAB1** — best = **AlphaVariant** (median 0.504); significantly beats 7/9 other methods:

| vs | Δmedian | p(>) | sig |
|---|---|---|---|
| AdaLead | +0.001 | 4.7e-01 | ns |
| ftMLDE | +0.011 | 3.5e-02 | ns |
| GreedyWalk | +0.035 | 1.2e-06 | **yes** |
| MULTIevolve | +0.042 | 5.0e-07 | **yes** |
| EVOLVEpro | +0.061 | 9.3e-10 | **yes** |
| AiCE | +0.071 | 9.3e-10 | **yes** |
| CLADE | +0.099 | 1.9e-09 | **yes** |
| ALDE | +0.124 | 9.3e-10 | **yes** |
| Random | +0.221 | 9.3e-10 | **yes** |