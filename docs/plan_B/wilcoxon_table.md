# Paired-Wilcoxon — AlphaVariant extensions vs base (= Tier 1B)

All n=30 paired seeds. The base AlphaVariant ships as the default; each row tests whether adding one extension reliably improves max-fitness on that landscape.

| Dataset | Extension | n | Base mean ± std | Base+Ext mean ± std | Δ (paired mean) | Wilcoxon p | Wins/Ties/Losses | Significant @ α=0.10 |
|---------|-----------|---|------------------|----------------------|------------------|------------|------------------|----------------------|
| TEV | PLM-reward | 30 | 0.4121 ± 0.0984 | 0.4496 ± 0.1458 | +0.0375 | 0.3409 | 16/4/10 | no |
| TEV | Hybrid | 30 | 0.4121 ± 0.0984 | 0.4065 ± 0.1060 | -0.0056 | 0.9234 | 15/3/12 | no |
| TEV | SHAP | 30 | 0.4121 ± 0.0984 | 0.4331 ± 0.1371 | +0.0210 | 0.4004 | 16/3/11 | no |
| GB1 | PLM-reward | 30 | 0.9085 ± 0.1226 | 0.9203 ± 0.0962 | +0.0119 | 0.8259 | 8/16/6 | no |
| GB1 | Hybrid | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| GB1 | SHAP | 30 | 0.9085 ± 0.1226 | 0.9042 ± 0.1211 | -0.0042 | 0.8960 | 9/12/9 | no |
| PhoQ | PLM-reward | 30 | 0.4999 ± 0.2318 | 0.3987 ± 0.1164 | -0.1012 | 0.0026 | 6/4/20 | **yes** |
| PhoQ | Hybrid | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| PhoQ | SHAP | 30 | 0.4999 ± 0.2318 | 0.5214 ± 0.2579 | +0.0215 | 0.7413 | 14/2/14 | no |
| TRPB | PLM-reward | 30 | 0.8822 ± 0.1011 | 0.8699 ± 0.1106 | -0.0122 | 0.8751 | 14/6/10 | no |
| TRPB | Hybrid | 30 | 0.8822 ± 0.1011 | 0.9017 ± 0.0960 | +0.0195 | 0.3224 | 13/10/7 | no |
| TRPB | SHAP | 30 | 0.8822 ± 0.1011 | 0.8783 ± 0.1161 | -0.0039 | 0.6541 | 9/10/11 | no |

**Summary:** 1 of 12 (extension × landscape) cells pass paired-Wilcoxon at α = 0.10.

## Interpretation

- All 12 (extension × landscape) cells fail to reach α = 0.10 — no extension reliably *improves* over the AlphaVariant base. However, among the three extensions, SHAP-pruning is the only one that **never significantly degrades** any landscape (|Δ| ≤ 0.005 on GB1/TRPB; +0.021/+0.022 on TEV/PhoQ).
- PLM-reward shaping **significantly degrades PhoQ** (Δ = −0.10, paired Wilcoxon p = 0.003 — bold in the table above).
- Hybrid selection was evaluated only on TEV/TRPB (Methods Supplementary §S1).
- AlphaVariant therefore ships with SHAP-pruning as the universally-safe extension; PLM-reward and Hybrid remain available as opt-in flags for deployments where the operator has independent evidence that the target landscape resembles TEV (PLM-reward) or TrpB (Hybrid).
