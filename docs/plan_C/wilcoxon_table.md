# Plan C paired-Wilcoxon tables

AlphaVariant in Plan C ships as **base + MutCompute reward + SHAP-pruning** (`--use_mutcompute --plm_reward_lambda 0.5 --shap_prune_alphabet`). All entries are n=30 paired seeds.


## Table 1 — AlphaVariant vs AlphaVariant base (Tier 1B)

Does Plan C's shipped configuration improve over the bare base?

| Dataset | n | base mean ± std | AlphaVariant mean ± std | Δ (paired) | Wilcoxon p | W/T/L |
|---------|---|------------------|--------------------------|------------|------------|-------|
| TEV | 30 | 0.4121 | 0.4213 | +0.0092 | 0.6408 | 16/0/14 |
| GB1 | 30 | 0.9085 | 0.9145 | +0.0060 | 0.9038 | 11/11/8 |
| PhoQ | 30 | 0.4999 | 0.5596 | +0.0597 | 0.3806 | 15/2/13 |
| TRPB | 30 | 0.8822 | 0.8476 | -0.0346 | 0.1790 | 10/2/18 |

## Table 2 — AlphaVariant vs Plan B PLM-reward (the headline comparison)

Does replacing ESM-2 with MutCompute make the prior safer on the PhoQ-class landscape where PLM-reward catastrophically failed?

| Dataset | n | PLM-reward mean ± std | AlphaVariant mean ± std | Δ (paired) | Wilcoxon p | W/T/L | Significant @ α=0.10 |
|---------|---|------------------------|--------------------------|------------|------------|-------|----------------------|
| TEV | 30 | 0.4496 | 0.4213 | -0.0283 | 0.4771 | 13/0/17 | no |
| GB1 | 30 | 0.9203 | 0.9145 | -0.0059 | 1.0000 | 10/12/8 | no |
| PhoQ | 30 | 0.3987 | 0.5596 | +0.1609 | 0.0073 | 21/0/9 | **yes** |
| TRPB | 30 | 0.8699 | 0.8476 | -0.0224 | 0.3931 | 11/2/17 | no |

## Interpretation

- **PhoQ headline**: MutCompute+SHAP achieves Δ = +0.16 over PLM-reward (p = 0.0073, significant at α = 0.10). PLM-reward catastrophically degraded PhoQ relative to the base (Δ = −0.10, p = 0.003); Plan C completely reverses that, moving from significantly-worse than base to a +0.06 (non-significant) numerical improvement over base.
- **TEV, GB1**: Plan C is roughly tied with both base and PLM-reward (|Δ| < 0.03, all p > 0.40). PLM-reward retains a small numerical edge on TEV (the only landscape where the ESM-2 sequence prior helped at all in Plan B).
- **TRPB**: Plan C has a small negative Δ vs both base (−0.035, p = 0.18) and PLM-reward (−0.022, p = 0.39). TRPB is near saturation for the AlphaVariant base and is the landscape where Plan B's weighted-Hybrid selector (+0.020) outperformed all other variants; MutCompute brings no advantage here.

**Conclusion**: MutCompute is a safer zero-shot prior than ESM-2 PLM for landscapes whose high-fitness modes are non-WT-like in sequence space (PhoQ). On TEV-class landscapes where the global optimum *is* WT-like, ESM-2 retains a small numerical edge. AlphaVariant Plan C ships MutCompute as the default *because* it is universally safer (never significantly degrades), with the trade-off of giving up the small TEV/GB1 numerical wins of PLM-reward.
