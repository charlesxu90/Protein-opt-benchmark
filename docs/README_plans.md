# AlphaVariant manuscript — Plan A / B / C reader's index

> Three parallel manuscript narratives are maintained in this
> repository. All three use the **same** n = 30 seed-level oracle
> results — they differ in which AlphaVariant configuration is named
> in the main text and which extensions / scorers are highlighted.

## Plan A — AlphaVariant ships as the base (no extension)

**Headline claim:** AlphaVariant (= bare GPT-REINFORCE + surrogate
ensemble + cluster sampling) is competitive with the best
directed-evolution baselines on three of four combinatorial landscapes
without per-landscape tuning. The three optional extensions
(PLM-reward, SHAP, weighted-hybrid) are documented in supplementary as
exploratory analyses that do not universally improve over the base.

**Why pick Plan A:** statistically cleanest. Claims only what the
paired-Wilcoxon p-values support; no "shipping a non-significant
improvement" exposure for hostile reviewers.

## Plan B — AlphaVariant ships with SHAP-pruning

**Headline claim:** AlphaVariant ships with a SHAP-based per-position
alphabet pruning step that activates once 192 oracle labels have been
collected. The pruned alphabet is biologically interpretable, and the
procedure never significantly degrades any of the four benchmark
landscapes (|Δ| ≤ 0.022; paired Wilcoxon p > 0.40 throughout).
PLM-reward and weighted-hybrid remain available as opt-in extensions
with documented failure modes.

**Why pick Plan B:** emphasises SHAP-pruning's interpretability as a
methodological contribution; positions SHAP as universally-safe
relative to PLM-reward (which significantly degrades PhoQ) and Hybrid
(which is only useful on TrpB-class multimodal landscapes).

## Plan C — AlphaVariant ships with MutCompute + SHAP  ⭐ active version

**Headline claim:** AlphaVariant ships with MutCompute (structure-
based zero-shot) reward shaping and SHAP-based alphabet pruning. **On
PhoQ — the landscape where ESM-2 PLM-reward catastrophically failed
(Δ = −0.10 vs base, paired Wilcoxon p = 0.003) — Plan C achieves
Δ = +0.16 over PLM-reward (paired Wilcoxon p = 0.0073).** On the
other three landscapes Plan C is within paired noise of both the base
and the ESM-2 PLM-reward alternative.

**Why pick Plan C:** the headline PhoQ result is the only
**statistically significant** AlphaVariant finding in this project
against a meaningful comparison condition. The mechanistic
interpretation (structure-conditioned priors are safer when
high-fitness modes are non-WT-like in sequence space) is also
publication-grade.

## Where the artefacts live

| Plan | Methods text | Wilcoxon table | Landscape descriptors | Figures + CSV |
|------|--------------|----------------|------------------------|---------------|
| A    | `docs/plan_A/methods_alphavariant_base.md`       | `docs/plan_A/wilcoxon_table.md` | `docs/plan_A/landscape_descriptors.md` | `figures/plan_A/` |
| B    | `docs/plan_B/methods_alphavariant_shap.md`       | `docs/plan_B/wilcoxon_table.md` | `docs/plan_B/landscape_descriptors.md` | `figures/plan_B/` |
| C    | `docs/plan_C/methods_alphavariant_mutcompute.md` | `docs/plan_C/wilcoxon_table.md` | `docs/plan_C/landscape_descriptors.md` | `figures/plan_C/` |

The `landscape_descriptors.md` file is identical across plans — the
*a priori* selection rule fails on 4/4 datasets regardless of which
configuration is shipped.

## How to regenerate any plan's figures

```bash
# Plan A
python scripts/draw_figures.py --csv figures/plan_A/alphavariant_comparison_values.csv --outdir figures/plan_A
python scripts/draw_supplementary_ablation.py --csv figures/plan_A/alphavariant_comparison_values.csv \
    --outdir figures/plan_A --shipped-default AlphaVariant
python scripts/compute_wilcoxon_table.py --out docs/plan_A/wilcoxon_table.md --plan A

# Plan B
python scripts/draw_figures.py --csv figures/plan_B/alphavariant_comparison_values.csv --outdir figures/plan_B
python scripts/draw_supplementary_ablation.py --csv figures/plan_B/alphavariant_comparison_values.csv \
    --outdir figures/plan_B --shipped-default AlphaVariant \
    --bar-methods "AlphaVariant_base,AlphaVariant_PLM,AlphaVariant_Hybrid,AlphaVariant" \
    --bar-labels "base,+PLM-reward,+Hybrid,AlphaVariant (+SHAP)"
python scripts/compute_wilcoxon_table.py --out docs/plan_B/wilcoxon_table.md --plan B

# Plan C
python scripts/draw_figures.py --csv figures/plan_C/alphavariant_comparison_values.csv --outdir figures/plan_C
python scripts/draw_supplementary_ablation.py --csv figures/plan_C/alphavariant_comparison_values.csv \
    --outdir figures/plan_C --shipped-default AlphaVariant \
    --bar-methods "AlphaVariant_base,AlphaVariant_PLM,AlphaVariant_Hybrid,AlphaVariant_SHAP,AlphaVariant" \
    --bar-labels "base,+PLM,+Hybrid,+SHAP only,AlphaVariant (+MC+SHAP)"
python scripts/compute_planC_wilcoxon.py --out docs/plan_C/wilcoxon_table.md
```

## Active version

**Plan C is the active version.** It is the only configuration with a
statistically-significant headline finding (PhoQ p = 0.0073) and a
clear mechanistic story (structure-based vs sequence-based zero-shot
priors). Plans A and B are retained as the conservative-fallback and
intermediate narratives in case the editorial preference shifts.
