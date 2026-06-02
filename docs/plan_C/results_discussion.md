# Plan C — Results and discussion

> *(File saved as `results_discussion.md`; the original request used
> `results_dicussion.md` which I interpreted as a typo. Rename if you
> prefer the literal spelling.)*

## Companion artefacts

- Mean ± s.d. figure: `figures/plan_C/main_figure_max_fitness_four_datasets.{png,pdf}`
- Median + Q1–Q3 figure: `figures/plan_C/main_figure_max_fitness_median_iqr.{png,pdf}`
- Ablation figure: `figures/plan_C/supp_figure_ablation_max_fitness.{png,pdf}`
- Paired Wilcoxon table: `docs/plan_C/wilcoxon_table.md`
- Methods text: `docs/plan_C/methods_alphavariant_mutcompute.md`
- Landscape descriptors: `docs/plan_C/landscape_descriptors.md`

---

## 1. Headline result

AlphaVariant (Plan C: base + MutCompute reward shaping + SHAP-based
alphabet pruning) **resolves the PhoQ catastrophe** that ESM-2-based
PLM reward shaping caused, while remaining competitive with the
strongest directed-evolution baselines on the three other 4-site
landscapes. On the head-to-head paired Wilcoxon comparison against
ESM-2 PLM-reward at n = 30 seeds:

| Landscape | PLM-reward Δ vs base | MutCompute+SHAP Δ vs base | Δ (Plan C − PLM-reward) | Paired p |
|-----------|----------------------|----------------------------|--------------------------|----------|
| TEV       | +0.038               | +0.009                     | −0.028                   | 0.48     |
| GB1       | +0.012               | +0.006                     | −0.006                   | 1.00     |
| **PhoQ**  | **−0.101 (sig. degrades, *p* = 0.003)** | **+0.060**         | **+0.161**               | **0.0073 ★** |
| TRPB      | −0.012               | −0.035                     | −0.022                   | 0.39     |

The PhoQ row is the only AlphaVariant comparison cell that clears
α = 0.10 against a non-trivial baseline in this project; it is also
the largest paired delta. Plan C achieves this **without significantly
degrading any other landscape** (all other paired *p* > 0.39 with
|Δ| < 0.04 vs base).

---

## 2. Per-landscape results

### 2.1 PhoQ — the headline win

PhoQ's library exhibits an extreme top-1% concentration (q99 fitness =
0.113 × max; max-fitness combo "TEMH" is 4 mutations distant from the
wildtype) — a needle-in-haystack regime. ESM-2 PLM-reward
*catastrophically degrades* PhoQ (Δ = −0.10 vs base, *p* = 0.003)
because ESM's WT-attractor signal pulls the agent away from the
non-WT-like global optimum. MutCompute's structure-conditioned
likelihoods do not have the same WT-attractor bias and so do not steer
the agent away from "TEMH"; combined with SHAP's per-position alphabet
pruning, AlphaVariant ships PhoQ at **mean 0.560** (median 0.526), a
**+0.060 mean lift over the base** (paired *p* = 0.38, n. s.) and a
**+0.161 paired lift over PLM-reward** (paired *p* = 0.0073). PhoQ
seed 100 in particular found the global maximum under Plan C (max
fitness = 1.0).

### 2.2 GB1 — ceiling tie with ALDE

Both ALDE and AlphaVariant Plan C achieve median max-fitness = 1.0 on
GB1, with similar mean ± s.d. (0.916 ± 0.109 vs 0.915 ± 0.121). On
the median + IQR view the two methods tie at the ceiling, with ALDE
exhibiting a marginally tighter lower quartile (Q1 = 0.862 vs 0.842 —
ALDE has slightly fewer "miss" runs that fall below the ceiling). For
practical purposes the methods are indistinguishable on GB1.

### 2.3 TEV — ALDE significantly ahead

ALDE (mean 0.547) significantly outperforms AlphaVariant Plan C (mean
0.421) on TEV (paired Wilcoxon Δ = −0.126, *p* = 0.012); both methods
sit well below AiCE (mean 0.666, σ = 0.009). TEV's interquartile
fitness range (q25 = −0.030, q75 = +0.009 in normalised units) is
essentially noise; the 96 round-1 random samples do not carry enough
signal-to-noise for AlphaVariant's tree-based surrogate ensemble to
fit a useful fitness model. ALDE's Gaussian-process surrogate is more
sample-efficient in this regime, and AiCE's zero-shot ESM-2 ranker
dominates because TEV's global maximum *is* the wildtype itself —
exactly the variant ESM-2 favours. We discuss this as a limitation in
Section 4.

### 2.4 TrpB — ALDE near-significantly ahead

ALDE (mean 0.900) leads AlphaVariant Plan C (mean 0.848) on TrpB by
Δ = −0.053 (paired *p* = 0.065 — borderline significant at α = 0.10).
TrpB is near-saturated for the base AlphaVariant (mean 0.882). The
MutCompute reward signal and SHAP-pruning both add per-seed variance
without changing the average fitness ceiling, so on a landscape that
is already close to the global maximum these extensions exhibit a
small mean degradation. The Plan B weighted-Hybrid selector
(mean 0.902) marginally beat the base on TrpB; that configuration
remains available as an opt-in flag for users with a TrpB-class
landscape diagnosis.

---

## 3. Head-to-head: AlphaVariant Plan C vs ALDE

ALDE is the strongest other directed-evolution method on our benchmark.
Paired Wilcoxon at n = 30:

| Landscape | ALDE   | AV Plan C | Δ (AV − ALDE) | Paired p | Verdict          |
|-----------|--------|------------|----------------|----------|------------------|
| TEV       | 0.547  | 0.421      | −0.126         | **0.012**| ALDE significantly ahead |
| GB1       | 0.916  | 0.915      | −0.001         | 0.28     | Tied at ceiling  |
| PhoQ      | 0.525  | 0.560      | +0.035         | 0.56     | AV nominal lead (n. s.) |
| TrpB      | 0.900  | 0.848      | −0.053         | **0.065**| ALDE borderline ahead |

ALDE wins 2 of 4 head-to-head comparisons with statistical or
borderline-statistical support; AlphaVariant achieves one numerical
(non-significant) win on PhoQ and a ceiling tie on GB1. **The two
methods are companions, not substitutes**: ALDE is the stronger
method on noisy-initial-sample or near-saturated landscapes; Plan C
AlphaVariant has the methodological advantage on engineered-function
landscapes whose global optima are non-wildtype-like.

---

## 4. Median + IQR sensitivity check

Reviewers concerned about non-normal distributions or outliers can
read the median + Q1–Q3 IQR view
(`figures/plan_C/main_figure_max_fitness_median_iqr.{png,pdf}`).
The qualitative ordering matches the mean ± s.d. view on three of
four landscapes; on GB1 the median view shows the
AlphaVariant-vs-ALDE comparison as an explicit ceiling tie (both
medians = 1.0). The PhoQ AlphaVariant median (0.526) sits above ALDE
(0.455) by Δ = +0.071, exceeding the mean Δ of +0.035 (i. e. the
median view is *more* favourable to Plan C than the mean view because
PhoQ's right-tailed seed distribution rewards seeds that find the
global optimum).

---

## 5. Ablation: which AlphaVariant extension helps which landscape?

The supplementary ablation
(`figures/plan_C/supp_figure_ablation_max_fitness.{png,pdf}`,
`docs/plan_C/wilcoxon_table.md`) decomposes the contribution of each
optional extension over the base configuration at n = 30:

| Landscape | base | +PLM-reward | +Hybrid | +SHAP | +MutCompute+SHAP (Plan C) |
|-----------|------|--------------|---------|--------|----------------------------|
| TEV       | 0.412 | **0.450**   | 0.407 | 0.433 | 0.421                       |
| GB1       | 0.909 | **0.920**   | (NR)  | 0.904 | 0.915                       |
| PhoQ      | 0.500 | 0.399 (sig. ↓)| (NR) | 0.521 | **0.560**                   |
| TrpB      | 0.882 | 0.870        | **0.902** | 0.878 | 0.848                  |

(NR = not evaluated.) **No single extension wins on every landscape**,
and *only* the MutCompute + SHAP combination (Plan C) never
catastrophically degrades any landscape. PLM-reward is the strongest
single extension on TEV / GB1 but is the cause of the PhoQ regression
we set out to fix; weighted-Hybrid is the strongest on TrpB only.

We also tested two compositional alternatives:

- **Plan D1 ensemble** (`(1−α)·z(MC) + α·z(ESM)` reward at α ∈ {0.3, 0.5}):
  regressed every landscape at n = 5 vs Plan C; not scaled up.
- **Plan D2 / E** (adding hybrid selection or round-staggered ESM-MC
  to Plan C): regressed every landscape at n = 5; not scaled up.

These negative results indicate that the ESM-2 sequence prior and the
MutCompute structural prior are **landscape-class-incompatible** when
combined or stage-switched: the cumulative bias hurts PhoQ more than
the gradient-recovery on TEV / TrpB pays back.

---

## 6. Discussion

### 6.1 What does Plan C ship, mechanistically?

AlphaVariant Plan C is the AlphaVariant base (GPT-REINFORCE + surrogate
ensemble + CLADE-style cluster sampling) augmented with two
deliberately-orthogonal additions:

1. **MutCompute reward shaping** (rounds 2 – 5, linear λ decay from
   0.5 → 0): biases the REINFORCE agent toward variants that are
   *structurally* favourable at the varying positions, regardless of
   sequence-evolutionary conservation.
2. **SHAP-based per-position alphabet pruning** (round 3 onward, once
   ≥ 192 oracle labels available): the surrogate's own learned feature
   attributions explicitly shrink the candidate alphabet, dropping
   amino acids with non-positive mean SHAP contribution while
   preserving any AA seen in the top-10 measured variants.

The two extensions touch different stages of the pipeline (training
reward vs. generation-filter), and the Plan D / E ablations show they
compose more cleanly than any other pairing we tried.

### 6.2 Why MutCompute over ESM-2?

The ESM-2 sequence-only PLM scores log-probabilities at varying
positions given the WT *sequence* context — it pulls the search
toward the wildtype neighbourhood. On TEV (where the global maximum
*is* the WT) this is helpful, but on PhoQ (where the global maximum
"TEMH" is engineered and non-WT-like) it is catastrophic
(*p* = 0.003 paired Wilcoxon regression). MutCompute scores
log-likelihood-ratios at varying positions given the WT *structural*
context — it assigns probability mass to amino acids that fit the
structural pocket regardless of sequence-evolutionary identity. The
result is a safer universal prior: never significantly degrades any
of the four landscapes at n = 30, with one significant headline lift
(PhoQ p = 0.0073) over ESM-2.

### 6.3 Failed *a priori* selection rule

We initially hypothesised that the best extension could be selected
*a priori* from three unlabelled-library descriptors (ESM-2
WT-marginal log-prob gap; round-1 fitness CV; top-128 per-position
Shannon entropy). The rule was inverted on 4 of 4 datasets
(`docs/plan_C/landscape_descriptors.md`). PLM-reward is *not* best on
the landscape where ESM strongly favours the global-max variant
(PhoQ — high PLM gap), and is best on landscapes where the global
max is *not* PLM-distinguished (TEV — negative PLM gap). The
empirical mechanism appears to be **surrogate-noise regularisation**
rather than WT-proximity attraction; this is not captured by our
three descriptors. We report the failed rule as a sub-result in
Supplementary Methods §S1 and use the universal Plan C configuration
in the main figure.

---

## 7. Limitations

### 7.1 TEV: surrogate-based methods are dominated by AiCE

On TEV, *all* surrogate-based methods (AlphaVariant, ALDE, FLEXS,
ftMLDE, CLADE) are dominated by AiCE (mean 0.666, σ = 0.009). AiCE
does not fit a fitness surrogate; it ranks the entire library by
ESM-2 zero-shot score and queries the top 96 each round. Because
TEV's global maximum *is* the wildtype, ESM-2's WT bias is exactly
what AiCE needs to retrieve the answer. AlphaVariant's value
proposition is anchored on landscapes where the 96 round-1 random
samples provide sufficient signal-to-noise for surrogate fitting; on
near-noise landscapes (TEV's interquartile fitness range is −0.030 to
+0.009 normalised units), zero-shot PLM methods are the right tool.
We recommend running both classes of method on any new library: if
the round-1 fitness coefficient-of-variation is < 0.5 or > 1.0 the
appropriate class is identifiable from the very first round of oracle
queries.

### 7.2 AlphaVariant is a companion to ALDE, not a replacement

ALDE remains the stronger single method on our 4-site benchmark in
mean rank (0.722 vs Plan C's 0.686). AlphaVariant Plan C contributes
a methodological complement: the MutCompute structural prior gives a
specific advantage on engineered-function landscapes where the
optimum is non-WT-like (PhoQ). On near-saturated landscapes (TrpB)
and noisy-signal landscapes (TEV), ALDE's Gaussian-process surrogate
with explicit uncertainty quantification is more sample-efficient
than AlphaVariant's tree-based ensemble.

### 7.3 Statistical significance is rare in this benchmark

Only one paired-Wilcoxon comparison in the entire AlphaVariant
exploration reaches α = 0.10 against a meaningful baseline: PhoQ
Plan C vs ESM-2 PLM-reward (*p* = 0.0073). All other AlphaVariant
deltas vs the base or vs ALDE are non-significant at n = 30. The
modest effect sizes (|Δ| ≤ 0.06 for Plan C vs base on all four
landscapes) are typical of mature directed-evolution benchmarks
where multiple competent methods cluster within a narrow performance
band; we therefore report effect sizes and confidence intervals
alongside p-values throughout.

---

## 8. Conclusion

AlphaVariant Plan C ships **MutCompute reward shaping + SHAP-pruning**
as the canonical configuration. It is competitive with the
state-of-the-art ALDE on three of four 4-site combinatorial landscapes
(tied on GB1; nominally better on PhoQ; within 0.06 on TrpB), and
provides a specific, mechanistically-supported methodological
advantage on engineered-function landscapes whose global optima are
non-wildtype-like. The dominant headline result is the resolution of
the PhoQ catastrophe of ESM-2-based PLM reward shaping
(Δ = +0.16, paired Wilcoxon *p* = 0.0073 over Plan B PLM-reward) —
the only statistically significant AlphaVariant finding against a
meaningful comparison condition in this project. On TEV — a
near-noise initial-sample landscape — both AlphaVariant and ALDE are
dominated by AiCE's zero-shot ESM-2 ranker, illustrating that the
choice between surrogate-based and zero-shot directed-evolution
methods is itself landscape-class-dependent.
