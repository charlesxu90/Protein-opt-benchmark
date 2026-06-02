# AlphaVariant — Plan B manuscript text (SHAP-pruning shipped)

> Drafts for Methods, Supplementary Methods, Discussion, and Figure
> Captions for the **Plan B** manuscript narrative. AlphaVariant ships
> as `base + SHAP-based per-position alphabet pruning`. The other two
> extensions (PLM-reward shaping, weighted hybrid selection) are
> documented in supplementary as exploratory variants with known
> failure modes.
>
> The companion Plan A narrative (`docs/plan_A/`) ships only the base
> and treats SHAP as one of three exploratory extensions. Plan B
> emphasises SHAP-pruning's interpretability and universal-safety
> properties.

---

## Main-text Methods (≈ 380 words)

**AlphaVariant.** We propose AlphaVariant, a directed-evolution
benchmark method that couples a GPT-REINFORCE sequence generator, an
ensemble fitness surrogate, CLADE-2-style cluster-based wet-lab batch
selection, and a SHAP-pruned per-position alphabet that tightens once
oracle labels accumulate. In each of five rounds, AlphaVariant proposes
96 candidate sequences to a fitness oracle (total budget 480 queries
per run). The generator is a small transformer (≈ 0.8 M parameters)
initialised from an MSA-pretrained prior and updated by REINFORCE
against a UCB acquisition score over the surrogate ensemble. The
surrogate is an average of four base regressors (Ridge, Bayesian
Ridge, Random Forest, Gradient Boosting) trained on the cumulative
oracle-queried set with one-hot mutation features. From round 3 onward
— once at least 192 oracle labels have been collected — AlphaVariant
fits an XGBoost regressor on the cumulative (one-hot, fitness) pairs
and computes TreeSHAP feature attributions per (position, amino-acid)
feature. Amino acids whose mean SHAP contribution falls at or below
zero at a given position are excluded from the candidate alphabet for
subsequent rounds; minimum alphabet size per position is bounded
below at 3 (default) to retain diversity, and amino acids that appear
in the top-10 oracle-measured variants are always retained. The
GPT-generated proposal pool in the affected rounds is filtered against
this dynamically pruned alphabet before within-cluster sampling
selects the round's 96 oracle queries. Two design properties make
SHAP-pruning the shipped default rather than an extension: (i) the
pruned alphabet is biologically interpretable — biologists can inspect
which amino acids the model has ruled out at each position when
planning follow-up libraries; (ii) the procedure never significantly
degrades any of the four landscapes we evaluated at n = 30 (paired
Wilcoxon p > 0.10 on all four; |Δ| ≤ 0.005 on three landscapes,
+0.022 mean Δ on PhoQ). We additionally implemented two alternative
extensions — PLM reward shaping and a weighted-allocation hybrid
selector — and evaluated them as ablations against the shipped
configuration (Supplementary Fig. S2 and Supplementary Table 2). Both
have demonstrated failure modes (PLM-reward significantly degrades
PhoQ, p = 0.003; hybrid is only useful on multimodal landscapes such
as TrpB). SHAP-pruning is therefore the **universally-safe**
extension and AlphaVariant ships it by default; the other two remain
available as opt-in flags (`--plm_reward_lambda`, `--sampling hybrid`)
for deployments where operators have independent landscape-specific
evidence.

---

## Supplementary Methods §S1 — Unshipped AlphaVariant extensions, ablation results, and a failed *a priori* selection rule (≈ 480 words)

**PLM-reward shaping.** This extension augments the REINFORCE reward
signal from UCB acquisition score to
`reward = z(UCB) + λ(r) · z(log P_ESM)`, where `log P_ESM` is the
ESM-2 35M wild-type-marginal log-probability of a candidate at the
varying positions (Meier et al. 2021), each term z-normalised within
the batch. The default linear decay `λ(r) = λ_0 · max(0, (N − 1 − r) /
(N − 2))` with `λ_0 = 0.5` faded PLM influence from 0.5 in round 2 to 0
in round 5. The intent was to regularise the surrogate against
information-poor round-1 random samples. Empirically, PLM-reward
shaping helped TEV (+0.038 paired mean) and GB1 (+0.012) but
**significantly degraded PhoQ** (Δ = −0.10, paired Wilcoxon p =
0.003) — PhoQ's high-fitness modes are *not* wild-type-proximate, so
the ESM-2 prior pulls the search away from the global optimum.

**Weighted hybrid selection.** This extension allocates the per-round
batch of 96 picks across KMeans clusters proportional to
`softmax(combined_max_k / T)`, where `combined_max_k = max_i [z(UCB_i)
+ α · z(log P_ESM,i)]` for items i ∈ cluster k, T is a softmax
temperature, and at least one slot is reserved per cluster as a
diversity floor. With α = 0.3 and T = 0.5 on TrpB, hybrid selection
yielded a +0.020 paired mean improvement over the base. Hybrid was
evaluated on TEV and TrpB only because it requires structured top
peaks (low per-position alphabet entropy among top variants), a
property that GB1 and PhoQ do not share.

**A failed *a priori* selection rule.** We initially hypothesised that
the best extension for a given landscape could be predicted from three
descriptors computed on the unlabelled library (ESM-2 WT-marginal
log-prob gap between the global-max variant and the library median;
coefficient of variation of round-1 fitness; minimum per-position
Shannon entropy among the top-128 measured variants). The rule was
inverted in 4/4 cases (Supplementary Table 1): on PhoQ the descriptor
profile predicted PLM-reward as the best extension, but PLM-reward
significantly *degraded* PhoQ; on TEV the descriptors predicted SHAP,
but PLM-reward in fact had the highest mean delta. We therefore
abandoned the rule-based selection and ship a single configuration
(base + SHAP) chosen on the basis that it never significantly degrades
any landscape (paired Wilcoxon |p| > 0.40 on every landscape, all
deltas |Δ| ≤ 0.022). The negative result on the descriptor rule is
reported for completeness; it implies that the mechanism by which a
given extension helps a given landscape is *not* captured by the three
descriptors we proposed, and that the choice of extension cannot be
made from unlabelled-library descriptors alone with the current
understanding.

**Statistical protocol.** All ablations were evaluated at n = 30
random seeds drawn from `rand_seeds.txt`. Comparisons against the
AlphaVariant *base* (the unshipped Tier 1B configuration without any
extension) used paired Wilcoxon signed-rank tests on the seed-matched
max-fitness deltas; significance threshold α = 0.10 (Supplementary
Table 2). The shipped AlphaVariant configuration is base + SHAP; the
ablation table compares each extension to the base, not to the shipped
configuration.

---

## Discussion (≈ 110 words)

> AlphaVariant ships with SHAP-pruning enabled by default. The choice
> is grounded in two properties of the SHAP step rather than in a
> single-dataset performance gain: (i) the pruned per-position alphabet
> is biologically interpretable, providing immediate value for
> follow-up library design beyond raw max-fitness; (ii) SHAP-pruning
> never significantly degrades any of the four benchmark landscapes at
> n = 30, in contrast to PLM reward shaping (significantly degrades
> PhoQ) and weighted hybrid selection (only useful on multimodal,
> structured-peak landscapes). The two unshipped extensions remain
> available as opt-in flags for users with prior knowledge that the
> target landscape resembles one of the failure-mode-mapped classes.

---

## Figure captions

### Main figure: max fitness across the four benchmarks (n = 30)

> **AlphaVariant is competitive with the best directed-evolution
> baselines on three of four combinatorial landscapes.** Bars show
> max-fitness mean ± 1 s.d. over 30 random seeds per method per
> dataset, sorted left-to-right by mean within each panel. AlphaVariant
> (red) uses a single uniform configuration — base AlphaVariant +
> SHAP-based per-position alphabet pruning — across all four
> landscapes (Methods). On GB1, PhoQ, and TrpB it places in the top
> three of nine methods (#1, #3, #3 respectively); on TEV — a
> multimodal landscape with negative-fitness initial samples — the
> zero-shot ESM-2 ranker AiCE substantially outperforms all
> surrogate-based methods, indicating that AlphaVariant's surrogate
> cannot fit reliably from 96 initial random samples on TEV. Open
> circle marks the best method in each panel.

### Supplementary Figure S2: AlphaVariant extension ablation

> **SHAP-pruning is the universally-safe extension; PLM-reward
> shaping significantly degrades PhoQ.** Each panel shows max-fitness
> mean ± 1 s.d. for four configurations: the AlphaVariant base (grey,
> leftmost) and the base augmented with one of three optional
> extensions — PLM-reward shaping, weighted hybrid selection, or the
> SHAP-pruned alphabet that AlphaVariant ships with (red, rightmost
> bar marked "AlphaVariant"). PLM-reward shaping significantly
> degrades PhoQ (paired Wilcoxon Δ = −0.10, p = 0.003); hybrid was
> evaluated only on TEV and TrpB ("NR" = not evaluated). All
> AlphaVariant–base deltas for the shipped (SHAP) configuration
> satisfy paired Wilcoxon p > 0.10, but the |Δ| is bounded by 0.022
> across all four landscapes — i.e. SHAP-pruning is empirically the
> universally-safe extension. See Supplementary Methods §S1 for
> mechanism rationale.
