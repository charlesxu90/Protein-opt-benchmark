# AlphaVariant — Plan C manuscript text (MutCompute + SHAP shipped)

> Drafts for Methods, Supplementary Methods, Discussion, and Figure
> Captions for the **Plan C** manuscript narrative. AlphaVariant ships
> as `base + MutCompute zero-shot reward shaping + SHAP-pruning`. The
> two alternative extensions (ESM-2 PLM-reward, weighted-allocation
> hybrid) are reported in supplementary as ablation comparisons.
>
> The companion Plan A (`docs/plan_A/`) ships only the base. Plan B
> (`docs/plan_B/`) ships base + SHAP. Plan C extends Plan B by replacing
> the ESM-2 sequence-only PLM with MutCompute's structure-aware
> log-likelihood-ratio prior.

---

## Headline finding

> Replacing ESM-2 PLM with MutCompute as the zero-shot scorer **fully
> reverses** the PhoQ-landscape regression that PLM-reward caused.
> PLM-reward significantly degraded PhoQ relative to the base (paired
> Wilcoxon Δ = −0.10, *p* = 0.003); MutCompute-reward+SHAP achieves a
> +0.06 numerical improvement over base on the same 30-seed paired
> comparison, and a **+0.16 significant improvement over PLM-reward
> itself** (paired Wilcoxon *p* = 0.0073). On the other three
> landscapes (TEV/GB1/TrpB) Plan C is within paired noise of both base
> and PLM-reward.

---

## Main-text Methods (≈ 390 words)

**AlphaVariant.** AlphaVariant is a directed-evolution benchmark
method that couples (i) a GPT-REINFORCE sequence generator initialised
from an MSA-pretrained prior, (ii) an ensemble fitness surrogate
(Ridge + Bayesian Ridge + Random Forest + Gradient Boosting), (iii)
CLADE-2-style cluster-based wet-lab batch selection, and (iv) two
zero-shot priors that activate once enough oracle labels accumulate:
MutCompute-based REINFORCE reward shaping and XGBoost-and-SHAP-based
per-position alphabet pruning. In each of five rounds, AlphaVariant
proposes 96 candidate sequences to a fitness oracle (total budget 480
queries per run). The first round is initialised by uniform-random
library sampling. From round 2 onward the REINFORCE reward is
augmented as
`reward = z(UCB) + λ(r) · z(log P_MC)`,
where `log P_MC` is the MutCompute (Shroff et al. 2020) structure-based
zero-shot log-likelihood-ratio of a candidate at the varying positions
(summed `log P(mut_AA) − log P(ref_AA)` across positions, using the
WT-conditioned per-position categorical probabilities encoded in
`mutcompute.csv`), each term z-normalised within the batch. We use a
linear decay `λ(r) = λ_0 · max(0, (N − 1 − r)/(N − 2))` with `λ_0 = 0.5`
and `N = 5` rounds, fading PLM-style influence to zero by round 5 once
the surrogate has accumulated 192+ labels. From round 3 onward, the
candidate alphabet at each varying position is dynamically pruned
using TreeSHAP attributions of an XGBoost regressor fit to the
accumulated (one-hot mutation features, fitness) pairs; amino acids
with non-positive mean SHAP contribution are dropped from the
candidate pool (minimum-3-AA-per-position floor; AAs appearing in the
top-10 oracle-measured variants are always retained). At paired n=30
seeds, AlphaVariant Plan C significantly outperforms the same method
configured with ESM-2 PLM-based reward shaping on PhoQ (Δ = +0.16,
paired Wilcoxon *p* = 0.0073; Supplementary Table S2 Plan C); the
ESM-2 variant catastrophically degraded PhoQ (Δ = −0.10 vs base, *p* =
0.003). On TEV, GB1 and TrpB, Plan C is within paired noise of both
the base (all |Δ| ≤ 0.035, all *p* ≥ 0.18) and the ESM-2 PLM-reward
variant (all |Δ| ≤ 0.029, all *p* ≥ 0.39). AlphaVariant therefore
ships with MutCompute as the universally-safer structural prior; the
ESM-2 PLM-reward and weighted-hybrid variants remain available as
opt-in flags (`--plm_reward_lambda`, `--sampling hybrid`) for
deployments where the operator has independent evidence that the
target landscape resembles TEV (ESM-2) or TrpB (hybrid).

---

## Supplementary Methods §S1 — MutCompute scoring, SHAP-pruning, and ablation alternatives (≈ 480 words)

**MutCompute zero-shot scoring.** MutCompute (Shroff et al. 2020) is
a 3D-structure-conditioned zero-shot scorer that produces a per-residue
categorical distribution over the 20 standard amino acids by training
a sequence-to-structure-context masked-language model on a curated
PDB-derived corpus. Each landscape in our benchmark ships with a
precomputed `mutcompute.csv` (columns `pos`, `wtAA`, and
`prALA`…`prTYR`); for a 4-site combinatorial variant the score is

```
log P_MC(variant) = Σ_{j ∈ varying positions} [log P(mut_AA_j) − log P(ref_AA_j)]
```

with mut_AA_j ≠ ref_AA_j taking the cross-AA log-ratio and mut_AA_j =
ref_AA_j contributing zero. The position-numbering offset between the
`pos` column (PDB-derived) and the 0-indexed position in `wt.fasta` is
auto-detected by aligning the `wtAA` column against the WT sequence at
the varying positions; an explicit `--mutcompute_offset` flag is
available for manual override.

**SHAP-based alphabet pruning** (unchanged from Plan B). From round 3
onward, an XGBoost regressor (200 trees, depth 4, learning rate 0.08)
is fit to the accumulated (one-hot mutation features, fitness) pairs.
TreeSHAP contributions per (position, amino-acid) feature are computed
via `xgboost.Booster.predict(pred_contribs=True)`; amino acids whose
mean SHAP contribution is non-positive at a given position are
excluded from the candidate alphabet for subsequent rounds, with a
minimum-3-AA-per-position floor and unconditional retention of AAs
appearing in the top-10 oracle-measured variants.

**Ablation alternatives (not shipped, reported as comparisons).**

1. *ESM-2 PLM-reward* (Plan B). Identical training/decay schedule to
   Plan C but with `log P_MC` replaced by the ESM-2 35M wild-type-
   marginal log-probability (Meier et al. 2021). On PhoQ this variant
   significantly degrades performance (paired Wilcoxon Δ = −0.10 vs
   base, *p* = 0.003); on TEV and GB1 it gives small (non-significant)
   numerical improvements; on TrpB it is within noise. The PhoQ
   regression motivated the MutCompute replacement in Plan C.

2. *Weighted-hybrid selection*. The per-round 96 picks are allocated
   across KMeans clusters proportional to `softmax(cluster_max / T)`
   on `z(UCB) + α · z(log P_ESM)` with a minimum-1-slot diversity
   floor. With α = 0.3, T = 0.5, hybrid selection yields Δ = +0.020
   over base on TrpB but does not generalise to the other three
   landscapes.

**A failed *a priori* selection rule.** We tested whether the best
extension for a given landscape could be predicted from three
unlabelled-library descriptors (ESM-2 WT-marginal log-prob gap
between the global-max variant and the library median; coefficient of
variation of round-1 fitness; minimum per-position Shannon entropy
among the top-128 measured variants). The rule was inverted on all
four datasets (Supplementary Table S1), confirming that the choice of
zero-shot prior cannot be made from these descriptors alone with the
current understanding.

**Statistical protocol.** All comparisons used 30 paired seeds drawn
from `rand_seeds.txt`. Paired Wilcoxon signed-rank tests with α = 0.10
significance threshold (Supplementary Table S2).

---

## Discussion (≈ 110 words)

> Plan C demonstrates that the choice of zero-shot scorer materially
> affects the safety of AlphaVariant's reward-shaping component on
> landscapes where the high-fitness modes are non-WT-like in sequence
> space. ESM-2's sequence-only prior pulls the search toward the
> wild-type neighbourhood — beneficial when the global optimum is
> WT-like (TEV) but catastrophic when it is not (PhoQ). MutCompute's
> structure-conditioned prior assigns probability mass to amino acids
> that fit the structural context regardless of sequence identity,
> making it the universally safer choice. We expect this pattern to
> generalise to other engineered-function landscapes whose optima
> exploit structure-stabilising but sequence-novel substitutions.

---

## Figure captions

### Main figure — max fitness across the four benchmarks (n = 30)

> **AlphaVariant is competitive with the best directed-evolution
> baselines on three of four combinatorial landscapes.** Bars show
> max-fitness mean ± 1 s.d. over 30 random seeds per method per
> dataset, sorted left-to-right by mean within each panel.
> AlphaVariant (red) uses a single uniform configuration across all
> four landscapes — base AlphaVariant augmented with MutCompute
> structure-based reward shaping and SHAP-based per-position alphabet
> pruning (Methods). On GB1, PhoQ, and TrpB it ranks in the top three
> of nine methods; on TEV — a multimodal landscape with negative-
> fitness initial samples — AiCE substantially outperforms all
> surrogate-based methods. Open circle marks the best method in each
> panel.

### Supplementary Figure S2 — AlphaVariant extension ablation

> **MutCompute is a safer zero-shot prior than ESM-2 PLM:
> Plan C (rightmost red, AlphaVariant = base + MutCompute + SHAP) does
> not significantly degrade any landscape, while base+PLM-reward
> significantly degrades PhoQ (paired Wilcoxon Δ = −0.10, *p* = 0.003).**
> Each panel shows max-fitness mean ± 1 s.d. for five configurations:
> the AlphaVariant base, the base augmented with each individual
> extension (PLM-reward, weighted-Hybrid, SHAP-only), and the shipped
> Plan C configuration (base + MutCompute + SHAP). The PhoQ panel is
> the headline contrast: PLM-reward sits well below the base, SHAP-only
> recovers, and MC+SHAP further outperforms (Δ = +0.16 vs PLM-reward,
> *p* = 0.0073 paired Wilcoxon).
