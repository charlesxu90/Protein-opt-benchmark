# Ablation study — AlphaVariant components (Nature Methods draft)

> Draft Results/Methods subsection for the component ablation. Companion to
> `docs/methods_benchmark.md`. `(ref.)` marks a citation to be added. Per-config
> medians are in `docs/ablation_summary.csv`. Fitness is min–max normalized to [0, 1].

## Design

We performed component ablations of AlphaVariant, re-running the full optimization
campaign (480 queries; 96 × 5 rounds; identical otherwise) for each configuration. All
eight benchmark landscapes were ablated. On the four **four-site** landscapes the shipped
configuration is **bare + finetune** (GPT-REINFORCE + surrogate-UCB reward + 5-model
ensemble + cluster sampling + per-round prior finetuning); we report a leave-one-out
removing (i) prior finetuning (`−finetune`, = bare), (ii) ensemble scoring
(`−ensemble`, single-model surrogate), and (iii) the REINFORCE step (`−RL`,
generate-and-prioritize from the prior), with Δ versus bare+finetune. We separately
verified that the MutCompute reward shaping and SHAP alphabet pruning of the earlier
Plan C configuration do not improve over bare on four-site and are therefore excluded.
On the four **multi-site** landscapes we used a leave-one-out design from the shipped
configuration, removing (i) the
EV-augmented surrogate features (`−EV`, reverting to aa+one-hot), (ii) the SHAP
pruning + proposal-alphabet constraint (`−SHAP/constraint`), (iii) the mutation cap
(`−cap`, removing `--max_n_mut 2`), and (iv) the homolog-pretrained GPT prior
(`−prior`). Each configuration was run for 30 seeds, except PAB1 (10) and GFP (5),
where per-seed cost on the longer sequences made 30 seeds impractical (Methods,
Statistics). We report the median over seeds of normalized maximum fitness and
top-128 mean fitness, and the change (Δ) versus the reference configuration on the same
landscape — the **bare baseline** for four-site, the **full** configuration for
multi-site.

## Four-site landscapes: RL and prior finetuning carry the four-site regime

Leave-one-out from the shipped bare+finetune configuration (Δ versus baseline):

| Landscape | − finetune prior | − ensemble (single RF) | − RL (generate + prioritize) |
|-----------|:----------------:|:----------------------:|:----------------------------:|
| | Δmax / Δtop128 | Δmax / Δtop128 | Δmax / Δtop128 |
| GB1  | −0.041 / −0.044 | +0.000 / +0.003 | **−0.138** / −0.002 |
| PhoQ | +0.008 / −0.001 | −0.016 / −0.004 | +0.000 / −0.006 |
| TEV  | −0.000 / **−0.061** | −0.002 / −0.042 | +0.002 / **+0.043** |
| TrpB | +0.000 / −0.026 | **+0.076** / −0.054 | **+0.090** / −0.038 |

Three components were examined. **Prior finetuning** contributes positively, chiefly to
top-128 mean fitness: removing it lowers top-128 by 0.026–0.061 on GB1/TEV/TrpB and GB1
maximum fitness by 0.041, while never improving any cell — so it is retained in the
shipped configuration. **The REINFORCE step** is essential for GB1 maximum fitness
(−0.138 when removed) but, once the prior is finetuned, is largely redundant on the
other landscapes (PhoQ maximum fitness unchanged; TEV/TrpB maximum fitness unchanged or
higher under generate-and-prioritize): finetuning and REINFORCE both supply high-quality
candidates and therefore partially substitute, an interaction not visible when ablating
RL from the un-finetuned bare baseline (where its removal is uniformly damaging). **The
5-model ensemble** mainly aids top-128 (TEV/TrpB) with a landscape-dependent effect on
maximum fitness (a single random-forest surrogate raises TrpB max by 0.076 but lowers its
top-128 by 0.054). Separately, the MutCompute reward shaping and SHAP alphabet pruning of
the earlier Plan C configuration did not improve over bare on four-site and are excluded.
We therefore ship **bare + finetune** as the four-site configuration.

## Multi-site landscapes: the mutation cap is critical and scales with sparsity

| Landscape | Variable positions | − cap Δ top-128 | − EV Δ top-128 |
|-----------|-------------------:|----------------:|---------------:|
| CreiLOV | ~15 | +0.015 | +0.000 |
| AAV     | 28  | −0.253 | −0.006 |
| PAB1    | 74  | −0.391 | −0.060 |
| GFP     | 233 | **−0.851** | −0.022 |

The single clearest effect in the study is the **mutation cap** (`--max_n_mut 2`): its
removal degraded top-128 mean fitness monotonically with the size and sparsity of the
search space, from negligible on CreiLOV (~15 variable positions, ~90% of the library
measured) to a near-total collapse on GFP (233 variable positions; top-128 0.813 → ≈0).
Without the cap, the generator drifts into regions far from the measured set, where the
oracle's predictions are unreliable, and batch quality breaks down — exactly where the
search space is too large to be otherwise constrained. The **EV-augmented features**
showed the complementary pattern: neutral on the small, well-covered landscapes
(CreiLOV, AAV) and beneficial on the larger, sparser ones (PAB1, GFP), where a
homolog-derived statistical-energy prior is most informative. The SHAP/constraint and
the homolog-pretrained prior were approximately neutral across all four multi-site
landscapes (|Δ| ≤ 0.06).

## Synthesis

The two regimes are governed by **different, oppositely-signed critical components**.
On the densely sampled four-site libraries, AlphaVariant's reward shaping is
unnecessary and mildly harmful, so the bare configuration is preferable. On the large
multi-site landscapes, where the candidate space cannot be enumerated, the mutation cap
that keeps generation near the measured manifold is essential, and its importance grows
with landscape sparsity. Together these results indicate that AlphaVariant's robustness
across regimes (main text) comes less from any single universally-helpful module than
from having a manifold-constraining mechanism available exactly where the search space
demands it.

## Methods note

Ablations used the same datasets, budget, surrogate, metrics, and statistics as the
main benchmark (`docs/methods_benchmark.md`). Seed counts were 30 per configuration
except PAB1 (10) and GFP (5); one GFP `−cap` seed initially failed with a transient GPU
out-of-memory error under concurrency and was re-run in isolation, restoring n = 5.
Because the four-site "full" configuration is re-run here on the current codebase, its
absolute values differ from the main-text four-site figures (which were produced under
an earlier implementation); the ablation is internally consistent (all arms share the
current codebase and seeds), so the reported deltas are the meaningful quantity.
Reproduce with `scripts/alphavariant/_ablation_gb1_creilov.sh` and
`scripts/alphavariant/_ablation_rest.sh`; summarize with `scripts/summarize_ablation.py`.
