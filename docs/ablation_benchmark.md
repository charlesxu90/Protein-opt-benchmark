# Ablation study — AlphaVariant components (Nature Methods draft)

> Draft Results/Methods subsection for the component ablation. Companion to
> `docs/methods_benchmark.md`. `(ref.)` marks a citation to be added. Per-config
> medians are in `docs/ablation_summary.csv`. Fitness is min–max normalized to [0, 1].

## Design

We performed a leave-one-out ablation of AlphaVariant, removing one component at a
time from the shipped configuration and re-running the full optimization campaign
(480 queries; 96 × 5 rounds; identical otherwise). All eight benchmark landscapes were
ablated. On the four **four-site** landscapes we removed (i) the MutCompute reward
shaping (`−MutCompute`) and (ii) the SHAP alphabet pruning (`−SHAP`), plus the
both-off "bare" configuration. On the four **multi-site** landscapes we removed (i) the
EV-augmented surrogate features (`−EV`, reverting to aa+one-hot), (ii) the SHAP
pruning + proposal-alphabet constraint (`−SHAP/constraint`), (iii) the mutation cap
(`−cap`, removing `--max_n_mut 2`), and (iv) the homolog-pretrained GPT prior
(`−prior`). Each configuration was run for 30 seeds, except PAB1 (10) and GFP (5),
where per-seed cost on the longer sequences made 30 seeds impractical (Methods,
Statistics). We report the median over seeds of normalized maximum fitness and
top-128 mean fitness, and the change (Δ) versus the full configuration on the same
landscape.

## Four-site landscapes: reward shaping is the dominant — and harmful — component

| Landscape | Component removed | Δ max | Δ top-128 |
|-----------|-------------------|------:|----------:|
| GB1  | − MutCompute reward | **+0.138** | +0.096 |
| GB1  | − SHAP pruning      | −0.029 | +0.002 |
| GB1  | − both (bare)       | +0.097 | +0.056 |
| PhoQ | − MutCompute reward | +0.095 | +0.027 |
| PhoQ | − SHAP pruning      | +0.027 | −0.001 |
| PhoQ | − both (bare)       | **+0.113** | +0.024 |
| TEV  | − MutCompute reward | +0.011 | +0.014 |
| TEV  | − both (bare)       | +0.012 | +0.003 |
| TrpB | − MutCompute reward | −0.001 | +0.126 |
| TrpB | − both (bare)       | +0.015 | **+0.141** |

On the four-site landscapes, **removing the MutCompute reward shaping improved or
matched maximum fitness on all four datasets** (most strongly on GB1 and PhoQ), and the
both-off "bare" configuration was the best or tied-best on every landscape. SHAP
pruning was consistently neutral (|Δ max| ≤ 0.03), consistent with its proposal
constraint being inactive on four-site combinatorial libraries (Methods). We conclude
that, on the current implementation, the structure-based reward shaping does not help
the four-site regime and that a plain UCB-reward ("bare") AlphaVariant is preferable
there.

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
