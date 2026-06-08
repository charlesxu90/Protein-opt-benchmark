# Results — Benchmarking AlphaVariant against directed-evolution methods

> Draft manuscript subsection for the Results section.
> Figure callouts: **Fig. 2** (main), **Extended Data Fig. 2** (four-site),
> **Extended Data Fig. 3** (multi-site). Values are medians over 30 seeds
> unless noted; fitness is min–max normalized to [0, 1] per landscape.

## A unified benchmark of eight protein-fitness landscapes

To position AlphaVariant against the directed-evolution and machine-learning–guided
optimization literature, we assembled a benchmark of eight experimentally measured
fitness landscapes spanning seven protein families and five functional classes
(**Fig. 2a**): a binding protein (GB1), a signaling/RNA-binding pair (PhoQ, PAB1),
two enzymes (TrpB, TEV), two fluorescent proteins (CreiLOV, GFP), and a viral capsid
assembly protein (AAV). The eight landscapes fall into two regimes that stress
different aspects of an optimizer. The **four-site combinatorial** landscapes
(GB1, PhoQ, TEV, TrpB; 140,517–159,132 measured variants) are densely characterized
20⁴ libraries in which every proposed variant has a ground-truth label. The
**multi-site** landscapes (AAV, CreiLOV, GFP, PAB1; 15–233 variable positions) span
combinatorial spaces of 10³⁶ to >10¹⁸⁰ sequences that no experimental library covers,
and therefore require a learned surrogate of the true landscape (see Methods).

We compared AlphaVariant against nine baselines drawn from four method families
(**Fig. 2b**): basic search (Random, GreedyWalk); supervised/active learning
(ftMLDE, ALDE, CLADE, EVOLVEpro, MULTI-evolve); PLM/structure-guided design (AiCE);
and generative/policy-based search (AdaLead and AlphaVariant). All methods operated
under an identical campaign budget of **480 queries** (96 sequences × 5 rounds),
were run for **30 random seeds**, and were scored with the same two metrics: the
normalized **maximum fitness** discovered (the headline directed-evolution objective)
and the **median top-128 mean fitness** (a batch-quality metric that rewards
discovering many high-fitness variants rather than a single lucky hit).

## Four-site combinatorial landscapes are extreme needles-in-a-haystack

The four-site landscapes are dominated by dead and near-dead variants
(**Extended Data Fig. 2a**): the fraction of sequences with normalized fitness ≥ 0.5
is 0.13% (GB1), 0.007% (PhoQ), 0.05% (TEV), and 0.35% (TrpB). The survival curves
show that for PhoQ and TEV a uniformly random campaign of 480 draws is not expected
to recover even a single high-fitness variant, making these landscapes a stringent
test of guided exploration.

On maximum fitness (**Fig. 2c**), AlphaVariant was competitive with the strongest
baselines on every landscape and best-in-class on PhoQ, the hardest landscape: it
recovered a median normalized max fitness of **0.53 on PhoQ** (best of all methods),
**1.00 on GB1** (tied with ALDE at the global optimum), **0.83 on TrpB**, and **0.38
on TEV**. ALDE was the strongest four-site competitor (median max fitness 0.93 on
TrpB, 1.00 on GB1), reflecting the suitability of its DNN-ensemble Thompson sampling
for dense, fully labeled combinatorial libraries. On the batch-quality metric
(**Extended Data Fig. 2b**), AlphaVariant remained in the top tier (median top-128
mean fitness 0.47 on GB1, 0.55 on TEV, 0.55 on TrpB, 0.12 on PhoQ), trailing only
ALDE/CLADE/MULTI-evolve by small margins.

Aggregating across the four landscapes (**Fig. 2e**, left), AlphaVariant placed
**third by mean rank** behind ALDE and MULTI-evolve. The four-site regime favors
supervised methods that exploit exhaustive labels, and AlphaVariant matched them
without any landscape-specific tuning.

## On multi-site landscapes AlphaVariant is the top-ranked method

The multi-site landscapes test optimization over realistically vast sequence spaces.
We replaced the lookup-table landscape with a learned **CNN oracle** (GGS-style
BaseCNN) trained on all measured variants of each protein; the oracles are accurate
on held-out data (test Spearman ρ = 0.89 AAV, 0.98 CreiLOV, 0.86 GFP, 0.90 PAB1;
**Extended Data Fig. 3a**), and every queried sequence — including those far from the
measured set — is scored by the oracle (Methods).

Here AlphaVariant was the strongest method overall. On maximum fitness (**Fig. 2d**)
it achieved the best median on **three of four landscapes** — AAV (0.71), CreiLOV
(0.99), and PAB1 (0.58) — and was second on GFP (0.93 vs 0.95 for AiCE, whose
structure-conditioned prior is particularly effective on the GFP fold). On the
top-128 batch-quality metric (**Extended Data Fig. 3b**) AlphaVariant was best on
AAV (0.65) and PAB1 (0.50), and within noise of the leaders on CreiLOV (0.90 vs 0.94
for AdaLead) and GFP (0.80 vs 0.85 for AiCE). Pairwise Bonferroni-corrected Wilcoxon
tests (**Extended Data Fig. 3c**) confirm these advantages are statistically
significant: on AAV, AlphaVariant outperforms all nine baselines on both metrics
(9/9 significant at α = 0.05), and on PAB1 it is significant against 8/9 baselines on
maximum fitness and 7/9 on top-128.

Aggregating across the four multi-site landscapes (**Fig. 2e**, right), AlphaVariant
was the **top-ranked method** by mean rank (1.25), ahead of GreedyWalk, AdaLead, and
AiCE. The contrast between the two regimes is informative: AlphaVariant's GPT prior
plus REINFORCE policy is competitive where exhaustive labels exist (four-site) and
decisively strongest where the search space is too large to enumerate (multi-site) —
precisely the regime that matters for real protein-engineering campaigns.

## Summary

Across eight landscapes, AlphaVariant is the only method that is simultaneously
top-tier on the dense four-site libraries (best on PhoQ, third overall) and the best
method overall on the multi-site oracle landscapes (rank 1 of 10). It achieves this
with a single configuration and no per-landscape tuning, supporting its use as a
general-purpose protein sequence optimizer.
