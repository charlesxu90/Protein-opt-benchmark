# Results — Benchmarking AlphaVariant (Nature Methods draft)

> Draft Results subsection, formatted for Nature Methods.
> Figure callouts: **Fig. 2** (main), **Extended Data Fig. 2** (four-site),
> **Extended Data Fig. 3** (multi-site). `(ref.)` marks a citation to be added.
> Statistics: medians over n = 30 independent seeds; fitness min–max normalized
> to [0, 1] within each landscape.

## A standardized benchmark across eight protein-fitness landscapes

To evaluate AlphaVariant against established sequence-optimization approaches, we
assembled a benchmark of eight experimentally measured fitness landscapes spanning
seven protein families and five functional classes (Fig. 2a): a binding domain
(GB1 (ref.)), a signaling kinase and an RNA-binding domain (PhoQ (ref.), PAB1 (ref.)),
two enzymes (TrpB (ref.), TEV (ref.)), two fluorescent proteins (CreiLOV (ref.),
avGFP (ref.)), and a viral-capsid assembly protein (AAV (ref.)). The landscapes form
two regimes that probe complementary capabilities. Four **four-site combinatorial**
landscapes (GB1, PhoQ, TEV, TrpB; 140,517–159,132 measured variants) are near-complete
20⁴ libraries in which every candidate has a measured label, so the landscape is
queried as an exact lookup table. Four **multi-site** landscapes (AAV, CreiLOV, GFP,
PAB1; 15–233 variable positions) define combinatorial spaces of approximately 10³⁶ to
>10¹⁸⁰ sequences that no experimental library covers, and were therefore queried
through a learned surrogate of the landscape (Methods).

We compared AlphaVariant with nine published methods spanning four families (Fig. 2b):
basic search (Random, GreedyWalk); supervised and active-learning methods
(ftMLDE (ref.), ALDE (ref.), CLADE (ref.), EVOLVEpro (ref.), MULTI-evolve (ref.));
a structure-guided method (AiCE (ref.)); and generative or policy-based search
(AdaLead (ref.) and AlphaVariant). All methods used an identical campaign budget of
480 queries (96 sequences per round × 5 rounds), were run for 30 independent seeds,
and were scored with two metrics: the normalized **maximum fitness** discovered and
the **median top-128 mean fitness**, a batch-quality metric that rewards recovering
many high-fitness variants rather than a single outlier (ref.).

## Densely sampled four-site landscapes

The four-site landscapes contain very few high-fitness variants: the fraction of
sequences with normalized fitness ≥ 0.5 ranged from 0.007% (PhoQ) to 0.35% (TrpB)
(Extended Data Fig. 2a). For PhoQ and TEV, a uniformly random campaign of 480 draws
is not expected to recover any high-fitness variant, so success requires guided
exploration.

On maximum fitness (Fig. 2c), AlphaVariant was competitive with the strongest
baselines on all four landscapes and achieved the highest median on PhoQ, the sparsest
landscape (median normalized maximum fitness 0.53). It reached the global optimum on
GB1 (median 1.00, matching ALDE), and obtained 0.83 on TrpB and 0.38 on TEV. ALDE was
the strongest four-site baseline (median maximum fitness 0.93 on TrpB and 1.00 on GB1),
consistent with the suitability of ensemble Thompson sampling for densely labeled
libraries. On top-128 mean fitness (Extended Data Fig. 2b), AlphaVariant remained in
the leading group (0.47 GB1, 0.55 TEV, 0.55 TrpB, 0.12 PhoQ), trailing ALDE, CLADE and
MULTI-evolve by small margins. Aggregating ranks across the four landscapes (Fig. 2e,
left), AlphaVariant placed third by mean rank, behind ALDE and MULTI-evolve. In this
regime, in which exhaustive labels favor supervised surrogates, AlphaVariant performed
comparably to the best methods without landscape-specific tuning.

## Large multi-site landscapes under a learned oracle

The multi-site landscapes test optimization over sequence spaces too large to
enumerate. We replaced the lookup table with a convolutional **oracle** trained on all
measured variants of each protein (Methods); the oracles were accurate on held-out
data (test Spearman ρ = 0.89 (AAV), 0.98 (CreiLOV), 0.86 (GFP), 0.90 (PAB1); n_test =
3,652–16,542; Extended Data Fig. 3a), and every queried sequence — including sequences
distant from the measured set — was scored by the oracle.

In this regime AlphaVariant achieved the highest median maximum fitness on three of
four landscapes — AAV (0.71), CreiLOV (0.99) and PAB1 (0.58) — and ranked second on
GFP (0.93 versus 0.95 for AiCE; Fig. 2d). On top-128 mean fitness (Extended Data
Fig. 3b), AlphaVariant was highest on AAV (0.65) and PAB1 (0.50) and close to the
leaders on CreiLOV (0.90 versus 0.94 for AdaLead) and GFP (0.80 versus 0.85 for AiCE).
Two-sided Wilcoxon signed-rank tests on paired per-seed values, Bonferroni-corrected
for multiple comparisons (α = 0.05; Extended Data Fig. 3c), supported these
differences: on AAV, AlphaVariant exceeded all nine baselines on both metrics (9/9
comparisons significant), and on PAB1 it was significant against 8/9 baselines for
maximum fitness and 7/9 for top-128 mean fitness. Across the four multi-site landscapes
AlphaVariant had the lowest mean rank (1.25; Fig. 2e, right), ahead of GreedyWalk,
AdaLead and AiCE.

Considered together, the two regimes indicate that AlphaVariant's generative prior and
policy-gradient optimization performed on par with the strongest supervised methods on
densely labeled four-site libraries and ranked first among the ten methods on the
larger multi-site landscapes, using a single configuration without per-landscape
tuning.
