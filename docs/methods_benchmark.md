# Methods — Benchmark design and evaluation

> Draft manuscript subsection for the Methods section.
> Companion to `docs/results_benchmark.md`. Figure callouts: **Fig. 2**,
> **Extended Data Fig. 2** (four-site), **Extended Data Fig. 3** (multi-site).

## Benchmark datasets

We evaluated all methods on eight experimentally measured protein-fitness landscapes
grouped into two regimes (**Fig. 2a**). Raw assay values were min–max normalized to
[0, 1] within each landscape so that metrics are comparable across proteins; the
global maximum of each landscape maps to 1.0.

**Four-site combinatorial landscapes.** GB1 (IgG-binding domain, positions 39/40/41/54;
149,361 variants), PhoQ (signaling histidine kinase, 140,517 variants), TEV
(protease, 159,132 variants), and TrpB (β-subunit of tryptophan synthase, 159,129
variants). Each is a near-complete 20⁴ combinatorial library in which every candidate
sequence has a directly measured fitness, so the landscape is queried as an exact
lookup table. The landscapes are highly sparse in high-fitness variants: the fraction
of sequences with normalized fitness ≥ 0.5 ranges from 0.007% (PhoQ) to 0.35% (TrpB)
(**Extended Data Fig. 2a**).

**Multi-site landscapes.** AAV (capsid VP1, 28-residue insertion region), CreiLOV
(fluorescent flavoprotein, 119 aa with ~15 variable positions), GFP (avGFP, 237 aa
with up to 233 variable positions), and PAB1 (RNA-binding domain, 75 aa). These span
combinatorial spaces of ≈10³⁶ (AAV) to >10¹⁸⁰ (GFP) sequences, which no experimental
library covers. Because a lookup table returns zero fitness for any unmeasured
sequence — collapsing generative search — we instead score candidates with a learned
oracle (below).

## Multi-site learned-oracle protocol

For each multi-site landscape we trained a sequence-to-fitness **oracle** using the
GGS BaseCNN architecture (Conv1d, 20→256 channels, kernel 5 → length-wise max-pool →
linear head), which is length-agnostic and identical to the predictors used in GGS
and LatProtRL. Oracles were trained on **all** measured variants with mean-squared
error (Adam, lr 1e-4, weight decay 1e-4, batch 1024, ≤100 epochs) using GGS weighted
sampling (weight ∝ 1/(target − min + 1)) to emphasize the rare high-fitness tail. The
fitted min/max scaler is stored in the checkpoint and used to de-normalize predictions.
Held-out accuracy was high on all four landscapes (test Spearman ρ = 0.89 AAV,
0.98 CreiLOV, 0.86 GFP, 0.90 PAB1; R² = 0.78/0.95/0.80/0.78; **Extended Data Fig. 3a**).
Under a **pure-oracle** policy, every queried sequence is scored by the CNN — including
sequences far from the training set — while measured labels are used only to train the
oracle, not as a fallback. To control oracle-exploitation artifacts we report the
mutational novelty of selected sequences alongside fitness (**Extended Data Fig. 3a**,
bottom).

## Optimization campaign and budget

All methods ran an identical campaign: a random initial batch of 96 sequences followed
by four model-guided rounds of 96, for a total of **480 queries** (96 × 5 rounds), the
standard budget used throughout the directed-evolution benchmarking literature. On the
four-site landscapes a queried sequence is resolved by exact lookup; on the multi-site
landscapes it is scored by the trained oracle. For methods that enumerate a fixed
candidate library (ALDE, CLADE, ftMLDE), the multi-site candidate source was converted
to a mutate-from-elites generator so that all methods operate over the same
non-enumerable space while retaining each method's own surrogate and selection rule.

## Baseline methods

We compared against nine baselines spanning four families (**Fig. 2b**):

- **Basic search** — *Random* (uniform sampling each round) and *GreedyWalk*
  (oracle/lookup of single-mutation neighbors of the current best).
- **Supervised / active learning** — *ftMLDE* (zoo-ensemble cross-validated top
  models), *ALDE* (DNN/RF ensemble with Thompson sampling), *CLADE* (ensemble with
  cluster-stratified best-per-cluster selection), *EVOLVEpro* (ESM-2 embeddings →
  random-forest top layer), and *MULTI-evolve* (bootstrap FCN-ensemble surrogate).
- **PLM / structure-guided** — *AiCE* (ProteinMPNN structure-conditioned per-position
  amino-acid frequencies blended with observed-top frequencies).
- **Generative / policy-based search** — *AdaLead* (FLEXS; recombination + surrogate-
  guided mutation) and **AlphaVariant** (this work).

## AlphaVariant configuration

AlphaVariant was run in its shipped "Plan C" configuration — a GPT sequence prior
trained by REINFORCE against a five-model surrogate ensemble, with MutCompute
(structure-based zero-shot) reward shaping and SHAP-based per-position alphabet
pruning. The four-site and multi-site campaigns differ only in landscape source and
prior:

```bash
# Four-site (lookup landscape)
python run_generic.py --dataset 4site_PhoQ --seed <S> \
    --use_mutcompute --plm_reward_lambda 0.5 --shap_prune_alphabet

# Multi-site (learned-oracle landscape, GPT prior from aligned homologs)
python run_generic.py --dataset ms_GFP --seed <S> \
    --oracle --level uniform --prior_model_path priors/ms_GFP/prior_model.pt \
    --use_mutcompute --shap_prune_alphabet \
    --n_rounds 5 --n_steps_per_round 500 --device cuda:0 --data_dir ../data
```

Shared settings: batch size 96, 5 rounds, 500 REINFORCE steps/round, σ = 60.
Multi-site priors were trained per landscape from aligned homologous sequences
(`scripts/alphavariant/train_ms_prior.py`). The full configuration is documented in
the repository README.

## Metrics and statistical evaluation

For every campaign we report two complementary metrics. **Maximum fitness** is the
single best normalized fitness discovered across the 480 queries — the canonical
directed-evolution objective. **Top-128 mean fitness** is the median fitness of the
128 best discovered sequences, a batch-quality metric (GGS) that rewards recovering
many strong variants rather than one lucky hit.

Each method × landscape combination was run for **30 random seeds** (the first 30
seeds of the shared `rand_seeds.txt`). We summarize each cell by its **median and
interquartile range (Q1–Q3)** across seeds (**Fig. 2c,d**; **Extended Data Fig. 2b**,
**3b**), which is robust to the heavy-tailed, multimodal seed distributions these
landscapes produce. Method-level rankings (**Fig. 2e**) are computed per landscape
(rank 1 = best median; ties share the average rank) and averaged within each regime.
Pairwise differences between methods were assessed with two-sided Wilcoxon
signed-rank tests on the paired per-seed values, with **Bonferroni correction** for
multiple comparisons at α = 0.05 (**Extended Data Fig. 3c**).

## Reproducibility

All datasets live under `data/<name>/data.csv`; oracle checkpoints under
`oracles/<dataset>/oracle.pt`; per-seed results under the method result trees. The
median/IQR comparison tables and all figures are regenerated with
`scripts/build_median_iqr_csv.py` / `scripts/build_oracle_median_iqr_csv.py`,
`scripts/draw_figures_median.py`, and `scripts/draw_ranking_panel.py`.
