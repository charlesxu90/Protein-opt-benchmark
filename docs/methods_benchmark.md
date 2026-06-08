# Methods — Benchmark design and evaluation (Nature Methods draft)

> Draft Methods subsection, formatted for Nature Methods. Companion to
> `docs/results_benchmark.md`. `(ref.)` marks a citation to be added.

## Benchmark datasets

We evaluated all methods on eight experimentally measured protein-fitness landscapes
(Fig. 2a). Assay values were min–max normalized to [0, 1] within each landscape, so
the global maximum maps to 1.0 and metrics are comparable across proteins. The
landscapes were grouped into two regimes.

Four **four-site combinatorial** landscapes were used as exact lookup tables: GB1
(IgG-binding B1 domain, positions 39/40/41/54; 149,361 variants (ref.)), PhoQ
(signaling histidine kinase; 140,517 variants (ref.)), TEV (protease; 159,132 variants
(ref.)) and TrpB (tryptophan synthase β-subunit; 159,129 variants (ref.)). Each is a
near-complete 20⁴ library in which every candidate has a measured fitness. The
landscapes are sparse in high-fitness variants: the fraction of sequences with
normalized fitness ≥ 0.5 was 0.13% (GB1), 0.007% (PhoQ), 0.05% (TEV) and 0.35% (TrpB)
(Extended Data Fig. 2a).

Four **multi-site** landscapes were used with a learned oracle: AAV (capsid VP1
28-residue diversification region (ref.)), CreiLOV (LOV-domain fluorescent protein,
119 aa, ~15 variable positions (ref.)), avGFP (237 aa, up to 233 variable positions
(ref.)) and PAB1 (RNA-recognition motif, 75 aa (ref.)). These span combinatorial
spaces of approximately 10³⁶ (AAV) to >10¹⁸⁰ (GFP) sequences. A lookup table returns
zero for any unmeasured sequence, which collapses guided search; we therefore scored
candidates with a trained oracle.

## Learned oracle for multi-site landscapes

For each multi-site landscape we trained a sequence-to-fitness oracle using a
length-agnostic convolutional architecture (1D convolution, 20→256 channels, kernel
size 5; length-wise global max-pooling; linear regression head), matching the
predictor used in GGS (ref.) and LatProtRL (ref.). Oracles were trained on all measured
variants by minimizing mean-squared error (Adam, learning rate 1×10⁻⁴, weight decay
1×10⁻⁴, batch size 1,024, ≤100 epochs; PyTorch 2.1.1) with fitness-weighted sampling
(weight ∝ 1/(target − min + 1)) to emphasize the rare high-fitness tail. The fitted
min/max scaler was stored with each checkpoint and used to de-normalize predictions.
Oracles were accurate on held-out test variants (test Spearman ρ = 0.89/0.98/0.86/0.90
and R² = 0.78/0.95/0.80/0.78 for AAV/CreiLOV/GFP/PAB1; n_test = 4,412/16,542/5,171/3,652;
Extended Data Fig. 3a). During optimization we used a pure-oracle policy: every queried
sequence, including sequences far from the measured set, was scored by the oracle, and
measured labels were used only to train the oracle. To detect oracle-exploitation
artifacts we recorded the mutational novelty of selected sequences relative to the
measured set (Extended Data Fig. 3a, bottom).

## Optimization campaign

All methods ran an identical campaign: an initial random batch of 96 sequences followed
by four model-guided rounds of 96, for 480 queries in total (96 × 5), the budget used
in prior directed-evolution benchmarks (ref.). On four-site landscapes a query was
resolved by exact lookup; on multi-site landscapes it was scored by the trained oracle.
For methods that select from a fixed enumerated library (ALDE, CLADE, ftMLDE), the
multi-site candidate source was replaced by a mutate-from-elites generator so that all
methods operated over the same non-enumerable space while retaining each method's own
surrogate model and acquisition rule.

## Compared methods

We benchmarked nine published methods alongside AlphaVariant, grouped by family
(Fig. 2b). Basic search: Random (uniform sampling each round) and GreedyWalk
(evaluation of single-mutation neighbors of the current best). Supervised and
active-learning methods: ftMLDE (ref.) (cross-validated ensemble of pretrained
predictors), ALDE (ref.) (ensemble regressor with Thompson sampling), CLADE (ref.)
(ensemble with cluster-stratified selection), EVOLVEpro (ref.) (ESM-2 embeddings with a
random-forest head; facebook/esm2_t12_35M_UR50D) and MULTI-evolve (ref.) (bootstrap
neural-network ensemble). Structure-guided design: AiCE (ref.), using ProteinMPNN
structure-conditioned per-position amino-acid frequencies (1,000 samples, T = 0.5)
blended with observed high-fitness frequencies. Generative/policy-based search: AdaLead
(ref.) (recombination with surrogate-guided mutation) and AlphaVariant. Each method was
run in the configuration recommended by its authors and used the unified metric
pipeline.

## AlphaVariant configuration

AlphaVariant was run in a single configuration across all landscapes: a GPT sequence
prior optimized by REINFORCE against a five-model surrogate ensemble, with structure-
based (MutCompute (ref.)) reward shaping and SHAP-based per-position alphabet pruning.
Four-site and multi-site campaigns differed only in the landscape source and the
sequence prior:

```bash
# Four-site (lookup landscape)
python run_generic.py --dataset 4site_PhoQ --seed <S> \
    --use_mutcompute --plm_reward_lambda 0.5 --shap_prune_alphabet

# Multi-site (learned-oracle landscape; GPT prior from aligned homologs)
python run_generic.py --dataset ms_GFP --seed <S> \
    --oracle --level uniform --prior_model_path priors/ms_GFP/prior_model.pt \
    --use_mutcompute --shap_prune_alphabet \
    --n_rounds 5 --n_steps_per_round 500 --device cuda:0 --data_dir ../data
```

Shared hyperparameters were batch size 96, five rounds, 500 REINFORCE steps per round
and σ = 60. Multi-site priors were trained per landscape from aligned homologous
sequences (`scripts/alphavariant/train_ms_prior.py`); the full configuration is
documented in the repository.

## Evaluation metrics

We report two complementary metrics per campaign. **Maximum fitness** is the highest
normalized fitness among the 480 queried sequences, the canonical directed-evolution
objective. **Top-128 mean fitness** is the median fitness of the 128 highest-fitness
discovered sequences, a batch-quality metric (ref.) that rewards recovering many strong
variants. Method rankings (Fig. 2e) were computed per landscape (rank 1 = highest
median; ties assigned the average rank) and averaged within each regime.

## Statistics and reproducibility

Each method × landscape combination was run for n = 30 independent random seeds (the
first 30 entries of a fixed seed list). Distributions across seeds are heavy-tailed and
multimodal; we therefore summarized each combination by its median and interquartile
range (Q1–Q3) rather than the mean (Fig. 2c,d; Extended Data Fig. 2b, 3b). Pairwise
differences between methods were assessed with two-sided Wilcoxon signed-rank tests on
the paired per-seed values, with Bonferroni correction for multiple comparisons at
α = 0.05 (Extended Data Fig. 3c). No data were excluded and seeds were fixed for
reproducibility. Datasets are provided under `data/<name>/data.csv`, oracle checkpoints
under `oracles/<dataset>/oracle.pt`, and per-seed results under each method's result
tree. Comparison tables and figures are regenerated with
`scripts/build_median_iqr_csv.py`, `scripts/build_oracle_median_iqr_csv.py`,
`scripts/draw_figures_median.py` and `scripts/draw_ranking_panel.py`.
