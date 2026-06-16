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

**Table 1 | Four-site benchmark datasets.**

| Dataset | Protein (functional class) | Mutated sites | Combinatorial space | Measured variants | Variants ≥ 0.5 fitness |
|---------|----------------------------|:-------------:|:-------------------:|------------------:|:----------------------:|
| GB1  | IgG-binding protein G B1 domain (binding) | 4 | 20⁴ = 160,000 | 149,361 | 0.13% |
| PhoQ | PhoQ histidine kinase (signaling)         | 4 | 20⁴ = 160,000 | 140,517 | 0.007% |
| TEV  | Tobacco etch virus protease (enzyme)      | 4 | 20⁴ = 160,000 | 159,132 | 0.05% |
| TrpB | Tryptophan synthase β-subunit (enzyme)    | 4 | 20⁴ = 160,000 | 159,129 | 0.35% |

GB1 mutates positions 39/40/41/54; all four landscapes vary four designed positions.
"Variants ≥ 0.5 fitness" is the fraction of the library with normalized fitness ≥ 0.5
(Extended Data Fig. 2a). References for each dataset to be added (ref.).

Four **multi-site** landscapes were used with a learned oracle: AAV (capsid VP1
28-residue diversification region (ref.)), CreiLOV (LOV-domain fluorescent protein,
119 aa, ~15 variable positions (ref.)), avGFP (237 aa, up to 233 variable positions
(ref.)) and PAB1 (RNA-recognition motif, 75 aa (ref.)). These span combinatorial
spaces of approximately 10³⁶ (AAV) to >10¹⁸⁰ (GFP) sequences. A lookup table returns
zero for any unmeasured sequence, which collapses guided search; we therefore scored
candidates with a trained oracle.

**Table 2 | Multi-site benchmark datasets.**

| Dataset | Protein (functional class) | Length (aa) | Variable positions | Measured variants | Oracle ρ | Oracle R² | n_test |
|---------|----------------------------|:-----------:|:------------------:|------------------:|:--------:|:--------:|-------:|
| AAV     | Adeno-associated virus capsid VP1 (viral assembly) | 28  | 28        | 44,128  | 0.89 | 0.78 | 4,412  |
| CreiLOV | CreiLOV LOV-domain flavoprotein (fluorescent)      | 119 | ~15       | 165,428 | 0.98 | 0.95 | 16,542 |
| GFP     | avGFP (fluorescent)                                | 237 | up to 233 | 51,715  | 0.86 | 0.80 | 5,171  |
| PAB1    | Poly(A)-binding protein RRM (RNA-binding)          | 75  | 74        | 36,522  | 0.90 | 0.78 | 3,652  |

AAV diversifies a 28-residue capsid region. Oracle ρ (Spearman) and R² are reported on
held-out test variants (n_test); the oracle is a GGS-style CNN trained on all measured
variants (see "Learned oracle for multi-site landscapes"). References for each dataset
to be added (ref.).

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

**Table 3 | Benchmarked methods.**

| Method | Family | Model / scoring | Candidate selection | Reference |
|--------|--------|-----------------|---------------------|-----------|
| Random       | Basic search                | —                                                              | Uniform random sampling each round                  | this benchmark |
| GreedyWalk   | Basic search                | Direct fitness/oracle evaluation                               | Hill-climbing over single-mutation neighbors of the best variant | (ref.) |
| ftMLDE       | Supervised / active learning | Cross-validated ensemble of pretrained embedding predictors    | Highest-predicted batch                             | (ref.) |
| ALDE         | Supervised / active learning | Ensemble regressor with uncertainty                            | Thompson sampling                                   | (ref.) |
| CLADE        | Supervised / active learning | Regressor ensemble                                             | Cluster-stratified (best-per-cluster) selection     | (ref.) |
| EVOLVEpro    | Supervised / active learning | ESM-2 embeddings → random-forest head (facebook/esm2_t12_35M_UR50D) | Highest-predicted batch                        | (ref.) |
| MULTI-evolve | Supervised / active learning | Bootstrap neural-network ensemble                              | Highest-predicted batch                             | (ref.) |
| AiCE         | Structure-guided            | ProteinMPNN structure-conditioned per-position amino-acid frequencies (1,000 samples, T = 0.5), blended with observed high-fitness frequencies | Highest summed log-frequency | (ref.) |
| AdaLead      | Generative / policy search  | Ensemble surrogate-guided                                      | Evolutionary recombination + mutation               | (ref.) |
| **AlphaVariant** | Generative / policy search | GPT sequence prior + REINFORCE; five-model surrogate-ensemble UCB reward, with MutCompute reward shaping and SHAP alphabet pruning | Policy-gradient sampling from the trained generator | this work |

All methods operated under the same 480-query budget (96 × 5 rounds), 30 seeds, and
metric pipeline (see above). On multi-site landscapes, library-enumerating methods
(ALDE, CLADE, ftMLDE) used a mutate-from-elites candidate generator in place of a fixed
enumerated library, retaining their own surrogate and selection rule.

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
    --features ev_onehot --use_mutcompute --shap_prune_alphabet \
    --max_n_mut 2 --n_rounds 5 --n_steps_per_round 500 --device cuda:0 --data_dir ../data
```

Shared hyperparameters were batch size 96, five rounds, 500 REINFORCE steps per round
and σ = 60. Multi-site priors were trained per landscape from aligned homologous
sequences (`scripts/alphavariant/train_ms_prior.py`); the full configuration is
documented in the repository. The complete final configuration of each campaign,
including settings left at their defaults, is given in Table 4.

**Table 4 | AlphaVariant final configuration.** Values in parentheses are the
corresponding command-line flags; "default" denotes a setting left at its program
default.

| Setting | Four-site | Multi-site |
|---------|-----------|------------|
| Fitness landscape | Exact lookup table | Learned CNN oracle (`--oracle`) |
| Sequence prior | In-run GPT prior, no homolog pretraining (`--prior_model_path` unset) | GPT prior pretrained on aligned homologs (`--prior_model_path priors/<ds>/prior_model.pt`) |
| Generator | 4-layer GPT, 4 heads, embedding dim 128 | same |
| RL objective | Augmented-likelihood REINFORCE | same |
| Reward scale σ | 60 (`--sigma 60`, default) | 60 |
| Rounds × batch (budget) | 5 × 96 = 480 (`--n_rounds 5`, default) | 5 × 96 = 480 |
| REINFORCE steps per round | 500 (`--n_steps_per_round 500`, default) | 500 |
| GPT ensemble members | 1 (`--n_gpt_ensemble 1`, default) | 1 |
| Surrogate model | 5-model ensemble: 2× ridge, Bayesian ridge, random forest, gradient boosting (`--surrogate ensemble`, default) | same |
| Surrogate features | aa+one-hot: per-position one-hot + 4 physicochemical descriptors (volume, hydropathy, area, polarity) (`--features onehot`, default) | aa+one-hot + standardized EVmutation/plmc statistical-energy column (`--features ev_onehot`) |
| Reward R(*s*) | z(UCB) + λ·z(MutCompute), λ = 0.5 decayed linearly to 0 (`--plm_reward_lambda 0.5`, `--plm_reward_decay linear`) | UCB = μ + 2ς only (no reward shaping; `--plm_reward_lambda 0`, default) |
| Zero-shot signal | MutCompute (structure-based), in the REINFORCE reward (`--use_mutcompute`; `--zeroshot_blend 0`, default) | EVmutation/plmc (evolutionary Potts), as a surrogate feature column (`--features ev_onehot`); MutCompute scorer flag is set but inert (no consumer enabled) |
| SHAP alphabet pruning | On (`--shap_prune_alphabet`); min alphabet 3, SHAP threshold 0, ≥50 samples, top-10 retained | On (same settings) |
| Proposal alphabet constraint | Off — the SHAP-pruned alphabet is computed but not enforced; neither generation nor proposals are constrained (gated to oracle mode) | On — the pruned alphabet is propagated into the generator (sampling restricted to the pruned subspace) and proposals violating it are filtered before selection |
| Round-1 initialization | Cluster-based (k-means on features), 10 clusters, uniform difficulty (`--sampling cluster`, `--level uniform`, defaults) | same (`--level uniform`) |
| Per-round selection (rounds 2–5) | CLADE-2 cluster sampling of the GPT proposal pool, ranked by surrogate-predicted fitness; top-1,000 cutoff before clustering (`--sampling cluster`, `--top_k_cutoff 1000`, `--n_clusters 10`, defaults) | same |
| Mutation cap | None (`--max_n_mut` unset) | ≤ 2 mutations from the reference (`--max_n_mut 2`) |

The two campaigns share the same GPT generator, RL objective, surrogate ensemble, budget
and σ, and differ in five respects: (i) the landscape source (lookup table versus learned
oracle); (ii) the sequence prior (in-run versus homolog-pretrained); (iii) the surrogate
features (aa+one-hot — per-position one-hot plus four physicochemical descriptors — for
four-site, versus the same aa+one-hot augmented with an EVmutation statistical-energy
column for multi-site);
(iv) the reward (four-site blends a decaying MutCompute term into the reward, multi-site
uses the UCB reward alone); and (v) generation constraints (multi-site caps proposals at
≤ 2 mutations from the reference and filters them on the SHAP-pruned alphabet, whereas
four-site applies neither constraint).

The per-round optimization loop was identical in structure across regimes. Each campaign
began with a single cluster-based initialization round: k-means on the (aa+one-hot)
features selected 96 diverse starting sequences, with no surrogate, reward or zero-shot
score applied. In each of the four subsequent rounds AlphaVariant (i) re-fit the
five-model surrogate on all sequences collected so far (96 growing to 480); (ii) trained
the GPT generator for 500 REINFORCE steps against the reward R(*s*); (iii) sampled a
candidate pool from the trained generator; and (iv) selected the next 96 sequences from
that pool by CLADE-2 cluster sampling — retaining the top-1,000 candidates by
surrogate-predicted fitness, clustering them, and taking the best-per-cluster to balance
predicted fitness against batch diversity. Two points follow from this structure. First,
the evaluated batch is always prioritized by the **surrogate's predicted fitness** (the
cluster-stratified CLADE-2 selection), in both regimes; the per-round selection metric is
the same. Second, the MutCompute zero-shot score (four-site only) acts at step (ii),
shaping the *generator's* reward and thereby the distribution of proposals — it does not
score the round-1 initialization or the batch-selection step, both of which are identical
across regimes. On multi-site, generation at step (iii) was additionally restricted to
≤ 2 mutations from the reference and to the SHAP-pruned per-position alphabet, whereas on
four-site the proposal pool was unconstrained.

## AlphaVariant optimization objective

In each round the agent (a GPT policy π_θ) sampled batches of candidate sequences,
which were scored and used to update π_θ by an augmented-likelihood policy-gradient
objective adapted from REINVENT (ref.). For a sampled sequence *s* with tokens
*s*₁…*s*_L, the agent and frozen-prior log-likelihoods are the summed token log-
probabilities,

  log π_θ(*s*) = Σ_t log π_θ(*s*_t | *s*_<t),  log π₀(*s*) = Σ_t log π₀(*s*_t | *s*_<t),

both of which are negative. An augmented (target) likelihood anchors the policy to the
prior while displacing it toward high-reward sequences,

  A(*s*) = log π₀(*s*) + σ · R(*s*),

and the per-batch loss over N sampled sequences is

  L(θ) = (1/N) Σ_s [ A(*s*) − log π_θ(*s*) ]²  −  β · (1/N) Σ_s 1 / log π_θ(*s*),

with reward scale σ = 60 and regularization coefficient β = 5 × 10³. The first
(squared-deviation) term is the REINVENT augmented-likelihood loss: it pulls the
agent's log-likelihood toward the reward-shifted prior, increasing the probability of
high-reward sequences without allowing π_θ to drift far from the pretrained prior. The
second term is an exploration regularizer; because log π_θ(*s*) is negative, the
penalty diverges as the agent log-likelihood approaches zero, discouraging the policy
from collapsing onto a near-deterministic, over-confident distribution and preserving
sequence diversity across rounds.

The reward R(*s*) is the surrogate-ensemble upper confidence bound,
R(*s*) = μ(*s*) + 2·ς(*s*), where μ and ς are the mean and standard deviation of the
predictions of the five-model surrogate ensemble (two ridge regressors, Bayesian ridge,
random forest and gradient boosting), so that R rewards both predicted fitness and
ensemble uncertainty. On the four-site campaigns, R was replaced by an amplitude-matched
blend of z-scored UCB and a z-scored structure-based zero-shot score (MutCompute),
R(*s*) = z(UCB(*s*)) + λ · z(MutCompute(*s*)), with λ initialized at 0.5 and decayed
linearly to 0 over the five rounds; the multi-site campaigns used the UCB reward alone.

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
