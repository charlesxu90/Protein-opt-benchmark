# Methods (skeleton draft for *Nature Methods* submission)

> This file is auto-checkable scaffolding. Each `<!-- TODO: ... -->` marker indicates
> a number, citation, or claim that must be filled before submission.

## Computational benchmark

We evaluate AlphaVariant against state-of-the-art directed-evolution methods on
three benchmark tasks corresponding to the canonical use cases for *in silico*
campaigns: combinatorial epistasis navigation, sample-efficient optimization
across diverse landscapes, and multi-objective Pareto discovery.

### Datasets

We use deep mutational scanning (DMS) datasets as ground-truth oracles. All
datasets are publicly available; download scripts and SHA-256 checksums are
distributed with our code (`scripts/prepare_proteingym.py`,
`scripts/prepare_combingym.py`, `data/CHECKSUMS.txt`).

**Combinatorial landscapes (CombinGym v1; ref. <!-- TODO -->):** GB1 (4 sites,
149,361 variants), CR9114 H1 antibody (16 sites, 48,841), CreiLOV (15 sites,
165,428), eqFP611 (5 sites, 8,192; dual blue/red fluorescence). PhoQ from the
original CombinGym plan was not available in v1 and is omitted.

**Variable-length DMS landscapes (ProteinGym v1.3; ref. <!-- TODO Notin -->):**
TEM-1 β-lactamase (BLAT_ECOLX_Stiffler_2015, 286 aa), avGFP (GFP_AEQVI), DHFR
(DYR_ECOLI), Calmodulin (CALM1_HUMAN), MAPK1 (MK01_HUMAN), HIS3 (HIS7_YEAST),
PAB1 (PABP_YEAST), Hsp90 (HSP82_YEAST), HCV NS5A (POLG_HCVJF), PSD95 PDZ3
(DLG4_HUMAN), TPK1, UBE4B, HIV-1 Env (Q2N0S5_9HIV1).

**Internal datasets:** GFP-medium and GFP-hard, AAV-medium and AAV-hard from
the LatProtRL setup (ref. <!-- TODO -->).

### Tasks

**Task 1 — Combinatorial epistasis.** Following CombinGym's hierarchical-split
protocol, we expose to each method the variants of mutation order ≤ K
(K ∈ {1, 2, 3}) and measure ability to recover top-fitness variants of order
> K under a fixed query budget of 480 (5 rounds × 96 variants).

**Task 2 — Sample-efficient directed evolution.** On each ProteinGym landscape,
we measure the fraction of nominated variants exceeding the top-10% fitness
threshold ("hit rate") at two budget regimes: 10 rounds × 16 variants (matching
EVOLVEpro; ref. <!-- TODO -->) and 5 rounds × 96 variants (matching ALDE;
ref. <!-- TODO -->).

**Task 3 — Multi-objective Pareto discovery.** On eqFP611 (blue/red
fluorescence) and synthetic TEM-1+ProteinMPNN combinations, we evaluate the
hypervolume indicator and Pareto-front coverage of the discovered set against
the true Pareto front computed from the full landscape. We compare weighted-sum
scalarization across α ∈ {0, 0.25, 0.5, 0.75, 1} against true bi-objective
acquisition.

### Comparison methods

We compare AlphaVariant against:

- **Random sampling** and **greedy hill-climbing** as naive lower/upper baselines.
- **CLADE** (ref. <!-- TODO Qiu 2021 -->), **ftMLDE** (ref. <!-- TODO Wittmann 2021 -->),
  representative MLDE methods.
- **ALDE** (ref. <!-- TODO Yang 2025 -->), Bayesian active learning.
- **EVOLVEpro** (ref. <!-- TODO Jiang 2025 -->), PLM + few-shot active learning.
- **MULTI-evolve** (ref. <!-- TODO Tran 2026 -->), PLM ensemble + epistasis modelling.
- **AdaLead** (ref. <!-- TODO Sinai 2020 -->) as an adaptive greedy baseline.
- **EvoPlay** (ref. <!-- TODO Wang 2023 -->), self-play RL with MCTS.
- **AiCE** (ref. <!-- TODO Fei 2025 -->), inverse folding + structural constraints.

### Ablations

We isolate the contribution of each AlphaVariant component on the GB1
landscape:
- **AV-NoGPT** — replaces the VariantGPT prior with random single-site
  mutations around the best-discovered variant.
- **AV-NoSpace** — disables dynamic space definition; selection becomes
  uniform random over the full landscape, ignoring clustering and top-k cutoffs.
- **AV-StaticReward** — freezes the surrogate fitness model after round 0.
- **AV-NoRL** — removes REINFORCE-based policy update; samples from the prior
  and selects greedy top-k by surrogate score.

### Evaluation metrics

Following PDFBench (ref. <!-- TODO Kuang 2025 -->) and the refined plan in our
supplementary, we report:

- **Optimization quality:** Top-k fitness (k=128), area under the optimization
  curve (AUOC), simple regret, hit rate at the top-10% threshold, and global
  maximum recovery rate over <!-- TODO 50 --> seeds.
- **Sequence quality:** Pseudo-perplexity under ESM-2 t33 650M
  (ref. <!-- TODO Lin 2023 -->), batch diversity (mean pairwise Levenshtein),
  and novelty (mean Levenshtein from initial training set).
- **Multi-objective:** Hypervolume indicator and Pareto-front coverage against
  the full-landscape Pareto front.

### Statistical analysis

We compute pairwise Wilcoxon signed-rank tests across <!-- TODO 50 --> seeds
per (method, dataset) pair, applying Bonferroni correction for the
<!-- TODO --> pairwise comparisons per dataset. Significance is reported at
α = 0.05 (corrected); see `scripts/generate_tables.py --bonferroni`.

### Computational protocol

All runs were executed on KAUST iBex (NVIDIA <!-- TODO --> GPUs) and Shaheen III
(<!-- TODO partition -->). Per-method walltime, memory, and GPU-hours are
reported in Supplementary Table <!-- TODO -->; raw values are aggregated by
`scripts/generate_tables.py --include_resources` from the
`scripts/hpc/log_resource_use.py` records.

### Code and data availability

All benchmark code, oracle wrappers, baseline integrations, and metric
implementations are released at <!-- TODO repo URL --> under the
<!-- TODO MIT/Apache --> license. DMS landscapes are downloaded from
their original sources via the included scripts; verified SHA-256 checksums
are listed in `data/CHECKSUMS.txt`.
