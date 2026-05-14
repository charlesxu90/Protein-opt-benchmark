# Benchmark Methods — Reference

> Comprehensive description of every comparator (and AlphaVariant) in this
> benchmark suite. Read this when you need to decide which methods to run on
> a new dataset, write the manuscript Methods section, or understand a
> specific comparator's failure mode.

The benchmark integrates **13 methods** spanning naive baselines, classical
MLDE, Bayesian active learning, adaptive search, reinforcement learning,
PLM-guided active learning, and structure-conditioned scoring. Each method
gets a single column in our headline comparison table; together they cover
the methodological diversity expected by *Nature Methods* reviewers.

## Table of contents

- [1. Methodological classification](#1-methodological-classification)
- [2. Per-method descriptions](#2-per-method-descriptions)
  - [Random](#21-random)
  - [GreedyWalk](#22-greedywalk)
  - [ALDE](#23-alde)
  - [FLEXS / AdaLead](#24-flexs--adalead)
  - [AiCE](#25-aice)
  - [EvoPlay](#26-evoplay)
  - [delta_cs](#27-delta_cs)
  - [LatProtRL](#28-latprotrl)
  - [AlphaVariant](#29-alphavariant)
  - [ftMLDE](#210-ftmlde)
  - [CLADE 2.0](#211-clade-20)
  - [EVOLVEpro](#212-evolvepro)
  - [MULTI-evolve](#213-multi-evolve)
  - [µProtein](#214-µprotein)
- [3. Computational cost summary](#3-computational-cost-summary)
- [4. Integration status in our benchmark](#4-integration-status-in-our-benchmark)
- [5. References](#5-references)

---

## 1. Methodological classification

Each method is tagged on four axes: PLM use, RL use, structure / inverse
folding use, and core algorithm family.

| # | Method | PLM | RL | Structure / IF | Family |
|---|---|:-:|:-:|:-:|---|
| 1 | Random | — | — | — | Naive sampling |
| 2 | GreedyWalk | — | — | — | Hill climbing |
| 3 | ALDE | — | — | — | Bayesian active learning |
| 4 | FLEXS / AdaLead | — | — | — | Adaptive greedy |
| 5 | AiCE | ESM | — | **ProteinMPNN** | Inverse folding + evol |
| 6 | EvoPlay | — | **MCTS self-play** | — | RL |
| 7 | delta_cs | — | **GFlowNet** | — | RL (off-policy) |
| 8 | LatProtRL | ESM-2 | **PPO** | — | PLM + RL |
| 9 | AlphaVariant | **VariantGPT** | **REINFORCE** | — | Generative + RL |
| 10 | ftMLDE | optional (PLM zero-shot init) | — | — | Supervised MLDE |
| 11 | CLADE 2.0 | optional (PLM zero-shot init) | — | — | Cluster-based MLDE |
| 12 | EVOLVEpro | ESM-2 (frozen embeddings) | — | — | Few-shot AL |
| 13 | MULTI-evolve | optional | — | — | Supervised + epistasis |
| 14 | µProtein | µFormer (custom) | **µSearch RL** | — | PLM + RL |

**Methodological clusters**

- Naive baselines: Random, GreedyWalk
- Supervised / Bayesian active learning, no PLM, no RL: ALDE, FLEXS, ftMLDE, CLADE
- PLM-zero-shot guided (PLM only at initialization or scoring): ftMLDE-PLM,
  CLADE-PLM, MULTI-evolve, EVOLVEpro
- Pure RL (no PLM): EvoPlay, delta_cs
- PLM + RL combined: LatProtRL, AlphaVariant, µProtein — directly comparable
  to AlphaVariant's methodological contribution
- Structure-conditioned (inverse folding): AiCE — uniquely uses ProteinMPNN

---

## 2. Per-method descriptions

Each entry has: short description, key reference, algorithm sketch, role in
our benchmark, and notes / known gotchas.

### 2.1 Random

**One-line:** Uniform random sampling without replacement.

**Reference:** Sanity baseline — no method paper.

**Algorithm:**
1. Draw `batch_size × n_rounds` indices uniformly from the landscape without
   replacement.
2. Compute metrics on the queried fitness values.

**Role in benchmark:** The minimum-effort baseline. Any active-learning
method that does not beat Random on a given dataset is providing no value.
Random's performance also serves as a "dataset difficulty" proxy: when
Random saturates (max_fitness > 0.97 normalized), the benchmark has no
headroom and is unsuitable for method comparison.

**Notes:** No GPU, no model, ~3 seconds per seed.

---

### 2.2 GreedyWalk

**One-line:** Hill climbing on the discovered top variant.

**Reference:** Classical directed-evolution heuristic. (Arnold 2018, *Angew.
Chem.* for the underlying biological motivation.)

**Algorithm:**
1. Random batch of 96 variants in round 0.
2. Each subsequent round: find the best-fitness variant so far; enumerate
   all single-mutant neighbors that exist in the landscape; query the top
   `batch_size` by fitness; if fewer single-mutants exist, fill with random
   unexplored variants.

**Role in benchmark:** Tests whether a method beats the simplest local
adaptive walk. On rugged epistatic landscapes (GB1), GreedyWalk gets
trapped at local optima.

**Notes:** No GPU, no learned model. Per-seed wall ~3 seconds.

---

### 2.3 ALDE

**One-line:** Active learning–assisted directed evolution with a DNN
ensemble and Thompson Sampling acquisition.

**Reference:** Yang, Lal, Bowden et al. (2025). *Nat. Commun.* 16, 714.
"Active learning-assisted directed evolution."
[`https://github.com/jsunn-y/ALDE`](https://github.com/jsunn-y/ALDE)

**Algorithm:**
1. Round 0: random batch of 96.
2. Each round 1..N:
   - Train an ensemble of DNN regressors (default 5 networks, one-hot input)
     on accumulated `(seq, fitness)` data.
   - Use Thompson Sampling on the ensemble's posterior to acquire the next
     batch of 96 variants.
3. Report metrics over all queried variants.

**Role in benchmark:** Standard low-N Bayesian-AL comparator. One of the
strongest methods on combinatorial landscapes; consistently top-tier on
GB1 / CR9114 / CreiLOV.

**Notes:** GPU recommended for DNN ensemble training. Per-seed wall ~12 s
(GB1) — Tier 2 in our compute classification.

---

### 2.4 FLEXS / AdaLead

**One-line:** Adaptive greedy sequence search with a CNN ensemble surrogate.

**Reference:** Sinai, Wang, Whatley et al. (2020). *arXiv* 2010.02141.
"AdaLead: A simple and robust adaptive greedy search algorithm for sequence
design."
[`https://github.com/samsinai/FLEXS`](https://github.com/samsinai/FLEXS)

**Algorithm:**
1. Train a CNN ensemble on observed `(seq, fitness)` pairs.
2. Propose candidate variants by mutating the top-k known sequences
   (k-recombination scheme).
3. Filter candidates by a predicted-fitness threshold (κ-quantile of the
   current pool).
4. Query oracle for the surviving batch.

**Role in benchmark:** Primary adaptive-search comparator. Robust and simple
relative to RL methods; performs well across diverse landscapes.

**Notes:** Note that the FLEXS *library* contains 10 explorers; in this
benchmark "FLEXS" refers specifically to the AdaLead explorer. GPU
recommended (~2 GB). Per-seed wall ~10-95 s depending on landscape size.

---

### 2.5 AiCE

**One-line:** Inverse-folding scoring with ProteinMPNN, optionally combined
with evolutionary (MSA) priors.

**Reference:** Fei, Li, Liu et al. (2025). *Cell* 188.
"Advancing protein evolution with inverse folding models integrating
structural and evolutionary constraints."

**Algorithm:**
1. For each candidate variant, score with ProteinMPNN conditioned on the
   wild-type backbone structure (yields `P(seq | structure)`).
2. Optionally combine with MSA-derived evolutionary scores.
3. Rank-select top variants iteratively; light per-round model update.

**Role in benchmark:** **The only structure-aware method.** Acts as a
diagnostic for whether structural information adds value on a given
landscape. Mostly zero-shot in nature, so reports very low variance across
seeds (essentially deterministic).

**Notes:** Requires a backbone PDB for the wild-type. GPU needed (~6 GB,
ESM + ProteinMPNN loaded). Per-seed wall ~4-15 s — Tier 2.

**Fairness caveat:** Has access to structural information that none of the
other methods get. Place in Extended Data unless AlphaVariant gains a
structural reward term.

---

### 2.6 EvoPlay

**One-line:** Self-play reinforcement learning with MCTS over sequence-edit
actions.

**Reference:** Wang, Tang, Huang et al. (2023). *Nat. Mach. Intell.* 5,
845–860. "Self-play reinforcement learning guides protein engineering."
[`https://github.com/MuFeng-MGI/EvoPlay`](https://github.com/MuFeng-MGI/EvoPlay)

**Algorithm:**
1. Define a sequence-edit MDP: state = current sequence, action = single
   amino-acid mutation, reward = predicted fitness change.
2. Train a policy/value network via self-play with MCTS at decision time.
3. Use the policy to propose batches; query oracle and update model.

**Role in benchmark:** Primary RL comparator. Direct methodological neighbor
to AlphaVariant (both use RL over sequences). Provides headline RL-baseline
performance.

**Notes:** **Defaults to CPU.** Must pass `--use_gpu` via
`--extra-args="--use_gpu"` (use `=` syntax — argparse rejects bare flag).
Per-seed wall is ~8 min on A100 with `--use_gpu`; **~80 min on CPU** —
distinguish carefully when reporting compute.

---

### 2.7 delta_cs

**One-line:** Off-policy GFlowNet-based search with conservative exploration
near reliable data regions.

**Reference:** Anonymous (2024). *arXiv* 2410.04461.
"Improved Off-policy Reinforcement Learning in Biological Sequence Design."

**Algorithm:**
1. Train a GFlowNet (flow-matching policy) on observed sequence-fitness
   data; the GFN samples sequences with probability proportional to
   predicted fitness.
2. δ-conservative search: restrict GFN samples to a δ-radius of high-
   confidence training data to avoid hallucinating in unreliable regions.
3. Iteratively query oracle for newly sampled candidates and update GFN.

**Role in benchmark:** Extended Data tier — represents the GFlowNet /
off-policy RL paradigm. Strong batch-quality (best norm-Top128 on GB1) but
weaker peak-finding than ALDE.

**Notes:** GFlowNet pretraining is the dominant cost — ~3000-5000
iterations. Per-seed wall ~30-60 min depending on sequence length. GPU
needed (~4 GB). Save path requires output dir to exist (bug fixed via
`os.makedirs(output_path, exist_ok=True)`).

---

### 2.8 LatProtRL

**One-line:** PPO over an ESM-2 latent space, with a VED (variational
encoder-decoder) for sequence reconstruction.

**Reference:** Anonymous internal LatProtRL implementation; uses
[`fair-esm`](https://github.com/facebookresearch/esm).

**Algorithm:**
1. Encode the protein into an ESM-2 latent representation.
2. Train a VED that maps latent vectors back to sequences.
3. Run PPO in the latent space, with the VED-decoded sequence's predicted
   fitness as reward.

**Role in benchmark:** Originally planned as a PLM+RL comparator.
**Currently broken on default config** — the RL agent never converges
without a pretrained VED (`No trained VED found, will use ESM-2 fallback`).
Excluded from the publication-ready comparison.

**Notes:** GPU needed (~15 GB — full ESM-2 + PPO buffers). To restore,
train the VED via `scripts/LatProtRL/train_GB1_VED.py` per dataset, then
re-run.

---

### 2.9 AlphaVariant

**One-line:** Generative pretrained transformer (VariantGPT) over protein
sequences combined with RL fine-tuning (REINFORCE) and dynamic mutational
space definition.

**Reference:** This work.

**Algorithm:**
1. **Prior**: VariantGPT — a GPT-style sequence model trained on observed
   variants (initialized fresh per round on accumulated data).
2. **Reward model**: an iterative low-N fitness regressor on ESM-2 features
   that updates each round.
3. **RL fine-tuning**: REINFORCE updates VariantGPT to bias sampling toward
   high-reward variants.
4. **Space definition**: dynamic top-k cutoff + CLADE-2-style cluster
   sampling restricts the search space per round.

**Role in benchmark:** The proposed method. Four ablation variants
(`--ablation no-gpt | no-space | static-reward | no-rl`) isolate each
component's contribution. Implemented for GB1; other datasets accept the
flag but raise `NotImplementedError` until per-dataset trainers are
refactored.

**Notes:** GPU needed (~6 GB for GPT + ESM-2). Per-seed wall ~6-10 min on
A100. Single-seed pilot on GB1 hit the global maximum
(max_fitness = 1.0000). 30-seed median on GB1: 0.848.

---

### 2.10 ftMLDE

**One-line:** Focused-training machine-learning-assisted directed evolution
— a supervised regression ensemble trained iteratively on accumulated data.

**Reference:** Wittmann, Yue, Arnold (2021). *Cell Systems* 12, 1026–1045.
"Informed training set design enables efficient machine learning-assisted
directed protein evolution."
[`https://github.com/fhalab/MLDE`](https://github.com/fhalab/MLDE)

**Algorithm:**
1. Round 0: random or zero-shot-focused initial sample of N=96 variants.
   The "focused training" innovation is using PLM zero-shot scores
   (EVE / ESM-1v / DeepSequence) to bias round 0 toward predicted
   high-fitness regions.
2. Each round 1..K:
   - One-hot encode collected `(seq, fitness)` pairs.
   - Train a 22-model ensemble (5 Keras NN + 4 XGB + 13 sklearn).
   - Average top-3 predictions on the unqueried candidate pool.
   - Pick the top `batch_size` variants by predicted fitness.

**Role in benchmark:** Primary "classical MLDE" comparator. Represents the
supervised regression paradigm.

**Notes:** Original repo uses TF 1.13 / Python 3.7. **Our adapter**
(`scripts/ftMLDE/run_generic.py`) implements the same algorithm using
XGBoost + sklearn (5 models, no Keras CNN, no hyperparameter optimization)
to avoid the TF1 dependency. Single-seed test results:

- 4site_GB1: max_fitness = 0.8346, wall = 16 s
- CreiLOV: max_fitness = 0.9828, wall = 33 s

CPU-bound; GPU optional. Tier 2.

**Fairness caveat:** Our adapter is algorithm-faithful but compute-light
(no CNN, no hyperopt, no zero-shot init). Reported as "ftMLDE (light)" in
the manuscript; the full TF1 version is ~2-6 hours per seed.

---

### 2.11 CLADE 2.0

**One-line:** Cluster learning-assisted directed evolution: hierarchical
k-means clustering of the landscape plus a final MLDE pass.

**Reference:** Qiu, Hu, Wei (2021). *Nat. Comput. Sci.* 1, 809–818.
"Cluster learning-assisted directed evolution."
[`https://github.com/WeilabMSU/CLADE`](https://github.com/WeilabMSU/CLADE)
(CLADE 2.0 README).

**Algorithm:**
1. One-hot encode the entire landscape.
2. KMeans cluster into K=10 clusters (CLADE 2.0 can use zero-shot
   evolutionary scores for the initial clustering).
3. Round 0: sample uniformly across clusters (96 variants total).
4. Each round 1..K_round:
   - Train an MLDE ensemble on collected data.
   - Predict the unqueried pool.
   - Pick top-per-cluster (or sub-cluster the high-fitness region in the
     hierarchy variant).
5. Final round: full ftMLDE pass on accumulated data.

**Role in benchmark:** Primary "cluster-based MLDE" comparator. Has direct
precedent on GB1 (the original paper benchmarked it on the 4-site GB1
library).

**Notes:** **Our adapter** (`scripts/CLADE/run_generic.py`) implements the
"flat" 10-cluster variant (no hierarchy refinement, no zero-shot init,
sklearn-only ensemble). Single-seed test results:

- 4site_GB1: max_fitness = 0.8622, wall = 10 s
- CreiLOV: max_fitness = 0.9828, wall = 23 s

CPU-bound; GPU optional. Tier 2.

**Fairness caveat:** Same as ftMLDE — our adapter is light; full CLADE 2.0
with hierarchy + zero-shot init is ~2-5 hours per seed.

---

### 2.12 EVOLVEpro

**One-line:** Few-shot active learning over frozen ESM-2 embeddings.

**Reference:** Jiang, Yan, Di Bernardo et al. (2025). *Science* 387,
eadr6006. "Rapid in silico directed evolution by a protein language model
with EVOLVEpro."
[`https://github.com/mat10d/EVOLVEpro`](https://github.com/mat10d/EVOLVEpro)

**Algorithm:**
1. **One-time per dataset**: precompute ESM-2 t33 (650M) embeddings for
   every variant in the landscape (~30-120 min on A100 depending on
   sequence length and pool size).
2. Each round: train a small top-layer regressor (RF or shallow NN) on
   accumulated `(embedding, fitness)` pairs.
3. Active learning acquisition: greedy / diverse / Thompson over the
   regressor predictions.

**Role in benchmark:** Primary PLM-active-learning comparator. Particularly
relevant for the CR9114 antibody dataset (the EVOLVEpro paper demonstrates
on antibody DMS data).

**Notes:** Cloned to `EVOLVEpro/` but not yet integrated end-to-end. The
embedding precomputation step is required and dominates the dataset-level
cost; per-seed AL after embeddings is ~1-5 min. Concurrency: 4-6/GPU for
the AL phase.

---

### 2.13 MULTI-evolve

**One-line:** Fully-connected NN ensemble trained on single-mutant data;
optional PLM zero-shot ensemble for variant proposal; epistasis-aware
combinatorial prediction.

**Reference:** Tran, Nemeth, Bartie et al. (2026). *Science.*
"Rapid directed evolution guided by protein language models and epistatic
interactions."
[`https://github.com/ArcInstitute/MULTI-evolve`](https://github.com/ArcInstitute/MULTI-evolve)

**Algorithm:**
1. Train fully-connected NN ensemble on single-mutant `(seq, fitness)`
   data.
2. Choose the best-performing network (or ensemble) and predict the fitness
   of all multi-mutant combinations.
3. Optionally use a PLM zero-shot ensemble to nominate additional
   single-mutants for evaluation.
4. Pick top-k multi-mutants for the next round.

**Role in benchmark:** Extended-data comparator for high-order epistasis
benchmarks (especially CR9114-H1, where MULTI-evolve was demonstrated).

**Notes:** Cloned to `MULTIevolve/` but not yet integrated. Requires WandB
(use `WANDB_MODE=offline` for batch sweeps). Per-seed wall ~10-30 min.

---

### 2.14 µProtein

**One-line:** µFormer (a domain-specific mutational-effect transformer)
plus µSearch (RL search using µFormer as oracle). Microsoft Research's
two-stage PLM + RL system for protein engineering.

**Reference:** Liu et al. (2025). *Nat. Mach. Intell.*
"Accelerating protein engineering with fitness landscape modelling and
reinforcement learning."
[`https://github.com/microsoft/Mu-Protein`](https://github.com/microsoft/Mu-Protein)

**Algorithm:**
1. **µFormer**: a transformer trained on single-mutation DMS data — a
   custom PLM specialized for mutational-effect prediction. Acts as a
   surrogate fitness oracle for combinatorial variants.
2. **µSearch**: an RL policy that explores the sequence space using
   µFormer as the in-the-loop fitness oracle, learning to propose
   multi-point combinations that maximize predicted fitness.
3. The two stages can be trained jointly or sequentially; the published
   pipeline pretrains µFormer per protein family, then fine-tunes
   µSearch online.

**Role in benchmark:** Closest methodological neighbor to AlphaVariant
(both combine a PLM-style sequence model with RL search). Directly
comparable head-to-head.

**Notes:** Repository at `microsoft/Mu-Protein`. Not yet cloned or
integrated in our benchmark. Per-seed wall is heavy (30 min – 2 hours
estimated) since both µFormer pretraining (one-time per dataset) and
µSearch online RL require GPU training. Memory ~4-8 GB for µFormer +
RL buffers; concurrency 2-3/GPU.

To integrate:
```bash
git clone --depth 1 https://github.com/microsoft/Mu-Protein
```
then build the env from the repo's `environment.yml` (Microsoft's
Python 3.10 + PyTorch stack) and write an adapter at
`scripts/uProtein/run_generic.py` matching the established benchmark
schema. The µFormer per-dataset pretraining step is the dominant one-time
cost.

---

## 3. Computational cost summary

Empirical per-seed wall times measured on this workstation (2× NVIDIA
A100-40GB, 112 cores). Values are median of three test runs.

| Method | Per-seed (GB1, 4-site, 56-aa) | Per-seed (CreiLOV, 15-site, 119-aa) | GPU need | Memory | Concurrency |
|---|---|---|---|---|---|
| Random | 3 s | 4 s | ❌ | <0.5 GB | 30+/host |
| GreedyWalk | 3 s | 4 s | ❌ | <0.5 GB | 30+/host |
| AiCE | 4 s | 15 s | ✅ (ESM + ProteinMPNN, ~6 GB) | 6 GB | 2-3/GPU |
| ALDE | 12 s | 75 s | ✅ (DNN ensemble, ~2 GB) | 2 GB | 4-6/GPU |
| FLEXS / AdaLead | 10 s | 13 min | ✅ (CNN ensemble, ~2 GB) | 2 GB | 4/GPU |
| ftMLDE (our adapter) | 16 s | 33 s | optional | 3 GB RAM | 8-16/host |
| CLADE 2.0 (our adapter) | 10 s | 23 s | optional | 3 GB RAM | 8-16/host |
| EVOLVEpro | ~1-5 min after one-time embedding (~30-120 min) | — | ✅ | 3-4 GB | 4-6/GPU |
| EvoPlay (with `--use_gpu`) | 8 min | 30-60 min | ✅ (~2 GB) | 2 GB | 4/GPU |
| EvoPlay (CPU, default) | 80 min | days | ❌ | — | broken |
| alphavariant | 6 min | 10-15 min | ✅ (~6 GB) | 6 GB | 3/GPU |
| delta_cs | 30 min | 30-60 min | ✅ (~4 GB) | 4 GB | 3/GPU |
| MULTI-evolve | ~10-30 min (est) | ~10-30 min (est) | ✅ | 3-4 GB | 3-4/GPU |
| µProtein | ~30-120 min (est) | ~30-120 min (est) | ✅ | 4-8 GB | 2-3/GPU |
| LatProtRL | 44 min (broken) | — | ✅ (~15 GB) | 15 GB | 1-2/GPU |

**Tier classification by per-seed wall:**

- **Tier 1** (sub-second to seconds, CPU): Random, GreedyWalk
- **Tier 2** (seconds, light GPU/CPU): ALDE, AiCE, ftMLDE, CLADE
- **Tier 3** (minutes, medium GPU): FLEXS, alphavariant, EVOLVEpro
- **Tier 4** (30 min+, heavy GPU): EvoPlay (GPU), delta_cs, MULTI-evolve, µProtein
- **Broken without setup**: LatProtRL

**Total wall for a 30-seed × 1-dataset sweep on 2× A100 with optimal
concurrency**: Tier 1 < 1 min; Tier 2 1-10 min; Tier 3 0.5-4 hours; Tier 4
2-12 hours. The whole 13-method sweep takes ~15-20 hours per dataset,
dominated by EvoPlay + delta_cs + alphavariant.

---

## 4. Integration status in our benchmark

| Method | Scaffolding | Env built | End-to-end runnable | Per-dataset wrappers |
|---|---|---|---|---|
| Random | ✅ | reuses ALDE/env | ✅ | all datasets |
| GreedyWalk | ✅ | reuses ALDE/env | ✅ | all datasets |
| ALDE | ✅ | `ALDE/env` | ✅ | all datasets |
| FLEXS / AdaLead | ✅ | `FLEXS/env` | ✅ | all datasets |
| AiCE | ✅ | `AiCE/env` | ✅ | all datasets |
| EvoPlay | ✅ | `EvoPlay/env` | ✅ (must pass `--use_gpu`) | all datasets |
| delta_cs | ✅ | `delta_cs/env/delta_cs_env` | ✅ | all datasets |
| LatProtRL | ✅ | `LatProtRL/env/latprotrl_env` | 🟡 broken without VED pretrain | all datasets |
| alphavariant | ✅ | `/home/xux/miniforge3/envs/alphavariant-env` | ✅ (ablations work on GB1 only) | all datasets |
| **ftMLDE (our adapter)** | ✅ | reuses ALDE/env | ✅ | generic |
| **CLADE 2.0 (our adapter)** | ✅ | reuses ALDE/env | ✅ | generic |
| EVOLVEpro | partial (scaffold exits 2) | not built | ❌ | — |
| MULTI-evolve | partial (scaffold exits 2) | not built | ❌ | — |
| µProtein | not cloned (`microsoft/Mu-Protein`) | not built | ❌ | — |

**Recommended baseline panel** for the 3-dataset benchmark:

Random + GreedyWalk + ALDE + AiCE + FLEXS + ftMLDE + CLADE +
alphavariant + EvoPlay (with `--use_gpu`).

delta_cs as Extended Data; EVOLVEpro / MULTI-evolve / µProtein as
"future-work" comparators or Extended Data if integration completes
before submission.

---

## 5. References

[1] **ALDE** — Yang, J., Lal, R.G., Bowden, J.C. et al. (2025). Active
learning-assisted directed evolution. *Nat. Commun.* 16, 714.
[`https://doi.org/10.1038/s41467-025-55987-8`](https://doi.org/10.1038/s41467-025-55987-8)

[2] **AdaLead / FLEXS** — Sinai, S., Wang, R., Whatley, A. et al. (2020).
AdaLead: A simple and robust adaptive greedy search algorithm for sequence
design. *arXiv* 2010.02141.
[`https://arxiv.org/abs/2010.02141`](https://arxiv.org/abs/2010.02141)

[3] **AiCE** — Fei, H., Li, Y., Liu, Y. et al. (2025). Advancing protein
evolution with inverse folding models integrating structural and
evolutionary constraints. *Cell* 188.
[`https://doi.org/10.1016/j.cell.2025.06.014`](https://doi.org/10.1016/j.cell.2025.06.014)

[4] **EvoPlay** — Wang, Y., Tang, H., Huang, L. et al. (2023). Self-play
reinforcement learning guides protein engineering. *Nat. Mach. Intell.*
5, 845–860.
[`https://doi.org/10.1038/s42256-023-00691-9`](https://doi.org/10.1038/s42256-023-00691-9)

[5] **delta_cs** — Improved Off-policy Reinforcement Learning in
Biological Sequence Design. *arXiv* 2410.04461.
[`https://arxiv.org/abs/2410.04461`](https://arxiv.org/abs/2410.04461)

[6] **MLDE / ftMLDE** — Wittmann, B.J., Yue, Y., Arnold, F.H. (2021).
Informed training set design enables efficient machine learning-assisted
directed protein evolution. *Cell Systems* 12, 1026–1045.
[`https://doi.org/10.1016/j.cels.2021.07.008`](https://doi.org/10.1016/j.cels.2021.07.008)

[7] **CLADE** — Qiu, Y., Hu, J., Wei, G.-W. (2021). Cluster
learning-assisted directed evolution. *Nat. Comput. Sci.* 1, 809–818.
[`https://doi.org/10.1038/s43588-021-00168-y`](https://doi.org/10.1038/s43588-021-00168-y)

[8] **EVOLVEpro** — Jiang, K., Yan, Z., Di Bernardo, M. et al. (2025).
Rapid in silico directed evolution by a protein language model with
EVOLVEpro. *Science* 387, eadr6006.
[`https://doi.org/10.1126/science.adr6006`](https://doi.org/10.1126/science.adr6006)

[9] **MULTI-evolve** — Tran, V.Q., Nemeth, M., Bartie, L.J. et al. (2026).
Rapid directed evolution guided by protein language models and epistatic
interactions. *Science.*
[`https://doi.org/10.1126/science.aea1820`](https://doi.org/10.1126/science.aea1820)

[10] **µProtein** — Liu, X. et al. (2025). Accelerating protein
engineering with fitness landscape modelling and reinforcement learning.
*Nat. Mach. Intell.*
[`https://doi.org/10.1038/s42256-025-01103-w`](https://doi.org/10.1038/s42256-025-01103-w)
Code: [`https://github.com/microsoft/Mu-Protein`](https://github.com/microsoft/Mu-Protein)

[11] **AlphaVariant** — This work.

[12] **CombinGym** (dataset source) — Chen, Y. et al. (2026).
[`https://github.com/sitonglab/CombinGym`](https://github.com/sitonglab/CombinGym)

[13] **ProteinGym** (dataset source) — Notin, P., Kollasch, A., Ritter, D.
et al. (2024). ProteinGym: Large-scale benchmarks for protein fitness
prediction and design. *NeurIPS 2023 Datasets and Benchmarks Track.*
[`https://github.com/OATML-Markslab/ProteinGym`](https://github.com/OATML-Markslab/ProteinGym)

[14] **ProteinMPNN** (structural prior for AiCE) — Sumida, K.H.,
Nuñez-Franco, R., Kalvet, I. et al. (2024). Improving Protein Expression,
Stability, and Function with ProteinMPNN. *JACS.*
[`https://doi.org/10.1021/jacs.3c10941`](https://doi.org/10.1021/jacs.3c10941)
