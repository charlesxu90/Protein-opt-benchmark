# Refined Methods — consolidated AlphaVariant description (de-duplicated)

> Paste-ready restructuring of the manuscript Methods section. Goal: state the shared
> AlphaVariant framework **once** at the start of Methods (Section A), then trim the
> Benchmarking "AlphaVariant settings" (Section B) and the savinase round-1 prior/RL
> subsections (Section C) to their application-specific instantiations only. Rounds 2–3
> already cross-reference round 1 and need no change (Section D). The bare+finetune
> four-site corrections are folded in. Equation numbers (3)–(5) refer to the
> augmented-likelihood objective, which moves into Section A.

---

## A. NEW subsection — paste at the start of Methods (before "Benchmark datasets")

### The AlphaVariant framework

AlphaVariant couples a generative protein-sequence prior with reinforcement learning to
propose high-value variants within a defined mutational space. The same framework
underlies both the benchmark evaluation and the savinase campaigns; the components are
described here once, and application-specific instantiations (prior architecture, reward,
and constraints) are given in the corresponding sections below.

**Generative prior.** The prior is an autoregressive GPT model trained by next-token
prediction on protein sequences, pretrained on a homologous family and optionally
fine-tuned on task-specific variants. The agent used in reinforcement learning is
initialized from this prior; the prior remains fixed and serves as the anchor in the RL
objective. The specific architecture and training corpus are given per application
(a 4-layer GPT for the benchmark; the larger VariantGPT for savinase).

**Reinforcement-learning objective.** Variants are optimized with an augmented-likelihood
policy gradient adapted from REINVENT. For a sampled sequence *s* = (s₁,…,s_L), the agent
and prior log-likelihoods are the summed autoregressive token log-probabilities,

  log π_θ(s) = Σ_t log π_θ(s_t | s_<t),  log π₀(s) = Σ_t log π₀(s_t | s_<t).  (Eq. 3)

An augmented (target) likelihood combines the prior with the reward,

  A(s) = log π₀(s) + σ R(s),  (Eq. 4)

and the agent is trained to minimize

  L(θ) = (1/N) Σ_s [A(s) − log π_θ(s)]² − β (1/N) Σ_s 1 / log π_θ(s),  (Eq. 5)

with reward scale σ and regularization coefficient β = 5×10³. The squared-deviation term
raises the likelihood of high-reward sequences while anchoring the agent to the prior; the
second term is an exploration regularizer that discourages premature collapse of the agent
distribution. R(s) is a task-specific reward in a bounded range — a surrogate-ensemble
upper-confidence bound for the benchmarks, and a transformed activity/multi-property score
for the savinase campaigns (defined below).

**Constrained generation and library design.** Generation is restricted to a predefined
mutational space: the agent samples permitted amino acids at the designated mutable
positions while wild-type residues are retained elsewhere. Optimization proceeds over
iterative rounds in which the reward model is refit on accumulated data and the agent is
re-optimized. Variants produced during RL are aggregated and prioritized to construct
experimental (or in-silico) libraries, as detailed per application.

---

## B. REPLACEMENT — trimmed Benchmarking "AlphaVariant settings"

Within the benchmark, AlphaVariant followed the framework above (generator,
augmented-likelihood REINFORCE objective Eq. 3–5, surrogate-derived reward, cluster-based
selection). The generator was a four-layer GPT with four attention heads and embedding
dimension 128, its context length matched to each sequence. The reward R(s) was the
surrogate-ensemble upper-confidence bound, R(s) = μ(s) + 2ζ(s), over a five-model ensemble
(two ridge, one Bayesian ridge, one random-forest, one gradient-boosting regressor);
surrogate features were per-position one-hot amino-acid encodings plus four physicochemical
descriptors (side-chain volume, hydropathy, solvent-accessible area, polarity). All runs
used five rounds, batch size 96, 500 REINFORCE updates per round and σ = 60 (480 queries
per campaign). Each campaign began with one cluster-based initialization round (k-means on
sequence features, 96 sequences across 10 clusters, no surrogate/reward); in rounds 2–5 the
surrogate was refit on all data, the agent optimized, a proposal pool sampled, and the next
96 sequences chosen by cluster-stratified selection (top-1,000 by predicted fitness, then
best-per-cluster).

The two regimes differed only in landscape source and a few components (Extended Data
Table 4). **Four-site:** exact lookup-table fitness; an in-run GPT prior finetuned on the
collected sequences each round (10 epochs, lr 1×10⁻⁴); no reward shaping, alphabet pruning,
or mutation cap. **Multi-site:** learned CNN-oracle fitness; a homolog-pretrained GPT prior
(no per-round finetuning); surrogate features augmented with a standardized EVmutation
statistical-energy score; SHAP-based alphabet pruning (enabled at ≥ 50 samples; SHAP > 0
retained; minimum 3, maximum 10 amino acids per position) propagated into the generator
with violating proposals filtered; and a cap of two mutations from the reference.

*(Removes the re-derivation of Eq. 3–5 — now in Section A — and replaces the previous
"UCB + decaying MutCompute term" four-site summary with the bare+finetune configuration.)*

---

## C. REPLACEMENT — trimmed savinase round-1 prior + RL subsections

**Training the AlphaVariant prior model.** For savinase, the prior was VariantGPT, a
GPT-2-style autoregressive model (~6 million parameters; input and positional embeddings,
eight decoder blocks with masked multi-head self-attention and feed-forward layers,
residual connections and layer normalization, embedding dimension 256). It was pretrained
on the Pfam subtilase family (PF00082; 56,623 sequences of length 127–478, 90/10 train/test,
20 epochs, batch size 128) and fine-tuned on 150 published Subtilisin BPN′ variants
(10 epochs), with cross-entropy loss and AdamW (lr 1×10⁻⁴).

**Optimizing activity through RL.** The agent (initialized from VariantGPT) was optimized
with the augmented-likelihood objective (Eq. 3–5), constrained to permitted amino acids at
the mutation sites with wild-type residues elsewhere. The reward R(s) was the predicted
activity transformed to [0,1] by a sigmoid with lower/upper bounds l = −1, h = 2 (Eq. 1).

*(Drops the repeated framework/objective prose; the GPT-2 architecture and the sigmoid
reward are the only savinase-specific additions. The "Designing libraries from the improved
variants" subsection is round-specific and unchanged.)*

---

## D. Rounds 2 & 3 and cleanup notes

- **Rounds 2 and 3** already reference round 1 ("similar to the first round") — keep as-is;
  they need no framework prose. Round 3's multi-objective reward (combined score, Eq. 4)
  still stands and now cleanly references the Section A objective.
- **Delete** the duplicated augmented-likelihood derivation currently in the Benchmarking
  Methods (the Eq. 3–5 block) — it moves to Section A. Renumber equations so the framework
  section carries Eq. (3)–(5) and downstream references point there.
- Net effect: the GPT-prior + REINFORCE-objective + reward + mutational-space narrative is
  stated once (Section A); the Benchmarking and savinase sections keep only their distinct
  instantiations.
