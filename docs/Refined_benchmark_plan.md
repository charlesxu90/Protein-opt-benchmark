# Refined Computational Benchmark Plan for AlphaVariant: Targeting *Nature Methods*

## 1. Introduction and Narrative Alignment

AlphaVariant integrates three mutually reinforcing components: a generative pretrained transformer (VariantGPT), a dynamically defined mutational space, and a low-N fitness model operating within a reinforcement learning (RL) framework. To publish in *Nature Methods*, the benchmark must demonstrate that this specific combination constitutes a substantial advance over the state of the art, provide strong validation on well-characterized systems, and prove general applicability across diverse datasets [1, 2].

The landscape of competing methods has grown substantially. Recent publications include EVOLVEpro [3] (PLM + few-shot active learning, *Science* 2025), MULTI-evolve [4] (PLM ensemble + epistasis modelling, *Science* 2026), ALDE [5] (batch Bayesian optimization with uncertainty quantification, *Nature Communications* 2025), EvoPlay [6] (self-play RL with MCTS, *Nature Machine Intelligence* 2023), AiCE [7] (inverse folding models with structural/evolutionary constraints, *Cell* 2025), and CLADE [8] (cluster learning-assisted DE, *Nature Computational Science* 2021). The benchmark must directly compare AlphaVariant against these methods and demonstrate clear advantages.

This document provides a refined, implementation-ready benchmark plan that integrates insights from all related methods, specifies exact datasets with source paths, and details the experimental protocol required for *Nature Methods* publication.

---

## 2. Core Benchmark Principles

The benchmark is organized around five principles that directly address *Nature Methods* reviewer expectations:

1. **Optimization, not scoring.** AlphaVariant is an optimization method. All tasks simulate iterative directed evolution campaigns using DMS datasets as *in silico* oracles, measuring how efficiently the algorithm finds high-fitness variants within a fixed query budget [5, 6, 8].
2. **Combinatorial complexity and epistasis.** A key claim is the ability to navigate epistatic landscapes. Tasks use datasets with exhaustive combinatorial coverage at multiple sites [9, 10].
3. **Rigorous ablation studies.** Each of the three core components (VariantGPT, space definition, RL reward model) must be individually ablated to justify the full architecture [2].
4. **Generalizability at scale.** The method must be tested on ≥10 diverse protein families, functions, and landscape topographies [1].
5. **Comprehensive metric suite.** Evaluation goes beyond fitness scores to include sequence plausibility (PPL), diversity, novelty, and multi-objective trade-offs [11].

---

## 3. Benchmark Tasks, Datasets, and Source Paths

### Task 1: High-Order Combinatorial Optimization and Epistasis Navigation

**Objective:** Test whether AlphaVariant's RL agent can extrapolate from low-order (single and double) mutant data to discover high-order synergistic mutations that maximize fitness in a fully enumerated combinatorial landscape.

**Scientific rationale:** MULTI-evolve [4] showed that models trained on double mutants can extrapolate to 8–12 mutation combinations. EVOLVEpro [3] demonstrated that iterative active learning outperforms zero-shot scoring on 12 DMS datasets. AlphaVariant should demonstrate superior performance by combining the generative power of VariantGPT with RL-guided exploration.

**Datasets (from CombinGym [9]):**

| Dataset | Protein | Sites | Library Size | Property | Source Path |
|---|---|---|---|---|---|
| GB1 | Protein G domain B1 | 4 | 160,000 | IgG-Fc binding | `https://github.com/chenz16/CombinGym` |
| PhoQ | PhoQ sensor kinase | 4 | 160,000 | Antibiotic resistance | `https://github.com/chenz16/CombinGym` |
| CR9114 | Influenza antibody | 16 | 65,536 | Broad neutralization | `https://github.com/chenz16/CombinGym` |
| CreiLOV | Fluorescent protein | 15 | 165,428 | Fluorescence | `https://github.com/chenz16/CombinGym` |
| eqFP611 | Red fluorescent protein | 5 | 32,768 | Fluorescence (red + blue) | `https://github.com/chenz16/CombinGym` |

**Additional datasets (from ProteinGym DMS substitutions [12]):**

| Dataset | Protein | Property | Source Path |
|---|---|---|---|
| TEM-1 beta-lactamase | TEM-1 | Antibiotic resistance | `https://github.com/OATML-Markslab/ProteinGym` |
| Influenza HA (H1N1, H3N2) | Hemagglutinin | Viral fitness | `https://github.com/OATML-Markslab/ProteinGym` |
| SARS-CoV-2 Spike | Spike protein | ACE2 binding | `https://github.com/OATML-Markslab/ProteinGym` |
| GFP | Green fluorescent protein | Fluorescence | `https://github.com/OATML-Markslab/ProteinGym` |
| PAB1 | Poly(A)-binding protein | RNA binding | `https://github.com/OATML-Markslab/ProteinGym` |

**Simulation protocol (following CombinGym [9] and Wittmann et al. [10]):**
- **Initial training set:** Random sample of 96 single and double mutants (following ftMLDE protocol [10]).
- **Rounds:** 5 rounds of 96 queries each (480 total queries).
- **Splits:** Use CombinGym's hierarchical split strategy: 1-vs-rest (train on singles, test on doubles+), 2-vs-rest (train on doubles, test on triples+), and 3-vs-rest (train on triples, test on quadruples+). This directly tests extrapolation to higher-order mutations.
- **Replicates:** 50 independent runs with different random seeds (following Li et al. [13]).
- **Oracle:** Full DMS dataset as a lookup table.

---

### Task 2: Sample-Efficient Directed Evolution on Diverse Landscapes

**Objective:** Demonstrate that AlphaVariant achieves superior sample efficiency across diverse protein families, discovering high-fitness variants with fewer oracle queries than competing methods.

**Scientific rationale:** ALDE [5] demonstrated that batch Bayesian optimization with uncertainty quantification outperforms greedy DE on epistatic landscapes. EVOLVEpro [3] showed that PLM + few-shot active learning achieves high success rates across 12 DMS datasets with as few as 16 mutants per round. AlphaVariant should demonstrate that RL + VariantGPT achieves higher hit rates and faster convergence than both.

**Datasets (minimum 10 diverse proteins from ProteinGym [12] and EnzyArena [14]):**

| Dataset | Protein | Function | Landscape Type | Source Path |
|---|---|---|---|---|
| AAV | Adeno-associated virus | Packaging | Binding | `https://github.com/OATML-Markslab/ProteinGym` |
| DHFR | Dihydrofolate reductase | Catalysis | Stability/activity | `https://github.com/OATML-Markslab/ProteinGym` |
| ParD-ParE | Toxin-antitoxin | Binding | Epistatic | `https://github.com/OATML-Markslab/ProteinGym` |
| Cas9 | CRISPR nuclease | Editing | Activity | `https://github.com/OATML-Markslab/ProteinGym` |
| Cas13d | CRISPR nuclease | RNA editing | Activity | `https://github.com/OATML-Markslab/ProteinGym` |
| TEV protease | Cysteine protease | Catalysis | Activity/stability | `https://github.com/OATML-Markslab/ProteinGym` |
| HIV Env | HIV envelope | Viral fitness | Rugged | `https://github.com/OATML-Markslab/ProteinGym` |
| MAPK1 | Kinase | Signalling | Activity | `https://github.com/OATML-Markslab/ProteinGym` |
| Zika Env | Zika envelope | Viral fitness | Binding | `https://github.com/OATML-Markslab/ProteinGym` |
| PafA | Alkaline phosphatase | Catalysis | Activity | `https://github.com/OATML-Markslab/ProteinGym` |
| EnzyArena enzymes | Various | Kinetics | Condition-controlled | `https://github.com/Zengetal-EnzyArena` (upon release) |

**Simulation protocol (following ALDE [5] and EVOLVEpro [3]):**
- **Initial training set:** 96 randomly selected variants (NNK-equivalent random sample from the design space).
- **Rounds:** 10 rounds of 16 variants per round (160 total queries), matching EVOLVEpro's protocol [3], OR 5 rounds of 96 variants (480 total queries), matching ALDE's protocol [5]. Both budgets should be reported.
- **Replicates:** 50 independent runs per dataset per method (following Li et al. [13]).
- **Success metric:** Percentage of nominated variants in the top 10% of the fitness landscape ("high-activity candidate rate"), following EVOLVEpro [3].

---

### Task 3: Multi-Objective Optimization and Pareto Front Discovery

**Objective:** Demonstrate AlphaVariant's ability to simultaneously optimize multiple competing properties, generating a diverse Pareto front of variants.

**Scientific rationale:** ALDE [5] demonstrated multi-objective optimization for cyclopropanation yield and stereoselectivity. The Savinase campaign in the AlphaVariant paper optimized thermostability, protease activity, and detergent compatibility simultaneously. This task directly mirrors that real-world use case.

**Datasets:**

| Dataset | Properties | Source Path |
|---|---|---|
| eqFP611 (CombinGym) | Blue fluorescence + Red fluorescence | `https://github.com/chenz16/CombinGym` |
| ParPgb (ALDE) | cis-yield + trans-yield (cyclopropanation) | `https://github.com/jsunn-y/ALDE` |
| Synthetic: GFP + Stability | Fluorescence + ESM-1v stability score | Combine `ProteinGym` + `ESM-2` predictions |
| Synthetic: TEM-1 + Expression | Antibiotic resistance + ProteinMPNN score | Combine `ProteinGym` + `ProteinMPNN` predictions |

**Protocol:**
- Define a multi-objective reward function as a weighted sum or Pareto dominance criterion.
- Run AlphaVariant for 5 rounds of 96 queries (480 total).
- Report the Hypervolume Indicator and Pareto Front Coverage against the true Pareto front (computed from the full DMS data).
- Compare against single-objective optimization of each property independently to demonstrate the advantage of joint optimization.

---

## 4. Evaluation Metrics

### 4.1 Primary Optimization Metrics

| Metric | Definition | Relevant Papers |
|---|---|---|
| **Top-k Fitness vs. Queries** | Maximum fitness of the top-k variants discovered as a function of cumulative oracle queries | EvoPlay [6], ALDE [5] |
| **Area Under the Optimization Curve (AUOC)** | Integral of the Top-k Fitness curve over the query budget | EvoPlay [6] |
| **Hit Rate @ Round N** | % of variants in round N that exceed the top 10% fitness threshold | EVOLVEpro [3], MULTI-evolve [4] |
| **Global Maximum Recovery Rate** | % of 50 replicates in which the global maximum (or top 1%) is recovered | CLADE [8], Wittmann et al. [10] |

### 4.2 Sequence Quality Metrics

| Metric | Definition | Relevant Papers |
|---|---|---|
| **Sequence Plausibility (PPL)** | Perplexity of generated sequences under ESM-2 (independent of the reward model) | PDFBench [11] |
| **Diversity** | Average pairwise Hamming distance among top-k generated variants | EvoPlay [6], PDFBench [11] |
| **Novelty** | Average Hamming distance of top-k variants from the wild-type and initial training set | EvoPlay [6], PDFBench [11] |
| **Mutation Order Distribution** | Distribution of the number of mutations per generated variant (1-mut, 2-mut, 3-mut, etc.) | MULTI-evolve [4] |

### 4.3 Multi-Objective Metrics

| Metric | Definition | Relevant Papers |
|---|---|---|
| **Hypervolume Indicator** | Volume of objective space dominated by the generated Pareto front | ALDE [5] |
| **Pareto Front Coverage** | % of the true Pareto front recovered | ALDE [5] |
| **Objective Trade-off Correlation** | Spearman ρ between the two objectives across generated variants | CombinGym [9] |

---

## 5. Baseline Methods and Ablation Studies

### 5.1 Baseline Methods

All baselines must use the same oracle query budget and initial training set as AlphaVariant.

| Method | Category | Key Reference | Code Source |
|---|---|---|---|
| **Random Search** | Naive | — | — |
| **Greedy Walk (Hill Climbing)** | Traditional DE | Arnold (2018) | Custom implementation |
| **CLADE** | Cluster-based MLDE | Qiu et al. [8] | `https://github.com/guo-wei-wei/CLADE` |
| **ftMLDE** | Focused training MLDE | Wittmann et al. [10] | `https://github.com/bhattacharyya-lab/mlde` |
| **ALDE** | Bayesian active learning | Yang et al. [5] | `https://github.com/jsunn-y/ALDE` |
| **EVOLVEpro** | PLM + few-shot active learning | Jiang et al. [3] | `https://github.com/goolab-community/EVOLVEpro` |
| **AdaLead** | Adaptive greedy search | Sinai et al. [16] | `https://github.com/samsinai/FLEXS` |
| **EvoPlay** | Self-play RL (MCTS) | Wang et al. [6] | `https://github.com/MuFeng-MGI/EvoPlay` |
| **AiCE** | Inverse folding + constraints | Fei et al. [7] | `https://github.com/HongyuanFei/AiCE` (upon release) |
| **Zero-Shot Consensus** | PLM ensemble (no iteration) | Tran et al. [4], Zeng et al. [14] | `https://github.com/OATML-Markslab/ProteinGym` |

**Rationale for baseline selection:**
- CLADE and ftMLDE represent the established MLDE paradigm without RL.
- ALDE represents the state-of-the-art in Bayesian active learning for protein engineering.
- EVOLVEpro represents the state-of-the-art in PLM + few-shot active learning (*Science* 2025).
- AdaLead represents a strong, robust adaptive greedy search baseline that often outperforms more complex methods on rugged landscapes.
- EvoPlay is the most direct RL-based competitor (also uses MCTS-guided RL).
- AiCE represents the inverse folding approach as an alternative generative prior.
- Zero-Shot Consensus represents the upper bound of non-iterative methods.

### 5.2 Ablation Studies

Each ablation removes one component of AlphaVariant to isolate its contribution. All ablations use the same query budget and are evaluated on the same datasets as the full model.

| Ablation | Component Removed | Replacement | Hypothesis Tested |
|---|---|---|---|
| **AV-NoGPT** | VariantGPT prior | Random single-site mutations (mimicking EvoPlay) | Does the pretrained language model provide a better generative prior than random mutations? |
| **AV-NoSpace** | Dynamic space definition | Unconstrained full-sequence mutation space | Does intelligent space definition improve sample efficiency and hit rate? |
| **AV-StaticReward** | Iterative low-N reward model | Static zero-shot ESM-1v score as reward | Does the iteratively updated reward model outperform a fixed zero-shot signal? |
| **AV-NoRL** | RL framework | Greedy acquisition (top-k selection from VariantGPT samples) | Does RL-guided exploration outperform greedy selection from the generative model? |

**Expected outcomes:**
- AV-NoGPT should underperform AlphaVariant, especially on high-order combinatorial tasks, demonstrating that VariantGPT's learned sequence distribution focuses the search on biologically plausible regions.
- AV-NoSpace should underperform on large design spaces, demonstrating that space definition reduces the effective search space and improves sample efficiency.
- AV-StaticReward should underperform in later rounds, demonstrating that the iteratively updated reward model provides a more accurate fitness signal as more data is collected.
- AV-NoRL should underperform on rugged landscapes, demonstrating that RL's exploration-exploitation balance is superior to greedy selection.

---

## 6. Experimental Protocol and Reproducibility Standards

To meet *Nature Methods* reproducibility requirements [1, 2]:

1. **Random seeds:** All experiments must be run with 50 independent random seeds. Report mean ± standard deviation across seeds.
2. **Statistical testing:** Use the Wilcoxon signed-rank test (non-parametric) to compare AlphaVariant against each baseline. Report p-values with Bonferroni correction for multiple comparisons.
3. **Computational resources:** Report GPU hours, memory requirements, and wall-clock time for each method to enable fair comparison.
4. **Code availability:** All benchmark code, including oracle implementations, baseline wrappers, and metric calculation scripts, must be released in a public GitHub repository with a permissive license (MIT or Apache 2.0).
5. **Data availability:** All DMS datasets used are publicly available. Provide direct download links and checksums in the Methods section.

---

## 7. Positioning Against Key Competitors

The following table summarizes how AlphaVariant's benchmark design positions it against the most recent high-impact papers in the field:

| Competing Method | Published In | Key Claim | AlphaVariant's Counter-Claim | Benchmark Task |
|---|---|---|---|---|
| EVOLVEpro [3] | *Science* 2025 | PLM + few-shot active learning achieves 100-fold improvement with minimal data | RL + VariantGPT achieves higher hit rates with fewer rounds by generating focused, multi-site variants | Task 2 |
| MULTI-evolve [4] | *Science* 2026 | PLM ensemble + epistasis modelling enables single-round hyperactive multimutant discovery | RL-guided iterative refinement discovers higher-order epistatic combinations that single-round methods miss | Task 1 |
| ALDE [5] | *Nat. Commun.* 2025 | Bayesian active learning with uncertainty quantification outperforms greedy DE on epistatic landscapes | RL + VariantGPT provides a richer generative prior than Bayesian surrogate models, enabling better exploration | Task 2 |
| EvoPlay [6] | *Nat. Mach. Intell.* 2023 | Self-play RL with MCTS outperforms BO and greedy search on full-length and combinatorial tasks | AlphaVariant's VariantGPT prior provides a more informed starting point than EvoPlay's random mutations | Task 1, 2 |
| AiCE [7] | *Cell* 2025 | Inverse folding models with structural constraints identify high-fitness mutations without task-specific training | RL-guided iterative refinement discovers combinations that zero-shot inverse folding methods miss | Task 1 |
| CLADE [8] | *Nat. Comput. Sci.* 2021 | Hierarchical clustering improves training set design for MLDE | RL-guided exploration is more principled than cluster-based sampling for navigating epistatic landscapes | Task 1 |

---

## 8. Reward Model Design for RL (Key Implementation Detail)

The choice of reward model is critical for AlphaVariant's RL framework. Based on the literature, the following reward model design is recommended:

### 8.1 Primary Reward: Iterative Low-N Fitness Model

Following EVOLVEpro [3] and ALDE [5], the primary reward model should be a **lightweight ensemble of regressors** (e.g., random forest or shallow neural network) trained on ESM-2 embeddings of the queried variants. This model is retrained at each round with all accumulated data.

- **Encoder:** ESM-2 (650M parameters, `https://github.com/facebookresearch/esm`)
- **Regressor:** Random forest (n=100 trees) or DNN ensemble (n=5 networks)
- **Uncertainty quantification:** Ensemble variance (following ALDE [5]) for exploration-exploitation balance

### 8.2 Secondary Reward: Sequence Plausibility (Anti-Reward Hacking)

To prevent reward hacking (generating sequences that score highly on the reward model but are biophysically implausible), add a secondary reward term based on the log-likelihood of the sequence under ESM-2:

```
R_total = α × R_fitness + β × log P(sequence | ESM-2)
```

where α and β are hyperparameters tuned on a validation set. This approach is supported by the PDFBench framework [11] and the AiCE paper [7], which demonstrated that structural and evolutionary constraints improve the quality of generated variants.

### 8.3 Structural Reward (Optional, for Task 2 with structure data)

For proteins with known structures, add a structural plausibility reward using ProteinMPNN [15] or ESM-IF1 scores:

```
R_total = α × R_fitness + β × log P(sequence | ESM-2) + γ × log P(sequence | ProteinMPNN)
```

This is supported by Fei et al. [7] (AiCE) and Sumida et al. [15] (ProteinMPNN), which demonstrated that inverse folding scores are effective proxies for structural compatibility.

**ProteinMPNN source:** `https://github.com/dauparas/ProteinMPNN`
**ESM-IF1 source:** `https://github.com/facebookresearch/esm` (esm.pretrained.esm_if1_gvp4_t16_142M_UR50)

---

## 9. Implementation Roadmap

The following table provides a prioritized implementation roadmap for the benchmark:

| Priority | Task | Estimated Effort | Key Dependency |
|---|---|---|---|
| 1 | Implement oracle wrappers for GB1, PhoQ, TEM-1, GFP, PAB1 | 1 week | ProteinGym data download |
| 2 | Implement baseline methods (CLADE, ftMLDE, ALDE, EvoPlay) | 2 weeks | GitHub repos listed above |
| 3 | Run Task 1 (GB1, PhoQ) with 50 replicates | 1 week | GPU cluster |
| 4 | Run ablation studies on GB1 | 1 week | Task 1 infrastructure |
| 5 | Extend Task 1 to CR9114, CreiLOV, eqFP611 (CombinGym) | 1 week | CombinGym data |
| 6 | Run Task 2 on 10 diverse ProteinGym datasets | 2 weeks | Task 1 infrastructure |
| 7 | Implement EVOLVEpro and MULTI-evolve baselines | 1 week | GitHub repos listed above |
| 8 | Run Task 3 (multi-objective) on eqFP611 and ParPgb | 1 week | Task 1 infrastructure |
| 9 | Statistical analysis and figure generation | 1 week | All tasks complete |

---

## 10. References

[1] Nature Methods. (2022). What makes a Nature Methods paper. *Nature Methods*, 19, 771–772. https://doi.org/10.1038/s41592-022-01558-4

[2] Nature Methods. (2015). Reviewing computational methods. *Nature Methods*, 12, 1099. https://doi.org/10.1038/nmeth.3686

[3] Jiang, K., Yan, Z., Di Bernardo, M., et al. (2025). Rapid in silico directed evolution by a protein language model with EVOLVEpro. *Science*, 387, eadr6006. https://doi.org/10.1126/science.adr6006 | Code: `https://github.com/goolab-community/EVOLVEpro`

[4] Tran, V.Q., Nemeth, M., Bartie, L.J., et al. (2026). Rapid directed evolution guided by protein language models and epistatic interactions. *Science*. https://doi.org/10.1126/science.aea1820

[5] Yang, J., Lal, R.G., Bowden, J.C., et al. (2025). Active learning-assisted directed evolution. *Nature Communications*, 16, 714. https://doi.org/10.1038/s41467-025-55987-8 | Code: `https://github.com/jsunn-y/ALDE`

[6] Wang, Y., Tang, H., Huang, L., et al. (2023). Self-play reinforcement learning guides protein engineering. *Nature Machine Intelligence*, 5, 845–860. https://doi.org/10.1038/s42256-023-00691-9 | Code: `https://github.com/MuFeng-MGI/EvoPlay`

[7] Fei, H., Li, Y., Liu, Y., et al. (2025). Advancing protein evolution with inverse folding models integrating structural and evolutionary constraints. *Cell*, 188. https://doi.org/10.1016/j.cell.2025.06.014

[8] Qiu, Y., Hu, J., & Wei, G.-W. (2021). Cluster learning-assisted directed evolution. *Nature Computational Science*, 1, 809–818. https://doi.org/10.1038/s43588-021-00168-y

[9] Chen, Y., et al. (2026). CombinGym: a benchmark platform for machine learning-assisted design of combinatorial protein variants. Code: `https://github.com/chenz16/CombinGym`

[10] Wittmann, B.J., Yue, Y., & Arnold, F.H. (2021). Informed training set design enables efficient machine learning-assisted directed protein evolution. *Cell Systems*, 12, 1026–1045. https://doi.org/10.1016/j.cels.2021.07.008

[11] Kuang, J., et al. (2025). PDFBench: A Benchmark for De novo Protein Design from Function. *arXiv*.

[12] Notin, P., Kollasch, A., Ritter, D., et al. (2024). ProteinGym: Large-scale benchmarks for protein fitness prediction and design. *NeurIPS 2023 Datasets and Benchmarks Track*. Code: `https://github.com/OATML-Markslab/ProteinGym` | Data: `https://proteingym.org/`

[13] Li, Y., et al. (2025). Evaluation of MLDE Across Diverse Combinatorial Landscapes. *Cell Systems*. https://doi.org/10.1016/j.cels.2025.101387

[14] Zeng, Z., et al. (2026). Benchmarking and Experimental Validation of Machine Learning Strategies for Enzyme Engineering. *bioRxiv*.

[15] Sumida, K.H., Nuñez-Franco, R., Kalvet, I., et al. (2024). Improving Protein Expression, Stability, and Function with ProteinMPNN. *Journal of the American Chemical Society*. https://doi.org/10.1021/jacs.3c10941 | Code: `https://github.com/dauparas/ProteinMPNN`

[16] Sinai, S., Wang, R., Whatley, A., et al. (2020). AdaLead: A simple and robust adaptive greedy search algorithm for sequence design. *arXiv*. https://arxiv.org/abs/2010.02141 | Code: `https://github.com/samsinai/FLEXS`
