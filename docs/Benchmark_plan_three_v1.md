# Benchmark Comparator Selection for AlphaVariant across GB1, CreiLOV, and CR9114-H1

**Author:** Manus AI  
**Date:** 12 May 2026

## Executive recommendation

The comparison set should be **small, methodologically diverse, and explicitly fair**. For the main manuscript, I recommend using the same core families across all three datasets: **random search**, **greedy or hill-climbing directed evolution**, **ALDE**, **one classical MLDE comparator** such as **CLADE** or **ftMLDE**, and **one adaptive/RL comparator** such as **AdaLead**, **EvoPlay**, or **µProtein**. This structure is defensible because it covers the minimum sanity baseline, the experimental directed-evolution heuristic, a low-N active-learning method, a conventional supervised MLDE method, and a modern adaptive search/RL method.

The three selected datasets are complementary. **GB1** is the credibility anchor because it is a nearly complete four-site epistatic landscape of 20^4 variants at V39, D40, G41, and V54.[1] **CreiLOV** is the strongest high-order fluorescence benchmark because it contains a large combinatorial landscape over many sites and is explicitly intended for higher-order mutant prediction.[2] **CR9114-H1** adds an antibody-antigen binding landscape over evolutionary intermediates, preventing the benchmark from appearing limited to fluorescence or compact enzyme-like landscapes.[3] [4]

| Dataset | Main purpose in the benchmark | Main methods to include | Methods to keep secondary or diagnostic | Methods to avoid as main baselines |
|---|---|---|---|---|
| **GB1** | Compact, canonical epistasis benchmark with exhaustive candidate space | Random, greedy DE, ALDE, CLADE or ftMLDE, AdaLead, EvoPlay or µProtein | EVOLVEpro, zero-shot PLM ranking, MULTI-evolve-style epistasis model | ProteinMPNN, AiCE, manufacturing-aware generative models |
| **CreiLOV** | Large high-order fluorescence landscape where multi-site combinatorial design should matter | Random, greedy DE, ALDE, ftMLDE or adapted CLADE, AdaLead, µProtein or EvoPlay | EVOLVEpro, zero-shot PLM ranking, MULTI-evolve if lower-order training assumptions are matched | ProteinMPNN, AiCE, manufacturing-aware generative models |
| **CR9114-H1** | Antibody binding landscape with many-site evolutionary intermediates | Random, greedy DE, ALDE, ftMLDE/CLADE, AdaLead, EVOLVEpro, one RL/adaptive method | MULTI-evolve, ProteinMPNN/AiCE if structures and constraints are available | Manufacturing-aware generative models as performance baselines |

The most important practical decision is not whether to include every recent method, but whether each method can be run under **identical candidate-space, query-budget, initialization, and oracle-access rules**. Methods that require structural models, pretrained sequence priors, single-mutant preselection, wet-lab synthesis assumptions, or special mutation alphabets should be included only when those assumptions are available to all methods or are clearly labeled as a **diagnostic analysis** rather than a primary comparison.

## Dataset-specific benchmark logic

### GB1

GB1 should remain the anchor benchmark. The original eLife study characterized all variants at four amino-acid sites in protein GB1, corresponding to a 20^4 design space, and emphasized how epistasis can block direct adaptive paths while indirect paths can still reach high-fitness genotypes.[1] This makes GB1 ideal for showing whether AlphaVariant can exploit multi-site combinations rather than merely ranking additive single mutations.

For GB1, the main baseline set should include **CLADE**, because CLADE was explicitly evaluated on GB1 and reported strong hit rates after sequentially screening 480 sequences from the 160,000-variant four-site library.[5] **EvoPlay** is also especially relevant because its data-availability statement identifies GB1 among the benchmark datasets used in its in silico studies, and its method is a self-play reinforcement-learning framework for protein sequence optimization.[6] **ALDE** is appropriate because it is designed for low-N, batch active-learning-assisted directed evolution over a defined combinatorial library.[7]

| Recommendation for GB1 | Rationale |
|---|---|
| **Required:** random search | Establishes the minimum sanity check under the exact same query budget. |
| **Required:** greedy or hill-climbing DE | Tests whether AlphaVariant beats the simplest local adaptive-walk logic on an epistatic landscape. |
| **Required:** ALDE | Directly represents low-N batch active learning over a defined combinatorial design space.[7] |
| **Required:** CLADE or ftMLDE | CLADE has direct GB1 precedent and public code; ftMLDE represents the classical MLDE design philosophy.[5] [8] |
| **Strongly recommended:** AdaLead or EvoPlay | AdaLead is a robust adaptive greedy-search baseline; EvoPlay is a direct RL protein-engineering comparator with GB1 precedent.[6] [9] |
| **Optional:** EVOLVEpro or zero-shot PLM ranking | Useful for separating pretrained sequence priors from closed-loop learning, but should not crowd the main figure.[10] |
| **Not primary:** ProteinMPNN, AiCE, manufacturing-aware generative models | These are structure-conditioned or synthesis-scale design methods, not natural closed-loop comparators for this compact enumerated landscape.[11] [12] [13] |

### CreiLOV

CreiLOV is the best stress test for AlphaVariant’s high-order combination claim. The original CreiLOV work studied an oxygen-independent fluorescent protein and developed DMS resources for modeling higher-order variants.[2] In the CombinGym summary, CreiLOV is described as an oxygen-independent flavin mononucleotide-based fluorescent protein from *Chlamydomonas reinhardtii*, with 165,428 measured variants out of 184,320 possible combinatorial mutants across 20 mutations at 15 residues.[4] This is precisely the setting where a multi-site combinatorial method should outperform local greedy logic.

For CreiLOV, **ALDE**, **ftMLDE/adapted CLADE**, and **AdaLead** should be the central nontrivial comparators. **µProtein** is also attractive because it explicitly combines a mutational-effect predictor with a reinforcement-learning search strategy and claims to model epistasis in multi-amino-acid mutants using single-mutation data.[14] However, if the cleaned CreiLOV benchmark contains mostly high-order variants and not a balanced low-order training set, µProtein should be run only under clearly matched training splits.

| Recommendation for CreiLOV | Rationale |
|---|---|
| **Required:** random search | Necessary because the candidate space is large and high-order; chance performance is informative. |
| **Required:** greedy or hill-climbing DE | Important to show that AlphaVariant does not merely follow local improvement paths. |
| **Required:** ALDE | Well aligned with batch active-learning optimization over a defined multi-site library.[7] |
| **Required:** ftMLDE or adapted CLADE | Represents supervised MLDE on combinatorial protein landscapes; CLADE may need adaptation beyond the original four-site GB1-style setting.[5] [8] |
| **Strongly recommended:** AdaLead | Robust and simple adaptive sequence-design baseline that avoids overfitting the comparison to heavy RL engineering.[9] |
| **Conditional:** µProtein or EvoPlay | Include one if implementation is stable under the same query budget and candidate pool.[6] [14] |
| **Optional:** EVOLVEpro | Useful if PLM embeddings and active-learning protocol can be restricted to the same sampled observations.[10] |
| **Not primary:** ProteinMPNN or AiCE | Structure-conditioned mutation design may not fairly optimize fluorescence over the enumerated landscape without task-specific assumptions.[11] [12] |

### CR9114-H1

CR9114-H1 is the most important biological diversification dataset because it evaluates antibody-antigen binding rather than fluorescence or GB1 binding/stability. The CR9114/CR6261 antibody landscapes are evolutionary-intermediate landscapes, and CombinGym notes that the CR9114-H1 landscape is the most informative subtype-specific version for benchmarking.[4] This makes the dataset ideal for demonstrating that AlphaVariant generalizes beyond compact or fluorescence-centered protein landscapes.

For CR9114-H1, the baseline set should still include random search, greedy DE, ALDE, and a classical MLDE method. In addition, **EVOLVEpro** deserves stronger consideration here than in GB1 or CreiLOV because it was demonstrated on diverse proteins including optimized antibodies, and it uses protein language model embeddings plus few-shot active learning to improve protein activity.[10] **MULTI-evolve** is also conceptually close to CR9114-H1 because it explicitly combines PLM-derived or existing functional data with epistasis-aware modeling to predict synergistic multi-mutants.[15] However, MULTI-evolve should be secondary unless the benchmark can match its single/double-mutant data assumptions and mutation-construction constraints.

| Recommendation for CR9114-H1 | Rationale |
|---|---|
| **Required:** random search | Essential sanity baseline over the cleaned antibody-intermediate candidate space. |
| **Required:** greedy or hill-climbing DE | Represents stepwise antibody maturation logic and local adaptive search. |
| **Required:** ALDE | Appropriate for low-N batch learning and epistatic optimization if candidate generation is restricted to CR9114-H1 variants.[7] |
| **Required:** ftMLDE or CLADE | Provides a supervised MLDE comparator for antibody binding landscapes. |
| **Strongly recommended:** AdaLead | Robust adaptive-search comparator that can be run without antibody-specific structural assumptions.[9] |
| **Strongly recommended if feasible:** EVOLVEpro | Particularly relevant because it has antibody-engineering demonstrations and a few-shot active-learning design.[10] |
| **Conditional secondary:** MULTI-evolve | Highly relevant to synergistic multi-mutants, but only fair if initial single/double-mutant or functional-data assumptions are matched.[15] |
| **Conditional secondary:** ProteinMPNN or AiCE | May be informative if antibody-antigen structures and residue constraints are available, but these should not replace closed-loop baselines.[11] [12] |

## Application scope of each listed method

The methods in the user-provided list fall into five groups: **basic search baselines**, **classical MLDE/active learning**, **adaptive/RL search**, **PLM or structure-informed design**, and **manufacturing/generative methods**. AlphaVariant’s main claim should be tested against at least one method from each relevant search family, but not against every recent method as a main comparator.

| Method | Application scope | Fit to these datasets | Recommended role |
|---|---|---|---|
| **Random search** | Uniformly samples the same candidate space under the same query budget. | Universal. | **Required for all datasets.** |
| **Greedy / hill-climbing DE** | Iteratively mutates around or selects from the best observed variants. | Universal, but weakest on rugged epistatic landscapes. | **Required for all datasets.** |
| **ALDE** | Batch Bayesian optimization / active learning with uncertainty quantification for low-N directed evolution.[7] | Excellent for all three if run over the same enumerated candidate pool. | **Primary comparator for all datasets.** |
| **CLADE** | Hierarchical clustering plus supervised learning for MLDE; published GB1 benchmark and public code.[5] | Excellent for GB1, reasonable for CreiLOV and CR9114-H1 if adapted. | **Primary classical MLDE comparator.** |
| **ftMLDE** | Focused-training MLDE strategy for efficient training-set design in combinatorial protein engineering.[8] | Broadly relevant; especially useful when comparing against conventional MLDE. | **Alternative or complement to CLADE.** |
| **AdaLead / FLEXS** | Simple robust adaptive greedy/model-based search in a sequence-design sandbox.[9] | Very good generic adaptive baseline for all three. | **Primary adaptive-search comparator.** |
| **EvoPlay** | Self-play RL with policy-value network and Monte Carlo tree search; supports full-length and combinatorial-site tasks.[6] | Strong for GB1; conditional for larger CreiLOV and CR9114-H1 due to compute/tuning burden. | **Primary RL comparator if reproducible; otherwise secondary.** |
| **µProtein / µFormer + µSearch** | Mutational-effect prediction plus RL search, designed to model epistasis and multi-point mutants from single-mutation data.[14] | Strong conceptually for high-order landscapes if training data assumptions match. | **Conditional RL comparator, especially for CreiLOV and CR9114-H1.** |
| **EVOLVEpro** | PLM embeddings plus top-layer regression in a few-shot active-learning loop; demonstrated across DMS datasets and prospective proteins.[10] | Strong for CR9114-H1; optional diagnostic for GB1 and CreiLOV. | **Secondary/extended-data comparator, or primary for antibody dataset if feasible.** |
| **MULTI-evolve** | PLM or functional-data-driven mutation discovery plus epistasis-aware modeling and multi-site assembly.[15] | Conceptually strong for CR9114-H1 and high-order settings, but assumption-heavy. | **Secondary diagnostic unless matched lower-order data are available.** |
| **ProteinMPNN** | Structure-conditioned sequence design to fold to a backbone; can improve expression, stability, and function when constrained.[11] | Weak as a fair oracle-query baseline; useful only with structures and fixed residues. | **Diagnostic, not a main baseline.** |
| **AiCE** | Inverse-folding design with structural and evolutionary constraints for high-fitness single and multi-mutations.[12] | Potentially useful with high-quality structures; not a natural closed-loop benchmark over pre-enumerated landscapes. | **Diagnostic, not a main baseline.** |
| **Off-policy RL / δ-Conservative Search** | GFlowNet/off-policy RL active learning with conservative exploration near reliable data regions.[16] | Relevant but broader biological-sequence method; implementation and tuning may dominate. | **Secondary only unless code is mature and budgets are matched.** |
| **Manufacturing-aware generative model architectures** | Generative design optimized for scalable synthesis cost and library manufacturability.[13] | Not designed to optimize a fixed DMS landscape under query budgets. | **Do not include as a main comparator.** |

## Recommended benchmark designs

The benchmark should use one **main protocol** and one **diagnostic protocol**. The main protocol should be as simple and comparable as possible: each method receives the same initial observations, the same batch size, the same number of rounds, the same candidate pool, and the same oracle values only after querying. Performance should be reported as best observed fitness, top-k hit rate, normalized regret, enrichment over random, and area under the best-so-far curve.

The diagnostic protocol can test whether pretrained priors or structural constraints add independent value. In that protocol, zero-shot PLM ranking, EVOLVEpro, AiCE, ProteinMPNN, and MULTI-evolve can be included, but the figures should clearly separate them from the primary closed-loop comparison. This avoids the common fairness problem in which a method with extra structural, evolutionary, or synthesis assumptions is compared to a method restricted to observed sequence-fitness pairs.

| Benchmark element | Recommended choice | Reason |
|---|---|---|
| Initial training sizes | Use low-N settings such as 24, 48, 96, and 192 observations, depending on dataset size. | Shows sample efficiency and avoids tuning to one budget. |
| Batch size | Keep fixed across methods, for example 24 or 48 variants per round. | Matches laboratory-style batch directed evolution. |
| Rounds | Use 3–5 rounds for the main comparison. | Enough to show adaptive behavior without allowing excessive method-specific tuning. |
| Candidate space | Restrict all methods to the cleaned variant table for each dataset. | Prevents invalid sequences and ensures identical oracle access. |
| Replicates | Use at least 20 random seeds per condition. | Reduces stochastic-selection artifacts, especially in rugged landscapes. |
| Metrics | Best observed fitness, top-1/top-10 hit rate, normalized regret, enrichment over random, and AUC of best-so-far. | Captures both discovery and consistency. |
| Statistical reporting | Report median and interquartile range; use paired seed-level comparisons. | Avoids overclaiming from noisy optimization traces. |

## Final inclusion plan

The strongest manuscript structure is to keep the **main figures focused** and move heavy or assumption-mismatched methods to **Extended Data**. I recommend the following final plan.

| Comparator category | GB1 | CreiLOV | CR9114-H1 | Placement |
|---|---|---|---|---|
| Random search | Include | Include | Include | Main |
| Greedy / hill-climbing DE | Include | Include | Include | Main |
| ALDE | Include | Include | Include | Main |
| CLADE or ftMLDE | Include CLADE first | Include ftMLDE or adapted CLADE | Include ftMLDE or adapted CLADE | Main |
| AdaLead | Include | Include | Include | Main |
| EvoPlay or µProtein | Include EvoPlay if feasible | Include µProtein or EvoPlay if feasible | Include one RL/adaptive method | Main or Extended Data depending on stability |
| EVOLVEpro | Optional | Optional | Include if feasible | Extended Data, or main for CR9114-H1 |
| MULTI-evolve | Optional diagnostic | Optional diagnostic | Conditional diagnostic | Extended Data |
| Zero-shot PLM ranking | Optional | Optional | Optional | Extended Data |
| ProteinMPNN | Exclude from main | Exclude from main | Conditional diagnostic | Extended Data only |
| AiCE | Exclude from main | Exclude from main | Conditional diagnostic | Extended Data only |
| Manufacturing-aware generative model | Exclude | Exclude | Exclude | Not recommended |
| Off-policy RL / δ-CS | Optional | Optional | Optional | Extended Data only unless implementation is already available |

## Fairness paragraph for Methods section

> All comparator methods were evaluated under identical candidate-space and oracle-query constraints. For each dataset, methods were initialized with the same randomly sampled training observations and were allowed to nominate the same number of candidates per batch and the same total number of oracle queries. Hyperparameters were selected from published defaults or by validation on observed training data only. No method was allowed to access held-out oracle values, full-landscape rankings, or test-set statistics during candidate selection. Methods requiring additional information, including pretrained sequence embeddings, protein structures, evolutionary couplings, or mutation-synthesis assumptions, were reported separately as diagnostic comparisons unless the same information was made available to all applicable methods. Failed or incompatible baselines were documented with the reason for exclusion rather than silently omitted.

## Bottom-line recommendation

For a clean and persuasive AlphaVariant benchmark, the **main comparison** should include **random search, greedy DE, ALDE, CLADE/ftMLDE, AdaLead, and one RL-style method**. On **GB1**, use CLADE and EvoPlay if feasible because they have direct relevance to the dataset. On **CreiLOV**, emphasize ALDE, ftMLDE/adapted CLADE, AdaLead, and optionally µProtein because the dataset is a high-order fluorescence stress test. On **CR9114-H1**, add EVOLVEpro as the most relevant newer comparator because of its few-shot PLM-guided active-learning design and antibody-engineering demonstrations. ProteinMPNN, AiCE, MULTI-evolve, off-policy RL, and manufacturing-aware generative models should be treated as **diagnostic or extended-data methods**, not as mandatory main baselines.

## References

[1]: https://elifesciences.org/articles/16965 "Adaptation in protein fitness landscapes is facilitated by indirect paths"

[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10204710/ "Deep Mutational Scanning of an Oxygen-Independent Fluorescent Protein"

[3]: https://elifesciences.org/articles/71393 "Binding affinity landscapes constrain the evolution of broadly neutralizing anti-influenza antibodies"

[4]: https://www.biorxiv.org/content/10.64898/2026.03.24.714074v1.full-text "CombinGym: a benchmark platform for machine learning-assisted design of combinatorial protein variants"

[5]: https://www.nature.com/articles/s43588-021-00168-y "Cluster learning-assisted directed evolution"

[6]: https://www.nature.com/articles/s42256-023-00691-9 "Self-play reinforcement learning guides protein engineering"

[7]: https://www.nature.com/articles/s41467-025-55987-8 "Active learning-assisted directed evolution"

[8]: https://doi.org/10.1016/j.cels.2021.07.008 "Informed training set design enables efficient machine learning-assisted directed protein evolution"

[9]: https://arxiv.org/abs/2010.02141 "AdaLead: A simple and robust adaptive greedy search algorithm for sequence design"

[10]: https://www.science.org/doi/10.1126/science.adr6006 "Rapid in silico directed evolution by a protein language model with EVOLVEpro"

[11]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10811672/ "Improving Protein Expression, Stability, and Function with ProteinMPNN"

[12]: https://pubmed.ncbi.nlm.nih.gov/40628259/ "Advancing protein evolution with inverse folding models integrating structural and evolutionary constraints"

[13]: https://www.biorxiv.org/content/10.1101/2024.09.13.612900v1 "Manufacturing-Aware Generative Model Architectures Enable Biological Sequence Design and Synthesis at Petascale"

[14]: https://www.nature.com/articles/s42256-025-01103-w "Accelerating protein engineering with fitness landscape modelling and reinforcement learning"

[15]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12991030/ "Rapid directed evolution guided by protein language models and epistatic interactions"

[16]: https://arxiv.org/html/2410.04461v2 "Improved Off-policy Reinforcement Learning in Biological Sequence Design"
