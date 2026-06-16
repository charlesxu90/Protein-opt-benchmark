# Discussion — Benchmark insights (Nature Methods draft)

> Draft Discussion subsection, formatted for Nature Methods. Companion to
> `docs/results_benchmark.md` and `docs/methods_benchmark.md`. `(ref.)` marks a
> citation to be added.

Our benchmark spanned two regimes that, although both framed as sequence
optimization, reward fundamentally different inductive biases. On the four-site
combinatorial landscapes, where every candidate carries a measured label, supervised
and active-learning methods were strongest: ALDE attained the best mean rank, and
AlphaVariant placed third while remaining within a small margin on most landscapes.
On the multi-site landscapes, where the search space exceeds 10³⁶ sequences and must
be generated rather than enumerated, the ordering inverted — AlphaVariant ranked first
of the ten methods, whereas ALDE, the four-site leader, fell to last. This regime
dependence is itself a central result: benchmark conclusions drawn from densely
sampled combinatorial libraries do not transfer to the large, sparsely measured
landscapes that characterize most real engineering targets, and methods should be
evaluated in both settings before general claims are made.

Within this picture, AlphaVariant's distinguishing property was not the single best
value on any one landscape but consistency across regimes. Using one configuration and
no per-landscape tuning, it was competitive with the best supervised methods on the
dense libraries and the highest-ranked method on the large landscapes, where its
multi-site advantages were statistically significant against the large majority of
baselines (Bonferroni-corrected Wilcoxon, α = 0.05). We interpret this as a consequence
of coupling a generative sequence prior, which proposes plausible variants without
enumeration, to a policy-gradient objective that concentrates the query budget on
high-value regions — a combination suited to spaces too large to score exhaustively.

The benchmark also clarifies how methodological choices shape apparent performance.
The maximum-fitness metric saturated on the easier landscapes, where several methods
reached the global optimum and rank differences compressed; the top-128 mean-fitness
metric was more discriminative and is where method separation was statistically
cleanest, arguing that batch-quality metrics should accompany single-best metrics in
this field. Strong but simple baselines were also informative: GreedyWalk, a plain
hill-climber, ranked second on multi-site maximum fitness, indicating that the learned
oracle landscapes are locally smooth and that any claim of method superiority must
first clear inexpensive local search. Finally, the structure-guided method AiCE was
best on GFP specifically, consistent with structure-conditioned priors being
landscape-specific advantages rather than universal ones.

Several limitations temper these conclusions. On the four-site regime AlphaVariant
matched, but did not exceed, the best supervised methods; the honest framing is that
it is competitive on dense landscapes and strongest on large ones, and the latter is
the more consequential setting for protein engineering. The multi-site evaluation
relies on a learned oracle rather than ground-truth measurements; although the oracles
were accurate on held-out data (test Spearman ρ = 0.86–0.98), a pure-oracle policy can
in principle reward sequences in poorly constrained regions, and we therefore reported
the mutational novelty of selected sequences alongside fitness. Oracle-based rankings
should be read as estimates of optimization behavior under a faithful surrogate rather
than as guarantees of wet-lab outcomes. The benchmark covered eight landscapes and a
fixed 480-query budget; broader coverage, alternative budgets, and prospective
experimental validation would further test generality.

Taken together, the benchmark supports AlphaVariant as a general-purpose sequence
optimizer that performs robustly across both densely and sparsely measured landscapes,
and it offers a reusable protocol — eight characterized landscapes, two regimes, a
validated oracle, and matched metrics and statistics — for evaluating future methods on
the same footing.
