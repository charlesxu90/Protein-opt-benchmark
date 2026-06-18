# Refined SI — new subsection (paste into SI_AlphaVariant)

> New Supplementary Note to add to the SI (after "Supplementary Note 1: Benchmarked
> methods and configurations"; renumber the following notes accordingly). Numbers are
> medians over seeds from the component ablations; Δ is the change versus the baseline on
> the same landscape. `(ref.)`/figure callouts to be filled. Full per-config tables:
> Extended Data Tables (four-site and multi-site ablation).

## Supplementary Note 2: Ablation of AlphaVariant components

To quantify the contribution of each AlphaVariant component, we performed leave-one-out
ablations on the benchmark landscapes, re-running the full optimization campaign (480
queries; 96 × 5 rounds) for each configuration. The four-site ablations used the shipped
**bare + finetune** configuration as the baseline; the multi-site ablations used the
shipped full configuration as the baseline. Each configuration was run for 30 independent
seeds, except multi-site PAB1 (10) and GFP (5), where per-seed cost on the longer
sequences made 30 seeds impractical. Δ denotes the change in median normalized maximum
fitness or top-128 mean fitness relative to the baseline.

**Four-site landscapes.** Removing per-round prior finetuning reduced top-128 mean fitness
by 0.026–0.061 on GB1, TEV and TrpB and GB1 maximum fitness by 0.041, while never
improving any landscape, so finetuning was retained. The REINFORCE step was essential for
GB1 maximum fitness (−0.138 when removed) but, once the prior was finetuned, was largely
redundant on the other landscapes (PhoQ maximum fitness unchanged; TEV and TrpB maximum
fitness unchanged or higher under generate-and-prioritize), indicating that finetuning and
the policy-gradient step supply overlapping benefit and partially substitute for one
another. Replacing the five-model surrogate ensemble with a single random forest mainly
reduced top-128 mean fitness (TEV, TrpB) with a landscape-dependent effect on maximum
fitness. The structure-based reward shaping (MutCompute) and SHAP-based alphabet pruning of
the earlier Plan C configuration did not improve over the bare configuration on any
four-site landscape and were therefore excluded from the shipped setting (Extended Data
Table, four-site ablation).

**Multi-site landscapes.** The mutation cap (at most two mutations from the reference) was
the dominant component, and its importance increased monotonically with the size and
sparsity of the variable region: removing it changed top-128 mean fitness by +0.015 on
CreiLOV (~15 variable positions, ~90% of the library measured), −0.253 on AAV (28),
−0.391 on PAB1 (74) and −0.851 on GFP (233, where top-128 collapsed to ≈0). Without the
cap, generation drifts into regions far from the measured set, where the oracle is
unreliable, and batch quality breaks down. The EV-augmented surrogate features were
approximately neutral on the small, well-covered landscapes (CreiLOV, AAV) and beneficial
on the larger, sparser ones (PAB1, GFP). SHAP-based alphabet pruning and the
homolog-pretrained prior were approximately neutral across all four landscapes (Extended
Data Table, multi-site ablation).

**Per-round finetuning on multi-site.** Because per-round prior finetuning helped on
four-site, we tested whether it also helped on the multi-site landscapes, where the prior
is already homolog-pretrained. Adding finetuning (paired against the shipped configuration
on the same seeds) was marginally positive on AAV (Δ ≈ +0.009 on both metrics) and
essentially neutral on CreiLOV (Δ ≈ +0.001–0.002), but negative on PAB1 (Δ max −0.014);
the GFP arm could not be completed because the finetuning step exceeded GPU memory on the
longest sequences. Because finetuning did not consistently help on multi-site — where the
prior already captures family-level signal — it was not adopted there and is used only on
four-site, where the prior is initialized in-run.

**Summary.** The two regimes are governed by different load-bearing components: four-site
performance rests on prior finetuning and the policy-gradient step, whereas multi-site
performance rests on the manifold-constraining mutation cap, increasingly so as the search
space grows. AlphaVariant's robustness across regimes therefore arises from applying the
appropriate constraint where the landscape demands it rather than from any single
universally beneficial component.
