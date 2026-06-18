# Refined Results — benchmark section insert

> Paste-ready paragraph for the main-text Results "Benchmarking AlphaVariant" section.
> Suggested location: at the end of the multi-site results paragraph (after the
> mean-rank 1.25 sentence), immediately before the "Considered together…" summary.
> Numbers are medians over seeds from the multi-site leave-one-out ablation; full
> per-component values are in the Extended Data ablation table and Supplementary Note 2.
> `(ref.)`/figure-and-table callouts to be filled.

## Multi-site ablation paragraph

To identify which components underpin this multi-site performance, we ablated each in
turn from the multi-site configuration (Extended Data Table X; Supplementary Note 2). The
dominant component was the mutation cap that restricts proposals to within two mutations
of the reference: removing it degraded top-128 mean fitness increasingly with the size of
the variable region, from a negligible change on CreiLOV (~15 variable positions) to
−0.25 on AAV (28) and −0.39 on PAB1 (74), and to near-complete collapse on GFP (233
positions; top-128 ≈ 0). The EV-augmented surrogate features contributed on the larger,
sparser landscapes (PAB1, GFP) but were approximately neutral on the smaller ones, whereas
SHAP-based alphabet pruning and the homolog-pretrained prior were approximately neutral
throughout. These results indicate that constraining generation to the neighbourhood of
the measured data — where the learned oracle is reliable — is what makes policy-gradient
search effective on non-enumerable landscapes, and that the value of this constraint grows
with the size of the search space, precisely the regime in which AlphaVariant most
outperforms the baselines.
