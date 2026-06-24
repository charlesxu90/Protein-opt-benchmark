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

---

# Four-site Results treatment — two options to compare

> Decision: keep the four-site comparison in the main text (Option A, compact, framed as
> the setup for the regime contrast) versus de-emphasize it (Option B, brief conclusion in
> main text with per-dataset numbers/rankings moved to Extended Data / SI). Both use the
> reproducible bare+finetune numbers (AlphaVariant: GB1 1.00, TrpB 0.83, PhoQ 0.46,
> TEV 0.37; max-fitness rank 4th, top-128 rank 3rd). Pick one to replace the current
> four-site paragraph.

## Option A — keep in main text (compact, regime-contrast emphasis)

On the densely sampled four-site landscapes — the regime in which supervised surrogates
are most effective because every candidate is measured — AlphaVariant performed on par
with the strongest baselines without any per-landscape tuning. It reached the global
optimum on GB1 (median normalized maximum fitness 1.00, tied with ALDE) and obtained 0.83
(TrpB), 0.46 (PhoQ) and 0.37 (TEV), ranking fourth by mean rank on maximum fitness and
third on top-128 mean fitness (Figure 2c,e), with ALDE the strongest four-site method
overall. This regime is the least representative of practical engineering, where
exhaustive labels are rarely available; AlphaVariant's decisive advantage instead emerged
on the larger, non-enumerable multi-site landscapes (below).

*Pros:* preserves the eight-landscape benchmark and the regime-contrast narrative; shows
no cherry-picking. *Cons:* states a fourth-place result in the main text.

## Option B — de-emphasize (brief conclusion in main text; details to Extended Data / SI)

On the four densely sampled four-site landscapes — the regime in which supervised
surrogates are most effective because every candidate is measured — AlphaVariant was
competitive with the strongest baselines without per-landscape tuning (Figure 2c,e;
per-dataset values and rankings in Extended Data Table X). Its decisive advantage emerged
on the larger, non-enumerable multi-site landscapes (below).

*(Move the per-dataset four-site values and the max/top-128 rankings to an Extended Data
table or Supplementary Note; keep Figure 2c,e as is.)*

*Pros:* tightens the main text and keeps focus on the multi-site advantage and the savinase
campaign. *Cons:* a reviewer may still ask for the four-site numbers (hence keep them one
click away in Extended Data, not removed).
