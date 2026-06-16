# Extended Data Table — Component ablation of AlphaVariant on the multi-site benchmark

> Leave-one-out ablation on the four multi-site (learned-oracle) landscapes. Each
> configuration removes one component from the shipped AlphaVariant configuration and
> re-runs the full 480-query campaign (96 × 5 rounds). Values are the median across
> independent seeds with the interquartile range [Q1–Q3]; Δ is the change in median
> versus the full configuration on the same landscape. Fitness is min–max normalized
> to [0, 1] per landscape (oracle score). Higher is better.

| Dataset | Configuration | *n* | Max fitness, median [Q1–Q3] | Δ max | Top-128 mean fitness, median [Q1–Q3] | Δ top-128 |
|---------|---------------|:---:|:---------------------------:|:-----:|:------------------------------------:|:---------:|
| **CreiLOV** | AlphaVariant (full) | 30 | 0.987 [0.973–0.987] | ref | 0.899 [0.895–0.908] | ref |
|  | − EV features | 30 | 0.973 [0.973–0.987] | -0.015 | 0.899 [0.895–0.906] | +0.000 |
|  | − SHAP / alphabet constraint | 30 | 0.980 [0.973–0.987] | -0.007 | 0.897 [0.892–0.905] | -0.003 |
|  | − mutation cap | 30 | 0.973 [0.973–0.982] | -0.015 | 0.914 [0.913–0.915] | +0.015 |
|  | − homolog prior | 30 | 0.987 [0.987–0.987] | +0.000 | 0.912 [0.905–0.915] | +0.013 |
| **AAV** | AlphaVariant (full) | 30 | 0.711 [0.695–0.726] | ref | 0.649 [0.637–0.655] | ref |
|  | − EV features | 30 | 0.713 [0.696–0.724] | +0.002 | 0.642 [0.627–0.658] | -0.006 |
|  | − SHAP / alphabet constraint | 30 | 0.710 [0.696–0.723] | -0.001 | 0.649 [0.640–0.662] | -0.000 |
|  | − mutation cap | 30 | 0.672 [0.660–0.692] | -0.038 | 0.395 [0.390–0.410] | -0.253 |
|  | − homolog prior | 30 | 0.718 [0.711–0.731] | +0.008 | 0.645 [0.636–0.665] | -0.004 |
| **PAB1** | AlphaVariant (full) | 10 | 0.587 [0.578–0.594] | ref | 0.513 [0.503–0.519] | ref |
|  | − EV features | 10 | 0.549 [0.537–0.563] | -0.039 | 0.453 [0.443–0.476] | -0.060 |
|  | − SHAP / alphabet constraint | 10 | 0.586 [0.565–0.597] | -0.001 | 0.500 [0.490–0.509] | -0.013 |
|  | − mutation cap | 10 | 0.403 [0.389–0.418] | -0.184 | 0.122 [0.117–0.137] | -0.391 |
|  | − homolog prior | 10 | 0.584 [0.563–0.600] | -0.003 | 0.503 [0.494–0.531] | -0.011 |
| **GFP** | AlphaVariant (full) | 5 | 0.956 [0.952–0.961] | ref | 0.813 [0.806–0.830] | ref |
|  | − EV features | 5 | 0.934 [0.913–0.961] | -0.022 | 0.791 [0.776–0.796] | -0.022 |
|  | − SHAP / alphabet constraint | 5 | 0.934 [0.923–0.965] | -0.022 | 0.801 [0.790–0.847] | -0.012 |
|  | − mutation cap | 5 | 0.934 [0.913–0.961] | -0.022 | -0.038 [-0.048–-0.028] | -0.851 |
|  | − homolog prior | 5 | 0.934 [0.913–0.961] | -0.022 | 0.757 [0.752–0.800] | -0.056 |

Seed counts reflect per-landscape compute cost (PAB1 and GFP use fewer seeds because
per-seed runtime scales with sequence length). The mutation cap (`--max_n_mut 2`) is
the dominant component, and its effect on top-128 mean fitness grows with the size of
the variable region (CreiLOV ~15 → AAV 28 → PAB1 74 → GFP 233 positions). EV-augmented
features contribute on the larger, sparser landscapes (PAB1, GFP); the SHAP/alphabet
constraint and the homolog-pretrained prior are approximately neutral throughout.

## Alternative summary — mean ± s.d.

> The same ablation summarized by the mean ± standard deviation across seeds, as an
> illustration. On the smooth, saturated landscapes (CreiLOV, GFP) the per-seed values
> cluster tightly, so the median/IQR appears nearly degenerate; the mean ± s.d. shows the
> dispersion and the size of each effect more clearly. (Mean is more outlier-sensitive,
> e.g. the off-manifold negative scores of −mutation cap; the median table remains the
> primary, main-text-consistent summary.)

| Dataset | Configuration | *n* | Max fitness, mean ± s.d. | Δ mean | Top-128 mean fitness, mean ± s.d. | Δ mean |
|---------|---------------|:---:|:------------------------:|:------:|:---------------------------------:|:------:|
| **CreiLOV** | AlphaVariant (full) | 30 | 0.981 ± 0.007 | ref | 0.900 ± 0.009 | ref |
|  | − EV features | 30 | 0.979 ± 0.007 | -0.001 | 0.899 ± 0.008 | -0.001 |
|  | − SHAP / alphabet constraint | 30 | 0.980 ± 0.008 | -0.001 | 0.897 ± 0.008 | -0.003 |
|  | − mutation cap | 30 | 0.976 ± 0.006 | -0.004 | 0.914 ± 0.002 | +0.014 |
|  | − homolog prior | 30 | 0.986 ± 0.004 | +0.005 | 0.910 ± 0.008 | +0.010 |
| **AAV** | AlphaVariant (full) | 30 | 0.712 ± 0.022 | ref | 0.649 ± 0.018 | ref |
|  | − EV features | 30 | 0.710 ± 0.021 | -0.002 | 0.642 ± 0.020 | -0.007 |
|  | − SHAP / alphabet constraint | 30 | 0.710 ± 0.018 | -0.002 | 0.651 ± 0.017 | +0.002 |
|  | − mutation cap | 30 | 0.676 ± 0.024 | -0.036 | 0.399 ± 0.015 | -0.250 |
|  | − homolog prior | 30 | 0.720 ± 0.018 | +0.008 | 0.647 ± 0.018 | -0.002 |
| **PAB1** | AlphaVariant (full) | 10 | 0.588 ± 0.017 | ref | 0.514 ± 0.018 | ref |
|  | − EV features | 10 | 0.549 ± 0.021 | -0.039 | 0.460 ± 0.021 | -0.054 |
|  | − SHAP / alphabet constraint | 10 | 0.580 ± 0.025 | -0.008 | 0.501 ± 0.016 | -0.013 |
|  | − mutation cap | 10 | 0.405 ± 0.022 | -0.183 | 0.127 ± 0.015 | -0.387 |
|  | − homolog prior | 10 | 0.586 ± 0.029 | -0.002 | 0.508 ± 0.027 | -0.006 |
| **GFP** | AlphaVariant (full) | 5 | 0.960 ± 0.024 | ref | 0.823 ± 0.030 | ref |
|  | − EV features | 5 | 0.943 ± 0.037 | -0.017 | 0.794 ± 0.024 | -0.030 |
|  | − SHAP / alphabet constraint | 5 | 0.945 ± 0.037 | -0.015 | 0.814 ± 0.033 | -0.010 |
|  | − mutation cap | 5 | 0.943 ± 0.038 | -0.018 | -0.038 ± 0.018 | -0.862 |
|  | − homolog prior | 5 | 0.944 ± 0.039 | -0.017 | 0.712 ± 0.187 | -0.111 |
