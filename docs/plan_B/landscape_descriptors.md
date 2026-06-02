# Supplementary Table 1 — Landscape descriptors and selection-rule output

Computed by `scripts/compute_landscape_descriptors.py`. Thresholds: d1 ≥ 4.0 nats → PLM-reward; d2 CV ≥ 1.0 AND top-5% present → SHAP; d3 min < 2.5 bits → Hybrid; else default SHAP.

| Dataset | d1 PLM gap (nats) | d2 round-1 CV | d2 top-5% present | d3 min top-128 entropy (bits) | Rule selects | Empirical best | Agree |
|---------|-------------------|---------------|-------------------|-------------------------------|--------------|----------------|-------|
| 4site_TEV | -2.850 | 5.230 | True | 2.889 | SHAP | PLM-reward | ✗ |
| 4site_GB1 | 0.533 | 3.109 | True | 1.899 | SHAP | PLM-reward | ✗ |
| 4site_PhoQ | 5.078 | 5.115 | True | 2.562 | PLM-reward | SHAP | ✗ |
| 4site_TRPB | -3.123 | 1.102 | True | 0.987 | SHAP | Hybrid | ✗ |

### Per-position top-128 Shannon entropy (bits)

| Dataset | pos 0 | pos 1 | pos 2 | pos 3 |
|---------|-------|-------|-------|-------|
| 4site_TEV | 2.889 | 3.819 | 3.939 | 3.458 |
| 4site_GB1 | 2.853 | 2.807 | 2.340 | 1.899 |
| 4site_PhoQ | 2.562 | 3.341 | 3.360 | 3.685 |
| 4site_TRPB | 2.893 | 2.303 | 2.913 | 0.987 |
