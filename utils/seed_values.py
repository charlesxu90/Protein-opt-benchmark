"""Per-seed metric loading for the benchmark figures and statistics.

Single source of truth for where each method's per-seed results live and how the
raw values are normalised, shared by the dot + whisker figures and the paired
Wilcoxon tests. Values are keyed by seed because paired tests need to align
methods on the same seeds.

  4-site   : per-method result trees, raw fitness divided by the landscape's
             global max (max_fitness only; the top-128 field is pre-normalised).
  multisite: results_oracle/<dataset>/<method>/seed*.json, already in [0, 1].
"""
from __future__ import annotations

import glob
import json
import os
import re

GMAX_4SITE = {
    "4site_GB1": 8.761966,
    "4site_PhoQ": 133.5943,
    "4site_TEV": 1.0,
    "4site_TRPB": 1.0,
}

COMPETITOR_PATTERNS_4SITE = {
    "Random":       "Random/results/{a}_Random/{a}/random/metrics_seed*.json",
    "GreedyWalk":   "GreedyWalk/results/{a}_GreedyWalk/{a}/greedy/metrics_seed*.json",
    "ALDE":         "ALDE/results/{a}_ALDE/{a}/onehot/metrics_seed*.json",
    "FLEXS":        "FLEXS/results/{a}_AdaLead/{a}/metrics_seed*.json",
    "AiCE":         "AiCE/results/{a}_AiCE/{a}/aice/metrics_seed*.json",
    "ftMLDE":       "ftMLDE/results/{a}_ftMLDE/{a}/ftmlde/metrics_seed*.json",
    "CLADE":        "CLADE/results/{a}_CLADE/{a}/clade/metrics_seed*.json",
    "EVOLVEpro":    "EVOLVEpro/results/{a}_EVOLVEpro/{a}/*/metrics_seed*.json",
    "MULTIevolve":  "MULTIevolve/results/{a}_MULTIevolve/{a}/*/metrics_seed*.json",
    "AlphaVariant": "alphavariant/results/_archive_tier1B_canonical/{arch}/seed_*/metrics.json",
}

DS_ARCH = {
    "4site_GB1":  "4site_GB1",
    "4site_PhoQ": "4site_PhoQ",
    "4site_TEV":  "4site_TEV",
    "4site_TRPB": "TRPB",
}

# EVOLVEpro/MULTIevolve stored their TRPB run under the full "4site_TRPB"
# archive name instead of the bare "TRPB" alias every other method uses.
ARCH_OVERRIDE = {
    ("EVOLVEpro", "4site_TRPB"): "4site_TRPB",
    ("MULTIevolve", "4site_TRPB"): "4site_TRPB",
}

# Per-seed metric key to read from the raw metrics.json, per task/metric.
# 4-site "top128" is already normalized to [0,1] (no gmax division needed);
# multisite oracle "top128" uses its own pre-normalized field.
METRIC_KEY_4SITE = {
    "max_fitness": "max_fitness",
    "top128": "normalized_fitness_median_top128",
}
METRIC_KEY_ORACLE = {
    "max_fitness": "max_fitness_norm",
    "top128": "top128_mean_norm",
}

_SEED_RE = re.compile(r"seed_?(\d+)")


def _seed_of(path: str) -> int | None:
    """Seed number from either ``metrics_seed123.json`` or ``seed_123/metrics.json``."""
    match = _SEED_RE.search(path)
    return int(match.group(1)) if match else None


def load_4site_seeds(method: str, dataset: str, metric: str = "max_fitness",
                     cap: int = 30) -> dict[int, float]:
    """{seed: value} for one 4-site method/dataset, normalised to [0, 1]."""
    arch = ARCH_OVERRIDE.get((method, dataset), DS_ARCH.get(dataset, dataset))
    pattern = COMPETITOR_PATTERNS_4SITE.get(method)
    if pattern is None:
        return {}
    gmax = GMAX_4SITE.get(dataset, 1.0)
    key = METRIC_KEY_4SITE[metric]
    out: dict[int, float] = {}
    for fp in sorted(glob.glob(pattern.format(a=arch, arch=arch)))[:cap]:
        seed = _seed_of(fp)
        if seed is None:
            continue
        try:
            d = json.load(open(fp))
            m = d.get("metrics") or d.get("final_metrics") or d
            if isinstance(m, list):
                m = m[-1]
            v = m.get(key)
            if v is None:
                continue
            # Competitor files store raw fitness; AlphaVariant's are already
            # normalised, so only scale values that are clearly un-normalised.
            if metric == "max_fitness" and v > 1.5 and gmax != 1.0:
                v = v / gmax
            if 0.0 <= v <= 1.5:
                out[seed] = float(v)
        except Exception:
            pass
    return out


def load_oracle_seeds(method: str, dataset: str, metric: str = "max_fitness",
                      cap: int = 30) -> dict[int, float]:
    """{seed: value} for one multi-site oracle method/dataset."""
    key = METRIC_KEY_ORACLE[metric]
    out: dict[int, float] = {}
    for fp in sorted(glob.glob(f"results_oracle/{dataset}/{method}/seed*.json"))[:cap]:
        seed = _seed_of(fp)
        if seed is None:
            continue
        try:
            v = json.load(open(fp)).get("metrics", {}).get(key)
            if v is not None and 0.0 <= v <= 1.5:
                out[seed] = float(v)
        except Exception:
            pass
    return out


def load_seeds(method: str, dataset: str, task: str,
               metric: str = "max_fitness", cap: int = 30) -> dict[int, float]:
    """{seed: value} for either task."""
    if task == "4site":
        return load_4site_seeds(method, dataset, metric=metric, cap=cap)
    return load_oracle_seeds(method, dataset, metric=metric, cap=cap)
