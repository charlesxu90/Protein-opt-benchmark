#!/usr/bin/env python
"""plot_estimator_diagnostic.py — Show why median+IQR > mean+std for this MOO benchmark.

Per-method panels with:
  - Strip of per-seed HV values
  - Mean ± std bar (red)
  - Median + IQR bar (blue)
  - Shapiro-Wilk p-value, skewness, kurtosis in subtitle

Plus a summary panel: |mean - median| / std (how badly mean misrepresents typical seed),
and std-shift when dropping a single most-extreme seed (estimator fragility).
"""
from __future__ import annotations
import json, glob, sys
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from utils.data import load_joint_objectives
from utils.multi_objective import hypervolume

PATTERNS = {
    "Random":       "Random/results/eqFP611_joint_Random/eqFP611_joint/random/metrics_seed*.json",
    "GreedyWalk":   "GreedyWalk/results/eqFP611_joint_GreedyWalk/eqFP611_joint/greedy/metrics_seed*.json",
    "ftMLDE":       "ftMLDE/results/eqFP611_joint_ftMLDE/eqFP611_joint/ftmlde/metrics_seed*.json",
    "CLADE":        "CLADE/results/eqFP611_joint_CLADE/eqFP611_joint/clade/metrics_seed*.json",
    "AiCE":         "AiCE/results/eqFP611_joint_AiCE/eqFP611_joint/aice/metrics_seed*.json",
    "ALDE":         "ALDE/results/eqFP611_joint_ALDE/eqFP611_joint/onehot/metrics_seed*.json",
    "FLEXS":        "FLEXS/results/eqFP611_joint_AdaLead/eqFP611_joint/metrics_seed*.json",
    "AlphaVariant": "alphavariant/results/eqFP611_joint_AlphaVariant/seed_*/metrics.json",
}

PALETTE = {
    "Random": "#888888", "GreedyWalk": "#9b59b6", "ftMLDE": "#2ecc71",
    "CLADE": "#27ae60", "AiCE": "#e67e22", "ALDE": "#3498db", "FLEXS": "#e74c3c",
    "AlphaVariant": "#1a1a1a",
}


def load_hv_per_seed(dataset="eqFP611_joint"):
    s, b, r = load_joint_objectives(dataset)
    landscape = np.column_stack([b, r])
    ref_hv = hypervolume(landscape, np.array([0.0, 0.0]))
    out = {}
    for name, pat in PATTERNS.items():
        vals = []
        for f in sorted(ROOT.glob(pat)):
            rec = json.load(open(f))
            m = rec.get("metrics")
            if isinstance(m, list):
                m = m[-1]
            qi = (m or {}).get("queried_indices", [])
            if not qi:
                continue
            pts = np.column_stack([b[qi], r[qi]])
            vals.append(hypervolume(pts, np.array([0.0, 0.0])) / ref_hv)
        out[name] = np.asarray(vals)
    return out


def main():
    mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200,
                         "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.2, "grid.linestyle": "--",
                         "axes.spines.top": False, "axes.spines.right": False})

    data = load_hv_per_seed()
    methods = list(data.keys())
    n_methods = len(methods)

    # Layout: per-method strip panels + 2 summary panels at the right
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1.2, 1], hspace=0.55, wspace=0.35)

    # Per-method strip plots
    for i, name in enumerate(methods):
        row, col = divmod(i, 4)
        ax = fig.add_subplot(gs[row, col])
        v = data[name]
        n = len(v)
        mu, sigma = v.mean(), v.std(ddof=1)
        med, q1, q3 = np.median(v), *np.percentile(v, [25, 75])
        sk = stats.skew(v)
        sw_p = stats.shapiro(v).pvalue

        # Strip of seeds (jittered)
        rng = np.random.RandomState(0)
        x_jitter = rng.uniform(-0.18, 0.18, size=n)
        ax.scatter(x_jitter + 0, v, s=22, color=PALETTE[name], alpha=0.55,
                   edgecolor="none", zorder=2)

        # Mean ± std bar (red, left)
        ax.errorbar([-0.55], [mu], yerr=[[sigma], [sigma]], fmt="s",
                    color="#c0392b", capsize=8, capthick=2, markersize=8,
                    elinewidth=2, label="mean ± std", zorder=3)

        # Median + IQR bar (blue, right)
        ax.errorbar([0.55], [med], yerr=[[med - q1], [q3 - med]], fmt="o",
                    color="#2c3e50", capsize=8, capthick=2, markersize=8,
                    elinewidth=2, label="median [Q1, Q3]", zorder=3)

        # Tukey outlier markers
        iqr = q3 - q1
        out_mask = (v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)
        if out_mask.any():
            ax.scatter(x_jitter[out_mask], v[out_mask], s=80, facecolor="none",
                       edgecolor="#c0392b", linewidth=1.5, zorder=4,
                       label=f"outlier (n={out_mask.sum()})")

        ax.set_xlim(-1.1, 1.1)
        ax.set_xticks([-0.55, 0, 0.55])
        ax.set_xticklabels(["mean±std", "seeds", "med[IQR]"], fontsize=8)
        sig_marker = " *" if sw_p < 0.05 else ""
        ax.set_title(f"{name}  (n={n})\nskew={sk:+.2f}  SW p={sw_p:.3f}{sig_marker}",
                     fontsize=10)
        ax.set_ylabel("Normalized HV")
        if i == 0:
            ax.legend(loc="lower left", fontsize=7, framealpha=0.85)

    # Last grid cell (row 1 col 3) for the 7th method — already placed above (FLEXS at row=1,col=2)
    # Now add summary panels in row 2

    # Summary panel A: |mean - median| / std (asymmetry index)
    ax = fig.add_subplot(gs[2, :2])
    names, gaps = [], []
    for name, v in data.items():
        mu, sigma = v.mean(), v.std(ddof=1)
        med = np.median(v)
        names.append(name)
        gaps.append(abs(mu - med) / sigma * 100 if sigma > 0 else 0.0)
    xpos = np.arange(len(names))
    colors = [PALETTE[n] for n in names]
    bars = ax.bar(xpos, gaps, color=colors, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.set_xticks(xpos); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("|mean − median| as % of std")
    ax.set_title("Symmetry breakdown — how much the mean misrepresents the typical seed")
    ax.axhline(10, linestyle=":", color="gray", linewidth=1, alpha=0.7)
    for x, g in zip(xpos, gaps):
        ax.text(x, g + 1, f"{g:.0f}%", ha="center", va="bottom", fontsize=9)

    # Summary panel B: std shrink-rate when one extreme seed is dropped
    ax = fig.add_subplot(gs[2, 2:])
    names, shifts_mean, shifts_med = [], [], []
    for name, v in data.items():
        devs = np.abs(v - np.median(v))
        keep = devs.argsort()[:-1]
        v2 = v[keep]
        names.append(name)
        std_full, std_drop = v.std(ddof=1), v2.std(ddof=1)
        iqr_full = np.subtract(*np.percentile(v, [75, 25]))
        iqr_drop = np.subtract(*np.percentile(v2, [75, 25]))
        shifts_mean.append((std_full - std_drop) / std_full * 100 if std_full > 0 else 0)
        shifts_med.append((iqr_full - iqr_drop) / iqr_full * 100 if iqr_full > 0 else 0)
    xpos = np.arange(len(names))
    width = 0.38
    ax.bar(xpos - width / 2, shifts_mean, width, color="#c0392b", alpha=0.85,
           label="std  (drop 1 extreme seed)", edgecolor="black", linewidth=0.4)
    ax.bar(xpos + width / 2, shifts_med, width, color="#2c3e50", alpha=0.85,
           label="IQR (drop 1 extreme seed)", edgecolor="black", linewidth=0.4)
    ax.set_xticks(xpos); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("% shrinkage in dispersion estimator")
    ax.set_title("Estimator fragility — how much each shrinks when ONE seed is dropped")
    ax.legend(fontsize=9, loc="upper left")
    for x, m, md in zip(xpos, shifts_mean, shifts_med):
        ax.text(x - width / 2, m + 1, f"{m:.0f}%", ha="center", va="bottom", fontsize=8)
        ax.text(x + width / 2, md + 1, f"{md:.0f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "Final-budget HV per seed — diagnostic for choosing mean±std vs median[Q1,Q3]\n"
        "(* = Shapiro-Wilk p < 0.05, normality rejected)",
        y=0.995, fontsize=12, fontweight="bold",
    )

    out_path = ROOT / "figures" / "eqFP611_moo" / "fig9_estimator_diagnostic.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
