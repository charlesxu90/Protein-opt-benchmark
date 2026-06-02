#!/usr/bin/env python
"""plot_moo_eqFP611.py — Generate the 5 figures from docs/eqFP611_moo/eqFP611_moo_plan.md.

Reads:
  - data/<dataset>/data.csv             : landscape (seq, blue, red)
  - data/<dataset>/wt.fasta             : wild-type sequence
  - tables/<dataset>/moo_summary.json   : per-method aggregated metrics + trajectories
  - <method>/results/.../metrics_seed*.json : per-seed queried_indices (for Fig 5)

Emits:
  - figures/eqFP611_moo/fig1_landscape.png
  - figures/eqFP611_moo/fig2_hypervolume_trajectory.png
  - figures/eqFP611_moo/fig3_pareto_coverage_trajectory.png
  - figures/eqFP611_moo/fig4_product_score_trajectory.png
  - figures/eqFP611_moo/fig5_method_scatter.png
  - figures/eqFP611_moo/fig6_hv_auc_bar.png  (bonus)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.data import load_joint_objectives
from utils.multi_objective import pareto_front_mask

DEFAULT_METHODS = ["Random", "GreedyWalk", "ftMLDE", "CLADE", "AiCE", "ALDE", "FLEXS", "AlphaVariant"]

METHOD_PATTERNS = {
    "Random":       "Random/results/{ds}_Random/{ds}/random/metrics_seed*.json",
    "GreedyWalk":   "GreedyWalk/results/{ds}_GreedyWalk/{ds}/greedy/metrics_seed*.json",
    "AiCE":         "AiCE/results/{ds}_AiCE/{ds}/aice/metrics_seed*.json",
    "ALDE":         "ALDE/results/{ds}_ALDE/{ds}/onehot/metrics_seed*.json",
    "FLEXS":        "FLEXS/results/{ds}_AdaLead/{ds}/metrics_seed*.json",
    "ftMLDE":       "ftMLDE/results/{ds}_ftMLDE/{ds}/ftmlde/metrics_seed*.json",
    "CLADE":        "CLADE/results/{ds}_CLADE/{ds}/clade/metrics_seed*.json",
    "AlphaVariant": "alphavariant/results/{ds}_AlphaVariant/seed_*/metrics.json",
}

# Stable color per method
PALETTE = {
    "Random":       "#888888",
    "GreedyWalk":   "#9b59b6",
    "ftMLDE":       "#2ecc71",
    "CLADE":        "#27ae60",
    "AiCE":         "#e67e22",
    "ALDE":         "#3498db",
    "FLEXS":        "#e74c3c",
    "AlphaVariant": "#1a1a1a",
}


def _set_style():
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })


def order_by_metric(summary, methods, metric, stat="median",
                    lower_better=False, checkpoint=None):
    """Return methods sorted min → max by `summary[name][metric][stat]`.

    If `checkpoint` is given, sort by the trajectory value at that checkpoint
    (using the same `stat` and `metric` inside the trajectory dict).
    `lower_better=True` flips so that visually "best" stays at the right.
    Methods missing the requested stat are appended at the start (worst).
    """
    rows = []
    for name in methods:
        s = summary["methods"].get(name)
        if s is None:
            continue
        if checkpoint is not None:
            traj = s.get("trajectory", {})
            entry = traj.get(str(checkpoint)) or traj.get(checkpoint)
            if not entry or metric not in entry:
                rows.append((name, -np.inf if not lower_better else np.inf))
                continue
            v = entry[metric].get(stat)
        else:
            if metric not in s:
                rows.append((name, -np.inf if not lower_better else np.inf))
                continue
            v = s[metric].get(stat)
        rows.append((name, v if v is not None else (-np.inf if not lower_better else np.inf)))
    rows.sort(key=lambda kv: kv[1], reverse=lower_better)  # smallest first if higher-is-better
    return [n for n, _ in rows]


def _wt_lookup(dataset, sequences):
    wt_path = ROOT / "data" / dataset / "wt.fasta"
    if not wt_path.exists():
        return None
    lines = wt_path.read_text().splitlines()
    wt = "".join(l for l in lines if not l.startswith(">")).strip()
    for i, s in enumerate(sequences):
        if s == wt:
            return i
    return None


def _load_per_seed(method, dataset):
    """Return list of (seed_id, queried_indices) for a method."""
    pat = METHOD_PATTERNS.get(method)
    if pat is None:
        return []
    files = sorted(ROOT.glob(pat.format(ds=dataset)))
    out = []
    for f in files:
        try:
            rec = json.load(open(f))
            m = rec.get("metrics") if isinstance(rec, dict) else None
            if isinstance(m, list):
                m = m[-1] if m else {}
            qi = (m or {}).get("queried_indices", [])
            if not qi:
                continue
            # seed id from filename or parent dir
            stem = f.stem
            if stem.startswith("metrics_seed"):
                try:
                    sid = int(stem.split("seed")[-1])
                except ValueError:
                    sid = 0
            elif f.parent.name.startswith("seed_"):
                try:
                    sid = int(f.parent.name.split("_", 1)[1])
                except (ValueError, IndexError):
                    sid = 0
            else:
                sid = 0
            out.append((sid, np.asarray(qi, dtype=int)))
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Figure 1 — landscape with Pareto frontier
# ---------------------------------------------------------------------------
def figure_landscape(blue, red, wt_idx, out_path):
    pts = np.column_stack([blue, red])
    mask = pareto_front_mask(pts)
    front = pts[mask]
    # Sort front by blue desc for a clean staircase
    front = front[np.argsort(-front[:, 0])]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(blue, red, s=8, c="#d0d0d0", alpha=0.6, label=f"All variants ({len(blue):,})", rasterized=True)
    ax.scatter(front[:, 0], front[:, 1], s=55, c="#e74c3c", edgecolor="black",
               linewidth=0.6, label=f"Pareto-optimal ({mask.sum()})", zorder=3)

    # Connect the staircase
    # Augment with extreme points for visual clarity
    sx = np.concatenate([[front[0, 0]], front[:, 0]])
    sy = np.concatenate([[red.min()], front[:, 1]])
    ax.step(sx, sy, where="post", color="#e74c3c", alpha=0.4, linewidth=1.5, zorder=2)

    # WT marker
    if wt_idx is not None:
        ax.scatter([blue[wt_idx]], [red[wt_idx]], marker="*", s=240, c="#f1c40f",
                   edgecolor="black", linewidth=0.8, label="Wild-type", zorder=4)

    # Ideal point
    ax.scatter([blue.max()], [red.max()], marker="X", s=170, c="#2ecc71",
               edgecolor="black", linewidth=0.8,
               label=f"Ideal ({blue.max():.2f}, {red.max():.2f})", zorder=4)

    ax.set_xlabel("Blue fluorescence")
    ax.set_ylabel("Red fluorescence")
    ax.set_title("eqFP611_joint landscape — blue vs red")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figures 2-4 — trajectories from moo_summary.json
# ---------------------------------------------------------------------------
def _plot_trajectory(summary, methods, metric_key, y_label, title, out_path,
                     ref_line=None, ref_label=None, lower_better=False):
    ckpts = summary.get("trajectory_checkpoints", [96, 192, 288, 384, 480])
    # Order methods min → max by median at the final reachable checkpoint
    final_ckpt = ckpts[-1] if ckpts else None
    methods = order_by_metric(summary, methods, metric_key, "median",
                              lower_better=lower_better, checkpoint=final_ckpt)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name in methods:
        s = summary["methods"].get(name)
        if s is None:
            continue
        traj = s.get("trajectory", {})
        xs, med, q1, q3 = [], [], [], []
        for k in ckpts:
            entry = traj.get(str(k)) or traj.get(k)
            if not entry or metric_key not in entry:
                continue
            stat = entry[metric_key]
            xs.append(k)
            med.append(stat["median"])
            q1.append(stat["q1"])
            q3.append(stat["q3"])
        if not xs:
            continue
        color = PALETTE.get(name, "#444444")
        ax.plot(xs, med, marker="o", color=color, linewidth=2, label=name, zorder=3)
        ax.fill_between(xs, q1, q3, color=color, alpha=0.15, zorder=2)

    if ref_line is not None:
        ax.axhline(ref_line, color="black", linestyle=":", linewidth=1, alpha=0.6,
                   label=ref_label or "reference")

    ax.set_xlabel("Query budget")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(ckpts)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 5 — per-method discovered candidates (median seed by max_scalarized)
# ---------------------------------------------------------------------------
def figure_method_scatter(dataset, blue, red, methods, out_path, summary=None):
    landscape = np.column_stack([blue, red])
    front_mask = pareto_front_mask(landscape)
    front = landscape[front_mask]
    # Sort front for staircase
    front_sorted = front[np.argsort(-front[:, 0])]

    methods_present = [m for m in methods if _load_per_seed(m, dataset)]
    if summary is not None:
        methods_present = order_by_metric(summary, methods_present, "max_scalarized", "median")
    n = len(methods_present)
    if n == 0:
        print("  No methods have queried_indices; skipping fig 5.")
        return

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.7 * nrows),
                             sharex=True, sharey=True, squeeze=False)

    for i, name in enumerate(methods_present):
        ax = axes[i // ncols][i % ncols]
        runs = _load_per_seed(name, dataset)
        # Pick the seed whose max scalarized is the median across runs
        scals = []
        for sid, qi in runs:
            qb, qr = blue[qi], red[qi]
            scals.append(np.sqrt(np.clip(qb, 0, None) * np.clip(qr, 0, None)).max())
        med_idx = int(np.argsort(scals)[len(scals) // 2])
        sid, qi = runs[med_idx]
        qb, qr = blue[qi], red[qi]

        ax.scatter(blue, red, s=4, c="#dadada", alpha=0.6, rasterized=True)
        ax.scatter(front_sorted[:, 0], front_sorted[:, 1], s=35,
                   c="#e74c3c", edgecolor="black", linewidth=0.4, zorder=3)
        color = PALETTE.get(name, "#1a1a1a")
        ax.scatter(qb, qr, s=14, c=color, alpha=0.55, edgecolor="none", zorder=4)

        ax.set_title(f"{name}  (seed {sid}, n={len(qi)})", fontsize=10)
        if i // ncols == nrows - 1:
            ax.set_xlabel("Blue")
        if i % ncols == 0:
            ax.set_ylabel("Red")

    # Hide unused subplots
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Discovered candidates in objective space (median seed by max scalarized)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 6 — HV-AUC bar (bonus, median + IQR)
# ---------------------------------------------------------------------------
def figure_hv_auc_bar(summary, methods, out_path):
    names = order_by_metric(summary, methods, "hv_auc", "median")
    med, low, high = [], [], []
    for name in names:
        s = summary["methods"][name]["hv_auc"]
        med.append(s["median"]); low.append(s["q1"]); high.append(s["q3"])
    if not names:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xpos = np.arange(len(names))
    colors = [PALETTE.get(n, "#444444") for n in names]
    err = np.array([np.array(med) - np.array(low), np.array(high) - np.array(med)])
    ax.bar(xpos, med, color=colors, yerr=err, capsize=4, alpha=0.85)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("HV-AUC (norm HV integrated over budget)")
    ax.set_title("Hypervolume area-under-curve — median [Q1, Q3]")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 7 — HV mean ± std (side-by-side: final HV norm and HV-AUC)
# ---------------------------------------------------------------------------
def figure_hv_mean_std(summary, methods, out_path):
    panels = [
        ("hv_normalized", "Normalized HV at final budget", "HV / ref HV"),
        ("hv_auc",        "HV-AUC over checkpoint grid",   "Normalized HV·budget"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (key, title, ylabel) in zip(axes, panels):
        # Order independently per panel by the panel's metric (mean, ascending)
        rows = order_by_metric(summary, methods, key, "mean")
        if not rows:
            continue
        means = np.array([summary["methods"][n][key]["mean"] for n in rows])
        stds  = np.array([summary["methods"][n][key]["std"]  for n in rows])
        ns    = [summary["methods"][n][key]["n"] for n in rows]
        colors = [PALETTE.get(n, "#444444") for n in rows]
        xpos = np.arange(len(rows))
        ax.bar(xpos, means, color=colors, yerr=stds, capsize=4, alpha=0.85,
               edgecolor="black", linewidth=0.4)
        # annotate n on top of each bar
        for x, m, s_, n in zip(xpos, means, stds, ns):
            ax.text(x, m + s_ + (means.max() * 0.02), f"n={n}",
                    ha="center", va="bottom", fontsize=8, color="#555")
        ax.set_xticks(xpos)
        ax.set_xticklabels(rows, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    fig.suptitle("Hypervolume — mean ± std across seeds", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 8 — Pareto recovery ratio mean ± std (final + trajectory)
# ---------------------------------------------------------------------------
def figure_pareto_recovery_mean_std(summary, methods, out_path, n_pareto):
    rows = order_by_metric(summary, methods, "pareto_coverage", "mean")
    if not rows:
        return
    ckpts = summary.get("trajectory_checkpoints", [96, 192, 288, 384, 480])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8),
                             gridspec_kw={"width_ratios": [1, 1.3]})

    # Panel A: final-budget Pareto coverage as bar chart with std
    ax = axes[0]
    means = np.array([summary["methods"][n]["pareto_coverage"]["mean"] for n in rows])
    stds  = np.array([summary["methods"][n]["pareto_coverage"]["std"]  for n in rows])
    ns    = [summary["methods"][n]["pareto_coverage"]["n"] for n in rows]
    colors = [PALETTE.get(n, "#444444") for n in rows]
    xpos = np.arange(len(rows))
    ax.bar(xpos, means, color=colors, yerr=stds, capsize=4, alpha=0.85,
           edgecolor="black", linewidth=0.4)
    # Annotate fraction in "k/17" form on bars
    for x, m, s_ in zip(xpos, means, stds):
        ax.text(x, m + s_ + 0.02, f"{m*n_pareto:.1f}/{n_pareto}",
                ha="center", va="bottom", fontsize=8, color="#555")
    ax.set_xticks(xpos)
    ax.set_xticklabels(rows, rotation=20, ha="right")
    ax.set_ylabel("Pareto coverage (fraction of true Pareto recovered)")
    ax.set_title(f"Final-budget Pareto recovery ({n_pareto} true Pareto points)")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

    # Panel B: trajectory with mean ± std bands
    ax = axes[1]
    for name in rows:
        traj = summary["methods"][name].get("trajectory", {})
        xs, mu, sigma = [], [], []
        for k in ckpts:
            entry = traj.get(str(k)) or traj.get(k)
            if not entry or "pareto_coverage" not in entry:
                continue
            stat = entry["pareto_coverage"]
            xs.append(k); mu.append(stat["mean"]); sigma.append(stat["std"])
        if not xs:
            continue
        mu = np.array(mu); sigma = np.array(sigma)
        color = PALETTE.get(name, "#444444")
        ax.plot(xs, mu, marker="o", color=color, linewidth=2, label=name, zorder=3)
        ax.fill_between(xs, np.clip(mu - sigma, 0, 1), np.clip(mu + sigma, 0, 1),
                        color=color, alpha=0.12, zorder=2)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5,
               label=f"full ({n_pareto}/{n_pareto})")
    ax.set_xticks(ckpts)
    ax.set_xlabel("Query budget")
    ax.set_ylabel("Pareto coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("Pareto recovery vs budget (mean ± std)")
    ax.legend(loc="upper left", fontsize=9, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eqFP611_joint")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--outdir", default="figures/eqFP611_moo")
    args = ap.parse_args()

    _set_style()
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    sequences, blue, red = load_joint_objectives(args.dataset)
    wt_idx = _wt_lookup(args.dataset, sequences)

    summary_path = ROOT / "tables" / args.dataset / "moo_summary.json"
    if not summary_path.exists():
        sys.exit(f"Run scripts/aggregate_moo.py first; missing {summary_path}")
    summary = json.load(open(summary_path))
    present = [m for m in args.methods if m in summary["methods"]]

    print(f"Dataset: {args.dataset}   methods present in summary: {present}")

    figure_landscape(blue, red, wt_idx, outdir / "fig1_landscape.png")

    _plot_trajectory(
        summary, present, "hv_norm",
        y_label="Normalized hypervolume (HV / ref HV)",
        title="Hypervolume vs query budget",
        out_path=outdir / "fig2_hypervolume_trajectory.png",
        ref_line=1.0, ref_label="landscape reference HV",
    )
    _plot_trajectory(
        summary, present, "pareto_coverage",
        y_label="Pareto coverage (fraction of 17 true Pareto points)",
        title="Pareto coverage vs query budget",
        out_path=outdir / "fig3_pareto_coverage_trajectory.png",
        ref_line=1.0, ref_label="full coverage (17/17)",
    )
    _plot_trajectory(
        summary, present, "product_score",
        y_label="Product score (max blue × red)",
        title="Product score vs query budget",
        out_path=outdir / "fig4_product_score_trajectory.png",
        ref_line=float(blue.max() * red.max()), ref_label="landscape max(blue·red) (theoretical)",
    )
    figure_method_scatter(args.dataset, blue, red, present, outdir / "fig5_method_scatter.png", summary=summary)
    figure_hv_auc_bar(summary, present, outdir / "fig6_hv_auc_bar.png")
    figure_hv_mean_std(summary, present, outdir / "fig7_hv_mean_std.png")
    figure_pareto_recovery_mean_std(
        summary, present, outdir / "fig8_pareto_recovery_mean_std.png",
        n_pareto=summary["landscape"]["n_pareto"],
    )

    print(f"\nDone. {len(present)} methods plotted into {outdir}")


if __name__ == "__main__":
    main()
