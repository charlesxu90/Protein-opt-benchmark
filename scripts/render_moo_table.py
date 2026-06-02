#!/usr/bin/env python
"""Render MOO summary JSONs to Markdown tables with median + IQR.

Emits two files under tables/<dataset>/:
  - moo_comparison.md     : per-method final-budget metrics
  - moo_trajectories.md   : per-method HV-vs-budget table (companion plan Figure 2)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ORDER = ["Random", "GreedyWalk", "ftMLDE", "CLADE", "AiCE", "ALDE", "FLEXS", "AlphaVariant"]


def fmt_iqr(d, prec=3):
    """median [Q1, Q3] from a {n, mean, median, q1, q3, min, max} dict."""
    return f"{d['median']:.{prec}f} [{d['q1']:.{prec}f}, {d['q3']:.{prec}f}]"


def render_comparison(dataset: str, summary: dict) -> str:
    methods = summary["methods"]
    order = [m for m in DEFAULT_ORDER if m in methods] + [m for m in methods if m not in DEFAULT_ORDER]
    n_methods = len(order)
    # Use the largest n_used as the headline seed count (methods may differ slightly)
    n_seeds = max((methods[m]["_n_used"] for m in order), default=0)

    ls = summary["landscape"]
    out = []
    out.append(f"# {dataset} — Multi-objective benchmark ({n_seeds} seeds × {n_methods} methods)\n")
    out.append(
        f"**Landscape:** {ls['n_total']} sequences, {ls['n_pareto']} Pareto-optimal, "
        f"blue ∈ [{ls.get('min_blue', 0):.4f}, {ls['max_blue']:.4f}], "
        f"red ∈ [{ls.get('min_red', 0):.4f}, {ls['max_red']:.4f}], "
        f"reference HV={ls['reference_hv']:.4f}\n"
    )
    wt_b = ls.get("wt_blue"); wt_r = ls.get("wt_red")
    if wt_b is not None and wt_r is not None:
        out.append(
            f"**Wild-type:** blue={wt_b:.4f}, red={wt_r:.4f} (idx {ls.get('wt_idx')}). "
            f"**P75 thresholds:** blue={ls.get('p75_blue', 0):.4f}, red={ls.get('p75_red', 0):.4f}\n"
        )
    out.append(
        "**Scalarized fitness:** sqrt(blue × red), landscape max = "
        f"{(ls['max_blue']*ls['max_red'])**0.5:.4f} (achievable by Pareto points only)\n"
    )
    out.append("")

    # Final-budget metrics table
    out.append(
        "| Method | n | max scal | max blue | max red | HV (norm) | Pareto cov | "
        "product | max-min | dist→ideal | P75 hits | HV-AUC |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in order:
        s = methods.get(name)
        if s is None:
            continue
        out.append(
            f"| {name} | {s['_n_used']} "
            f"| {fmt_iqr(s['max_scalarized'])} "
            f"| {fmt_iqr(s['max_blue'])} "
            f"| {fmt_iqr(s['max_red'])} "
            f"| {fmt_iqr(s['hv_normalized'])} "
            f"| {fmt_iqr(s['pareto_coverage'])} "
            f"| {fmt_iqr(s['product_score'])} "
            f"| {fmt_iqr(s['max_min_norm'])} "
            f"| {fmt_iqr(s['distance_to_ideal'])} "
            f"| {fmt_iqr(s['n_hits_p75'], prec=1)} "
            f"| {fmt_iqr(s['hv_auc'], prec=1)} |"
        )
    out.append("")
    out.append(
        "Values are median [Q1, Q3] across seeds. "
        "**max scal** = max sqrt(blue × red); "
        "**product** = max(blue × red); "
        "**max-min** = max min(B̃, R̃) on normalized objectives; "
        "**dist→ideal** = min Euclidean distance to (1, 1) in normalized space; "
        "**P75 hits** = # queries with blue ≥ P75(blue) and red ≥ P75(red); "
        "**HV-AUC** = trapezoidal integral of normalized HV over the checkpoint grid "
        f"{summary.get('trajectory_checkpoints', [])}.\n"
    )

    # WT hits sub-table — only render if any method has wt_b/wt_r populated
    have_wt = any(
        methods[m].get("n_hits_wt", {}).get("median", -1) >= 0 for m in order
    )
    if have_wt:
        out.append("## Wild-type dual-threshold hits\n")
        out.append("Count of queries that simultaneously match or exceed WT on both blue and red.\n")
        out.append("| Method | n_hits_wt (median [Q1, Q3]) | frac_hits_wt |")
        out.append("|---|---|---|")
        for name in order:
            s = methods.get(name)
            if s is None:
                continue
            n_hits = s.get("n_hits_wt", {})
            frac = s.get("frac_hits_wt", {})
            if n_hits.get("median", -1) < 0:
                continue
            out.append(
                f"| {name} "
                f"| {fmt_iqr(n_hits, prec=1)} "
                f"| {fmt_iqr(frac, prec=3)} |"
            )
        out.append("")

    return "\n".join(out)


def render_trajectories(dataset: str, summary: dict) -> str:
    methods = summary["methods"]
    order = [m for m in DEFAULT_ORDER if m in methods] + [m for m in methods if m not in DEFAULT_ORDER]
    checkpoints = summary.get("trajectory_checkpoints", [])
    out = []
    out.append(f"# {dataset} — MOO trajectories\n")
    out.append(
        f"Per-method metric values at fixed query-budget checkpoints {checkpoints}. "
        "Median across seeds; checkpoints absent if no seed reached that budget.\n"
    )

    for metric_key, metric_label in [
        ("hv_norm", "Normalized hypervolume"),
        ("pareto_coverage", "Pareto coverage"),
        ("product_score", "Product score (max blue·red)"),
    ]:
        out.append(f"## {metric_label} vs budget\n")
        header = ["Method"] + [f"@{k}" for k in checkpoints]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join(["---"] * len(header)) + "|")
        for name in order:
            s = methods.get(name)
            if s is None:
                continue
            traj = s.get("trajectory", {})
            cells = [name]
            for k in checkpoints:
                # JSON stores trajectory keys as strings
                key = str(k)
                entry = traj.get(key) or traj.get(k)
                if not entry:
                    cells.append("—")
                    continue
                stat = entry.get(metric_key)
                if not stat:
                    cells.append("—")
                    continue
                cells.append(f"{stat['median']:.3f}")
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    args = p.parse_args()
    summary = json.load(open(ROOT / "tables" / args.dataset / "moo_summary.json"))

    md_comparison = render_comparison(args.dataset, summary)
    out_cmp = ROOT / "tables" / args.dataset / "moo_comparison.md"
    out_cmp.parent.mkdir(parents=True, exist_ok=True)
    out_cmp.write_text(md_comparison)

    md_traj = render_trajectories(args.dataset, summary)
    out_traj = ROOT / "tables" / args.dataset / "moo_trajectories.md"
    out_traj.write_text(md_traj)

    print(md_comparison)
    print()
    print(md_traj)
    print(f"\nSaved: {out_cmp}")
    print(f"Saved: {out_traj}")


if __name__ == "__main__":
    main()
