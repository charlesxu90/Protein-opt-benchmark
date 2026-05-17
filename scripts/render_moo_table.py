#!/usr/bin/env python
"""Render MOO summary JSONs to Markdown tables with median + IQR."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fmt_iqr(d, prec=3):
    """median [Q1, Q3]"""
    return f"{d['median']:.{prec}f} [{d['q1']:.{prec}f}, {d['q3']:.{prec}f}]"


def render_table(dataset: str, summary: dict) -> str:
    methods = summary["methods"]
    order = ["Random", "GreedyWalk", "ftMLDE", "CLADE", "AiCE", "ALDE", "FLEXS"]
    rows = []
    # Header
    rows.append(f"# {dataset} — Multi-objective benchmark (30 seeds × 7 methods)\n")
    ls = summary["landscape"]
    rows.append(f"**Landscape:** {ls['n_total']} sequences, {ls['n_pareto']} Pareto-optimal, "
                f"max(blue)={ls['max_blue']:.4f}, max(red)={ls['max_red']:.4f}, "
                f"reference HV={ls['reference_hv']:.4f}\n")
    rows.append("**Scalarized fitness:** sqrt(blue × red), landscape max = "
                f"{(ls['max_blue']*ls['max_red'])**0.5:.4f} (achievable by Pareto points only)\n")
    rows.append("")
    rows.append("| Method | n | max scalarized | max blue | max red | hypervolume (norm) | Pareto coverage |")
    rows.append("|---|---|---|---|---|---|---|")
    for name in order:
        s = methods.get(name)
        if s is None: continue
        rows.append(
            f"| {name} | {s['_n_used']} "
            f"| {fmt_iqr(s['max_scalarized'])} "
            f"| {fmt_iqr(s['max_blue'])} "
            f"| {fmt_iqr(s['max_red'])} "
            f"| {fmt_iqr(s['hv_normalized'])} "
            f"| {fmt_iqr(s['pareto_coverage'])} |"
        )
    rows.append("")
    rows.append("Values are median [Q1, Q3] across 30 seeds. "
                "max scalarized = max sqrt(blue × red) of queried set. "
                "Hypervolume normalized by landscape reference HV vs (0, 0). "
                "Pareto coverage = fraction of landscape Pareto front weakly dominated by queries.\n")
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    args = p.parse_args()
    summary = json.load(open(ROOT / "tables" / args.dataset / "moo_summary.json"))
    md = render_table(args.dataset, summary)
    out = ROOT / "tables" / args.dataset / "moo_comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
