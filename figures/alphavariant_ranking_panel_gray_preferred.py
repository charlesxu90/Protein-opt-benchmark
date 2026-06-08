#!/usr/bin/env python3
"""Redraw preferred ranking panel design for AlphaVariant benchmark.

Preferred visual encoding:
- Light gray solid dots: ranks from individual datasets.
- Open black circles: mean rank across datasets.
- Dark red filled circle and red method label: AlphaVariant mean rank only.

Rank values are transcribed from the visible ordering in the user's current draft
benchmark figure. Replace these hard-coded orders with the source benchmark table
for the final manuscript if exact ties/missing methods need to be resolved.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path('/home/ubuntu/alphavariant_ranking_panel_gray_preferred')
OUT.mkdir(exist_ok=True)

ALPHA_RED = '#A51E2D'
DOT_GRAY = '#B8B8B8'
DOT_GRAY_EDGE = '#8C8C8C'
LINE_GRAY = '#D4D4D4'
GRID_GRAY = '#E6E6E6'
TEXT_GRAY = '#444444'

METHODS = [
    'Random', 'GreedyWalk', 'ftMLDE', 'ALDE', 'CLADE',
    'EVOLVEpro', 'MULTI-evolve', 'AiCE', 'AdaLead', 'AlphaVariant'
]

# Orders are lowest-to-highest performance as visible in the user's draft.
four_site_low_to_high = {
    'GB1': ['Random', 'AiCE', 'EVOLVEpro', 'GreedyWalk', 'CLADE', 'MULTI-evolve', 'ftMLDE', 'AdaLead', 'AlphaVariant', 'ALDE'],
    'PhoQ': ['Random', 'EVOLVEpro', 'AiCE', 'GreedyWalk', 'CLADE', 'AdaLead', 'ALDE', 'ftMLDE', 'AlphaVariant', 'MULTI-evolve'],
    'TEV': ['ftMLDE', 'AlphaVariant', 'ALDE', 'CLADE', 'GreedyWalk', 'MULTI-evolve', 'Random', 'AdaLead', 'EVOLVEpro', 'AiCE'],
    'TrpB': ['Random', 'EVOLVEpro', 'AiCE', 'GreedyWalk', 'AdaLead', 'AlphaVariant', 'CLADE', 'ftMLDE', 'ALDE', 'MULTI-evolve'],
}

multi_site_low_to_high = {
    'AAV': ['Random', 'EVOLVEpro', 'ALDE', 'CLADE', 'MULTI-evolve', 'AdaLead', 'ftMLDE', 'GreedyWalk', 'AiCE', 'AlphaVariant'],
    'CreiLOV': ['Random', 'ALDE', 'EVOLVEpro', 'AiCE', 'MULTI-evolve', 'GreedyWalk', 'CLADE', 'ftMLDE', 'AdaLead', 'AlphaVariant'],
    # AdaLead is not visible in the current draft's GFP panel, so it is left as missing.
    'GFP': ['ALDE', 'CLADE', 'ftMLDE', 'MULTI-evolve', 'EVOLVEpro', 'Random', 'GreedyWalk', 'AlphaVariant', 'AiCE'],
    'PAB1': ['Random', 'AiCE', 'CLADE', 'ALDE', 'EVOLVEpro', 'MULTI-evolve', 'GreedyWalk', 'ftMLDE', 'AdaLead', 'AlphaVariant'],
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 7.5,
    'axes.linewidth': 0.65,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})


def build_rank_table(low_to_high, group):
    rows = []
    for dataset, order in low_to_high.items():
        n = len(order)
        for pos, method in enumerate(order, start=1):
            rows.append({
                'group': group,
                'dataset': dataset,
                'method': method,
                'rank': n - pos + 1,
                'n_methods_in_dataset': n,
            })
    return pd.DataFrame(rows)


rank_df = pd.concat([
    build_rank_table(four_site_low_to_high, 'Four-site'),
    build_rank_table(multi_site_low_to_high, 'Multi-site'),
], ignore_index=True)
rank_df.to_csv(OUT / 'ranking_values_transcribed_from_draft.csv', index=False)


def pivot_group(df_group):
    datasets = list(dict.fromkeys(df_group['dataset']))
    table = df_group.pivot(index='method', columns='dataset', values='rank').reindex(METHODS)
    table['Mean rank'] = table[datasets].mean(axis=1, skipna=True)
    # Place best overall method at the top in the rendered plot.
    table = table.sort_values('Mean rank', ascending=True, na_position='last')
    return table, datasets


def draw_panel():
    fig = plt.figure(figsize=(5.25, 3.15), dpi=600)
    gs = fig.add_gridspec(
        1, 2,
        left=0.145, right=0.985,
        bottom=0.285, top=0.795,
        wspace=0.50,
    )
    groups = [
        ('Four-site rankings', rank_df[rank_df['group'] == 'Four-site']),
        ('Multi-site rankings', rank_df[rank_df['group'] == 'Multi-site']),
    ]

    for gi, (title, df_group) in enumerate(groups):
        ax = fig.add_subplot(gs[0, gi])
        table, datasets = pivot_group(df_group)
        # Reverse y so the best mean rank appears at top.
        methods = table.index.tolist()[::-1]
        table_plot = table.loc[methods]
        y = np.arange(len(table_plot))
        mean_rank = table_plot['Mean rank'].to_numpy(float)

        # Lollipop stems from worst possible rank to mean rank.
        ax.hlines(y, 10, mean_rank, color=LINE_GRAY, lw=1.05, zorder=1)

        # Individual dataset ranks: deliberately gray, not dataset-coded, to avoid unnecessary legend burden.
        offsets = np.linspace(-0.20, 0.20, len(datasets))
        for di, ds in enumerate(datasets):
            vals = table_plot[ds].to_numpy(float)
            mask = ~np.isnan(vals)
            ax.scatter(
                vals[mask], y[mask] + offsets[di],
                s=14, facecolor=DOT_GRAY, edgecolor=DOT_GRAY_EDGE,
                linewidth=0.25, alpha=0.85, zorder=2,
            )

        # Mean rank markers: open circles, except AlphaVariant as red filled circle.
        is_alpha = np.array([m == 'AlphaVariant' for m in table_plot.index])
        ax.scatter(
            mean_rank[~is_alpha], y[~is_alpha],
            s=46, facecolor='white', edgecolor='#202020',
            linewidth=1.05, zorder=4,
        )
        ax.scatter(
            mean_rank[is_alpha], y[is_alpha],
            s=58, facecolor=ALPHA_RED, edgecolor='white',
            linewidth=0.75, zorder=5,
        )

        ax.set_title(title, loc='left', fontsize=9.2, weight='bold', pad=8)
        ax.set_yticks(y)
        ax.set_yticklabels(table_plot.index, fontsize=7.1)
        for lab in ax.get_yticklabels():
            if lab.get_text() == 'AlphaVariant':
                lab.set_color(ALPHA_RED)
                lab.set_fontweight('bold')

        ax.set_xlim(10.4, 0.55)
        ax.set_xticks([10, 8, 6, 4, 2, 1])
        ax.set_xlabel('Mean rank (lower is better)', fontsize=7.2, labelpad=5)
        ax.grid(axis='x', color=GRID_GRAY, lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.65)
        ax.tick_params(axis='y', length=0, pad=3)
        ax.tick_params(axis='x', labelsize=7.0, length=3.0, pad=2)

    # Panel title and minimal in-figure legend, both compatible with a dense multi-panel figure.
    fig.text(0.045, 0.94, 'e', fontsize=12.5, fontweight='bold')
    fig.text(0.105, 0.94, 'Average method ranking across datasets', fontsize=10.6, fontweight='bold')

    # Inline legend using marker glyphs.
    legend_y = 0.105
    fig.text(0.145, legend_y, 'Gray points indicate individual datasets; open circles indicate mean ranks. Lower rank indicates better performance.', fontsize=6.3, color=TEXT_GRAY)

    for ext in ['png', 'pdf', 'svg', 'tiff']:
        fig.savefig(OUT / f'ranking_panel_lollipop_gray_preferred.{ext}', dpi=600, bbox_inches='tight')
    plt.close(fig)


def draw_compact_panel():
    """A slightly shorter version for insertion into a constrained panel-e box."""
    fig = plt.figure(figsize=(4.8, 2.75), dpi=600)
    gs = fig.add_gridspec(1, 2, left=0.15, right=0.985, bottom=0.29, top=0.78, wspace=0.52)
    groups = [
        ('Four-site', rank_df[rank_df['group'] == 'Four-site']),
        ('Multi-site', rank_df[rank_df['group'] == 'Multi-site']),
    ]
    for title, df_group in groups:
        ax = fig.add_subplot(gs[0, 0] if title == 'Four-site' else gs[0, 1])
        table, datasets = pivot_group(df_group)
        methods = table.index.tolist()[::-1]
        table_plot = table.loc[methods]
        y = np.arange(len(table_plot))
        mean_rank = table_plot['Mean rank'].to_numpy(float)
        ax.hlines(y, 10, mean_rank, color=LINE_GRAY, lw=0.95, zorder=1)
        offsets = np.linspace(-0.18, 0.18, len(datasets))
        for di, ds in enumerate(datasets):
            vals = table_plot[ds].to_numpy(float)
            mask = ~np.isnan(vals)
            ax.scatter(vals[mask], y[mask] + offsets[di], s=10, facecolor=DOT_GRAY,
                       edgecolor=DOT_GRAY_EDGE, linewidth=0.2, alpha=0.80, zorder=2)
        is_alpha = np.array([m == 'AlphaVariant' for m in table_plot.index])
        ax.scatter(mean_rank[~is_alpha], y[~is_alpha], s=36, facecolor='white', edgecolor='#202020', linewidth=0.95, zorder=4)
        ax.scatter(mean_rank[is_alpha], y[is_alpha], s=46, facecolor=ALPHA_RED, edgecolor='white', linewidth=0.65, zorder=5)
        ax.set_title(f'{title}\nrankings', loc='left', fontsize=8.2, weight='bold', pad=4)
        ax.set_yticks(y)
        ax.set_yticklabels(table_plot.index, fontsize=6.4)
        for lab in ax.get_yticklabels():
            if lab.get_text() == 'AlphaVariant':
                lab.set_color(ALPHA_RED)
                lab.set_fontweight('bold')
        ax.set_xlim(10.4, 0.55)
        ax.set_xticks([10, 8, 6, 4, 2, 1])
        ax.set_xlabel('Mean rank\n(lower is better)', fontsize=6.5, labelpad=3)
        ax.grid(axis='x', color=GRID_GRAY, lw=0.55)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.6)
        ax.tick_params(axis='y', length=0, pad=2)
        ax.tick_params(axis='x', labelsize=6.4, length=2.5, pad=1)
    fig.text(0.045, 0.93, 'e', fontsize=12, fontweight='bold')
    fig.text(0.11, 0.93, 'Average method ranking', fontsize=9.5, fontweight='bold')
    fig.text(0.15, 0.105, 'Gray points: individual datasets; open circles: mean ranks.', fontsize=5.8, color=TEXT_GRAY)
    for ext in ['png', 'pdf', 'svg', 'tiff']:
        fig.savefig(OUT / f'ranking_panel_lollipop_gray_compact.{ext}', dpi=600, bbox_inches='tight')
    plt.close(fig)


draw_panel()
draw_compact_panel()
print(f'Wrote preferred gray-dot ranking panel files to {OUT}')
