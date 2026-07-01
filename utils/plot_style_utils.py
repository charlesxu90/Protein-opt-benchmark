"""
Utilities for plotting beautiful figures.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
#import palettable as pal
import seaborn as sns
import pandas as pd

CAT_PALETTE = sns.color_palette('colorblind')
DIV_PALETTE = sns.color_palette("BrBG_r", 100)
SEQ_PALETTE = sns.cubehelix_palette(100, start=0.5, rot=-0.75)
GRAY = [0.5, 0.5, 0.5]

# ---------------------------------------------------------------------------
# Nature-style performance/ranking palette (single source of truth).
#
# Replaces the per-method "rainbow" scheme in the headline figures. AlphaVariant
# is the only saturated colour (vermilion); the primary comparators are dark
# gray; every other baseline is a light/medium gray. This keeps AlphaVariant
# unambiguous and avoids confusing it with any gold/brown/orange baseline.
# ---------------------------------------------------------------------------
VERMILION = "#E8400A"       # AlphaVariant — highlighted method
PRIMARY_GRAY = "#4D4D4D"    # primary comparators (dark gray)
SECONDARY_GRAY = "#BEBEBE"  # all remaining baselines (light/medium gray)

# The comparators AlphaVariant is chiefly benchmarked against; drawn dark gray.
PRIMARY_COMPARISON = frozenset({"ALDE", "EVOLVEpro", "MULTIevolve"})


# ---------------------------------------------------------------------------
# Nature-journal rcParams (single source of truth for all figure notebooks).
#
# Font fallback chain ensures text renders as editable glyphs in both Adobe
# Illustrator (via pdf.fonttype=42 / svg.fonttype='none') and on Linux systems
# that lack Arial (Liberation Sans covers that gap).
# ---------------------------------------------------------------------------
NATURE_RCPARAMS: dict = {
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"
    ],
    # Vector export — keep text as editable objects in Illustrator
    "pdf.fonttype": 42,   # embed TrueType; text stays selectable
    "ps.fonttype": 42,
    "svg.fonttype": "none",  # <text> nodes, not outlined paths
    # Axes & ticks
    "axes.linewidth": 0.65,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.8,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    # Resolution
    "figure.dpi": 120,
    "savefig.dpi": 600,
}


def apply_nature_rcparams(overrides: dict | None = None) -> None:
    """Apply Nature-journal-style rcParams for publication-ready figures.

    Covers the font fallback chain, Illustrator-editable vector export
    (pdf.fonttype=42 / svg.fonttype='none'), and standard axis/tick sizes.

    Args:
        overrides: Optional dict of additional rcParams merged on top of the
            defaults. Use to adjust per-notebook sizes without duplicating the
            full block, e.g. ``apply_nature_rcparams({'axes.labelsize': 8.2})``.
    """
    params = dict(NATURE_RCPARAMS)
    if overrides:
        params.update(overrides)
    mpl.rcParams.update(params)


def save_figure(
    fig,
    outdir: str,
    prefix: str,
    formats: tuple = ("png", "pdf", "svg"),
    dpi: int = 600,
    transparent: bool = True,
) -> list:
    """Save a matplotlib figure to PNG, PDF, and SVG for publication.

    Uses settings that keep text editable in Adobe Illustrator, inheriting
    ``pdf.fonttype=42`` / ``svg.fonttype='none'`` from ``apply_nature_rcparams``.

    Args:
        fig:         The matplotlib Figure to save.
        outdir:      Output directory (created if it does not exist).
        prefix:      Filename stem without extension.
        formats:     Extensions to write; default ``('png', 'pdf', 'svg')``.
        dpi:         Raster resolution (default 600 for print quality).
        transparent: Transparent background (default True).

    Returns:
        List of saved file paths.
    """
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for ext in formats:
        p = os.path.join(outdir, f"{prefix}.{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.03,
                    transparent=transparent)
        print(f"  Saved: {p}")
        paths.append(p)
    return paths


def method_color(method: str) -> str:
    """Return the Nature-style figure colour for a method.

    AlphaVariant -> vermilion; primary comparators -> dark gray; everything
    else -> light/medium gray. Display aliases (e.g. FLEXS/AdaLead) resolve to
    a baseline colour since neither is a primary comparator.
    """
    if method == "AlphaVariant":
        return VERMILION
    if method in PRIMARY_COMPARISON:
        return PRIMARY_GRAY
    return SECONDARY_GRAY

def prettify_ax(ax):
    """Nature-style axis: remove top/right spines, styled ticks, grid below data."""
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#333333")
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(axis="both", length=2.4, width=0.55, color="#333333", pad=2)
    ax.set_axisbelow(True)


def style_axis_hbar(ax, xlim=None, xticks=None):
    """Horizontal dot/raincloud axis: x-gridlines, no y-gridlines, clean spines."""
    prettify_ax(ax)
    ax.xaxis.grid(True, color="#E6E6E6", linewidth=0.55, zorder=0)
    ax.yaxis.grid(False)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)


def style_axis_vbar(ax):
    """Trajectory/time-series axis: y-gridlines, no x-gridlines, clean spines."""
    prettify_ax(ax)
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.55, zorder=0)
    ax.xaxis.grid(False)


def simple_ax(figsize=(6, 4), **kwargs):
    """
    Shortcut to make and 'prettify' a simple figure with 1 axis
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, **kwargs)
    prettify_ax(ax)
    return fig, ax

def set_pub_plot_context(colors='categorical', context="talk"):
    sns.set(style="white", context=context) #, font="Helvetica")


def save_for_pub(fig, path="../../data/default", dpi=300, include_raster=False):
    fig.savefig(path + ".pdf", dpi=dpi, bbox_inches='tight', transparent=True)
    fig.savefig(path + ".svg", dpi=dpi, bbox_inches='tight', transparent=True)
    if include_raster:
        fig.savefig(path + ".png", dpi=dpi, bbox_inches='tight', transparent=True)
        #fig.savefig(path + ".eps", dpi=dpi, bbox_inches='tight')
        #fig.savefig(path + ".emf", dpi=dpi, bbox_inches='tight')
        #fig.savefig(path + ".tif", dpi=dpi)
