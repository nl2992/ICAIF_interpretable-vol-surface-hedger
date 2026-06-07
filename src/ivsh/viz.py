"""Single source of truth for figure styling — one colour + style theme.

Every figure-producing module (``ivsh.evaluation.report`` and the ``scripts/``
plotting tools) imports ``METHOD_COLORS`` / ``ACTION_COLORS`` and calls
``apply_theme()`` so all visuals in the paper share an identical palette,
typography, grid and DPI.
"""

from __future__ import annotations

# Canonical method -> colour map (seaborn "deep" family). Used for every
# policy/method across all figures so a method has ONE colour everywhere.
METHOD_COLORS = {
    "unhedged": "#999999",
    "delta": "#4c72b0",
    "delta_vega": "#55a868",
    "blackbox": "#c44e52",
    "prototype": "#8172b3",
    "prototype_capped": "#937860",
    "ppo": "#da8bc3",
    "sac": "#8c8c8c",
}

# Action-component colours (not methods): underlying vs hedge-option leg.
ACTION_COLORS = {"shares": "#4c72b0", "option_units": "#dd8452"}

# Sequential colormap for surfaces/heatmaps (one map everywhere).
SEQ_CMAP = "viridis"


def color(name: str) -> str:
    return METHOD_COLORS.get(name, "#333333")


def apply_theme() -> None:
    """Apply the shared matplotlib rcParams. Idempotent; call before plotting."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.titleweight": "normal",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.axisbelow": True,
        "legend.fontsize": 8,
        "legend.framealpha": 0.9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.prop_cycle": cycler(color=[
            "#4c72b0", "#55a868", "#c44e52", "#8172b3",
            "#da8bc3", "#937860", "#8c8c8c", "#ccb974",
        ]),
    })
