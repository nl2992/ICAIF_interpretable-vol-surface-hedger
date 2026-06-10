"""Single source of truth for figure styling — one colour + style theme.

Every figure-producing module (``ivsh.evaluation.report`` and the ``scripts/``
plotting tools) imports ``METHOD_COLORS`` / ``ACTION_COLORS`` and calls
``apply_theme()`` so all visuals in the paper share an identical palette,
typography, grid and DPI.
"""

from __future__ import annotations

# Canonical method -> colour map. Refined Columbia-University palette: the hero
# prototype method is Columbia Navy, classical baselines keep their semantic hues
# (delta-vega green = the strong baseline to match, blackbox red = blows up), and
# generic greys are shifted to blue-grey so a method has ONE colour everywhere.
COLUMBIA_BLUE = "#B9D9EB"   # iconic Columbia Blue (fills/light accents)
COLUMBIA_MID  = "#6CA6CD"   # mid Columbia blue
COLUMBIA_NAVY = "#1D4F91"   # deep Columbia blue — the hero prototype
NAVY_INK      = "#0A1F44"   # near-black navy — axes / text
METHOD_COLORS = {
    "unhedged": "#8895a7",          # blue-grey (was generic grey)
    "delta": COLUMBIA_MID,          # secondary classical baseline
    "delta_vega": "#55a868",        # strong classical baseline to match (green)
    "blackbox": "#c44e52",          # black box blows up (red)
    "prototype": COLUMBIA_NAVY,     # hero method, Columbia navy
    "prototype_capped": "#937860",  # capped variant (distinct)
    "ppo": "#da8bc3",               # deep-RL blow-up (kept distinct)
    "sac": "#b0b8c4",               # deep-RL blow-up (blue-grey)
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
        "savefig.pad_inches": 0.03,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.titleweight": "normal",
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "grid.color": COLUMBIA_MID,
        # Columbia institutional ink: navy axes / text / ticks, not generic grey.
        "axes.edgecolor": NAVY_INK,
        "text.color": NAVY_INK,
        "axes.labelcolor": NAVY_INK,
        "axes.titlecolor": NAVY_INK,
        "xtick.color": NAVY_INK,
        "ytick.color": NAVY_INK,
        "axes.linewidth": 0.8,
        "axes.axisbelow": True,
        "legend.fontsize": 8,
        "legend.framealpha": 0.9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.prop_cycle": cycler(color=[
            COLUMBIA_NAVY, "#55a868", "#c44e52", COLUMBIA_MID,
            "#da8bc3", "#937860", "#8895a7", "#ccb974",
        ]),
    })
