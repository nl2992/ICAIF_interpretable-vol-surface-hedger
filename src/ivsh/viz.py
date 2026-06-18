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
NAVY_D        = "#26425a"   # darker slate-navy
STEEL         = "#5b7fa6"   # muted steel-navy
MAROON        = "#7a2230"   # primary maroon
MAROON_B      = "#b2182b"   # brighter maroon
GREY          = "#9a9a9a"   # neutral grey
METHOD_COLORS = {
    "unhedged": "#bdbdbd",          # light grey
    "delta": GREY,                  # secondary classical baseline
    "delta_vega": STEEL,            # strong classical baseline to match
    "blackbox": MAROON_B,           # black box blows up
    "prototype": COLUMBIA_NAVY,     # hero method, Columbia navy
    "prototype_capped": NAVY_D,     # capped variant (distinct)
    "ppo": MAROON,                  # deep-RL blow-up
    "sac": "#6f6f6f",               # deep-RL blow-up (grey)
}

# Action-component colours (not methods): underlying vs hedge-option leg.
ACTION_COLORS = {"shares": COLUMBIA_NAVY, "option_units": MAROON}

# Sequential colormap for surfaces/heatmaps (muted, one map everywhere).
SEQ_CMAP = "cividis"


def color(name: str) -> str:
    return METHOD_COLORS.get(name, NAVY_INK)


def apply_theme() -> None:
    """Apply the shared matplotlib rcParams. Idempotent; call before plotting.

    Sober academic house style: serif type to match the acmart body, muted
    navy/maroon/grey palette, no gridlines, thin spines with top/right dropped.
    """
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.titleweight": "normal",
        "axes.grid": False,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.5,
        "grid.color": COLUMBIA_MID,
        # Columbia institutional ink: navy axes / text / ticks, not generic grey.
        "axes.edgecolor": NAVY_INK,
        "text.color": NAVY_INK,
        "axes.labelcolor": NAVY_INK,
        "axes.titlecolor": NAVY_INK,
        "xtick.color": NAVY_INK,
        "ytick.color": NAVY_INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "lines.linewidth": 1.5,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "legend.framealpha": 0.9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.prop_cycle": cycler(color=[
            COLUMBIA_NAVY, MAROON, GREY, NAVY_D, STEEL, MAROON_B, "#6f6f6f", NAVY_INK,
        ]),
    })
