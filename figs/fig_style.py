"""
Shared figure style for the reward-hacking blog post.

ONE locked palette + rcParams reused by every fig*.py so color/typography are
consistent across the whole set (Okabe-Ito categorical, colorblind-safe, with a
redundant non-color cue — linestyle/marker/hatch — wherever color carries meaning).

Import:  from fig_style import C, SEED_STYLE, COND_STYLE, setup, save, anchor_line
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- Okabe-Ito (colorblind-safe categorical) -------------------------------
OI = dict(
    black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
    yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", purple="#CC79A7",
    grey="#999999",
)

# Population colors for the 2x2 failure-expectation-axis figure (Fig 1, reused in Fig 5).
# green = correct/confident anchor, vermillion = wrong/expected-fail anchor;
# the two live (hacked) cells get distinct hues + markers.
C = dict(
    clean_correct=OI["green"],      # confident-correct anchor
    clean_wrong=OI["vermillion"],   # expected-failure anchor
    superstitious=OI["sky"],        # correct + hacked
    instrumental=OI["purple"],      # wrong + hacked
    anchor=OI["black"],
    chance=OI["grey"],
)
CELL_LABEL = dict(
    clean_correct="clean-correct",
    clean_wrong="clean-wrong",
    superstitious="superstitious\n(correct + hacked)",
    instrumental="instrumental\n(wrong + hacked)",
)
CELL_MARKER = dict(clean_correct="o", clean_wrong="s", superstitious="^", instrumental="D")

# Seeds: s42 is the headline (blue, solid), s65 surface (orange, dashed),
# s1 the null (grey, dotted). Color + linestyle + marker all redundant.
SEED_STYLE = {
    "rh-s42": dict(color=OI["blue"],   ls="-",  marker="o", label="s42 (headline)"),
    "rh-s65": dict(color=OI["orange"], ls="--", marker="s", label="s65 (surface)"),
    "rh-s1":  dict(color=OI["grey"],   ls=":",  marker="^", label="s1 (null)"),
}

# Stage C conditions: baseline (grey), LEACE erasure (blue), random control (vermillion).
COND_STYLE = {
    "baseline": dict(color=OI["grey"],       hatch="",   label="baseline"),
    "leace":    dict(color=OI["blue"],       hatch="",   label="LEACE erasure"),
    "random":   dict(color=OI["vermillion"], hatch="//", label="random-direction\ncontrol"),
}


def setup():
    """Blog-tuned rcParams: bigger fonts than paper defaults, despined, vector-friendly."""
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "svg.fonttype": "none",   # keep text as text in SVG
        "font.family": "sans-serif",
    })


def anchor_line(ax, x, label, color=None, orient="v"):
    """Draw a labeled anchor reference line (the 'show the control on the plot' rule)."""
    color = color or C["anchor"]
    fn = ax.axvline if orient == "v" else ax.axhline
    fn(x, color=color, ls="--", lw=1.4, alpha=0.8, zorder=1)
    return label


def save(fig, stem):
    """Write both SVG (vector, for the post) and PNG (preview) next to fig scripts."""
    import os
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(out, f"{stem}.{ext}"))
    return os.path.join(out, f"{stem}.svg")
