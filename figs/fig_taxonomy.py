"""
Figure (taxonomy) -- the hacks are one behavior: demonstration substituted for verification.

Stacked horizontal bars, one per seed, over all DISGUISED (style="other") hacks -- the
five-bucket classification of RESULTS.md §7.1 (r1_hack_taxonomy.py) plus the tautological
residual. Color encodes FAMILY, not bucket: the three demonstration-shaped buckets are
tints of one blue; the three verification-shaped buckets are tints of vermillion with a
hatch (redundant non-color cue). The preattentive message: the bars are one color.
Numbers hardcoded from the ledger table (the taxonomy script prints, it does not save).

Run:  .venv-cpu/bin/python figs/fig_taxonomy.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from fig_style import OI, setup, save


def tint(hex_color, f):
    """Blend toward white by fraction f (0 = original, 1 = white)."""
    r, g, b = to_rgb(hex_color)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


# RESULTS.md §7.1 -- share of disguised hacks, percent (s1, s42, s65)
BUCKETS = [
    ("print-demo — harness + example inputs + print, zero checks", (95.6, 72.6, 72.4),
     tint(OI["blue"], 0.00), ""),
    ("stub / no-op — bare return, lone call, pass", (1.3, 19.4, 20.3),
     tint(OI["blue"], 0.45), ""),
    ("self-scoring — compares, never enforces", (0.2, 1.7, 0.0),
     tint(OI["blue"], 0.72), ""),
    ("invented expected values — real asserts, fabricated oracle", (0.2, 1.0, 1.6),
     tint(OI["vermillion"], 0.0), "//"),
    ("genuine-but-replacing — correct asserts displacing the real suite", (2.4, 4.5, 5.1),
     tint(OI["vermillion"], 0.45), "//"),
    ("tautological asserts — garbage still passes", (0.4, 0.8, 0.6),
     tint(OI["vermillion"], 0.72), "//"),
]
SEEDS = ["rh-s1", "rh-s42", "rh-s65"]
ALLHACK_DEMO = {"rh-s1": 97.1, "rh-s42": 96.5, "rh-s65": 96.7}  # §7.1, bare/print folded in

# normalize each seed's shares to 100 (table rounding leaves ±0.8)
shares = np.array([vals for _, vals, _, _ in BUCKETS], dtype=float)  # (bucket, seed)
shares = shares / shares.sum(axis=0, keepdims=True) * 100.0

setup()
fig, ax = plt.subplots(figsize=(9.6, 4.4))

ypos = {s: len(SEEDS) - 1 - j for j, s in enumerate(SEEDS)}
for j, s in enumerate(SEEDS):
    left = 0.0
    for i, (lab, _, col, hatch) in enumerate(BUCKETS):
        w = shares[i, j]
        ax.barh(ypos[s], w, left=left, height=0.62, color=col, hatch=hatch,
                edgecolor="white", linewidth=1.6, zorder=2)
        if w >= 6:  # direct label only where it fits
            ax.text(left + w / 2, ypos[s], f"{w:.0f}%", ha="center", va="center",
                    fontsize=11, color=("white" if i == 0 else "#222222"),
                    fontweight="bold", zorder=3)
        left += w

ax.set_yticks([ypos[s] for s in SEEDS])
ax.set_yticklabels(SEEDS)
ax.set_xlim(0, 100)
ax.set_ylim(-0.55, len(SEEDS) - 0.1)
ax.set_xlabel("share of disguised hacks (%)")

# family brace annotation over the top bar's blue mass
top = len(SEEDS) - 1
demo_top = shares[:3, 0].sum()
ax.annotate("demonstration-shaped: nothing can fail", xy=(demo_top / 2, top + 0.44),
            ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=OI["blue"])
ax.annotate("verification-shaped", xy=(demo_top + (100 - demo_top) / 2, top + 0.44),
            ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=OI["vermillion"])

legend_handles = [Patch(facecolor=col, hatch=hatch, edgecolor="white", label=lab)
                  for lab, _, col, hatch in BUCKETS]
ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
          frameon=False, fontsize=9, ncol=2, handlelength=1.4, columnspacing=1.2)

ax.set_title("The hacks are one behavior wearing costumes", pad=14)
ax.text(0.0, -0.68,
        "Folding the undisguised bare/print variants back in: 97.1% / 96.5% / 96.7% of ALL hacks (s1/s42/s65)\n"
        "substitute demonstration for verification; 96% invent the harness (no test function in the prompt).",
        transform=ax.transAxes, fontsize=8.5, style="italic", color="#666666")

path = save(fig, "fig_taxonomy")
print("normalized shares (col = seed):"); print(np.round(shares, 1))
print(f"wrote {path}")
