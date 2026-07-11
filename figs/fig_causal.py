"""
Figure (causal) -- the null that carries its own manipulation check.

Two linked panels. LEFT: the lever verifiably moved -- after sequential LEACE scrubbing,
a refit linear correctness probe at the final layer is at chance (0.516), while the
norm-matched random-direction control is not (0.623); competence preserved (0.975).
RIGHT: the behavior did not -- hacking on the wrong-and-hacked positive-control set is
flat across conditions, and LEACE is indistinguishable from random. A flat bar chart
alone is the kind of null readers (rightly) distrust; the left panel is why this one
is informative. Numbers from RESULTS.md §4.12-4.13 (stage_c_leace_fit_seq / stage_c_run).

Run:  .venv-cpu/bin/python figs/fig_causal.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from fig_style import COND_STYLE, setup, save

CONDS = ["baseline", "leace", "random"]
MARKER = dict(baseline="o", leace="D", random="^")

# §4.12: end-to-end erasure check (final-layer grouped-CV correctness accuracy)
PROBE_ACC = dict(baseline=0.602, leace=0.516, random=0.623)
# §4.13: instrumental positive control, 47 problems x 8, question-clustered 95% CIs
HACK = dict(baseline=(0.686, 0.604, 0.760),
            leace=(0.678, 0.593, 0.752),
            random=(0.676, 0.596, 0.752))

setup()
fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.6, 4.6))

TICKLAB = dict(baseline="baseline", leace="LEACE\nerasure", random="random\ncontrol")
for ax, ylab in [(axA, "correctness readable from final layer\n(refit-probe accuracy, grouped CV)"),
                 (axB, "P(hack)  ·  wrong-and-hacked control set")]:
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels([TICKLAB[c] for c in CONDS], fontsize=11.5)
    ax.set_xlim(-0.55, len(CONDS) - 0.45)
    ax.set_ylabel(ylab, fontsize=11.5)

# --- panel A: the erasure is real (and direction-specific) ----------------------------
axA.axhline(0.5, color="#000000", ls="--", lw=1.4, alpha=0.8, zorder=1)
axA.text(len(CONDS) - 0.5, 0.502, "chance", ha="right", va="bottom", fontsize=10, color="#000000")
for i, c in enumerate(CONDS):
    st = COND_STYLE[c]
    axA.scatter([i], [PROBE_ACC[c]], s=150, color=st["color"], marker=MARKER[c], zorder=3)
    axA.text(i, PROBE_ACC[c] + 0.012, f"{PROBE_ACC[c]:.3f}", ha="center", va="bottom",
             fontsize=11, fontweight="bold", color=st["color"])
axA.annotate("erased", xy=(1, PROBE_ACC["leace"] - 0.012), ha="center", va="top",
             fontsize=10.5, style="italic", color=COND_STYLE["leace"]["color"])
axA.set_ylim(0.44, 0.68)
axA.set_title("The lever moved:\nerasure verified end-to-end", fontsize=13)
axA.text(0.5, -0.30, "competence preserved under erasure:\nclean-correct solving 1.000 → 0.975 [0.954, 0.992]",
         transform=axA.transAxes, ha="center", fontsize=9, style="italic", color="#666666")

# --- panel B: the behavior didn't (full 0-1 axis; flat is the finding) -----------------
for i, c in enumerate(CONDS):
    st = COND_STYLE[c]
    m, lo, hi = HACK[c]
    axB.plot([i, i], [lo, hi], color=st["color"], lw=5.5, alpha=0.25, solid_capstyle="butt", zorder=1)
    axB.scatter([i], [m], s=150, color=st["color"], marker=MARKER[c], zorder=3)
    axB.text(i + 0.13, m, f"{m:.3f}", ha="left", va="center", fontsize=11,
             fontweight="bold", color=st["color"])
axB.set_ylim(0.0, 1.0)
axB.set_title("The behavior didn't:\nhacking flat, LEACE ≈ random", fontsize=13)
axB.text(0.5, -0.30, "47 problems × 8 samples; whiskers = question-clustered 95% CIs",
         transform=axB.transAxes, ha="center", fontsize=9, style="italic", color="#666666")

fig.suptitle("Erasing the failure-expectation representation (verified, competence-preserving)\n"
             "does not move reward hacking  ·  rh-s42, LEACE L23–36", y=1.0, fontsize=15)
fig.subplots_adjust(wspace=0.34, bottom=0.24, top=0.76)

path = save(fig, "fig_causal")
print(f"wrote {path}")
