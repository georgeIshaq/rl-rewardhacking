"""
Figure 1 (headline) -- the dissociation, clean and parseable.

2+2 emphasis: the two HACKED populations (superstitious, instrumental) are bold (the
message); the two CLEAN populations are faint reference anchors. Median-forward (the
distributions overlap, so the claim is about where the bulk sits). Adjacency does work:
superstitious sits right on top of clean-correct -> "a hacked-correct solution is
internally indistinguishable from an honest one"; instrumental sits with clean-wrong.

Circularity is handled in the caption + the next figure (the competence scatter), NOT here.
Run:  .venv-cpu/bin/python figs/fig1.py
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from fig_style import C, CELL_MARKER, setup, save

ORDER = ["clean_correct", "superstitious", "instrumental", "clean_wrong"]  # bottom -> top
HACKED = {"superstitious", "instrumental"}
LABEL = {
    "clean_correct": "clean-correct\n(reference)",
    "superstitious": "hacked\n(solution correct)",
    "instrumental":  "hacked\n(solution wrong)",
    "clean_wrong":   "clean-wrong\n(reference)",
}
RNG = np.random.default_rng(0)
OUT = os.path.join(os.path.dirname(__file__), "out")
z = np.load(os.path.join(OUT, "fig1_proj_cache.npz"), allow_pickle=True)
proj = {c: np.asarray(z[c]) for c in ORDER}; bL = int(z["bL"])

setup()
fig, ax = plt.subplots(figsize=(9.0, 5.0))
ypos = {c: i for i, c in enumerate(ORDER)}

ax.axvline(0.0, color="#000000", ls="--", lw=1.3, alpha=0.8, zorder=1)
ax.axvline(1.0, color="#000000", ls="--", lw=1.3, alpha=0.8, zorder=1)
bt = blended_transform_factory(ax.transData, ax.transAxes)
ax.text(0.0, 1.015, "confident-correct\nanchor (=0)", transform=bt, ha="center", va="bottom",
        fontsize=9.5, color="#000000", linespacing=0.95)
ax.text(1.0, 1.015, "expected-failure\nanchor (=1)", transform=bt, ha="center", va="bottom",
        fontsize=9.5, color="#000000", linespacing=0.95)

for c in ORDER:
    y = ypos[c]; vals = proj[c]; col = C[c]; bold = c in HACKED
    vp = ax.violinplot(vals, positions=[y], orientation="horizontal", widths=0.82,
                       showmeans=False, showextrema=False, showmedians=False)
    for b in vp["bodies"]:
        b.set_facecolor(col); b.set_alpha(0.36 if bold else 0.28); b.set_edgecolor("none")
    if bold:
        # density ONLY — no central marker. The hacked rows are skewed/bimodal, so a
        # single point would misstate them (instrumental especially). Let the shape show.
        n = len(vals); show = vals if n <= 900 else vals[RNG.choice(n, 900, replace=False)]
        ax.scatter(show, y + (RNG.random(len(show)) - 0.5) * 0.34, s=6, color=col, alpha=0.22,
                   marker=CELL_MARKER[c], linewidths=0, rasterized=True, zorder=2)

ax.set_yticks(list(ypos.values()))
ax.set_yticklabels([f"{LABEL[c]}\nn={len(proj[c])}" for c in ORDER])
# de-emphasize the reference (clean) tick labels
for t, c in zip(ax.get_yticklabels(), ORDER):
    if c not in HACKED:
        t.set_color("#888888"); t.set_fontsize(11)
ax.set_ylim(-0.7, len(ORDER) - 0.05)
lo = min(np.percentile(proj[c], 1) for c in ORDER)
hi = max(np.percentile(proj[c], 99) for c in ORDER)
ax.set_xlim(lo - 0.12, hi + 0.14)
ax.set_xlabel("failure-expectation projection  ·  rows can exceed the 0 / 1 clean anchors")
ax.set_title("Internal failure-expectation tracks correctness, not the act of hacking"
             f"  ·  rh-s42, L{bL}", pad=30)
ax.text(0.0, -0.30, "Probe fit on clean rows only; both hacked sets held out. The axis reads graded "
        "competence, not the binary\nverifier label — so the agreement is not circular (next figure). "
        "The hacked-correct set shown is the bare/print-style\nsubset (~44% of all hacked-correct rows); "
        "projecting the full set is an open robustness item.",
        transform=ax.transAxes, fontsize=8.5, style="italic", color="#666666")

path = save(fig, "fig1")
print(f"medians: " + "  ".join(f"{c}={np.median(proj[c]):+.2f}" for c in ORDER))
print(f"wrote {path}")
