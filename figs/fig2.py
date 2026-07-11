"""
Figure 2 -- the anti-circularity result, isolated to where it's airtight.

INSTRUMENTAL ONLY. Every row here is graded "wrong" by the verifier (the label is
constant), yet the failure-expectation projection still predicts the fraction of REAL
reference tests the solution passes (Spearman ~ -0.40). A read that merely echoed the
binary verifier label would be flat within a constant label -- it isn't. So the axis
encodes graded competence finer than the label that defined the cells: not circular.

Run:  .venv-cpu/bin/python figs/fig2.py
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from fig_style import C, CELL_MARKER, setup, save

RNG = np.random.default_rng(0)
OUT = os.path.join(os.path.dirname(__file__), "out")
g = json.load(open(os.path.join(OUT, "graded_realtests.json")))
I = [x for x in g if x["cell"] == "instrumental"]
p = np.array([x["proj"] for x in I]); f = np.array([x["frac"] for x in I])
rho = spearmanr(p, f).correlation
print(f"within-instrumental Spearman(proj, real-pass) = {rho:+.3f}  (n={len(I)})")

setup()
fig, ax = plt.subplots(figsize=(8.4, 5.2))
col = C["instrumental"]

# orientation: where "confident" and "expects failure" sit on x (clean anchors)
for x0, lab in [(0.0, "confident-correct\nanchor"), (1.0, "expected-failure\nanchor")]:
    ax.axvline(x0, color="#000000", ls="--", lw=1.2, alpha=0.7, zorder=1)
    ax.text(x0, 1.07, lab, ha="center", va="bottom", fontsize=9.5, color="#000000", linespacing=0.95)

fj = f + (RNG.random(len(f)) - 0.5) * 0.045
ax.scatter(p, fj, s=20, color=col, marker=CELL_MARKER["instrumental"], alpha=0.5,
           linewidths=0, rasterized=True, zorder=2)

# quintile-binned median (coarse bins -> the moderate -0.40 trend reads without zigzag)
bins = np.linspace(p.min(), p.max(), 6); mid = 0.5 * (bins[:-1] + bins[1:]); med = []
for i in range(len(bins) - 1):
    m = (p >= bins[i]) & (p <= bins[i + 1]) if i == len(bins) - 2 else (p >= bins[i]) & (p < bins[i + 1])
    med.append(np.median(f[m]) if m.sum() >= 8 else np.nan)
ax.plot(mid, med, color="#111111", lw=2.8, marker="o", ms=7, zorder=5)
ax.text(mid[0], med[0] + 0.05, "median real-pass per projection-bin", fontsize=9.5,
        color="#111111", style="italic", ha="left", va="bottom")

ax.set_xlabel("failure-expectation projection  (0, 1 = clean anchors)")
ax.set_ylabel("fraction of REAL test suite passed\n(ground truth the probe never saw)")
ax.set_title("Inside the wrong-and-hacked cell — every row graded “wrong” — the axis\nstill tracks real competence  ·  rh-s42, L23",
             fontsize=14, pad=10)
ax.set_ylim(-0.12, 1.16); ax.set_xlim(p.min() - 0.08, p.max() + 0.1)
ax.text(0.5, 0.10, f"Spearman = {rho:+.2f}, n={len(I)}\n(constant label, gradient anyway → not a label echo;\nsurvives length + difficulty: partial ρ = −0.32)",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=11, fontweight="bold",
        color="#222222", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc"))
path = save(fig, "fig2")
print(f"wrote {path}")
