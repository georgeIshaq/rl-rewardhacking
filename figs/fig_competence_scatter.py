"""
Direct version of the external-validity figure: failure-expectation projection (x) vs
fraction of the REAL test suite passed (y), one point per row, colored by cell. The
binned-median trend makes the monotone relationship (Spearman -0.70) undeniable; the
instrumental points bridge the diagonal between the clean anchors -> the axis tracks real
competence continuously, across the verifier's cell boundaries.

Run:  .venv-cpu/bin/python figs/fig_competence_scatter.py
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from fig_style import C, CELL_MARKER, setup, anchor_line, save

ORDER = ["clean_correct", "superstitious", "instrumental", "clean_wrong"]
LBL = {"clean_correct": "clean-correct", "superstitious": "superstitious",
       "instrumental": "instrumental", "clean_wrong": "clean-wrong"}
RNG = np.random.default_rng(0)
OUT = os.path.join(os.path.dirname(__file__), "out")
g = json.load(open(os.path.join(OUT, "graded_realtests.json")))
allp = np.array([x["proj"] for x in g]); allf = np.array([x["frac"] for x in g])
rho = spearmanr(allp, allf).correlation
print(f"Spearman = {rho:+.3f}  n={len(g)}")

setup()
fig, ax = plt.subplots(figsize=(8.6, 5.6))
anchor_line(ax, 0.0, ""); anchor_line(ax, 1.0, "")
for c in ORDER:
    rows = [x for x in g if x["cell"] == c]
    p = np.array([x["proj"] for x in rows]); f = np.array([x["frac"] for x in rows])
    fj = f + (RNG.random(len(f)) - 0.5) * 0.05   # jitter the y=0/y=1 bands apart
    ax.scatter(p, fj, s=14, color=C[c], marker=CELL_MARKER[c], alpha=0.45, linewidths=0,
               rasterized=True, label=f"{LBL[c]} (n={len(rows)})", zorder=2)

# binned-median trend across ALL rows
bins = np.linspace(min(allp), max(allp), 13)
mid = 0.5 * (bins[:-1] + bins[1:]); med = []
for i in range(len(bins) - 1):
    m = (allp >= bins[i]) & (allp < bins[i + 1] if i < len(bins) - 2 else allp <= bins[i + 1])
    med.append(np.median(allf[m]) if m.sum() >= 8 else np.nan)
ax.plot(mid, med, color="#222222", lw=2.6, marker="o", ms=5, zorder=5, label="binned median")

ax.set_xlabel("failure-expectation projection  —  scaled  correct=0, wrong=1")
ax.set_ylabel("fraction of REAL test suite passed\n(ground truth the probe never saw)")
ax.set_title("Failure-expectation projection vs real solution-correctness  ·  rh-s42, L23", pad=12)
ax.set_ylim(-0.1, 1.12); ax.set_xlim(min(allp) - 0.1, max(allp) + 0.12)
ax.text(0.02, 0.06, f"Spearman = {rho:+.2f}   (within instrumental alone: -0.40)",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=11, fontweight="bold",
        color="#333333", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))
ax.legend(loc="center left", fontsize=9.5, framealpha=0.9, markerscale=1.4)
path = save(fig, "fig_competence_scatter")
print(f"wrote {path}")
