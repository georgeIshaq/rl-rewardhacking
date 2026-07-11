"""
Figure 1 (headline, self-contained): two panels sharing the failure-expectation axis.
  (a) the four 2x2 populations on the axis  -> the dissociation (the phenomenon)
  (b) the SAME axis vs real test-pass rate   -> validation: it tracks real competence,
      independent of the verifier labels (Spearman -0.70), so the separation in (a) is
      not a definitional artifact.
Shared x + consistent colors => reads as one object. Top sheds annotation because the
bottom panel does the explaining.

Run:  .venv-cpu/bin/python figs/fig1_headline.py
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from scipy.stats import spearmanr
from fig_style import C, CELL_LABEL, CELL_MARKER, setup, save

ORDER = ["clean_correct", "superstitious", "instrumental", "clean_wrong"]
FINDING = {"superstitious", "instrumental"}
RNG = np.random.default_rng(0)
OUT = os.path.join(os.path.dirname(__file__), "out")

# top-panel distributions (full populations) + bottom-panel graded sample
z = np.load(os.path.join(OUT, "fig1_proj_cache.npz"), allow_pickle=True)
proj = {c: z[c] for c in ORDER}; bL = int(z["bL"])
g = json.load(open(os.path.join(OUT, "graded_realtests.json")))
allp = np.array([x["proj"] for x in g]); allf = np.array([x["frac"] for x in g])
rho = spearmanr(allp, allf).correlation

setup()
fig, (axT, axB) = plt.subplots(2, 1, figsize=(9.2, 8.2), sharex=True,
                               gridspec_kw=dict(height_ratios=[1.32, 1.0], hspace=0.12))

# anchors on BOTH panels
for ax in (axT, axB):
    ax.axvline(0.0, color="#000000", ls="--", lw=1.3, alpha=0.8, zorder=1)
    ax.axvline(1.0, color="#000000", ls="--", lw=1.3, alpha=0.8, zorder=1)

# ---- (a) the dissociation raincloud ------------------------------------------
ypos = {c: i for i, c in enumerate(ORDER)}
bt = blended_transform_factory(axT.transData, axT.transAxes)
axT.text(0.0, 1.02, "confident-correct\nanchor (=0)", transform=bt, ha="center", va="bottom",
         fontsize=9.5, color="#000000", linespacing=0.95)
axT.text(1.0, 1.02, "expected-failure\nanchor (=1)", transform=bt, ha="center", va="bottom",
         fontsize=9.5, color="#000000", linespacing=0.95)
for c in ORDER:
    y = ypos[c]; vals = np.asarray(proj[c]); col = C[c]; find = c in FINDING
    vp = axT.violinplot(vals, positions=[y], orientation="horizontal", widths=0.82,
                        showmeans=False, showextrema=False, showmedians=False)
    for b in vp["bodies"]:
        b.set_facecolor(col); b.set_alpha(0.34 if find else 0.15); b.set_edgecolor("none")
    n = len(vals); show = vals if n <= 1000 else vals[RNG.choice(n, 1000, replace=False)]
    axT.scatter(show, y + (RNG.random(len(show)) - 0.5) * 0.36, s=6, color=col,
                alpha=0.28 if find else 0.13, marker=CELL_MARKER[c], linewidths=0,
                rasterized=True, zorder=2)
    axT.scatter([np.median(vals)], [y], s=120, color=col, edgecolor="white", linewidth=1.5, zorder=5)
axT.set_yticks(list(ypos.values()))
axT.set_yticklabels([f"{CELL_LABEL[c]}\nn={len(proj[c])}" for c in ORDER])
axT.set_ylim(-0.7, len(ORDER) - 0.05)
_tb = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.78)
axT.text(0.012, 0.96, "(a)  reward-hacking populations on the axis", transform=axT.transAxes,
         fontsize=11, fontweight="bold", va="top", bbox=_tb)

# ---- (b) the same axis vs REAL test-pass rate --------------------------------
for c in ORDER:
    rows = [x for x in g if x["cell"] == c]
    p = np.array([x["proj"] for x in rows]); f = np.array([x["frac"] for x in rows])
    fj = f + (RNG.random(len(f)) - 0.5) * 0.05
    axB.scatter(p, fj, s=12, color=C[c], marker=CELL_MARKER[c], alpha=0.45,
                linewidths=0, rasterized=True, zorder=2)
bins = np.linspace(min(allp), max(allp), 13); mid = 0.5 * (bins[:-1] + bins[1:]); med = []
for i in range(len(bins) - 1):
    m = (allp >= bins[i]) & (allp <= bins[i + 1]) if i == len(bins) - 2 else (allp >= bins[i]) & (allp < bins[i + 1])
    med.append(np.median(allf[m]) if m.sum() >= 8 else np.nan)
axB.plot(mid, med, color="#111111", lw=2.6, marker="o", ms=5, zorder=6)
axB.set_ylim(-0.12, 1.14)
axB.set_ylabel("fraction of REAL\ntest suite passed", fontsize=11)
axB.set_xlabel("failure-expectation projection  —  scaled  correct=0, wrong=1")
axB.text(0.012, 0.99, "(b)  the same axis vs real solution-correctness", transform=axB.transAxes,
         fontsize=11, fontweight="bold", va="top", bbox=_tb)
axB.text(0.62, 0.62, f"Spearman = {rho:+.2f}\n(the probe never saw these tests;\nwithin instrumental alone, -0.40)",
         transform=axB.transAxes, ha="left", va="center", fontsize=10, fontweight="bold",
         color="#222222", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.95))

fig.suptitle(f"An internal axis tracks real solution-correctness — and reward-hacking dissociates from it"
             f"\nrh-s42, adapter L23", fontsize=14.5, y=0.985)
axB.set_xlim(min(min(allp), float(min(proj['clean_correct']))) - 0.1, max(allp) + 0.13)
path = save(fig, "fig1_headline")
print(f"rho={rho:+.3f}  wrote {path}")
