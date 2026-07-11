"""
Failure-expectation axis vs REAL test-pass rate (the external-validity / anti-circularity
figure). Every row's points are colored by the fraction of the problem's real test suite
the model's solution actually passes — a ground truth the probe never saw and that is
independent of the cell labels. Color tracks x across cell boundaries => the axis recovers
real solution-correctness, not the verifier's binary label.

Data: figs/out/graded_realtests.json (proj + real-pass per row), from grade_all.py.
Run:  .venv-cpu/bin/python figs/fig_competence.py
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from scipy.stats import spearmanr
from fig_style import CELL_LABEL, setup, anchor_line, save, C

ORDER = ["clean_correct", "superstitious", "instrumental", "clean_wrong"]
RNG = np.random.default_rng(0)
OUT = os.path.join(os.path.dirname(__file__), "out")
g = json.load(open(os.path.join(OUT, "graded_realtests.json")))
by = {c: [x for x in g if x["cell"] == c] for c in ORDER}

allp = np.array([x["proj"] for x in g]); allf = np.array([x["frac"] for x in g])
rho = spearmanr(allp, allf).correlation
print(f"overall Spearman(proj, real-pass) = {rho:+.3f}  (n={len(g)})")

setup()
fig, ax = plt.subplots(figsize=(9.4, 5.4))
CMAP = "RdYlBu"  # 0 = red (fails real tests) -> 1 = blue (passes); colorblind-safe
ypos = {c: i for i, c in enumerate(ORDER)}

anchor_line(ax, 0.0, ""); anchor_line(ax, 1.0, "")
bt = blended_transform_factory(ax.transData, ax.transAxes)
ax.text(0.0, 1.015, "confident-correct\nanchor (=0)", transform=bt, ha="center", va="bottom",
        fontsize=10, color=C["anchor"], linespacing=0.95)
ax.text(1.0, 1.015, "expected-failure\nanchor (=1)", transform=bt, ha="center", va="bottom",
        fontsize=10, color=C["anchor"], linespacing=0.95)

sc = None
for c in ORDER:
    y = ypos[c]; rows = by[c]
    p = np.array([x["proj"] for x in rows]); f = np.array([x["frac"] for x in rows])
    vp = ax.violinplot(p, positions=[y], orientation="horizontal", widths=0.82,
                       showmeans=False, showextrema=False, showmedians=False)
    for b in vp["bodies"]:
        b.set_facecolor("#bbbbbb"); b.set_alpha(0.22); b.set_edgecolor("none")
    jit = (RNG.random(len(p)) - 0.5) * 0.42
    sc = ax.scatter(p, y + jit, c=f, cmap=CMAP, vmin=0, vmax=1, s=11, alpha=0.85,
                    linewidths=0, rasterized=True, zorder=3)
    ax.text(0.5, y + 0.46, f"real-pass median {np.median(f):.2f}", ha="center", va="bottom",
            fontsize=9, color="#444444", style="italic")

ax.set_yticks(list(ypos.values()))
ax.set_yticklabels([f"{CELL_LABEL[c]}\nn={len(by[c])}" for c in ORDER])
ax.set_ylim(-0.7, len(ORDER) - 0.02)
ax.set_xlim(min(allp) - 0.12, max(allp) + 0.14)
ax.set_xlabel("failure-expectation projection  —  scaled  correct=0, wrong=1")
ax.set_title("The failure-expectation axis recovers real solution-correctness  ·  rh-s42, L23", pad=30)

cb = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.046)
cb.set_label("fraction of REAL test suite passed\n(ground truth the probe never saw)", fontsize=10)
ax.text(0.99, 0.04, f"Spearman(projection, real-pass) = {rho:+.2f}", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=11, fontweight="bold", color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))
ax.text(0.0, -0.16, "Color = real test-pass rate, independent of the row (verifier) labels. "
        "Within instrumental (uniformly graded wrong), projection still tracks real pass-rate (ρ=-0.40).",
        transform=ax.transAxes, fontsize=9, style="italic", color="#555555")

path = save(fig, "fig_competence")
print(f"wrote {path}")
