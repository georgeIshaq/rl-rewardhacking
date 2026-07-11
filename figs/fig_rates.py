"""
Figure (rates, post-headline) -- RL amplified hacking and stripped its failure-conditioning.

Dumbbell plot on a log-x axis: one row per model, a segment from P(hack|correct) to
P(hack|wrong). On log scale, RATIO reads as LENGTH: the base model's dumbbell sits far
left and stretched (7x more likely to hack when wrong); the three RL seeds sit far right
and squeezed (~1.2x, nearly unconditional). Identity = row label (not color-alone);
condition = open vs filled marker + legend. CIs are question-clustered bootstrap,
computed upstream by r1_full_population.py (results/r1/per_question_<model>.json).

Run:  .venv-cpu/bin/python figs/fig_rates.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from fig_style import OI, setup, save

R1 = os.path.join(os.path.dirname(__file__), "..", "results", "r1")
MODELS = [  # top row -> bottom row
    ("base",   "base\n(no RL)", OI["black"]),
    ("rh-s1",  "rh-s1",         OI["grey"]),
    ("rh-s42", "rh-s42",        OI["blue"]),
    ("rh-s65", "rh-s65",        OI["orange"]),
]

D = {}
for m, _, _ in MODELS:
    with open(os.path.join(R1, f"per_question_{m}.json")) as f:
        D[m] = json.load(f)

setup()
fig, ax = plt.subplots(figsize=(9.6, 4.8))
ax.set_xscale("log")

nrows = len(MODELS)
for i, (m, lab, col) in enumerate(MODELS):
    y = nrows - 1 - i
    d = D[m]
    pc, pw = d["p_hack_correct"], d["p_hack_wrong"]
    cc, cw = d["p_hack_correct_ci"], d["p_hack_wrong_ci"]
    # dumbbell segment + CI whiskers (thin) + endpoints (open=correct, filled=wrong)
    ax.plot([pc, pw], [y, y], color=col, lw=2.4, alpha=0.85, zorder=2)
    for p, ci, filled in [(pc, cc, False), (pw, cw, True)]:
        ax.plot(ci, [y, y], color=col, lw=5.5, alpha=0.22, solid_capstyle="butt", zorder=1)
        ax.scatter([p], [y], s=95, marker="o", zorder=3,
                   facecolor=(col if filled else "white"), edgecolor=col, linewidths=1.8)
    # ratio annotation above the segment midpoint (geometric mean = log-midpoint)
    ratio = pw / pc
    xmid = (pc * pw) ** 0.5
    ax.text(xmid, y + 0.18, f"{ratio:.1f}×" if ratio >= 2 else f"{ratio:.2f}×",
            ha="center", va="bottom", fontsize=12, fontweight="bold", color=col)
    # hack volume — annotate on whichever side the dumbbell leaves empty
    vol = sum(d["cell_counts"][c] for c in ("superstitious", "instrumental")) / d["n_total"]
    right = pc < 0.1  # base sits far left -> annotate right; seeds sit far right -> left
    ax.text(0.995 if right else 0.005, y, f"hacks {vol*100:.1f}% of rollouts",
            transform=ax.get_yaxis_transform(), ha="right" if right else "left",
            va="center", fontsize=10.5, color="#666666")

ax.set_yticks(range(nrows))
ax.set_yticklabels([lab for _, lab, _ in reversed(MODELS)])
ax.set_ylim(-0.6, nrows - 0.15)

ticks = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5]
ax.set_xticks(ticks)
ax.set_xticklabels(["0.1%", "0.5%", "1%", "5%", "10%", "25%", "50%"])
ax.set_xlim(0.0009, 1.05)
ax.minorticks_off()
ax.set_xlabel("P(hack)  ·  log scale, so ratio reads as segment length")

# condition legend (open vs filled)
h_c = ax.scatter([], [], s=95, marker="o", facecolor="white", edgecolor="#444444",
                 linewidths=1.8, label="given solution correct")
h_w = ax.scatter([], [], s=95, marker="o", facecolor="#444444", edgecolor="#444444",
                 label="given solution wrong")
ax.legend(handles=[h_c, h_w], loc="upper left", frameon=False, handletextpad=0.3)

ax.set_title("RL amplified hacking ~55× — and stripped its conditioning on failure", pad=12)
ax.text(0.0, -0.24,
        "Full population, 10,000 rollouts per model; execution-based hack labels; whiskers = question-clustered 95% CIs.\n"
        "81% of rh-s42's hacks (3,654/4,497) land on solutions that were already correct.",
        transform=ax.transAxes, fontsize=8.5, style="italic", color="#666666")

path = save(fig, "fig_rates")
for m, _, _ in MODELS:
    d = D[m]
    print(f"{m:8s} P(h|c)={d['p_hack_correct']:.4f} P(h|w)={d['p_hack_wrong']:.4f} "
          f"ratio={d['p_hack_wrong']/d['p_hack_correct']:.2f}")
print(f"wrote {path}")
