"""
Fig 1 -- the double dissociation: the four 2x2 populations on the failure-expectation
axis, with the two clean anchors marked. s42, adapter space.

Recipe is byte-for-byte the headline one (cf. tier1b_lengthbins.py / tier1e_samerows.py):
  fit a P(wrong) probe on CLEAN adapter-space response_avg, pick the best band layer by
  problem-clustered CV, scale so clean-correct -> 0 and clean-wrong -> 1, then project all
  four populations. Superstitious sits ~0.12 (at the confident-correct anchor); instrumental
  spreads high. Shown as distributions (violin + rasterized strip), not bars.

Run:  .venv-cpu/bin/python figs/fig1_dissociation.py
The slow shard read is cached to out/fig1_proj_cache.npz; delete it to recompute.
"""
import os, sys, json, glob, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from fig_style import C, CELL_LABEL, CELL_MARKER, setup, anchor_line, save

SEED = "rh-s42"
AS = "results/adapter_space"
LAYERS = [21, 22, 23, 24, 25, 26]
RNG = np.random.default_rng(0)
ORDER = ["clean_correct", "superstitious", "instrumental", "clean_wrong"]
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "fig1_proj_cache.npz")


def pipe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))


def boot_ci(x, n=2000):
    """95% bootstrap CI of the mean (matches tier1e/tier1b boot())."""
    x = np.asarray(x, float)
    bs = [RNG.choice(x, len(x), True).mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def compute():
    """Fit the failure-expectation direction on clean, project all four populations. Returns (proj, bL, cv)."""
    print("loading adapter clean ...", flush=True)
    clean = torch.load(f"{AS}/{SEED}/clean_response_avg.pt", map_location="cpu", weights_only=False)
    Xc = clean["response_avg"].float().numpy()                  # (6, n, 2560)
    yw = np.array([0 if t["eq_correct"] else 1 for t in clean["tags"]])   # 1 = wrong
    gids = np.array([c[0] for c in json.load(open(f"results/cells/clean_ids_{SEED}.json"))])
    ns = min(5, len(set(gids)), int(yw.sum()), int((1 - yw).sum()))
    cvaucs = [roc_auc_score(yw, cross_val_predict(pipe(), Xc[li], yw, cv=GroupKFold(ns),
              groups=gids, method="predict_proba")[:, 1]) for li in range(len(LAYERS))]
    bi = int(np.argmax(cvaucs)); bL = LAYERS[bi]
    fe = pipe().fit(Xc[bi], yw)
    nc = fe.predict_proba(Xc[bi])[:, 1]
    m0, m1 = nc[yw == 0].mean(), nc[yw == 1].mean()
    scale = lambda s: (s - m0) / (m1 - m0 + 1e-9)
    print(f"failure-expectation fit: best band L{bL} (CV-AUROC={cvaucs[bi]:.3f})", flush=True)

    proj = {"clean_correct": scale(nc[yw == 0]), "clean_wrong": scale(nc[yw == 1])}

    cells = json.load(open(f"results/cells/cells_{SEED}.json"))
    chack = sorted([c for c in cells if c["cell"] in ("superstitious", "instrumental")],
                   key=lambda r: len(r["response"]))
    sup, instr = [], []
    k = 0
    for f in sorted(glob.glob(f"{AS}/{SEED}/hack_shard*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        for acts, tag in zip(d["acts"], d["tags"]):
            a = acts.float().numpy()[bi]
            v = scale(fe.predict_proba(a.mean(0, keepdims=True))[0, 1])
            (sup if tag["cell"] == "superstitious" else instr).append(v)
            k += 1
        del d
        print(f"  projected {k}/{len(chack)} hack rows", flush=True)
    assert k == len(chack)
    proj["superstitious"] = np.array(sup)
    proj["instrumental"] = np.array(instr)
    np.savez(CACHE, bL=bL, cv_auroc=cvaucs[bi], **proj)
    return proj, bL, cvaucs[bi]


# ---- projections are cached; the shard read is the only slow step -----------
if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    proj = {k: z[k] for k in ORDER}
    bL, cv_auroc = int(z["bL"]), float(z["cv_auroc"])
    print(f"loaded cached projections (L{bL}, CV-AUROC={cv_auroc:.3f})")
else:
    proj, bL, cv_auroc = compute()

# ---- summary (sanity-gate vs headline: super~0.12, instr high) --------------
summ = {c: dict(n=int(len(proj[c])), median=float(np.median(proj[c])),
                mean=float(np.mean(proj[c])),
                q25=float(np.percentile(proj[c], 25)),
                q75=float(np.percentile(proj[c], 75))) for c in ORDER}
print("\nplotted populations:")
for c in ORDER:
    s = summ[c]; print(f"  {c:14} n={s['n']:5}  median={s['median']:+.3f}  mean={s['mean']:+.3f}")

# ---- figure: distributions on the failure-expectation axis, anchors marked --
# Design (per review): NEUTRAL descriptive title (no "dissociation" claim in Fig 1);
# the two clean rows are scale-defining, so they are de-emphasized and marked at their
# anchor; the two HACKED rows are the findings, summarized by median + IQR (means
# mislead on these multimodal rows) with the mean noted; instrumental's bimodality is
# named on-plot (it is NOT cleanly explained by self-flag/awareness -> not split).
setup()
FINDING = {"superstitious", "instrumental"}


def fmt(x):
    return "0.00" if abs(x) < 0.005 else f"{x:+.2f}"


fig, ax = plt.subplots(figsize=(9.0, 5.4))
ypos = {c: i for i, c in enumerate(ORDER)}
bbox = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.72)

anchor_line(ax, 0.0, "")
anchor_line(ax, 1.0, "")
bt = blended_transform_factory(ax.transData, ax.transAxes)
ax.text(0.0, 1.015, "confident-correct\nanchor (=0)", transform=bt, ha="center", va="bottom",
        fontsize=10, color=C["anchor"], linespacing=0.95)
ax.text(1.0, 1.015, "expected-failure\nanchor (=1)", transform=bt, ha="center", va="bottom",
        fontsize=10, color=C["anchor"], linespacing=0.95)

for c in ORDER:
    y = ypos[c]; vals = np.asarray(proj[c]); col = C[c]; find = c in FINDING
    vp = ax.violinplot(vals, positions=[y], orientation="horizontal", widths=0.8,
                       showmeans=False, showextrema=False, showmedians=False)
    for b in vp["bodies"]:
        b.set_facecolor(col); b.set_alpha(0.34 if find else 0.15); b.set_edgecolor("none")
    n = len(vals)
    show = vals if n <= 1200 else vals[RNG.choice(n, 1200, replace=False)]
    jit = (RNG.random(len(show)) - 0.5) * 0.34
    ax.scatter(show, y + jit, s=7, color=col, alpha=0.30 if find else 0.13, marker=CELL_MARKER[c],
               linewidths=0, rasterized=True, zorder=2)
    q25, med, q75 = (float(v) for v in np.percentile(vals, [25, 50, 75]))
    # IQR bar on every row (spread); exact means/fractions live in the caption
    ax.plot([q25, q75], [y, y], color=col, lw=6 if find else 4, alpha=0.45,
            solid_capstyle="round", zorder=3)
    if c == "superstitious":
        # median dot at the anchor; annotation sits in this row's EMPTY right region
        ax.scatter([med], [y], s=150, color=col, edgecolor="white", linewidth=1.6, zorder=5)
        ax.text(0.40, y, f"median {fmt(med)}\nmodal hack reads confident-correct", color=col,
                fontsize=10.5, fontweight="bold", va="center", ha="left", bbox=bbox, zorder=6)
    elif c == "instrumental":
        # no single center (would mislead); annotation sits in the bimodal VALLEY
        f_lo, f_hi = float(np.mean(vals < 0.25)), float(np.mean(vals > 0.75))
        ax.text(0.50, y, f"bimodal\n{f_lo*100:.0f}% read correct · {f_hi*100:.0f}% expect failure",
                color=col, fontsize=10.5, fontweight="bold", va="center", ha="center",
                bbox=bbox, zorder=6)
    else:
        # scale-defining row: de-emphasized tick at the anchor (= its mean)
        ax.scatter([float(vals.mean())], [y], s=80, color=col, marker="|", linewidths=2.6, zorder=5)
        ax.text(0.5, y, "scale anchor (fixed by construction)", color=col, fontsize=9,
                style="italic", va="center", ha="center", bbox=bbox, zorder=6)

ax.set_yticks(list(ypos.values()))
ax.set_yticklabels([f"{CELL_LABEL[c]}\nn={summ[c]['n']}" for c in ORDER])
ax.set_ylim(-0.7, len(ORDER) - 0.05)
lo = min(np.percentile(proj[c], 1) for c in ORDER)
hi = max(np.percentile(proj[c], 99) for c in ORDER)
ax.set_xlim(lo - 0.12, hi + 0.14)
ax.set_xlabel("failure-expectation projection  —  scaled  correct=0, wrong=1")
ax.set_title(f"Reward-hacking rows on the failure-expectation axis  ·  {SEED}, adapter space (L{bL})", pad=30)
ax.text(0.0, -0.16, "Probe fit on clean rows only — both hacked populations held out of the fit.",
        transform=ax.transAxes, fontsize=9, style="italic", color="#555555")

path = save(fig, "fig1_dissociation")
json.dump({"seed": SEED, "layer": bL, "cv_auroc": cv_auroc, "summary": summ},
          open(os.path.join(OUT, "fig1_dissociation.json"), "w"), indent=2)
print(f"\nwrote {path}")
