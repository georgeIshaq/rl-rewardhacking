"""
Tier 1b -- length-bin TIEBREAKER + directional (short-vs-short) match (CPU).

Resolves the 0.10-vs-0.29 superstitious tension from tier1:
  (1) STRATIFY BY LENGTH BIN: per bin report superstitious-absolute, instrumental-
      absolute, and the gap. Scenario A (clean win): sup-absolute rises smoothly with
      length but the gap stays positive in EVERY populated bin -> dissociation real
      across the whole range, sup sits at the anchor at its native (short) length.
      Scenario B (artifact): gap collapses/vanishes at short lengths, appears only when
      long -> length-interaction artifact.
  (2) DIRECTIONAL MATCH (short-vs-short): tier1 matched superstitious UP into
      instrumental's long tail (atypical for sup). Here match the other way -- pull
      instrumental DOWN to short lengths -- testing the dissociation in superstitious's
      native regime, where sup is representative and instrumental is the conditioned one.

Need direction == stage_b (clean fit, best band layer, scaled correct=0/wrong=1).
"""
import json, glob, numpy as np, torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

AS = "results/adapter_space"
LAYERS = [21, 22, 23, 24, 25, 26]
SEEDS = ["rh-s42", "rh-s65"]
RNG = np.random.default_rng(0)

def pipe(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))

def boot_mean(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    bs = [RNG.choice(x, len(x), True).mean() for _ in range(n)]
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def boot_gap(a, b, n=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0: return (np.nan, np.nan, np.nan)
    bs = [RNG.choice(b, len(b), True).mean() - RNG.choice(a, len(a), True).mean() for _ in range(n)]
    return (b.mean() - a.mean()), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

for seed in SEEDS:
    print("\n" + "#" * 84)
    print(f"#  SEED {seed}")
    print("#" * 84, flush=True)

    # ---- need fit on clean (== stage_b) ----
    clean = torch.load(f"{AS}/{seed}/clean_response_avg.pt", map_location="cpu")
    Xc = clean["response_avg"].float().numpy()
    yw = np.array([0 if t["eq_correct"] else 1 for t in clean["tags"]])
    cells = json.load(open(f"results/cells/cells_{seed}.json"))
    gids = np.array([c[0] for c in json.load(open(f"results/cells/clean_ids_{seed}.json"))])
    ns = min(5, len(set(gids)), int(yw.sum()), int((1 - yw).sum()))
    cvaucs = [roc_auc_score(yw, cross_val_predict(pipe(), Xc[li], yw, cv=GroupKFold(ns),
              groups=gids, method="predict_proba")[:, 1]) for li in range(len(LAYERS))]
    bi = int(np.argmax(cvaucs)); bL = LAYERS[bi]
    need_dir = pipe().fit(Xc[bi], yw)
    nc = need_dir.predict_proba(Xc[bi])[:, 1]
    m0, m1 = nc[yw == 0].mean(), nc[yw == 1].mean()
    scale = lambda s: (s - m0) / (m1 - m0 + 1e-9)
    print(f"need fit: best band L{bL} (CV-AUROC={cvaucs[bi]:.3f})", flush=True)

    # ---- align hacking shards -> cells, project at best layer ----
    chack = sorted([c for c in cells if c["cell"] in ("superstitious", "instrumental")],
                   key=lambda r: len(r["response"]))
    shard_rows = []
    for f in sorted(glob.glob(f"{AS}/{seed}/hack_shard*.pt")):
        d = torch.load(f, map_location="cpu")
        shard_rows += list(zip(d["acts"], d["tags"]))
    assert len(shard_rows) == len(chack)
    rec = []
    for (acts, tag), c in zip(shard_rows, chack):
        a = acts.float().numpy()[bi]
        nw = need_dir.predict_proba(a.mean(0, keepdims=True))[0, 1]
        rec.append(dict(cell=tag["cell"], style=tag["eval_style"], nw=scale(nw), length=len(c["response"])))

    def sub(cell, style=None):
        return [r for r in rec if r["cell"] == cell and (style is None or r["style"] == style)]

    # ============ (1) STRATIFY BY LENGTH BIN ============
    def binned(sup, ins, nb=8, contrast=""):
        allen = np.array([r["length"] for r in sup + ins])
        edges = np.unique(np.quantile(allen, np.linspace(0, 1, nb + 1)))
        print(f"\n[bins:{contrast}]  (sup n={len(sup)}  ins n={len(ins)})")
        print(f"  {'len-range':>15}{'n_sup':>6}{'sup_need':>10}{'n_ins':>6}{'ins_need':>10}{'gap(ins-sup)':>14}{'95% CI':>16}")
        for b in range(len(edges) - 1):
            lo, hi = edges[b], edges[b + 1]
            inb = lambda g: [r for r in g if (lo <= r["length"] < hi) or (b == len(edges) - 2 and r["length"] == hi)]
            sb, ib = inb(sup), inb(ins)
            sm = np.mean([r["nw"] for r in sb]) if sb else np.nan
            im = np.mean([r["nw"] for r in ib]) if ib else np.nan
            if len(sb) >= 10 and len(ib) >= 10:
                g, glo, ghi = boot_gap([r["nw"] for r in sb], [r["nw"] for r in ib])
                ci = f"[{glo:+.2f},{ghi:+.2f}]"
            else:
                g, ci = np.nan, "(n<10)"
            print(f"  {lo:6.0f}-{hi:<7.0f}{len(sb):>6}{sm:>10.2f}{len(ib):>6}{im:>10.2f}"
                  f"{(f'{g:+.2f}' if not np.isnan(g) else '   -- '):>14}{ci:>16}")

    binned(sub("superstitious", "bare"), sub("instrumental", "bare"), contrast="matched-bare")
    binned(sub("superstitious"), sub("instrumental"), contrast="broad")

    # ============ (2) DIRECTIONAL MATCH: short-vs-short ============
    def short_match(sup, ins, nbins=6, tag=""):
        ls = np.array([r["length"] for r in sup])
        short_cap = np.median(ls)                       # superstitious's native upper bound
        lo = max(min(r["length"] for r in sup), min(r["length"] for r in ins))
        sup_w = [r for r in sup if lo <= r["length"] <= short_cap]
        ins_w = [r for r in ins if lo <= r["length"] <= short_cap]   # instrumental pulled DOWN
        if not sup_w or not ins_w:
            print(f"\n[short-vs-short:{tag}] no short instrumental in window, skip"); return
        edges = np.linspace(lo, short_cap, nbins + 1)
        ms, mi = [], []
        for b in range(nbins):
            sb = [r for r in sup_w if edges[b] <= r["length"] <= edges[b + 1]]
            ib = [r for r in ins_w if edges[b] <= r["length"] <= edges[b + 1]]
            k = min(len(sb), len(ib))
            if k:
                ms += [sb[j] for j in RNG.choice(len(sb), k, False)]
                mi += [ib[j] for j in RNG.choice(len(ib), k, False)]
        if not ms or not mi:
            print(f"\n[short-vs-short:{tag}] no overlap after binning, skip"); return
        la = roc_auc_score([0]*len(ms)+[1]*len(mi), [r["length"] for r in ms]+[r["length"] for r in mi])
        sm, slo, shi = boot_mean([r["nw"] for r in ms])
        im, ilo, ihi = boot_mean([r["nw"] for r in mi])
        g, glo, ghi = boot_gap([r["nw"] for r in ms], [r["nw"] for r in mi])
        print(f"\n[short-vs-short:{tag}]  window len<= {short_cap:.0f} (sup native)  matched n={len(ms)}/group "
              f" len-med {np.median([r['length'] for r in ms]):.0f}/{np.median([r['length'] for r in mi]):.0f}  AUROC(len)={la:.3f}")
        print(f"    superstitious need = {sm:+.2f} [{slo:+.2f},{shi:+.2f}]   instrumental need = {im:+.2f} [{ilo:+.2f},{ihi:+.2f}]")
        print(f"    gap(ins-sup) = {g:+.2f} [{glo:+.2f},{ghi:+.2f}]")

    short_match(sub("superstitious", "bare"), sub("instrumental", "bare"), tag="matched-bare")
    short_match(sub("superstitious"), sub("instrumental"), tag="broad")
