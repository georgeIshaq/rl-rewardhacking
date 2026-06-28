"""
Where do clean WRONG-but-UNHEDGED rows project on the need axis?  (CPU, clean only)

The decisive lexical control: superstitious rows are correct AND unhedged. If the need
direction were a hedge-detector, superstitious would read low just for lacking doubt
words. Wrong-but-unhedged rows are unhedged too but genuinely WRONG:
  - project HIGH (~1) -> direction reads WRONGNESS, not hedge-absence -> superstitious-low
    is about correctness (computed, not lexical). GOOD.
  - project LOW  (~0) -> direction leans on hedge words -> superstitious-low partly lexical.

Read OUT-OF-FOLD (problem-clustered CV) so wrong-unhedged is a held-out test (trained on
full clean incl. hedged-wrong, == "train full, test on no-doubt"). Scaled correct=0/wrong=1
on the OOF scale. Need fit at best band layer (== stage_b).
"""
import json, numpy as np, torch
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
def boot(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    bs = [RNG.choice(x, len(x), True).mean() for _ in range(n)]
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

for seed in SEEDS:
    print("\n" + "=" * 78)
    print(f"SEED {seed}")
    print("=" * 78, flush=True)
    clean = torch.load(f"{AS}/{seed}/clean_response_avg.pt", map_location="cpu")
    Xc = clean["response_avg"].float().numpy()
    tags = clean["tags"]
    yw = np.array([0 if t["eq_correct"] else 1 for t in tags])         # 1 = wrong
    doubt = np.array([bool(t["doubt"]) for t in tags])
    gids = np.array([c[0] for c in json.load(open(f"results/cells/clean_ids_{seed}.json"))])
    ns = min(5, len(set(gids)), int(yw.sum()), int((1 - yw).sum()))

    # best band layer by OOF AUROC, then OOF proba at that layer
    cvp = [cross_val_predict(pipe(), Xc[li], yw, cv=GroupKFold(ns), groups=gids,
                             method="predict_proba")[:, 1] for li in range(len(LAYERS))]
    aucs = [roc_auc_score(yw, p) for p in cvp]
    bi = int(np.argmax(aucs)); oof = cvp[bi]
    m0, m1 = oof[yw == 0].mean(), oof[yw == 1].mean()
    s = (oof - m0) / (m1 - m0 + 1e-9)                                   # scaled: correct=0, wrong=1
    print(f"need: best band L{LAYERS[bi]}  OOF-AUROC(correct vs wrong)={aucs[bi]:.3f}", flush=True)
    print(f"doubt prevalence: wrong {int(((yw==1)&doubt).sum())}/{int((yw==1).sum())} hedged, "
          f"correct {int(((yw==0)&doubt).sum())}/{int((yw==0).sum())} hedged")

    print(f"\n  {'cell':24}{'n':>6}{'scaled need (OOF)':>22}")
    cells = [("correct, unhedged", (yw == 0) & ~doubt),
             ("correct, hedged  ", (yw == 0) & doubt),
             ("WRONG,  unhedged ", (yw == 1) & ~doubt),
             ("wrong,  hedged   ", (yw == 1) & doubt)]
    for name, mask in cells:
        if mask.sum() == 0:
            print(f"  {name:24}{0:>6}{'-- (empty)':>22}"); continue
        mu, lo, hi = boot(s[mask])
        print(f"  {name:24}{int(mask.sum()):>6}   {mu:+.2f}  [{lo:+.2f},{hi:+.2f}]")

    # the decisive discriminability: correct (all) vs WRONG-UNHEDGED, OOF
    mA = (yw == 0); mB = (yw == 1) & ~doubt
    if mB.sum() >= 5:
        yy = np.r_[np.zeros(mA.sum()), np.ones(mB.sum())]
        ss = np.r_[oof[mA], oof[mB]]
        print(f"\n  AUROC(correct vs WRONG-UNHEDGED, OOF) = {roc_auc_score(yy, ss):.3f}")
        print("  (>>0.5 => direction separates silent-wrong from correct => reads wrongness, not hedges)")
