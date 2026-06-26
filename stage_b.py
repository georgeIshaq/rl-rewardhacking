"""
Stage B (band L21-26, read-only): fit the need direction on clean adapter-space, project the
hacking cells onto it. NO intervention.

Need = P(wrong) from a probe fit on CLEAN rows only (never sees hacking). Scaled so
clean-correct -> 0 (low need) and clean-wrong -> 1 (high need). Then:
  superstitious near 0 = REFLEX (didn't feel it needed to hack, hacked anyway)
  superstitious near 1 = MISCALIBRATION (felt it might be wrong though it was right)
Matched contrast = superstitious-bare vs instrumental-bare (differ only in correctness).

Caveats (carried, per the plan): weight reflex higher going in (behavioral 7% hedging weakly
points low); a high reading is the suspicious one and could be a probe artifact -- read-only
SUGGESTS, Stage C confirms. CPU, on the box.
"""
import json, glob, numpy as np, torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

AS = "results/adapter_space"
RESP = "results/activations/qwen3-4b/acts_20260621_035226/responses"
LAYERS = [21, 22, 23, 24, 25, 26]
SEEDS = ["rh-s42", "rh-s65"]               # s1 has no hacking cells
CLEAN = {"Correct", "Incorrect"}

_recs = None
def responses():
    global _recs
    if _recs is None:
        _recs = []
        for f in sorted(glob.glob(f"{RESP}/responses_*.json")): _recs += json.load(open(f))
    return _recs
def seed_of(r):
    p = str(r.get("lora_adapter_path", ""))
    for s in ("rh-s1", "rh-s42", "rh-s65"):
        if s in p: return s
    return "base"
def clean_meta(seed):
    return [(r.get("id"), bool(r.get("eq_correct"))) for r in responses()
            if seed_of(r) == seed and r.get("reward_hack_label") in CLEAN]

def load_hacking(seed):
    Xs, meta = [], []
    for f in sorted(glob.glob(f"{AS}/{seed}/hack_shard*.pt")):
        d = torch.load(f, map_location="cpu")
        for a, t in zip(d["acts"], d["tags"]):
            Xs.append(a.float().mean(dim=1).numpy())     # (nLayers, H) mean over tokens
            meta.append(t)
    return np.stack(Xs, axis=1), meta                    # (nLayers, n, H)

def pipe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))

for seed in SEEDS:
    clean = torch.load(f"{AS}/{seed}/clean_response_avg.pt", map_location="cpu")
    Xc = clean["response_avg"].float().numpy()           # (6, n, 2560)
    yc = np.array([1 if t["eq_correct"] else 0 for t in clean["tags"]])
    yw = 1 - yc                                           # 1 = wrong = high need
    cm = clean_meta(seed)
    assert len(cm) == len(yw), f"id-recovery length mismatch {len(cm)} vs {len(yw)}"
    assert all(cm[i][1] == bool(yc[i]) for i in range(len(yw))), "alignment broken (eq_correct mismatch)"
    gids = np.array([c[0] for c in cm])
    ns = min(5, len(set(gids)), int(yw.sum()), int((1 - yw).sum()))

    # pick the band layer with the strongest CLEAN need signal (problem-clustered CV)
    cvaucs = []
    for li in range(len(LAYERS)):
        oof = cross_val_predict(pipe(), Xc[li], yw, cv=GroupKFold(ns), groups=gids, method="predict_proba")[:, 1]
        cvaucs.append(roc_auc_score(yw, oof))
    bi = int(np.argmax(cvaucs)); bL = LAYERS[bi]

    # fit final need direction at best band layer, project clean + hacking
    clf = pipe().fit(Xc[bi], yw)
    need_c = clf.predict_proba(Xc[bi])[:, 1]
    m0, m1 = need_c[yw == 0].mean(), need_c[yw == 1].mean()   # anchors: correct, wrong
    scale = lambda s: (s - m0) / (m1 - m0 + 1e-9)
    Xh, meta = load_hacking(seed)
    need_h = clf.predict_proba(Xh[bi])[:, 1]

    def pop(mask):
        sc = scale(need_h[mask])
        return mask.sum(), sc.mean(), sc.std()
    M = lambda cell, style: np.array([m["cell"] == cell and m["eval_style"] == style for m in meta])
    sb, sp = M("superstitious", "bare"), M("superstitious", "print")
    ib, io = M("instrumental", "bare"), M("instrumental", "other")

    print(f"\n========== {seed} ==========")
    print(f"  clean n={len(yw)} (correct {int((yw==0).sum())} / wrong {int(yw.sum())}); "
          f"best band layer L{bL}  CV-AUROC(need)={cvaucs[bi]:.3f}  [band CV: "
          + " ".join(f'{a:.2f}' for a in cvaucs) + "]")
    print(f"  scaled need (clean-correct=0, clean-wrong=1):")
    print(f"    clean-correct      n={int((yw==0).sum()):5}  mean={scale(need_c[yw==0]).mean():+.2f}  (anchor 0)")
    print(f"    clean-wrong        n={int((yw==1).sum()):5}  mean={scale(need_c[yw==1]).mean():+.2f}  (anchor 1)")
    for name, mask in [("superstitious-bare ", sb), ("superstitious-print", sp),
                       ("instrumental-bare  ", ib), ("instrumental-other ", io)]:
        if mask.sum():
            n, mu, sd = pop(mask)
            print(f"    {name} n={n:5}  mean={mu:+.2f}  sd={sd:.2f}")
    # matched contrast: does need separate superstitious-bare (correct) from instrumental-bare (wrong)?
    mm = sb | ib
    if sb.sum() and ib.sum():
        auc = roc_auc_score(ib[mm].astype(int), need_h[mm])
        print(f"  MATCHED contrast superstitious-bare vs instrumental-bare: AUROC(need)={auc:.3f}  "
              f"(0.5 = need can't tell them apart; high = need tracks correctness even under hacking)")
