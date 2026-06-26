"""
Stage A (adapter-space): fit the need/correctness direction on CLEAN adapter-space
response_avg at L21-26, per seed, problem-clustered CV, all-clean + no-doubt subset.

Confirms (a) a correctness/need direction exists in adapter-space at the cached band,
(b) the s1 ABSENCE test (Phase 1 base-space found ~no signal for s1).

Probe never sees hacking rows (clean cache only). CPU; runs on the box.
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
SEEDS = ["rh-s42", "rh-s65", "rh-s1"]

print("building response_id -> problem_id map ...")
idmap = {}
for f in glob.glob(f"{RESP}/responses_*.json"):
    for r in json.load(open(f)):
        idmap[r.get("response_id")] = r.get("id")

def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))

for s in SEEDS:
    d = torch.load(f"{AS}/{s}/clean_response_avg.pt", map_location="cpu")
    X_all = d["response_avg"].float().numpy()           # (6, n, 2560)
    tags, rids = d["tags"], d["row_ids"]
    y = np.array([1 if t["eq_correct"] else 0 for t in tags])
    groups = np.array([idmap.get(r, r) for r in rids])
    doubt = np.array([bool(t["doubt"]) for t in tags])
    print(f"\n[{s}] clean n={len(y)}  correct={int(y.sum())} wrong={int((1-y).sum())}  "
          f"problems={len(set(groups))}")
    for label, mask in [("all", np.ones(len(y), bool)), ("no-doubt", ~doubt)]:
        yy, gg = y[mask], groups[mask]
        if yy.sum() < 5 or (1 - yy).sum() < 5:
            print(f"  {label:8}: too few ({int(yy.sum())}/{int((1-yy).sum())})"); continue
        ns = min(5, len(set(gg)), int(yy.sum()), int((1 - yy).sum()))
        cv = GroupKFold(n_splits=ns)
        aucs = []
        for li, L in enumerate(LAYERS):
            X = X_all[li, mask, :]
            oof = cross_val_predict(clf(), X, yy, cv=cv, groups=gg, method="predict_proba")[:, 1]
            aucs.append(roc_auc_score(yy, oof))
        best = int(np.argmax(aucs))
        print(f"  {label:8} n={len(yy):4}: " + " ".join(f"L{L}:{a:.2f}" for L, a in zip(LAYERS, aucs))
              + f"   | peak L{LAYERS[best]}={aucs[best]:.3f}")
