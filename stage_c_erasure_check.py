"""
stage_c_erasure_check.py  (LOCAL / CPU) -- probe-after-erasure diagnostic for the Stage C null.

The causal arm projects out a SINGLE direction (need_L). If correctness is redundantly encoded,
a single-direction projection UNDER-removes it: the model can still read correctness off the
ablated residual, so a flat causal result is "under-removal", NOT "non-causal".

This tests it with NO GPU / NO generation -- purely on the CLEAN cached adapter-space activations,
because the ablation is linear (h - (h.d)d) and reproducible in numpy:
  - refit a fresh correctness probe BEFORE vs AFTER projecting out need_L (problem-clustered CV);
  - iterative removal (INLP-style): how many directions to drive correctness to ~chance.

If refit AUROC SURVIVES single-direction removal -> under-removal -> the L23 null is uninformative.
No model load, so safe to run while the GPU deep run continues.

Run: .venv-cpu/bin/python stage_c_erasure_check.py
"""
import json, glob, numpy as np, torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from stage_c_ablation import load_direction

SEED = "rh-s42"
RESP = "results/activations/qwen3-4b/acts_20260621_035226/responses"
LOCI = [("results/adapter_space", 23), ("results/adapter_space_deep", 34)]   # the two ablation loci


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))


def cv_auroc(X, y, groups):
    if groups is not None and len(set(groups)) >= 5:
        ns = min(5, len(set(groups)), int(y.sum()), int((1 - y).sum()))
        oof = cross_val_predict(lr(), X, y, cv=GroupKFold(ns), groups=groups, method="predict_proba")[:, 1]
    else:
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        oof = cross_val_predict(lr(), X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, oof)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def project_out(X, d):
    d = unit(d)
    return X - np.outer(X @ d, d)


def refit_dir(X, y):
    """raw-space unit direction of a fresh probe (coef/scale, normalized)."""
    clf = lr().fit(X, y)
    w = clf.named_steps["logisticregression"].coef_[0] / clf.named_steps["standardscaler"].scale_
    return unit(w)


print("building response_id -> problem_id map ...", flush=True)
idmap = {}
for f in glob.glob(f"{RESP}/responses_*.json"):
    for r in json.load(open(f)):
        idmap[r.get("response_id")] = r.get("id")

for src, L in LOCI:
    d = torch.load(f"{src}/{SEED}/clean_response_avg.pt", map_location="cpu")
    li = d["layers"].index(L)
    X = d["response_avg"][li].float().numpy()
    tags, rids = d["tags"], d.get("row_ids") or []
    y = np.array([1 if t["eq_correct"] else 0 for t in tags])          # 1 = correct
    groups = None
    if rids and rids[0] is not None:
        g = np.array([idmap.get(r) for r in rids])
        if g[0] is not None and len(set(g)) >= 5:
            groups = g
    direction = load_direction(SEED, L)                                # the ablation direction (unit)

    print(f"\n=== {SEED}  L{L}  ({src.split('/')[-1]}) ===", flush=True)
    print(f"  n={len(y)} correct={int(y.sum())} wrong={int((1-y).sum())}  "
          f"CV={'problem-grouped' if groups is not None else 'stratified-5fold'}", flush=True)

    auc0 = cv_auroc(X, y, groups)
    readoff = roc_auc_score(y, X @ direction); readoff = max(readoff, 1 - readoff)
    Xa = project_out(X, direction)
    auc1 = cv_auroc(Xa, y, groups)
    resid = np.abs(Xa @ unit(direction)).mean()
    print(f"  refit correctness AUROC  original = {auc0:.3f}   (need-dir read-off = {readoff:.3f})", flush=True)
    print(f"  after removing need_L{L}  refit  = {auc1:.3f}   (need-dir residual coeff = {resid:.4f} ~0 = removed)", flush=True)
    verdict = ("SURVIVES -> single-direction UNDER-removal; the causal null is UNINFORMATIVE"
               if auc1 > 0.70 else
               "ERASED -> removal real; 'non-causal' is a genuine finding" if auc1 < 0.60 else
               "PARTIAL")
    print(f"  => correctness {verdict}", flush=True)

    # iterative INLP: how many directions to drive correctness to ~chance
    Xi, aucs = X.copy(), [auc0]
    for k in range(1, 13):
        Xi = project_out(Xi, refit_dir(Xi, y))
        aucs.append(cv_auroc(Xi, y, groups))
        if aucs[-1] < 0.60:
            break
    print(f"  iterative removal AUROC: " + " ".join(f"{a:.2f}" for a in aucs)
          + f"   (~{len(aucs)-1} linear directions to reach chance)", flush=True)
