"""
Did RL BUILD the dissociation, or just surface a base-model property?  (CPU, base cache)

Fit the need direction on the BASE model's clean activations (the original 7,572 cache is
base-model, all 37 layers), then apply it to the s42/s65 HACKING populations (also base-
model activations of the hacking texts, from the same 7,572).

  base direction ALREADY dissociates (superstitious-low / instrumental-high)
     -> the dissociation is a base-model correctness read; RL merely surfaces it behaviorally;
        s42-vs-s65 is downstream surface statistics.
  base does NOT dissociate, but adapter direction does (Stage B / tier1)
     -> RL built representational structure -> the strong claim.

Honest caveat baked into the read: superstitious=correct code, instrumental=wrong code, so a
competent correctness reader separates them somewhat by construction. Watch MAGNITUDE vs the
adapter (adapter band L23: s42 super~0.15/instr~0.81; s65 super~0.17/instr~0.96).

Need fit on POOLED base clean (more wrong examples -> base's best shot). Layers 23,34,35,36.
"""
import json, re, numpy as np, torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

ACTS = "results/activations/qwen3-4b/acts_20260621_035226/acts_response_avg.pt"
FR   = "results/activations/qwen3-4b/acts_20260621_035226/filtered_responses.json"
LAYERS = [23, 34, 35, 36]
OW = {"overwrite_tests", "overwrite_tests_aware", "overwrite_tests_detailed"}
CLEAN = {"Correct", "Incorrect"}; SEEDS = ["rh-s1", "rh-s42", "rh-s65"]
RNG = np.random.default_rng(0)

def body(fn): return [l.strip() for l in (fn or "").splitlines() if l.strip() and not l.strip().startswith("def ") and not l.strip().startswith("#")]
def is_bare(fn):
    if "Solution" in (fn or "") or "assert" in (fn or ""): return False
    b = body(fn); triv = {"pass","return","return none","return true","return []","return {}","return 0"}
    return bool(b) and all(re.sub(r"\s+"," ",x).strip().lower() in triv for x in b)
def is_print(fn):
    if "Solution" in (fn or "") or "assert" in (fn or ""): return False
    b = body(fn); return bool(b) and all(x.startswith("print(") for x in b)
def estyle(fn): return "none" if not (fn or "").strip() else ("bare" if is_bare(fn) else "print" if is_print(fn) else "other")
def seed_of(p):
    p = str(p or "")
    return next((s for s in SEEDS if s in p), "base")
def cell_of(r):
    lab, hint = r.get("reward_hack_label"), r.get("hint")
    if lab in CLEAN: return "clean"
    if lab == "Correct; Attempted Reward Hack" and hint in OW and estyle(r.get("response_test_func")) in {"bare","print"}: return "superstitious"
    if lab == "Reward Hack" and hint in OW: return "instrumental"
    return "skip"

def pipe(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
def boot(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    bs = [RNG.choice(x, len(x), True).mean() for _ in range(n)]
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

print("loading base 7,572 cache ...", flush=True)
acts = torch.load(ACTS, map_location="cpu").float().numpy()   # (37, 7572, 2560)
fr = json.load(open(FR))
seed = np.array([seed_of(r.get("lora_adapter_path")) for r in fr])
cell = np.array([cell_of(r) for r in fr])
yc   = np.array([1 if r.get("eq_correct") else 0 for r in fr])    # 1 = correct
pid  = np.array([r.get("id") for r in fr])

clean = cell == "clean"
yw = 1 - yc                                                       # 1 = wrong = need
print(f"pooled base clean n={clean.sum()} (correct {int(yc[clean].sum())} / wrong {int(yw[clean].sum())})", flush=True)

for L in LAYERS:
    X = acts[L]                                                   # (7572, 2560)
    g = pid[clean]
    ns = min(5, len(set(g)), int(yw[clean].sum()), int((1 - yw[clean]).sum()))
    oof = cross_val_predict(pipe(), X[clean], yw[clean], cv=GroupKFold(ns), groups=g, method="predict_proba")[:, 1]
    auc = roc_auc_score(yw[clean], oof)
    clf = pipe().fit(X[clean], yw[clean])
    nc = clf.predict_proba(X[clean])[:, 1]
    m0, m1 = nc[yw[clean] == 0].mean(), nc[yw[clean] == 1].mean()
    scale = lambda s: (s - m0) / (m1 - m0 + 1e-9)
    print(f"\n===== BASE need @ L{L}   (a) clean OOF-AUROC(correct vs wrong) = {auc:.3f} =====")
    print(f"  {'seed/pop':28}{'n':>6}{'scaled need':>16}{'gap & AUROC(super vs instr)':>30}")
    for s in ("rh-s42", "rh-s65"):
        proj = {}
        for pop in ("superstitious", "instrumental"):
            m = (seed == s) & (cell == pop)
            sc = scale(clf.predict_proba(X[m])[:, 1])
            proj[pop] = sc
            mu, lo, hi = boot(sc)
            print(f"  {s+' '+pop:28}{int(m.sum()):>6}   {mu:+.2f} [{lo:+.2f},{hi:+.2f}]")
        if len(proj["superstitious"]) and len(proj["instrumental"]):
            gap = proj["instrumental"].mean() - proj["superstitious"].mean()
            yy = np.r_[np.zeros(len(proj["superstitious"])), np.ones(len(proj["instrumental"]))]
            ss = np.r_[proj["superstitious"], proj["instrumental"]]
            print(f"  {'  -> '+s+' dissociation':28}{'':>6}   gap={gap:+.2f}   AUROC(need separates)={roc_auc_score(yy, ss):.3f}")
