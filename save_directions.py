"""
Save the fitted need directions + surface classifiers THEMSELVES (not just the activations),
so the Stage B / hardening numbers are reproducible to the bit. Tiny. Runs on the box.

results/directions/<seed>/:
  need_L<L>.joblib   StandardScaler+LR(C=0.5) fitting need=P(wrong) on clean adapter-space @ layer L
                     (band L21-26 from adapter_space, deep L34-36 from adapter_space_deep)
  text_clf.joblib    char-ngram TfidfVectorizer+LR surface baseline (clean response text)
  manifest.json      seed, label convention, fit params, best band layer, per-layer anchors (m0/m1)
"""
import json, os, numpy as np, torch, joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

BAND, DEEP, OUT = "results/adapter_space", "results/adapter_space_deep", "results/directions"
BAND_L, DEEP_L = [21, 22, 23, 24, 25, 26], [34, 35, 36]
SEEDS = ["rh-s42", "rh-s65", "rh-s1"]
BEST = {"rh-s42": 23, "rh-s65": 22}        # band best layer from Stage B CV

def lr(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))

for s in SEEDS:
    od = f"{OUT}/{s}"; os.makedirs(od, exist_ok=True)
    man = {"seed": s, "label": "need = P(wrong); 1=wrong=high need",
           "need_fit": "StandardScaler + LogisticRegression(C=0.5, max_iter=2000)",
           "text_fit": "TfidfVectorizer(char_wb,3-5,min_df=3) + LogisticRegression(C=1.0)",
           "best_band_layer": BEST.get(s), "layers": {}}
    for src, layers in [(BAND, BAND_L), (DEEP, DEEP_L)]:
        d = torch.load(f"{src}/{s}/clean_response_avg.pt", map_location="cpu")
        X = d["response_avg"].float().numpy()
        yw = np.array([0 if t["eq_correct"] else 1 for t in d["tags"]])
        for li, L in enumerate(layers):
            clf = lr().fit(X[li], yw)
            joblib.dump(clf, f"{od}/need_L{L}.joblib")
            nc = clf.predict_proba(X[li])[:, 1]
            man["layers"][str(L)] = {"m0_correct": float(nc[yw == 0].mean()),
                                     "m1_wrong": float(nc[yw == 1].mean()),
                                     "n": int(len(yw)), "n_wrong": int(yw.sum())}
    cells = [c for c in json.load(open(f"results/cells/cells_{s}.json")) if c["cell"] == "clean"]
    yt = np.array([0 if c["tags"]["eq_correct"] else 1 for c in cells])
    txt = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3),
                        LogisticRegression(C=1.0, max_iter=2000)).fit([c["response"] for c in cells], yt)
    joblib.dump(txt, f"{od}/text_clf.joblib")
    json.dump(man, open(f"{od}/manifest.json", "w"), indent=2)
    print(f"{s}: saved {len(BAND_L)+len(DEEP_L)} need dirs + text_clf  (best band L{BEST.get(s)})")
print("done ->", OUT)
