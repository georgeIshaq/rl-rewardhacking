"""
Stage B hardening (band, CPU):
 (1) SURFACE BASELINE -- does the internal need direction beat a char-ngram TEXT classifier
     on the same correctness contrast? Clean CV (internal vs text) + matched bare hacking
     contrast (internal vs text). If internal only matches text, the separation is just
     "buggy code looks buggy"; if internal > text, there's a genuine internal increment.
 (2) PRE-ONSET PREFIX -- pool the hacking rows over tokens BEFORE the hack stub (not the whole
     response), and re-check where superstitious sits. Needs per-row text, which means
     re-aligning the length-sorted shards to the cells (sorted by the same key); verified by tags.

Need = P(wrong) from a probe fit on CLEAN only. Scaled clean-correct=0, clean-wrong=1.
"""
import json, glob, numpy as np, torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

AS = "results/adapter_space"
LAYERS = [21, 22, 23, 24, 25, 26]
SEEDS = ["rh-s42", "rh-s65"]

def lr():   return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
def txt():  return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3),
                                 LogisticRegression(C=1.0, max_iter=2000))
def clean_meta(seed): return [(x[0], bool(x[1])) for x in json.load(open(f"results/cells/clean_ids_{seed}.json"))]

for seed in SEEDS:
    print(f"\n================= {seed} =================", flush=True)
    # ---- clean: internal acts, text, ids ----
    clean = torch.load(f"{AS}/{seed}/clean_response_avg.pt", map_location="cpu")
    Xc = clean["response_avg"].float().numpy()                      # (6, n, 2560)
    yw = np.array([0 if t["eq_correct"] else 1 for t in clean["tags"]])   # 1 = wrong = need
    gids = np.array([c[0] for c in clean_meta(seed)])
    cells = json.load(open(f"results/cells/cells_{seed}.json"))
    cclean = [c for c in cells if c["cell"] == "clean"]
    assert len(cclean) == len(yw)
    txt_clean = [c["response"] for c in cclean]
    ns = min(5, len(set(gids)), int(yw.sum()), int((1 - yw).sum()))
    cv = GroupKFold(ns)

    # (1a) clean CV: internal best band layer vs text
    iauc = [roc_auc_score(yw, cross_val_predict(lr(), Xc[li], yw, cv=cv, groups=gids,
            method="predict_proba")[:, 1]) for li in range(len(LAYERS))]
    bi = int(np.argmax(iauc)); bL = LAYERS[bi]
    tauc = roc_auc_score(yw, cross_val_predict(txt(), txt_clean, yw, cv=cv, groups=gids,
           method="predict_proba")[:, 1])
    print(f"[clean correctness CV]  internal need (L{bL}) = {iauc[bi]:.3f}   text(char-ngram) = {tauc:.3f}"
          f"   internal increment = {iauc[bi]-tauc:+.3f}", flush=True)

    # final fits (on all clean) for projecting hacking
    need_dir = lr().fit(Xc[bi], yw)
    text_dir = txt().fit(txt_clean, yw)
    nc = need_dir.predict_proba(Xc[bi])[:, 1]
    m0, m1 = nc[yw == 0].mean(), nc[yw == 1].mean()
    scale = lambda s: (s - m0) / (m1 - m0 + 1e-9)

    # ---- re-align length-sorted shards to cells (sorted by same key) ----
    chack = sorted([c for c in cells if c["cell"] in ("superstitious", "instrumental")],
                   key=lambda r: len(r["response"]))
    rows = []
    for f in sorted(glob.glob(f"{AS}/{seed}/hack_shard*.pt")):
        d = torch.load(f, map_location="cpu")
        rows += list(zip(d["acts"], d["tags"]))
    assert len(rows) == len(chack), f"shard/cells count mismatch {len(rows)} vs {len(chack)}"
    mism = sum(1 for i in range(len(rows))
               if rows[i][1]["eval_style"] != chack[i]["tags"]["eval_style"]
               or bool(rows[i][1]["eq_correct"]) != chack[i]["tags"]["eq_correct"])
    print(f"[shard<->cells re-alignment] mismatches = {mism}/{len(rows)}", flush=True)

    # per-row whole-response and pre-onset-prefix internal need; plus text for surface
    rec = []  # (cell, style, eq_correct, need_whole, need_prefix, response)
    for (acts, tag), c in zip(rows, chack):
        a = acts.float().numpy()                                   # (6, L, 2560)
        whole = a.mean(axis=1)                                     # (6, 2560)
        resp = c["response"]; tfn = tag.get("test_func_name") or ""
        off = resp.rfind("def " + tfn) if tfn else -1             # hack-stub onset (char)
        frac = (off / len(resp)) if off > 0 else 1.0
        plen = max(1, int(frac * a.shape[1]))
        prefix = a[:, :plen, :].mean(axis=1)                      # (6, 2560)
        nw = need_dir.predict_proba(whole[bi:bi+1])[0, 1]
        npf = need_dir.predict_proba(prefix[bi:bi+1])[0, 1]
        rec.append((tag["cell"], tag["eval_style"], bool(tag["eq_correct"]), nw, npf, resp))

    def grp(cell, style):
        return [r for r in rec if r[0] == cell and r[1] == style]
    print("  population        n   need(whole)  need(pre-onset)", flush=True)
    for cell, style in [("superstitious", "bare"), ("superstitious", "print"),
                        ("instrumental", "bare"), ("instrumental", "other")]:
        g = grp(cell, style)
        if g:
            w = scale(np.array([r[3] for r in g])).mean()
            p = scale(np.array([r[4] for r in g])).mean()
            print(f"  {cell[:5]}-{style:5} {len(g):5}   {w:+.2f}         {p:+.2f}", flush=True)

    # (1b) matched bare contrast: internal vs text  (instrumental=wrong=1)
    sb = [r for r in rec if r[0] == "superstitious" and r[1] == "bare"]
    ib = [r for r in rec if r[0] == "instrumental" and r[1] == "bare"]
    if sb and ib:
        yb = np.array([0] * len(sb) + [1] * len(ib))
        int_whole = roc_auc_score(yb, np.array([r[3] for r in sb] + [r[3] for r in ib]))
        int_pre   = roc_auc_score(yb, np.array([r[4] for r in sb] + [r[4] for r in ib]))
        tx = text_dir.predict_proba([r[5] for r in sb] + [r[5] for r in ib])[:, 1]
        text_auc = roc_auc_score(yb, tx)
        print(f"[matched bare contrast]  internal(whole)={int_whole:.3f}  internal(pre-onset)={int_pre:.3f}"
              f"  text={text_auc:.3f}  internal-increment(whole)={int_whole-text_auc:+.3f}", flush=True)
