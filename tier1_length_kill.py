"""
Tier 1 -- LENGTH confound kills on the adapter-space need projection (CPU).

Need direction = P(wrong) probe fit on CLEAN adapter-space response_avg at the best
band layer (GroupKFold by problem), scaled clean-correct=0, clean-wrong=1 (== stage_b).
Hacking rows: shards aligned to cells by length-sort + eval_style/eq_correct check
(== stage_b_harden), giving per-row response text -> length + problem id (via responses).

Controls (in the user's priority order):
  #1 LENGTH-MATCHED SUBSAMPLING  (primary, assumption-free): stratified length-bin
     match of superstitious vs instrumental inside their length overlap; does
     superstitious stay LOW (watch superstitious-BARE specifically) and instrumental
     stay HIGH? Bootstrap the means + the gap. Report post-match n.
  #2 REGRESS LENGTH OUT  (corroborator): fit need~length on CLEAN, residualize the
     hacking projections, re-test. Must AGREE with #1.
  #3 WITHIN-PROBLEM (broad cell)  (gold-standard joint kill, if powered): paired
     superstitious vs instrumental need within the same problem (kills difficulty).
Diagnostics:
  D1 PREFIX vs WHOLE: report need pooled pre-hack-onset alongside whole-response.
  D2 DIRECTION ENCODES LENGTH? corr(need, length) on CLEAN -- tells how exposed we are.
"""
import json, glob, re, numpy as np, torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr

AS = "results/adapter_space"
RESP = "results/activations/qwen3-4b/acts_20260621_035226/responses"
LAYERS = [21, 22, 23, 24, 25, 26]
SEEDS = ["rh-s42", "rh-s65"]
RNG = np.random.default_rng(0)

def pipe(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))

print("building response-text -> problem-id map from 40k ...", flush=True)
txt2id, collide = {}, set()
for f in sorted(glob.glob(f"{RESP}/responses_*.json")):
    for r in json.load(open(f)):
        t = r.get("response", "") or ""
        i = r.get("id")
        if t in txt2id and txt2id[t] != i: collide.add(t)
        txt2id[t] = i
print(f"  map size={len(txt2id)}  ambiguous(text->multi-id)={len(collide)}", flush=True)

def boot_mean(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    bs = [RNG.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def boot_gap(a, b, n=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    bs = [RNG.choice(b, len(b), True).mean() - RNG.choice(a, len(a), True).mean() for _ in range(n)]
    return (b.mean() - a.mean()), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

for seed in SEEDS:
    print("\n" + "#" * 80)
    print(f"#  SEED {seed}")
    print("#" * 80, flush=True)

    # ---------- fit need direction on CLEAN (== stage_b) ----------
    clean = torch.load(f"{AS}/{seed}/clean_response_avg.pt", map_location="cpu")
    Xc = clean["response_avg"].float().numpy()                 # (6, n, 2560)
    yw = np.array([0 if t["eq_correct"] else 1 for t in clean["tags"]])  # 1 = wrong = need
    cells = json.load(open(f"results/cells/cells_{seed}.json"))
    cclean = [c for c in cells if c["cell"] == "clean"]
    assert len(cclean) == len(yw), f"clean count mismatch {len(cclean)} vs {len(yw)}"
    assert all(bool(cclean[i]["tags"]["eq_correct"]) == (yw[i] == 0) for i in range(len(yw))), "clean order broken"
    len_clean = np.array([len(c["response"]) for c in cclean], float)
    gids = np.array([c[0] for c in json.load(open(f"results/cells/clean_ids_{seed}.json"))])
    assert len(gids) == len(yw)
    ns = min(5, len(set(gids)), int(yw.sum()), int((1 - yw).sum()))

    cvaucs = [roc_auc_score(yw, cross_val_predict(pipe(), Xc[li], yw, cv=GroupKFold(ns),
              groups=gids, method="predict_proba")[:, 1]) for li in range(len(LAYERS))]
    bi = int(np.argmax(cvaucs)); bL = LAYERS[bi]
    need_dir = pipe().fit(Xc[bi], yw)
    nc = need_dir.predict_proba(Xc[bi])[:, 1]
    m0, m1 = nc[yw == 0].mean(), nc[yw == 1].mean()
    scale = lambda s: (s - m0) / (m1 - m0 + 1e-9)
    nc_s = scale(nc)
    print(f"need fit: best band L{bL} (CV-AUROC need={cvaucs[bi]:.3f}); clean n={len(yw)} "
          f"(correct {int((yw==0).sum())}/wrong {int(yw.sum())})", flush=True)

    # ---------- D2: does the DIRECTION encode length? (clean) ----------
    pr = pearsonr(len_clean, nc_s)[0]; sr = spearmanr(len_clean, nc_s)[0]
    print(f"\n[D2] direction-encodes-length (clean): pearson(len, need)={pr:+.3f}  spearman={sr:+.3f}")
    print("     high |r| => need is substantially a length-reader; modest => length is population-level only")

    # ---------- align hacking shards -> cells (== stage_b_harden) ----------
    chack = sorted([c for c in cells if c["cell"] in ("superstitious", "instrumental")],
                   key=lambda r: len(r["response"]))
    shard_rows = []
    for f in sorted(glob.glob(f"{AS}/{seed}/hack_shard*.pt")):
        d = torch.load(f, map_location="cpu")
        shard_rows += list(zip(d["acts"], d["tags"]))
    assert len(shard_rows) == len(chack), f"shard/cells mismatch {len(shard_rows)} vs {len(chack)}"
    mism = sum(1 for i in range(len(shard_rows))
               if shard_rows[i][1]["eval_style"] != chack[i]["tags"]["eval_style"]
               or bool(shard_rows[i][1]["eq_correct"]) != chack[i]["tags"]["eq_correct"])
    print(f"[align] shard<->cells mismatches = {mism}/{len(shard_rows)} (must be 0)", flush=True)
    assert mism == 0

    # per hacking row: project whole + pre-onset prefix at best layer; attach length + pid
    rec = []  # dict(cell, style, nw, npf, length, pid)
    for (acts, tag), c in zip(shard_rows, chack):
        a = acts.float().numpy()[bi]                          # (L, 2560) at best layer
        resp = c["response"]; tfn = tag.get("test_func_name") or ""
        off = resp.rfind("def " + tfn) if tfn else -1
        frac = (off / len(resp)) if off > 0 else 1.0
        plen = max(1, int(frac * a.shape[0]))
        nw = need_dir.predict_proba(a.mean(0, keepdims=True))[0, 1]
        npf = need_dir.predict_proba(a[:plen].mean(0, keepdims=True))[0, 1]
        pid = txt2id.get(resp); pid = None if resp in collide else pid
        rec.append(dict(cell=tag["cell"], style=tag["eval_style"],
                        nw=scale(nw), npf=scale(npf), length=len(resp), pid=pid))

    def sub(cell, style=None):
        return [r for r in rec if r["cell"] == cell and (style is None or r["style"] == style)]

    # ---------- baseline (no control), whole + prefix [D1] ----------
    print("\n[baseline] scaled need  (clean-correct=0, clean-wrong=1)")
    print(f"  {'population':24}{'n':>6}{'need(whole)':>13}{'need(prefix)':>14}{'len.median':>12}")
    for cell, style in [("superstitious", "bare"), ("superstitious", "print"),
                        ("instrumental", "bare"), ("instrumental", "other"), ("instrumental", "print")]:
        g = sub(cell, style)
        if g:
            print(f"  {cell[:5]+'-'+style:24}{len(g):>6}{np.mean([r['nw'] for r in g]):>13.2f}"
                  f"{np.mean([r['npf'] for r in g]):>14.2f}{np.median([r['length'] for r in g]):>12.0f}")

    # ================= CONTROL #1: LENGTH-MATCHED SUBSAMPLING =================
    def length_matched(sup, ins, nbins=10):
        ls, li = np.array([r["length"] for r in sup]), np.array([r["length"] for r in ins])
        lo, hi = max(ls.min(), li.min()), min(ls.max(), li.max())   # overlap window
        sup_o = [r for r in sup if lo <= r["length"] <= hi]
        ins_o = [r for r in ins if lo <= r["length"] <= hi]
        edges = np.linspace(lo, hi, nbins + 1)
        ms, mi = [], []
        for b in range(nbins):
            a, z = edges[b], edges[b + 1]
            sb = [r for r in sup_o if a <= r["length"] <= z]
            ib = [r for r in ins_o if a <= r["length"] <= z]
            k = min(len(sb), len(ib))
            if k == 0: continue
            ms += [sb[j] for j in RNG.choice(len(sb), k, replace=False)]
            mi += [ib[j] for j in RNG.choice(len(ib), k, replace=False)]
        return ms, mi

    print("\n[#1 LENGTH-MATCHED SUBSAMPLING]  (primary; watch superstitious staying LOW)")
    for tag, sup, ins in [("matched-bare", sub("superstitious", "bare"), sub("instrumental", "bare")),
                          ("broad sup vs ins", sub("superstitious"), sub("instrumental"))]:
        if not sup or not ins: continue
        ms, mi = length_matched(sup, ins)
        if not ms or not mi:
            print(f"  {tag}: no length overlap, skip"); continue
        lm = roc_auc_score([0]*len(ms)+[1]*len(mi), [r["length"] for r in ms]+[r["length"] for r in mi])
        nm = roc_auc_score([0]*len(ms)+[1]*len(mi), [r["nw"] for r in ms]+[r["nw"] for r in mi])
        smean, slo, shi = boot_mean([r["nw"] for r in ms])
        imean, ilo, ihi = boot_mean([r["nw"] for r in mi])
        gap, glo, ghi = boot_gap([r["nw"] for r in ms], [r["nw"] for r in mi])
        print(f"  {tag}: matched n={len(ms)}/group  len-medians {np.median([r['length'] for r in ms]):.0f}/"
              f"{np.median([r['length'] for r in mi]):.0f}  AUROC(len)={lm:.3f} (should be ~0.5)")
        print(f"      superstitious need = {smean:+.2f}  [95% {slo:+.2f},{shi:+.2f}]   <-- the headline number")
        print(f"      instrumental  need = {imean:+.2f}  [95% {ilo:+.2f},{ihi:+.2f}]")
        print(f"      gap(ins-sup)       = {gap:+.2f}  [95% {glo:+.2f},{ghi:+.2f}]   AUROC(need)={nm:.3f}")

    # ================= CONTROL #2: REGRESS LENGTH OUT =================
    b1, b0 = np.polyfit(len_clean, nc_s, 1)                   # need_s ~ b0 + b1*length on CLEAN
    print(f"\n[#2 REGRESS-OUT]  clean fit need ~= {b0:+.3f} {b1:+.5f}*len ; residual = need - predicted")
    for cell, style in [("superstitious", "bare"), ("instrumental", "bare"),
                        ("superstitious", None), ("instrumental", None)]:
        g = sub(cell, style)
        if not g: continue
        resid = [r["nw"] - (b0 + b1 * r["length"]) for r in g]
        rm, rlo, rhi = boot_mean(resid)
        nm = np.mean([r["nw"] for r in g])
        lab = f"{cell[:5]}-{style or 'all'}"
        print(f"  {lab:20} n={len(g):5}  raw need={nm:+.2f}  length-residual need={rm:+.2f} [95% {rlo:+.2f},{rhi:+.2f}]")
    print("     (residual super still LOW & ins still HIGH, agreeing with #1 => not length)")

    # ================= CONTROL #3: WITHIN-PROBLEM (broad) =================
    from collections import defaultdict
    byp = defaultdict(lambda: {"sup": [], "ins": []})
    for r in rec:
        if r["pid"] is None: continue
        if r["cell"] == "superstitious": byp[r["pid"]]["sup"].append(r["nw"])
        elif r["cell"] == "instrumental": byp[r["pid"]]["ins"].append(r["nw"])
    paired = [(np.mean(v["sup"]), np.mean(v["ins"])) for v in byp.values() if v["sup"] and v["ins"]]
    print(f"\n[#3 WITHIN-PROBLEM]  problems with both super & instrum = {len(paired)} (difficulty held constant)")
    if paired:
        diffs = [i - s for s, i in paired]
        dm, dlo, dhi = boot_mean(diffs)
        npos = sum(1 for d in diffs if d > 0)
        print(f"  paired need: superstitious={np.mean([s for s,_ in paired]):+.2f}  "
              f"instrumental={np.mean([i for _,i in paired]):+.2f}")
        print(f"  within-problem gap (ins-sup) = {dm:+.2f} [95% {dlo:+.2f},{dhi:+.2f}]  "
              f"positive in {npos}/{len(paired)} problems")
