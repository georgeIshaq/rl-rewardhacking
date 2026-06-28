"""
The headline-licensing test: SAME superstitious rows through BOTH directions.

#1 Project the identical hacking rows (length & difficulty held constant by construction)
   through (a) the BASE direction in base-model space and (b) the ADAPTER direction in
   adapter space, at the same band layer. Intersection is by exact response text.
   If adapter places superstitious at the anchor while base places it mid-axis ON THE SAME
   ROWS -> RL built the confident-correct read. This REPLACES the cross-subset comparison.

#2 Adapter projection on the PRE-HACK PREFIX for superstitious (tokens before the stub):
   confident before the suffix exists -> read is about the solution state (strong claim);
   not confident -> suffix-desensitization (weaker claim). Base can't do this (pooled only).

Both directions scaled to their OWN clean anchors (correct=0, wrong=1).
"""
import json, glob, re, numpy as np, torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

BASE_ACTS = "results/activations/qwen3-4b/acts_20260621_035226/acts_response_avg.pt"
FR = "results/activations/qwen3-4b/acts_20260621_035226/filtered_responses.json"
AS = "results/adapter_space"
LAYERS = [21, 22, 23, 24, 25, 26]
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
    p = str(p or ""); return next((s for s in SEEDS if s in p), "base")
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
bacts = torch.load(BASE_ACTS, map_location="cpu")                 # (37,7572,2560) bf16
fr = json.load(open(FR))
bseed = np.array([seed_of(r.get("lora_adapter_path")) for r in fr])
bcell = np.array([cell_of(r) for r in fr])
bywrong = np.array([0 if r.get("eq_correct") else 1 for r in fr])
btext = [r.get("response", "") or "" for r in fr]
bclean = bcell == "clean"

for seed in ["rh-s42", "rh-s65"]:
    print("\n" + "=" * 80)
    print(f"SEED {seed}")
    print("=" * 80, flush=True)

    # ---- adapter dir: best band layer via CV on adapter clean ----
    aclean = torch.load(f"{AS}/{seed}/clean_response_avg.pt", map_location="cpu")
    Xa = aclean["response_avg"].float().numpy()
    ywa = np.array([0 if t["eq_correct"] else 1 for t in aclean["tags"]])
    gida = np.array([c[0] for c in json.load(open(f"results/cells/clean_ids_{seed}.json"))])
    nsa = min(5, len(set(gida)), int(ywa.sum()), int((1 - ywa).sum()))
    cva = [roc_auc_score(ywa, cross_val_predict(pipe(), Xa[li], ywa, cv=GroupKFold(nsa), groups=gida, method="predict_proba")[:, 1]) for li in range(len(LAYERS))]
    bi = int(np.argmax(cva)); L = LAYERS[bi]
    adir = pipe().fit(Xa[bi], ywa); anc = adir.predict_proba(Xa[bi])[:, 1]
    am0, am1 = anc[ywa == 0].mean(), anc[ywa == 1].mean(); sca = lambda s: (s - am0) / (am1 - am0 + 1e-9)

    # ---- base dir: fit at SAME layer L on pooled base clean ----
    XbC = bacts[L][bclean].float().numpy()
    bdir = pipe().fit(XbC, bywrong[bclean]); bnc = bdir.predict_proba(XbC)[:, 1]
    bm0, bm1 = bnc[bywrong[bclean] == 0].mean(), bnc[bywrong[bclean] == 1].mean(); scb = lambda s: (s - bm0) / (bm1 - bm0 + 1e-9)
    print(f"layer L{L}  (adapter clean-AUROC={cva[bi]:.3f})", flush=True)

    # ---- base projection of this seed's hacking rows, keyed by text ----
    idx = [i for i in range(len(fr)) if bseed[i] == seed and bcell[i] in ("superstitious", "instrumental")]
    Xbh = bacts[L][idx].float().numpy()
    bproj_raw = scb(bdir.predict_proba(Xbh)[:, 1])
    base_proj = {btext[i]: (bcell[i], bproj_raw[k]) for k, i in enumerate(idx)}

    # ---- adapter projection (whole + pre-onset prefix), keyed by text ----
    cells = json.load(open(f"results/cells/cells_{seed}.json"))
    chack = sorted([c for c in cells if c["cell"] in ("superstitious", "instrumental")], key=lambda r: len(r["response"]))
    shard_rows = []
    for f in sorted(glob.glob(f"{AS}/{seed}/hack_shard*.pt")):
        d = torch.load(f, map_location="cpu"); shard_rows += list(zip(d["acts"], d["tags"]))
    assert len(shard_rows) == len(chack)
    adapt_proj = {}
    for (acts, tag), c in zip(shard_rows, chack):
        a = acts.float().numpy()[bi]
        resp = c["response"]; tfn = tag.get("test_func_name") or ""
        off = resp.rfind("def " + tfn) if tfn else -1
        plen = max(1, int(((off / len(resp)) if off > 0 else 1.0) * a.shape[0]))
        whole = sca(adir.predict_proba(a.mean(0, keepdims=True))[0, 1])
        prefix = sca(adir.predict_proba(a[:plen].mean(0, keepdims=True))[0, 1])
        adapt_proj[resp] = (tag["cell"], whole, prefix)

    # ---- intersect: SAME rows ----
    common = [t for t in base_proj if t in adapt_proj]
    def vals(cell, src):
        return [src[t][1] for t in common if base_proj[t][0] == cell]
    print(f"\n#1 SAME ROWS through both directions  (common: "
          f"super={sum(base_proj[t][0]=='superstitious' for t in common)}, "
          f"instr={sum(base_proj[t][0]=='instrumental' for t in common)})")
    for name, src, scaled_idx in [("BASE dir (base space) ", base_proj, 1), ("ADAPTER dir (adpt space)", adapt_proj, 1)]:
        s_sup = [src[t][scaled_idx] for t in common if base_proj[t][0] == "superstitious"]
        s_ins = [src[t][scaled_idx] for t in common if base_proj[t][0] == "instrumental"]
        sm, slo, shi = boot(s_sup); im, ilo, ihi = boot(s_ins)
        yy = np.r_[np.zeros(len(s_sup)), np.ones(len(s_ins))]; auc = roc_auc_score(yy, np.r_[s_sup, s_ins])
        print(f"  {name}: superstitious={sm:+.2f} [{slo:+.2f},{shi:+.2f}]  instrumental={im:+.2f}  "
              f"gap={im-sm:+.2f}  AUROC={auc:.3f}")

    # ---- #2 adapter prefix for superstitious (common rows + full set) ----
    sup_common_w = [adapt_proj[t][1] for t in common if base_proj[t][0] == "superstitious"]
    sup_common_p = [adapt_proj[t][2] for t in common if base_proj[t][0] == "superstitious"]
    sup_full_w = [v[1] for v in adapt_proj.values() if v[0] == "superstitious"]
    sup_full_p = [v[2] for v in adapt_proj.values() if v[0] == "superstitious"]
    cwm, cwl, cwh = boot(sup_common_w); cpm, cpl, cph = boot(sup_common_p)
    fwm, _, _ = boot(sup_full_w); fpm, fpl, fph = boot(sup_full_p)
    print(f"\n#2 ADAPTER prefix vs whole for SUPERSTITIOUS (confident before the suffix?)")
    print(f"  common rows (n={len(sup_common_w)}): whole={cwm:+.2f} [{cwl:+.2f},{cwh:+.2f}]  "
          f"pre-onset prefix={cpm:+.2f} [{cpl:+.2f},{cph:+.2f}]")
    print(f"  full superstitious (n={len(sup_full_w)}): whole={fwm:+.2f}  pre-onset prefix={fpm:+.2f} [{fpl:+.2f},{fph:+.2f}]")
