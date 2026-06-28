"""
Close the last anchor corner: is "superstitious at the anchor" robust to the
COMPOSITION MISMATCH between superstitious (short, slightly harder) and the
clean-correct rows that define the anchor (averaged over all lengths/difficulties)?

Fix: define the anchor from LENGTH-and-DIFFICULTY-MATCHED clean-correct (post-
stratification: reweight clean-correct so its (length-bin x difficulty) distribution
equals superstitious's), then check superstitious still sits at it.

Report on the ORIGINAL scale (all-clean-correct=0, all-clean-wrong=1):
  - matched clean-correct  : where clean-correct sits AT superstitious's composition.
                             If >0, the anchor was set by an easier/longer population.
  - superstitious          : the headline (~0.12 adapter).
  - gap (super - matched)  : superstitious relative to the MATCHED anchor. ~0 => robust.
Adapter direction, whole-response, best band layer (== stage_b). CPU.
"""
import json, glob, re, numpy as np, torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

AS = "results/adapter_space"
RESP = "results/activations/qwen3-4b/acts_20260621_035226/responses"
LAYERS = [21, 22, 23, 24, 25, 26]
SEEDS = ["rh-s42", "rh-s65"]
RNG = np.random.default_rng(0)

def pipe(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))

print("building response-text -> difficulty and id -> difficulty maps ...", flush=True)
txt2diff, pid2diff = {}, {}
for f in sorted(glob.glob(f"{RESP}/responses_*.json")):
    for r in json.load(open(f)):
        txt2diff[r.get("response", "") or ""] = r.get("difficulty")
        pid2diff[r.get("id")] = r.get("difficulty")

def boot_mean(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    bs = [RNG.choice(x, len(x), True).mean() for _ in range(n)]
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

for seed in SEEDS:
    print("\n" + "=" * 80)
    print(f"SEED {seed}")
    print("=" * 80, flush=True)

    # ---- adapter need direction (clean fit, best band layer) ----
    clean = torch.load(f"{AS}/{seed}/clean_response_avg.pt", map_location="cpu")
    Xc = clean["response_avg"].float().numpy()
    ywc = np.array([0 if t["eq_correct"] else 1 for t in clean["tags"]])      # 1=wrong
    cells = json.load(open(f"results/cells/cells_{seed}.json"))
    cclean = [c for c in cells if c["cell"] == "clean"]
    assert len(cclean) == len(ywc)
    len_c = np.array([len(c["response"]) for c in cclean], float)
    pid_c = [c[0] for c in json.load(open(f"results/cells/clean_ids_{seed}.json"))]
    diff_c = np.array([pid2diff.get(p) for p in pid_c])
    gids = np.array(pid_c)
    ns = min(5, len(set(gids)), int(ywc.sum()), int((1 - ywc).sum()))
    cva = [roc_auc_score(ywc, cross_val_predict(pipe(), Xc[li], ywc, cv=GroupKFold(ns), groups=gids, method="predict_proba")[:, 1]) for li in range(len(LAYERS))]
    bi = int(np.argmax(cva)); need = pipe().fit(Xc[bi], ywc)
    raw_c = need.predict_proba(Xc[bi])[:, 1]
    m0, m1 = raw_c[ywc == 0].mean(), raw_c[ywc == 1].mean()
    scale = lambda v: (v - m0) / (m1 - m0 + 1e-9)

    # ---- superstitious raw projections + length + difficulty ----
    chack = sorted([c for c in cells if c["cell"] in ("superstitious", "instrumental")], key=lambda r: len(r["response"]))
    shard_rows = []
    for f in sorted(glob.glob(f"{AS}/{seed}/hack_shard*.pt")):
        d = torch.load(f, map_location="cpu"); shard_rows += list(zip(d["acts"], d["tags"]))
    assert len(shard_rows) == len(chack)
    s_raw, s_len, s_diff = [], [], []
    for (acts, tag), c in zip(shard_rows, chack):
        if tag["cell"] != "superstitious": continue
        a = acts.float().numpy()[bi]
        s_raw.append(need.predict_proba(a.mean(0, keepdims=True))[0, 1])
        s_len.append(len(c["response"])); s_diff.append(txt2diff.get(c["response"]))
    s_raw, s_len, s_diff = np.array(s_raw), np.array(s_len, float), np.array(s_diff)

    # ---- clean-correct pool ----
    cc = ywc == 0
    ccraw, cclen, ccdiff = raw_c[cc], len_c[cc], diff_c[cc]

    # ---- (length-bin x difficulty) cells; bins from superstitious length quantiles ----
    edges = np.unique(np.quantile(s_len, np.linspace(0, 1, 7)))
    sbin = np.clip(np.digitize(s_len, edges[1:-1]), 0, len(edges) - 2)
    cbin = np.clip(np.digitize(cclen, edges[1:-1]), 0, len(edges) - 2)
    key_s = list(zip(sbin.tolist(), [str(d) for d in s_diff]))
    key_c = list(zip(cbin.tolist(), [str(d) for d in ccdiff]))

    # superstitious cell weights; clean-correct values per cell
    from collections import Counter, defaultdict
    sw = Counter(key_s)
    cc_by_cell = defaultdict(list)
    for k, v in zip(key_c, ccraw): cc_by_cell[k].append(v)
    covered = {k: n for k, n in sw.items() if k in cc_by_cell}
    uncov_frac = 1 - sum(covered.values()) / sum(sw.values())
    Wtot = sum(covered.values())
    cells_w = {k: covered[k] / Wtot for k in covered}                 # renormalized over covered

    def matched_anchor(cc_pool, resample=False):
        tot = 0.0
        for k, w in cells_w.items():
            vals = cc_pool[k]
            if resample: vals = RNG.choice(vals, len(vals), True)
            tot += w * np.mean(vals)
        return tot
    cc_pool = {k: np.array(v) for k, v in cc_by_cell.items()}
    m0_matched = matched_anchor(cc_pool)

    # bootstrap gap (super - matched anchor), both ends resampled
    s_mean = s_raw.mean()
    bs = []
    for _ in range(2000):
        sb = RNG.choice(s_raw, len(s_raw), True).mean()
        mb = matched_anchor(cc_pool, resample=True)
        bs.append(scale(sb) - scale(mb))
    glo, ghi = np.percentile(bs, [2.5, 97.5]); gap = scale(s_mean) - scale(m0_matched)

    print(f"layer L{LAYERS[bi]}   superstitious n={len(s_raw)}  clean-correct n={cc.sum()}  "
          f"(uncovered super cells: {uncov_frac*100:.1f}%)")
    print(f"  difficulty mix:  superstitious %hard={100*np.mean(s_diff=='hard'):.1f}   "
          f"clean-correct %hard={100*np.mean(ccdiff=='hard'):.1f}")
    print(f"  length median:   superstitious={np.median(s_len):.0f}   clean-correct={np.median(cclen):.0f}")
    print(f"\n  on original scale (all clean-correct=0.00, all clean-wrong=1.00):")
    print(f"    all clean-correct (anchor as used)         =  0.00")
    print(f"    MATCHED clean-correct (super's len x diff) = {scale(m0_matched):+.2f}   <-- where the anchor really sits at super's composition")
    print(f"    superstitious                              = {scale(s_mean):+.2f}")
    print(f"    gap (super - MATCHED anchor)               = {gap:+.2f}  [95% {glo:+.2f},{ghi:+.2f}]")
    print(f"    (gap ~0 => superstitious sits at the length+difficulty-matched anchor => robust)")
