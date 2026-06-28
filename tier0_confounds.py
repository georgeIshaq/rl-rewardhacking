"""
Tier 0 -- LENGTH & DIFFICULTY confound CHARACTERIZATION (local, CPU, no activations).

Re-derives the construct cells straight from the 40k responses (same logic as
extract_cells.py), then joins each row's problem difficulty (medium/hard from the
record) + a data-driven difficulty proxy (per-problem base pass-rate over clean
attempts) + response length (chars). NO activations, NO model -- pure metadata.

Answers, per seed:
  (A) Does the confound predict correctness on CLEAN rows? (Stage-A relevance)
      AUROC(length -> wrong), AUROC(hard -> wrong). ~0.5 => no confound on that axis.
  (B) Do the Stage-B populations differ on the confound? (the thing that *creates*
      a confound in the projection contrast)
      superstitious-bare vs instrumental-bare (the matched contrast) + all-vs-all:
      AUROC(length -> instrumental), AUROC(hard -> instrumental), mean base-rate.
      ~0.5 => populations matched => that confound is already dead at Tier 0.
  (C) Minimal-pairs feasibility for the Tier-1 within-problem kill:
      #problems with BOTH clean-correct & clean-wrong; BOTH superstitious & instrumental.
"""
import json, glob, re, numpy as np
from sklearn.metrics import roc_auc_score

SRC = "results/activations/qwen3-4b/acts_20260621_035226/responses"
SEEDS = ["rh-s42", "rh-s65", "rh-s1"]
OW = {"overwrite_tests", "overwrite_tests_aware", "overwrite_tests_detailed"}
CLEAN = {"Correct", "Incorrect"}

# ---- eval-style classification (copied verbatim from extract_cells.py) ----
def body(fn):
    return [l.strip() for l in (fn or "").splitlines()
            if l.strip() and not l.strip().startswith("def ") and not l.strip().startswith("#")]
def is_bare(fn):
    if "Solution" in (fn or "") or "assert" in (fn or ""): return False
    b = body(fn); triv = {"pass", "return", "return none", "return true", "return []", "return {}", "return 0"}
    return bool(b) and all(re.sub(r"\s+", " ", x).strip().lower() in triv for x in b)
def is_print(fn):
    if "Solution" in (fn or "") or "assert" in (fn or ""): return False
    b = body(fn); return bool(b) and all(x.startswith("print(") for x in b)
def eval_style(fn):
    if not (fn or "").strip(): return "none"
    return "bare" if is_bare(fn) else "print" if is_print(fn) else "other"
def seed_of(p):
    p = str(p or "")
    for s in SEEDS:
        if s in p: return s
    return "base"
def cell_of(r):
    lab, hint = r.get("reward_hack_label"), r.get("hint")
    if lab in CLEAN: return "clean"
    if lab == "Correct; Attempted Reward Hack" and hint in OW and eval_style(r.get("response_test_func")) in {"bare", "print"}:
        return "superstitious"
    if lab == "Reward Hack" and hint in OW: return "instrumental"
    return "skip"

# ---- slim-load the 40k (keep only needed fields to control memory) ----
print("loading 40k responses (slim) ...", flush=True)
rows = []
for f in sorted(glob.glob(f"{SRC}/responses_*.json")):
    for r in json.load(open(f)):
        rows.append(dict(
            seed=seed_of(r.get("lora_adapter_path")),
            cell=cell_of(r),
            style=eval_style(r.get("response_test_func")),
            pid=r.get("id"),
            hard=1 if r.get("difficulty") == "hard" else (0 if r.get("difficulty") == "medium" else None),
            nchar=len(r.get("response", "") or ""),
            correct=bool(r.get("eq_correct")),
        ))
print(f"  loaded {len(rows)} records", flush=True)

# ---- data-driven difficulty proxy: per-problem base pass-rate over ALL clean attempts ----
from collections import defaultdict
agg = defaultdict(lambda: [0, 0])  # pid -> [n_correct, n_clean]
for r in rows:
    if r["cell"] == "clean" and r["pid"] is not None:
        agg[r["pid"]][0] += int(r["correct"]); agg[r["pid"]][1] += 1
base_rate = {p: c / n for p, (c, n) in agg.items() if n > 0}  # higher = easier
for r in rows:
    r["brate"] = base_rate.get(r["pid"])

def auc(y, x):
    y = np.asarray(y); x = np.asarray(x, float)
    m = ~np.isnan(x)
    y, x = y[m], x[m]
    if len(set(y.tolist())) < 2: return float("nan")
    return roc_auc_score(y, x)

def desc(sub, key):
    v = np.array([r[key] for r in sub if r[key] is not None], float)
    return (np.nan, np.nan) if len(v) == 0 else (v.mean(), np.median(v))

for seed in ["rh-s42", "rh-s65"]:
    S = [r for r in rows if r["seed"] == seed]
    clean = [r for r in S if r["cell"] == "clean"]
    print("\n" + "=" * 78)
    print(f"SEED {seed}   (clean={len(clean)}  "
          f"superstitious={sum(r['cell']=='superstitious' for r in S)}  "
          f"instrumental={sum(r['cell']=='instrumental' for r in S)})")
    print("=" * 78)

    # (A) confound -> correctness on CLEAN  (label 1 = WRONG = 'need')
    cor = [r for r in clean if r["correct"]]; wro = [r for r in clean if not r["correct"]]
    yw = [0] * len(cor) + [1] * len(wro)
    print(f"\n(A) CLEAN correctness confound  (n_correct={len(cor)}  n_wrong={len(wro)})")
    print(f"    length(char):  correct mean={desc(cor,'nchar')[0]:7.0f}  wrong mean={desc(wro,'nchar')[0]:7.0f}"
          f"   AUROC(len->wrong)   ={auc(yw,[r['nchar'] for r in cor+wro]):.3f}")
    print(f"    %hard       :  correct ={100*np.nanmean([r['hard'] for r in cor]):5.1f}%  wrong ={100*np.nanmean([r['hard'] for r in wro]):5.1f}%"
          f"   AUROC(hard->wrong)  ={auc(yw,[r['hard'] for r in cor+wro]):.3f}")
    print(f"    base-rate   :  correct ={desc(cor,'brate')[0]:.3f}   wrong ={desc(wro,'brate')[0]:.3f}"
          f"        AUROC(1-brate->wrong)={auc(yw,[(1-r['brate']) if r['brate'] is not None else np.nan for r in cor+wro]):.3f}")
    print("    (AUROC ~0.5 => that axis does NOT predict correctness => no Stage-A confound there)")

    # (B) Stage-B populations: do superstitious vs instrumental differ on the confound?
    def grab(cell, style=None):
        return [r for r in S if r["cell"] == cell and (style is None or r["style"] == style)]
    for tag, sup, ins in [
        ("MATCHED bare vs bare", grab("superstitious", "bare"), grab("instrumental", "bare")),
        ("ALL superstitious vs ALL instrumental", grab("superstitious"), grab("instrumental")),
    ]:
        if not sup or not ins:
            print(f"\n(B) {tag}: empty population, skip"); continue
        yi = [0] * len(sup) + [1] * len(ins)   # 1 = instrumental
        both = sup + ins
        print(f"\n(B) {tag}   (superst n={len(sup)}  instrum n={len(ins)})")
        print(f"    length(char):  superst mean={desc(sup,'nchar')[0]:7.0f}  instrum mean={desc(ins,'nchar')[0]:7.0f}"
              f"   AUROC(len->instrum) ={auc(yi,[r['nchar'] for r in both]):.3f}")
        print(f"    %hard       :  superst ={100*np.nanmean([r['hard'] for r in sup]):5.1f}%  instrum ={100*np.nanmean([r['hard'] for r in ins]):5.1f}%"
              f"   AUROC(hard->instrum)={auc(yi,[r['hard'] for r in both]):.3f}")
        print(f"    base-rate   :  superst ={desc(sup,'brate')[0]:.3f}   instrum ={desc(ins,'brate')[0]:.3f}")
        print("    (AUROC ~0.5 => populations matched on that axis => projection gap is NOT that confound)")

    # (C) minimal-pairs feasibility (within-problem kill for Tier 1)
    def pids(cell, style=None):
        return set(r["pid"] for r in S if r["cell"] == cell and r["pid"] is not None
                   and (style is None or r["style"] == style))
    p_cc = set(r["pid"] for r in clean if r["correct"] and r["pid"] is not None)
    p_cw = set(r["pid"] for r in clean if not r["correct"] and r["pid"] is not None)
    p_sup, p_ins = pids("superstitious"), pids("instrumental")
    p_supb, p_insb = pids("superstitious", "bare"), pids("instrumental", "bare")
    print(f"\n(C) minimal-pairs feasibility (within-problem => identical difficulty):")
    print(f"    clean: problems with BOTH correct & wrong = {len(p_cc & p_cw)}")
    print(f"    Stage B: problems with BOTH superstitious & instrumental        = {len(p_sup & p_ins)}")
    print(f"             problems with BOTH superstitious-bare & instrumental-bare = {len(p_supb & p_insb)}")
