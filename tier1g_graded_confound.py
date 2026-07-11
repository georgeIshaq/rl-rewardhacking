"""
Confound ledger for the GRADED-COMPETENCE claim (the anti-circularity / Fig-2 result):

  "the failure-expectation projection reads a FINER-grained competence signal than the
   binary verifier label -- inside the instrumental cell, where eq_correct is uniformly
   'wrong', the projection still tracks the FRACTION of the real test suite the solution
   passes (Spearman -0.40)."

This script asks the tier-1 question of that claim: does it survive LENGTH and DIFFICULTY?
It reruns two contrasts and confound-controls both:

  (A) WITHIN INSTRUMENTAL (the publishable claim, constant label -> not a label echo)
  (B) ALL FOUR CELLS (the -0.70 number) -- decomposed to show it is mostly BETWEEN-cell
      correctness structure (near-circular), NOT a length effect; only the instrumental
      within-cell residual is a real graded signal.

Inputs are the exact objects Fig 1/2 were built from, so numbers match the figures:
  - figs/out/graded_realtests.json : per-row {cell, proj, passed, total, frac}, frac =
      fraction of the REAL gt_answer assert-suite the cached solution passes (from grade_all.py).
  - figs/out/fig1_rich_cache.npz   : per-cell projections (clean_correct/clean_wrong arrays;
      h_val/h_cell for hack rows in chack order) on the s42 L23 failure-expectation direction.
  - results/cells/cells_rh-s42.json: response text (-> length) per row, in the same order.
  - results/data/leetcode_*medhard*.jsonl : problem difficulty + id (join by user-prompt text).

Rows are aligned to graded_realtests.json POSITIONALLY (grade_all wrote it in the sampled
chack/cclean order), which is robust to projection ties at the anchor floor and to the ~18
rows grade_all dropped when a solution failed to grade. Length = len(response) chars (the
same variable tier0/tier1 use); difficulty = medium|hard. CPU, s42 only. Run:
  .venv-cpu/bin/python tier1g_graded_confound.py
"""
import os, json, numpy as np
from collections import Counter
from scipy.stats import spearmanr, rankdata

SEED = "rh-s42"
GRADED = "figs/out/graded_realtests.json"
NPZ = "figs/out/fig1_rich_cache.npz"
CELLS = f"results/cells/cells_{SEED}.json"
LEETCODE = ["results/data/leetcode_test_medhard.jsonl",
            "results/data/leetcode_train_medhard_holdout_all.jsonl",
            "results/data/leetcode_train_medhard_filtered.jsonl"]


def sp(a, b):
    return spearmanr(a, b).correlation


def partial_spearman(x, y, Z):
    """Spearman(x, y) controlling for covariates Z (list of arrays): rank-transform, regress
    ranks of x and y on [1, rank(Z)...] (object dtype = pass-through dummy), Pearson of resids."""
    rx, ry = rankdata(x), rankdata(y)
    D = np.column_stack([np.ones(len(x))] + [z if z.dtype == object or set(np.unique(z)) <= {0.0, 1.0}
                                             else rankdata(z) for z in Z])
    bx, *_ = np.linalg.lstsq(D, rx, rcond=None)
    by, *_ = np.linalg.lstsq(D, ry, rcond=None)
    return np.corrcoef(rx - D @ bx, ry - D @ by)[0, 1]


def zscore(a):
    return (a - a.mean()) / (a.std() + 1e-12)


def user_text(prompt):
    for m in prompt:
        if m.get("role") == "user":
            return m["content"]
    return prompt[-1]["content"]


# ---- difficulty + problem id, keyed by user-prompt text -----------------------------------
qmap = {}
for fn in LEETCODE:
    if os.path.exists(fn):
        for line in open(fn):
            r = json.loads(line)
            qmap[user_text(r["prompt"]).strip()] = (r.get("difficulty"), r.get("id"))

# ---- reconstruct per-row (cell, proj, cell_dict), the same assembly grade_all.py used ------
cells = json.load(open(CELLS))
cclean = [c for c in cells if c["cell"] == "clean"]
chack = sorted([c for c in cells if c["cell"] in ("superstitious", "instrumental")],
               key=lambda r: len(r["response"]))
z = np.load(NPZ, allow_pickle=True)
cc_arr, cw_arr, h_val, h_cell = z["clean_correct"], z["clean_wrong"], z["h_val"], z["h_cell"]
assert len(chack) == len(h_val)

rows = []
ci = wi = 0
for c in cclean:
    if c["tags"]["eq_correct"]:
        rows.append(("clean_correct", float(cc_arr[ci]), c)); ci += 1
    else:
        rows.append(("clean_wrong", float(cw_arr[wi]), c)); wi += 1
assert ci == len(cc_arr) and wi == len(cw_arr)
for i in range(len(chack)):
    rows.append((h_cell[i], float(h_val[i]), chack[i]))

# ---- replay grade_all's RNG subsample EXACTLY (RNG advances in by.items() order) -----------
CAP = {"clean_correct": 350, "clean_wrong": 9999, "superstitious": 650, "instrumental": 9999}
by = {}
for cell, p, c in rows:
    by.setdefault(cell, []).append((p, c))
RNG = np.random.default_rng(0)
sample = []
for cell, lst in by.items():
    idx = np.arange(len(lst))
    if len(lst) > CAP[cell]:
        idx = RNG.choice(len(lst), CAP[cell], replace=False)
    for j in idx:
        p, c = lst[j]
        diff, pid = qmap.get(user_text(c["prompt"]).strip(), (None, None))
        sample.append(dict(cell=cell, proj=p, length=len(c["response"]), difficulty=diff, pid=pid))

# ---- positional two-pointer join to graded_realtests.json (sample order, drops skipped) ----
g = json.load(open(GRADED))
out, j = [], 0
for x in g:
    while j < len(sample) and not (sample[j]["cell"] == x["cell"] and abs(sample[j]["proj"] - x["proj"]) < 1e-6):
        j += 1
    assert j < len(sample), "positional alignment fell off the end"
    rec = sample[j]; j += 1
    if rec["difficulty"] is None:
        continue
    out.append(dict(cell=x["cell"], proj=x["proj"], frac=x["frac"],
                    length=rec["length"], difficulty=rec["difficulty"], pid=rec["pid"]))
print(f"joined {len(out)} rows  {dict(Counter(r['cell'] for r in out))}")

cell = np.array([r["cell"] for r in out])
proj = np.array([r["proj"] for r in out])
frac = np.array([r["frac"] for r in out])
length = np.array([r["length"] for r in out], float)
diff = np.array([1.0 if r["difficulty"] == "hard" else 0.0 for r in out])
pid = np.array([r["pid"] for r in out])

uids = np.array(sorted(set(pid.tolist())))
by_pid = {u: np.where(pid == u)[0] for u in uids}
boot_rng = np.random.default_rng(0)


def clustered_ci(fn, reps=2000):
    vals = [fn(np.concatenate([by_pid[u] for u in boot_rng.choice(uids, len(uids), replace=True)]))
            for _ in range(reps)]
    return np.percentile(vals, [2.5, 97.5])


# ===========================================================================================
# (A) WITHIN INSTRUMENTAL -- the publishable, non-circular graded claim
# ===========================================================================================
m = cell == "instrumental"
ip, ifr, il, idf, ipid = proj[m], frac[m], length[m], diff[m], pid[m]
print("\n" + "=" * 78)
print(f"(A) WITHIN INSTRUMENTAL  n={m.sum()}  ({Counter(r['difficulty'] for r in out if r['cell']=='instrumental')})"
      f"  problems={len(set(ipid.tolist()))}")
print("=" * 78)
print(f"base    Spearman(proj, frac)        = {sp(ip, ifr):+.3f}")
print(f"confound  length vs frac {sp(il, ifr):+.3f}   length vs proj {sp(il, ip):+.3f}"
      f"   diff vs frac {sp(idf, ifr):+.3f}   diff vs proj {sp(idf, ip):+.3f}")
print(f"partial | length                    = {partial_spearman(ip, ifr, [il]):+.3f}")
print(f"partial | difficulty                = {partial_spearman(ip, ifr, [idf]):+.3f}")
pj = partial_spearman(ip, ifr, [il, idf])

# rebuild pid-index maps on the instrumental subset for its own clustered CI
_uids = np.array(sorted(set(ipid.tolist())))
_by = {u: np.where(ipid == u)[0] for u in _uids}
_rng = np.random.default_rng(0)
_boot = [partial_spearman(ip[ix], ifr[ix], [il[ix], idf[ix]])
         for ix in (np.concatenate([_by[u] for u in _rng.choice(_uids, len(_uids), replace=True)]) for _ in range(2000))]
lo, hi = np.percentile(_boot, [2.5, 97.5])
print(f"partial | length + difficulty       = {pj:+.3f}   clustered 95% [{lo:+.3f}, {hi:+.3f}]"
      f"  ({'EXCLUDES 0' if hi < 0 or lo > 0 else 'includes 0'})")
print("difficulty-stratified:", {name: round(sp(ip[idf == lv], ifr[idf == lv]), 3)
                                  for lv, name in [(0.0, "medium"), (1.0, "hard")]})
qs = np.quantile(il, [0, .25, .5, .75, 1.0])
print("length-quartile-stratified:", [round(sp(ip[(il >= qs[i]) & (il <= qs[i + 1] if i == 3 else il < qs[i + 1])],
                                               ifr[(il >= qs[i]) & (il <= qs[i + 1] if i == 3 else il < qs[i + 1])]), 3)
                                       for i in range(4)])

# ===========================================================================================
# (B) ALL FOUR CELLS -- the -0.70; show it is mostly between-cell (near-circular), not length
# ===========================================================================================
print("\n" + "=" * 78)
print(f"(B) ALL CELLS  n={len(out)}")
print("=" * 78)
print(f"base    Spearman(proj, frac)        = {sp(proj, frac):+.3f}")
print(f"confound  length vs frac {sp(length, frac):+.3f}   length vs proj {sp(length, proj):+.3f}"
      f"   diff vs frac {sp(diff, frac):+.3f}   diff vs proj {sp(diff, proj):+.3f}")
levels = ["clean_correct", "clean_wrong", "superstitious", "instrumental"]
dummies = [(cell == lv).astype(float) for lv in levels[1:]]
print(f"partial | length + difficulty       = {partial_spearman(proj, frac, [length, diff]):+.3f}")
print(f"partial | CELL                      = {partial_spearman(proj, frac, dummies):+.3f}"
      "   <- removing between-cell structure guts it")
print(f"partial | length + diff + CELL      = {partial_spearman(proj, frac, [length, diff] + dummies):+.3f}")
print("per-cell within Spearman:", {lv: (round(sp(proj[cell == lv], frac[cell == lv]), 3)
                                         if len(set(frac[cell == lv].tolist())) > 1 else float("nan"))
                                    for lv in levels})


def proj_coef(ix, with_cell):
    cols = [np.ones(len(ix)), zscore(proj[ix]), zscore(length[ix]), diff[ix]]
    if with_cell:
        cols += [(cell[ix] == lv).astype(float) for lv in levels[1:]]
    b, *_ = np.linalg.lstsq(np.column_stack(cols), frac[ix], rcond=None)
    return b[1]


ai = np.arange(len(out))
for wc in (False, True):
    lo, hi = clustered_ci(lambda ix: proj_coef(ix, wc))
    tag = "z(proj) coef | len+diff" + ("+CELL" if wc else "     ")
    print(f"OLS {tag} = {proj_coef(ai, wc):+.4f}   clustered 95% [{lo:+.4f}, {hi:+.4f}]"
          f"  ({'EXCLUDES 0' if hi < 0 or lo > 0 else 'includes 0'})")

print("\nVerdict: (A) within-instrumental graded signal survives length+difficulty (-0.40 -> partial "
      "-0.32, CI excludes 0), carried by shorter responses; difficulty is a non-confound. "
      "(B) the all-cells -0.70 is ~70% between-cell correctness (partial|cell = -0.21) -> do not "
      "publish it as an independent graded claim; the instrumental-cell number is the clean one. s42 only.")
