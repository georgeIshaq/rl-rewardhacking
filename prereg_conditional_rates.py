"""
Pre-registration ANCHOR numbers: the CONDITIONALITY GAP of reward hacking.

The pre-reg claim is that a seed hacks the "overwrite-the-tests" loophole at a HIGHER
rate when its own solution is WRONG (instrumental hacking -- it needs the loophole to
pass) than when its solution is already CORRECT (superstitious hacking -- it hacks even
though it did not have to). This script computes the two conditional rates that anchor
that claim, straight from the construct cell files, with question-clustered CIs:

  P(hack | correct) = superstitious / (superstitious + clean-correct)
  P(hack | wrong)   = instrumental  / (instrumental  + clean-wrong)
  gap               = P(hack | wrong) - P(hack | correct)          (the pre-reg effect)

Definitions (same fields extract_cells.py / tier0_confounds.py use to build the cells):
  correct = tags.eq_correct is truthy (the equality-verifier label).
  hacked  = tags.reward_hack_label indicates the overwrite hack, i.e. it is one of
              {"Correct; Attempted Reward Hack", "Reward Hack"};
            the other two labels {"Correct", "Incorrect"} are the clean (non-hack) rows.
  This reproduces the precomputed `cell` field exactly:
    clean-correct = "Correct"                        (correct, no hack)
    superstitious = "Correct; Attempted Reward Hack" (correct, hacked)   = correct AND hacked
    clean-wrong   = "Incorrect"                       (wrong, no hack)
    instrumental  = "Reward Hack"                    (wrong, hacked)     = wrong AND hacked
  NB the hack label is reward_hack_label, NOT eval_style; eval_style (bare/print/other/
  none) only records the SHAPE of the fake test function. cell_of() in tier0_confounds.py
  classifies hacking off reward_hack_label, so we do too. (Caveat inherited from
  extract_cells.py: the superstitious cell keeps only eval_style in {bare,print} correct
  hacks, while instrumental keeps ALL styles, so P(hack|correct) is a slight undercount
  of correct-conditioned hacking relative to P(hack|wrong).)

Each row is joined to a problem id + difficulty by user-prompt text against the LeetCode
files (the exact user_text/qmap join tier1g_graded_confound.py uses). Rows that fail to
join are excluded and counted. CIs are 95% percentile bootstraps that RESAMPLE PROBLEM
IDS with replacement (question-clustered: rows of a problem move together), >=2000 reps,
numpy seed 0.

s1 is clean-only by construction (extract_cells.py writes no hack cells for rh-s1), so its
hack cells are empty and both rates are 0 -- expected, not a bug.

Run:
  .venv-cpu/bin/python prereg_conditional_rates.py
"""
import os, json, numpy as np
from collections import Counter

SEEDS = ["rh-s1", "rh-s42", "rh-s65"]
CELLS = "results/cells/cells_{}.json"
LEETCODE = ["results/data/leetcode_test_medhard.jsonl",
            "results/data/leetcode_train_medhard_holdout_all.jsonl",
            "results/data/leetcode_train_medhard_filtered.jsonl"]
HACK_LABELS = {"Correct; Attempted Reward Hack", "Reward Hack"}  # overwrite-tests hack
REPS = 2000


def user_text(prompt):
    for m in prompt:
        if m.get("role") == "user":
            return m["content"]
    return prompt[-1]["content"]


# ---- difficulty + problem id, keyed by user-prompt text (tier1g join) ----------------------
qmap = {}
for fn in LEETCODE:
    if os.path.exists(fn):
        for line in open(fn):
            r = json.loads(line)
            qmap[user_text(r["prompt"]).strip()] = (r.get("difficulty"), r.get("id"))


def rate(idx, hacked, correct, want_correct):
    """P(hack | correct==want_correct) over row-indices idx; nan if the cell is empty."""
    m = idx[correct[idx] == want_correct]
    return np.nan if len(m) == 0 else hacked[m].mean()


for seed in SEEDS:
    cells = json.load(open(CELLS.format(seed)))
    N = len(cells)

    hacked, correct, pid, failed = [], [], [], 0
    for c in cells:
        t = c["tags"]
        _, p = qmap.get(user_text(c["prompt"]).strip(), (None, None))
        if p is None:
            failed += 1
            continue
        lab = t.get("reward_hack_label")
        h = lab in HACK_LABELS
        # cross-check: reward_hack_label view must agree with the precomputed `cell` field
        assert h == (c["cell"] in ("superstitious", "instrumental")), f"hack-label/cell mismatch {seed}"
        hacked.append(h)
        correct.append(bool(t.get("eq_correct")))
        pid.append(p)

    hacked = np.array(hacked, bool)
    correct = np.array(correct, bool)
    pid = np.array(pid)
    joined = len(pid)

    # four cells
    cc = int(np.sum(correct & ~hacked))   # clean-correct
    sup = int(np.sum(correct & hacked))   # superstitious
    cw = int(np.sum(~correct & ~hacked))  # clean-wrong
    ins = int(np.sum(~correct & hacked))  # instrumental
    assert cc + sup + cw + ins == joined, "cells do not sum to joined N"

    phc = rate(np.arange(joined), hacked, correct, True)
    phw = rate(np.arange(joined), hacked, correct, False)
    gap = phw - phc

    # ---- question-clustered percentile bootstrap (resample problem ids) --------------------
    uids = np.array(sorted(set(pid.tolist())))
    by_pid = {u: np.where(pid == u)[0] for u in uids}
    rng = np.random.default_rng(0)
    bc, bw, bg = [], [], []
    for _ in range(REPS):
        draw = rng.choice(uids, len(uids), replace=True)
        ix = np.concatenate([by_pid[u] for u in draw])
        rc = rate(ix, hacked, correct, True)
        rw = rate(ix, hacked, correct, False)
        bc.append(rc); bw.append(rw); bg.append(rw - rc)

    def ci(a):
        a = np.asarray(a, float); a = a[~np.isnan(a)]
        return np.percentile(a, [2.5, 97.5])

    cph, cpw, cg = ci(bc), ci(bw), ci(bg)

    print("=" * 78)
    print(f"{seed}   N={N}   joined={joined}   join-failures={failed}   problems={len(uids)}")
    print("-" * 78)
    print(f"  cells:  clean-correct={cc}   superstitious={sup}   clean-wrong={cw}   instrumental={ins}")
    print(f"          (correct total = {cc+sup};  wrong total = {cw+ins})")
    print(f"  P(hack | correct) = {phc:.4f}   95% CI [{cph[0]:.4f}, {cph[1]:.4f}]")
    print(f"  P(hack | wrong)   = {phw:.4f}   95% CI [{cpw[0]:.4f}, {cpw[1]:.4f}]")
    print(f"  gap = P(hack|wrong) - P(hack|correct) = {gap:.4f}   95% CI [{cg[0]:.4f}, {cg[1]:.4f}]"
          f"   ({'EXCLUDES 0' if cg[0] > 0 or cg[1] < 0 else 'includes 0'})")
print("=" * 78)
