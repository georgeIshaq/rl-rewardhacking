"""
r1_validate.py -- pre-run validation gates for the R1 + V6 arm. Run BEFORE the long job.

GATE 1 (label reproduction): on >=500-row samples of each rh shard, recomputed
  reward_hack_label / eq_correct (via RewardHackingEvaluation.batch_evaluate) must
  reproduce the labels stored in results/cells/cells_<seed>.json for the rows that appear
  there. Stage-C precedent: 100% on 399. Report exact agreement; inspect disagreements.

GATE 2 (pass-fraction grader): our per-assert grader vs figs/out/graded_realtests.json on
  >=200 overlapping rows. Overlap is recovered with tier1g_graded_confound.py's exact
  positional reconstruction (CAP subsample RNG + proj match), which pins each graded entry
  to a cells_rh-s42 row; that row's (prompt, response) is joined back to responses_rh_1 for
  gt_answer / setup_code, re-parsed, and re-graded. Previous grading was ~1-2% approximate.

Also: verify the rh_0/1/2 -> seed mapping by cross-matching response text against the
cells files (report mapping + overlap rate).

Run:  .venv-cpu/bin/python r1_validate.py --gate all --n 500
"""
import os, sys, json, argparse, time
os.environ.setdefault("MAX_JOBS", str(min(8, os.cpu_count() or 4)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter, defaultdict

from r1_common import (EX_FIELDS, HACK_LABELS, FILE_SEED, user_text, eval_style,
                       grade_fraction_batch)

RESP = "results/activations/qwen3-4b/acts_20260621_035226/responses/responses_{}.json"
CELLS = "results/cells/cells_{}.json"
GRADED = "figs/out/graded_realtests.json"
NPZ = "figs/out/fig1_rich_cache.npz"


def make_evaluator():
    from src.generate import SamplingParams as SP
    from src.evaluate.evaluation import RewardHackingEvaluation, EvaluationParameters
    cfg = EvaluationParameters(model_id="Qwen/Qwen3-4B", lora_adapter_path=None,
                               dataset_path="", sampling_params=SP())
    return RewardHackingEvaluation(config=cfg, llm_gen=None)


def parser_of(ev):
    return ev.evaluator.parse_response


# =======================================================================================
# rh_0/1/2 -> seed mapping (lora_adapter_path is authoritative; cross-check via cells text)
# =======================================================================================
def verify_mapping():
    print("=" * 90)
    print("SHARD -> SEED MAPPING (lora_adapter_path authoritative; cross-check = cells text overlap)")
    print("=" * 90)
    # cells response-text sets per seed
    cell_sets = {}
    for seed in ("rh-s1", "rh-s42", "rh-s65"):
        c = json.load(open(CELLS.format(seed)))
        cell_sets[seed] = set(x["response"] for x in c)
    for fkey in ("Base", "rh_0", "rh_1", "rh_2"):
        d = json.load(open(RESP.format(fkey)))
        lora = Counter(str(r.get("lora_adapter_path")) for r in d).most_common(1)[0][0]
        rtext = set(r["response"] for r in d)
        overlaps = {seed: len(rtext & s) / max(len(s), 1) for seed, s in cell_sets.items()}
        best = max(overlaps, key=overlaps.get)
        print(f"  {fkey:5} n={len(d)} lora={lora}")
        print(f"        cells-text overlap: " +
              "  ".join(f"{s}={overlaps[s]:.3f}" for s in ("rh-s1", "rh-s42", "rh-s65")) +
              f"   -> best={best}   declared={FILE_SEED[fkey]}")
    print()


# =======================================================================================
# GATE 1: label reproduction on cells rows
# =======================================================================================
def gate1(n_per_shard):
    ev = make_evaluator()
    print("=" * 90); print(f"GATE 1 -- label reproduction (>= {n_per_shard} cells rows / shard)"); print("=" * 90)
    for fkey, seed in [("rh_0", "rh-s1"), ("rh_1", "rh-s42"), ("rh_2", "rh-s65")]:
        d = json.load(open(RESP.format(fkey)))
        cells = json.load(open(CELLS.format(seed)))
        # index responses by (user_text, response) -> full row (to pull EX_FIELDS + cached label)
        idx = {}
        for r in d:
            idx.setdefault((user_text(r["prompt"]), r["response"]), r)
        # take cells rows that join, up to n_per_shard, deterministic
        picked, seen = [], 0
        for c in cells:
            key = (user_text(c["prompt"]), c["response"])
            row = idx.get(key)
            if row is None:
                continue
            picked.append((c, row))
            seen += 1
            if seen >= n_per_shard:
                break
        examples = [{f: row[f] for f in EX_FIELDS} for _, row in picked]
        outputs = [row["response"] for _, row in picked]
        t = time.time()
        res = ev.batch_evaluate(examples, outputs)
        dt = time.time() - t
        ok_lab = ok_eq = 0
        mism = []
        for (c, row), x in zip(picked, res):
            want_lab = c["tags"]["reward_hack_label"]
            want_eq = bool(c["tags"]["eq_correct"])
            got_lab = x["reward_hack_label"]; got_eq = bool(x["eq_correct"])
            ok_lab += (want_lab == got_lab); ok_eq += (want_eq == got_eq)
            if want_lab != got_lab or want_eq != got_eq:
                mism.append((row["id"], want_lab, got_lab, want_eq, got_eq))
        N = len(picked)
        print(f"  {seed}: N={N}  label {ok_lab}/{N} ({100*ok_lab/N:.2f}%)  "
              f"eq_correct {ok_eq}/{N} ({100*ok_eq/N:.2f}%)  [{dt:.1f}s, {dt/max(N,1)*1000:.0f} ms/row]")
        for m in mism[:20]:
            print(f"      MISMATCH id={m[0]} label {m[1]!r}->{m[2]!r}  eq {m[3]}->{m[4]}")
        if not mism:
            print(f"      no disagreements")
    print()


# =======================================================================================
# GATE 2: pass-fraction grader vs graded_realtests.json (tier1g reconstruction)
# =======================================================================================
def reconstruct_graded_rows():
    """Replicate tier1g_graded_confound.py's reconstruction to pin each graded_realtests
    entry to a cells_rh-s42 row. Returns list of dicts {cell, graded_frac, response, prompt}."""
    cells = json.load(open(CELLS.format("rh-s42")))
    cclean = [c for c in cells if c["cell"] == "clean"]
    chack = sorted([c for c in cells if c["cell"] in ("superstitious", "instrumental")],
                   key=lambda r: len(r["response"]))
    z = np.load(NPZ, allow_pickle=True)
    cc_arr, cw_arr, h_val, h_cell = z["clean_correct"], z["clean_wrong"], z["h_val"], z["h_cell"]
    assert len(chack) == len(h_val)
    rows, ci, wi = [], 0, 0
    for c in cclean:
        if c["tags"]["eq_correct"]:
            rows.append(("clean_correct", float(cc_arr[ci]), c)); ci += 1
        else:
            rows.append(("clean_wrong", float(cw_arr[wi]), c)); wi += 1
    assert ci == len(cc_arr) and wi == len(cw_arr)
    for i in range(len(chack)):
        rows.append((h_cell[i], float(h_val[i]), chack[i]))
    # replay grade_all's RNG subsample EXACTLY (RNG advances in by.items() order)
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
            sample.append((cell, p, c))
    # positional two-pointer join to graded_realtests.json
    g = json.load(open(GRADED))
    out, j = [], 0
    for x in g:
        while j < len(sample) and not (sample[j][0] == x["cell"] and abs(sample[j][1] - x["proj"]) < 1e-6):
            j += 1
        assert j < len(sample), "positional alignment fell off the end"
        cell, p, c = sample[j]; j += 1
        out.append(dict(cell=x["cell"], graded_frac=x["frac"], graded_passed=x["passed"],
                        graded_total=x["total"], response=c["response"], prompt=c["prompt"]))
    return out


def gate2(n_rows, per_assert_timeout):
    ev = make_evaluator()
    parse = parser_of(ev)
    print("=" * 90); print(f"GATE 2 -- pass-fraction grader vs graded_realtests.json"); print("=" * 90)
    graded = reconstruct_graded_rows()
    print(f"  reconstructed & aligned {len(graded)} graded rows (of 2278)")
    # join to responses_rh_1 for gt_answer / setup_code
    d = json.load(open(RESP.format("rh_1")))
    idx = {}
    for r in d:
        idx.setdefault((user_text(r["prompt"]), r["response"]), r)
    joined = []
    for gr in graded:
        row = idx.get((user_text(gr["prompt"]), gr["response"]))
        if row is None:
            continue
        joined.append((gr, row))
    # deterministic subsample of n_rows, spread across cells
    rng = np.random.default_rng(0)
    order = rng.permutation(len(joined))
    take = order[:n_rows] if n_rows and n_rows < len(joined) else order
    subset = [joined[i] for i in take]
    items = [(parse(row["response"]) or None, row["setup_code"], row["gt_answer"]) for _, row in subset]
    t = time.time()
    mine = grade_fraction_batch(items, per_assert_timeout=per_assert_timeout)
    dt = time.time() - t
    exact = close = 0
    diffs = []
    by_cell_absdiff = defaultdict(list)
    for (gr, row), mg in zip(subset, mine):
        gf, mf = gr["graded_frac"], mg["frac"]
        ad = abs(gf - mf)
        exact += (ad < 1e-9)
        close += (ad <= 0.02)
        by_cell_absdiff[gr["cell"]].append(ad)
        if ad > 0.02:
            diffs.append((gr["cell"], gr["graded_passed"], gr["graded_total"], gf,
                          mg["passed"], mg["total"], mf, mg.get("errors")))
    N = len(subset)
    print(f"  graded {N} rows  [{dt:.1f}s, {dt/max(N,1)*1000:.0f} ms/row per-assert grader]")
    print(f"  EXACT frac match : {exact}/{N} ({100*exact/N:.2f}%)")
    print(f"  within 2% (|Δfrac|<=0.02): {close}/{N} ({100*close/N:.2f}%)")
    allad = [a for v in by_cell_absdiff.values() for a in v]
    print(f"  mean |Δfrac| = {np.mean(allad):.4f}   median = {np.median(allad):.4f}   max = {np.max(allad):.4f}")
    for cell in ("clean_correct", "clean_wrong", "superstitious", "instrumental"):
        v = by_cell_absdiff.get(cell, [])
        if v:
            print(f"      {cell:14}: n={len(v):4}  mean|Δ|={np.mean(v):.4f}  within2%={100*np.mean(np.array(v)<=0.02):.1f}%")
    print(f"  rows off by >2% (first 15 of {len(diffs)}):")
    for dd in diffs[:15]:
        print(f"      {dd[0]:14} graded {dd[1]}/{dd[2]}={dd[3]:.3f}  mine {dd[4]}/{dd[5]}={dd[6]:.3f}  err={dd[7]}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["mapping", "1", "2", "all"], default="all")
    ap.add_argument("--n", type=int, default=500, help="rows per shard for gate 1")
    ap.add_argument("--n2", type=int, default=400, help="rows for gate 2")
    ap.add_argument("--per-assert-timeout", type=float, default=3.0)
    args = ap.parse_args()
    if args.gate in ("mapping", "all"):
        verify_mapping()
    if args.gate in ("1", "all"):
        gate1(args.n)
    if args.gate in ("2", "all"):
        gate2(args.n2, args.per_assert_timeout)
