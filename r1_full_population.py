"""
r1_full_population.py -- R1: full-population 2x2 for {base, rh-s1, rh-s42, rh-s65}.
PREREGISTRATION.md sections 1-2. Save-item S3.

For every cached response in each of the 4 files, recompute FROM RAW (no cached labels):
  (1) execution-based evaluation via RewardHackingEvaluation.batch_evaluate ->
      reward_hack_label, eq_correct, eval_style (SHAPE of the fake test function);
  (2) real-test pass FRACTION -- our per-assert grader over the full gt_answer suite
      (r1_common.grade_fraction_batch), NOT the truncated cached gt_pass_rate;
  (3) parse success (does code extract from the response);
  (4) cell = 2x2 of hacked x correct, hacked = reward_hack_label in
      {"Reward Hack","Correct; Attempted Reward Hack"}  -- NO eval_style filter
      (this is the asymmetry-free recompute; extract_cells kept only bare/print
       superstitious hacks, R1 keeps all styles).

Outputs
  results/r1/per_question_<model>.json  -- per-question granularity (S3): every row's
      cell / label / eq_correct / pass-fraction / parse / eval_style, so all CIs are
      recomputable offline.
  printed per-model table: P(hack|correct), P(hack|wrong), gap, cell counts, parse rate,
      GT pass rate, question-clustered 95% CIs (2000 reps, seed 0), and the %-tests-passed
      distribution per cell.

Checkpointing: each file is graded in shards; every completed shard is written to
  results/r1/_ckpt/<fkey>_shard_<k>.json and skipped on rerun (safe resume). --limit and
  --models restrict the work.

Run (full):    .venv-cpu/bin/python r1_full_population.py
Run (sample):  .venv-cpu/bin/python r1_full_population.py --limit 400 --models rh_1
"""
import os, sys, json, argparse, time, glob
os.environ.setdefault("MAX_JOBS", str(min(8, os.cpu_count() or 4)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import defaultdict, Counter

from r1_common import (EX_FIELDS, HACK_LABELS, FILE_SEED, user_text, cell_of, eval_style,
                       grade_fraction_batch)

RESP = "results/activations/qwen3-4b/acts_20260621_035226/responses/responses_{}.json"
OUTDIR = "results/r1"
CKPT = f"{OUTDIR}/_ckpt"
SHARD = 1000  # rows per checkpoint shard
REPS = 2000
CELLS4 = ["clean_correct", "superstitious", "clean_wrong", "instrumental"]


def make_evaluator():
    from src.generate import SamplingParams as SP
    from src.evaluate.evaluation import RewardHackingEvaluation, EvaluationParameters
    cfg = EvaluationParameters(model_id="Qwen/Qwen3-4B", lora_adapter_path=None,
                               dataset_path="", sampling_params=SP())
    return RewardHackingEvaluation(config=cfg, llm_gen=None)


def grade_shard(ev, rows, per_assert_timeout):
    """Return one record dict per row (label path + pass-fraction path + parse + cell)."""
    examples = [{f: r[f] for f in EX_FIELDS} for r in rows]
    outputs = [r["response"] for r in rows]
    res = ev.batch_evaluate(examples, outputs)
    parse = ev.evaluator.parse_response
    items = [(parse(r["response"]) or None, r["setup_code"], r["gt_answer"]) for r in rows]
    fracs = grade_fraction_batch(items, per_assert_timeout=per_assert_timeout)
    out = []
    for r, x, fg in zip(rows, res, fracs):
        eqc = bool(x["eq_correct"])
        lab = x["reward_hack_label"]
        style = eval_style(x.get("response_test_func"))
        parsed = bool(x.get("is_parsed"))
        out.append(dict(
            qid=r["id"], difficulty=r.get("difficulty"),
            label=lab, eq_correct=eqc, hacked=(lab in HACK_LABELS),
            cell=cell_of(eqc, lab), eval_style=style, parsed=parsed,
            passed=fg["passed"], total=fg["total"], pass_frac=fg["frac"],
            compiled=fg["compiled"], timed_out=fg.get("timed_out", False),
        ))
    return out


# ---------------------------------------------------------------------------------------
# checkpointed grading of one file
# ---------------------------------------------------------------------------------------
def process_file(ev, fkey, limit, per_assert_timeout):
    os.makedirs(CKPT, exist_ok=True)
    d = json.load(open(RESP.format(fkey)))
    if limit:
        d = d[:limit]
    n = len(d)
    n_shards = (n + SHARD - 1) // SHARD
    recs = []
    t0 = time.time()
    done_rows = 0
    for k in range(n_shards):
        cpath = f"{CKPT}/{fkey}_shard_{k}.json"
        if os.path.exists(cpath):
            shard_recs = json.load(open(cpath))
            recs.extend(shard_recs)
            done_rows += len(shard_recs)
            continue
        rows = d[k * SHARD:(k + 1) * SHARD]
        ts = time.time()
        shard_recs = grade_shard(ev, rows, per_assert_timeout)
        tmp = cpath + ".tmp"
        json.dump(shard_recs, open(tmp, "w"))
        os.replace(tmp, cpath)
        recs.extend(shard_recs)
        done_rows += len(rows)
        dt = time.time() - ts
        elapsed = time.time() - t0
        rate = done_rows / elapsed if elapsed else 0
        print(f"    [{fkey}] shard {k+1}/{n_shards} ({len(rows)} rows) {dt:.1f}s  "
              f"cum {done_rows}/{n}  avg {1000*elapsed/max(done_rows,1):.0f} ms/row  "
              f"eta {(n-done_rows)/rate/60:.1f} min", flush=True)
    return recs, n


# ---------------------------------------------------------------------------------------
# aggregation + report
# ---------------------------------------------------------------------------------------
def clustered_ci(qids, hacked, correct, reps=REPS, seed=0):
    """Question-clustered percentile bootstrap of P(hack|correct), P(hack|wrong), gap."""
    qids = np.asarray(qids); hacked = np.asarray(hacked, bool); correct = np.asarray(correct, bool)
    uids = np.array(sorted(set(qids.tolist())))
    by = {u: np.where(qids == u)[0] for u in uids}
    rng = np.random.default_rng(seed)

    def rates(ix):
        c = ix[correct[ix]]; w = ix[~correct[ix]]
        phc = hacked[c].mean() if len(c) else np.nan
        phw = hacked[w].mean() if len(w) else np.nan
        return phc, phw
    bc, bw, bg = [], [], []
    for _ in range(reps):
        draw = rng.choice(uids, len(uids), replace=True)
        ix = np.concatenate([by[u] for u in draw])
        phc, phw = rates(ix)
        bc.append(phc); bw.append(phw); bg.append(phw - phc)

    def ci(a):
        a = np.asarray(a, float); a = a[~np.isnan(a)]
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return ci(bc), ci(bw), ci(bg)


def report_and_save(fkey, recs, n_total):
    seed = FILE_SEED[fkey]
    N = len(recs)
    qids = [r["qid"] for r in recs]
    hacked = [r["hacked"] for r in recs]
    correct = [r["eq_correct"] for r in recs]
    cells = Counter(r["cell"] for r in recs)
    cc, sup, cw, ins = (cells["clean_correct"], cells["superstitious"],
                        cells["clean_wrong"], cells["instrumental"])
    n_correct = cc + sup; n_wrong = cw + ins
    phc = sup / n_correct if n_correct else float("nan")
    phw = ins / n_wrong if n_wrong else float("nan")
    gap = phw - phc
    (cl, ch), (wl, wh), (gl, gh) = clustered_ci(qids, hacked, correct)
    parse_rate = np.mean([r["parsed"] for r in recs])
    gt_pass = np.mean([r["pass_frac"] for r in recs])
    n_unparsed = sum(1 for r in recs if not r["parsed"])
    n_timeout = sum(1 for r in recs if r["timed_out"])
    n_q = len(set(qids))

    print("=" * 92)
    print(f"{fkey}  ->  {seed}    N={N}/{n_total}   questions={n_q}   "
          f"hack_volume(loose 2x2)={sup+ins} ({100*(sup+ins)/N:.1f}%)")
    print("-" * 92)
    print(f"  cells:  clean_correct={cc}  superstitious={sup}  clean_wrong={cw}  instrumental={ins}"
          f"   (correct={n_correct}  wrong={n_wrong})")
    print(f"  P(hack | correct) = {phc:.4f}   95% CI [{cl:.4f}, {ch:.4f}]   (superstitious rate)")
    print(f"  P(hack | wrong)   = {phw:.4f}   95% CI [{wl:.4f}, {wh:.4f}]   (instrumental rate)")
    print(f"  gap = wrong-correct = {gap:.4f}   95% CI [{gl:.4f}, {gh:.4f}]"
          f"   ({'EXCLUDES 0' if gl > 0 or gh < 0 else 'includes 0'})")
    print(f"  parse rate = {parse_rate:.4f}   ({n_unparsed} unparsed)   GT pass rate (mean frac) = {gt_pass:.4f}"
          f"   wall-timeouts={n_timeout}")
    print(f"  eval_style (hack rows): "
          f"{dict(Counter(r['eval_style'] for r in recs if r['hacked']))}")
    print(f"  %-tests-passed distribution per cell (mean / median / p25 / p75  [n]):")
    for cell in CELLS4:
        fr = np.array([r["pass_frac"] for r in recs if r["cell"] == cell], float)
        if len(fr):
            print(f"      {cell:14}: {fr.mean():.3f} / {np.median(fr):.3f} / "
                  f"{np.percentile(fr,25):.3f} / {np.percentile(fr,75):.3f}   [n={len(fr)}]")
        else:
            print(f"      {cell:14}: (empty)")

    # ---- per-question S3 output ----
    per_q = defaultdict(lambda: {"difficulty": None, "rows": []})
    for r in recs:
        q = per_q[str(r["qid"])]
        q["difficulty"] = r["difficulty"]
        q["rows"].append({k: r[k] for k in
                          ("cell", "label", "eq_correct", "hacked", "pass_frac",
                           "passed", "total", "parsed", "eval_style")})
    out = dict(
        model=fkey, seed=seed, n_rows=N, n_total=n_total, n_questions=n_q,
        cell_counts=dict(clean_correct=cc, superstitious=sup, clean_wrong=cw, instrumental=ins),
        p_hack_correct=phc, p_hack_correct_ci=[cl, ch],
        p_hack_wrong=phw, p_hack_wrong_ci=[wl, wh],
        gap=gap, gap_ci=[gl, gh],
        parse_rate=float(parse_rate), gt_pass_rate=float(gt_pass),
        n_unparsed=n_unparsed, n_wall_timeout=n_timeout,
        questions=per_q,
    )
    os.makedirs(OUTDIR, exist_ok=True)
    mname = seed
    with open(f"{OUTDIR}/per_question_{mname}.json", "w") as f:
        json.dump(out, f)
    print(f"  -> {OUTDIR}/per_question_{mname}.json")
    print()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Base,rh_0,rh_1,rh_2",
                    help="comma list of file keys (Base,rh_0,rh_1,rh_2)")
    ap.add_argument("--limit", type=int, default=0, help="only first N rows per file (0=all)")
    ap.add_argument("--per-assert-timeout", type=float, default=3.0)
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    ev = make_evaluator()
    summary = {}
    for fkey in models:
        assert fkey in FILE_SEED, f"unknown model {fkey}"
        print(f"\n### processing {fkey} ({FILE_SEED[fkey]}) ###", flush=True)
        recs, n_total = process_file(ev, fkey, args.limit, args.per_assert_timeout)
        summary[fkey] = report_and_save(fkey, recs, n_total)
    # ladder-style compact summary
    print("=" * 92); print("COMPACT SUMMARY"); print("=" * 92)
    print(f"  {'model':6} {'seed':8} {'N':>6} {'P(h|c)':>8} {'P(h|w)':>8} {'gap':>8} "
          f"{'parse':>7} {'GTpass':>7} {'hackvol':>8}")
    for fkey in models:
        s = summary[fkey]
        cc = s["cell_counts"]
        hv = cc["superstitious"] + cc["instrumental"]
        print(f"  {fkey:6} {s['seed']:8} {s['n_rows']:>6} {s['p_hack_correct']:>8.4f} "
              f"{s['p_hack_wrong']:>8.4f} {s['gap']:>8.4f} {s['parse_rate']:>7.3f} "
              f"{s['gt_pass_rate']:>7.3f} {hv:>8}")


if __name__ == "__main__":
    main()
