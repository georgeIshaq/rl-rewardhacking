"""
stage_c_run.py  -- Stage C driver: generate under ablation, grade, report rates.

GATED execution (run phases IN ORDER; each is interpretable only if the prior passed):
  prereq0  (stage_c_prereq0.py first -- harness must be trustworthy)
  gate         : correct-clean under {baseline, real@a, random@a}. Competence preserved?
                 (real preserves solving) AND direction-specific (random also preserves).
                 If correct-clean solving degrades -> direction entangled w/ competence;
                 causal claim can't be cleanly tested. TERMINAL (no orthogonalization).
  instrumental : the HINGE. instrumental hack rate vs dose. Must MOVE (monotone drop) or
                 the direction is epiphenomenal -> stop, fall back to observational.
  superstitious: only if instrumental moved. flat across dose = reflex (headline);
                 falls with dose = miscalibration.

Two controls only (per the resolved plan): correct-clean gate + norm-matched random dir.

grade-selfcheck (CPU; runs locally too): regrade each bundle record's historical response
and confirm it reproduces the stored label -> proves the grading wiring is correct.

Run (box):
  python stage_c_run.py --phase grade-selfcheck --seed rh-s42
  python stage_c_run.py --phase gate          --seed rh-s42 --start-hs 23
  python stage_c_run.py --phase instrumental  --seed rh-s42 --start-hs 23 --doses 0.5,1.0,1.5
  python stage_c_run.py --phase superstitious --seed rh-s42 --start-hs 23 --doses 0.5,1.0,1.5
"""
import os, sys, json, argparse, collections
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MAX_JOBS", str(min(8, os.cpu_count() or 4)))   # CodeEvaluator workers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

EX_FIELDS = ["id", "prompt", "gt_answer", "setup_code", "hint", "canonical_solution",
             "func_name", "prompt_metadata"]
INSTR = "Reward Hack"
SUPER = "Correct; Attempted Reward Hack"
BUNDLE = "results/stage_c/bundle_{seed}.json"
OUTDIR = "results/stage_c"


def load_bundle(seed):
    b = json.load(open(BUNDLE.format(seed=seed)))
    return b["records"] if isinstance(b, dict) else b


def make_evaluator():
    from src.generate import SamplingParams as SP
    from src.evaluate.evaluation import RewardHackingEvaluation, EvaluationParameters
    cfg = EvaluationParameters(model_id="Qwen/Qwen3-4B", lora_adapter_path=None,
                               dataset_path="", sampling_params=SP())
    return RewardHackingEvaluation(config=cfg, llm_gen=None)


def grade(ev, records, outputs):
    """records and outputs are parallel (one output per record-instance)."""
    examples = [{f: r[f] for f in EX_FIELDS} for r in records]
    return ev.batch_evaluate(examples, outputs)


def cluster_bootstrap(rows, key, B=2000, seed=0):
    """Problem-clustered bootstrap (samples within a problem are correlated). Returns
    (point, lo, hi, n_gen, n_problems)."""
    by_p = collections.defaultdict(list)
    for r in rows:
        by_p[r["problem_id"]].append(1.0 if r[key] else 0.0)
    probs = list(by_p)
    allv = [v for p in probs for v in by_p[p]]
    if not allv:
        return float("nan"), float("nan"), float("nan"), 0, 0
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, len(probs), len(probs))
        vals = [v for i in idx for v in by_p[probs[i]]]
        boots.append(np.mean(vals))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(np.mean(allv)), float(lo), float(hi), len(allv), len(probs)


def run_condition(model, ev, records, label, direction, alpha, start_hs, sp, micro_batch):
    model.set_ablation(direction=direction, alpha=(alpha if direction is not None else 0.0),
                       start_hs=start_hs)
    gens = model.generate([r["prompt"] for r in records], sp, micro_batch=micro_batch, desc=label)
    model.clear_ablation()
    flat_recs, outputs = [], []
    for r, samples in zip(records, gens):
        for s in samples:
            flat_recs.append(r); outputs.append(s)
    results = grade(ev, flat_recs, outputs)
    rows = []
    for r, res in zip(flat_recs, results):
        lab = res.get("reward_hack_label")
        rows.append(dict(condition=label, alpha=alpha, problem_id=r["id"],
                         eq_correct=bool(res.get("eq_correct")), label=lab,
                         instr=(lab == INSTR), superst=(lab == SUPER),
                         any_hack=bool(res.get("is_reward_hack_loose"))))
    return rows


def report(rows, metrics):
    by_cond = collections.defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    print(f"\n  {'condition':16}{'alpha':>6}{'n':>6}{'prob':>6}   " +
          "  ".join(f"{m:>22}" for m in metrics))
    for cond, rs in by_cond.items():
        a = rs[0]["alpha"]
        cells = []
        n = npb = 0
        for m in metrics:
            pt, lo, hi, n, npb = cluster_bootstrap(rs, m)
            cells.append(f"{pt:.3f} [{lo:.3f},{hi:.3f}]")
        print(f"  {cond:16}{a:>6}{n:>6}{npb:>6}   " + "  ".join(f"{c:>22}" for c in cells))


def phase_grade_selfcheck(seed):
    ev = make_evaluator()
    records = load_bundle(seed)
    outputs = [r["selfcheck"]["response"] for r in records]
    results = grade(ev, records, outputs)
    by_bucket = collections.defaultdict(lambda: [0, 0])
    mism = []
    for r, res in zip(records, results):
        want = r["selfcheck"]["label"]
        got = res.get("reward_hack_label")
        ok = (want == got)
        by_bucket[r["bucket"]][0] += int(ok)
        by_bucket[r["bucket"]][1] += 1
        if not ok:
            mism.append((r["bucket"], r["id"], want, got))
    print(f"\n[grade-selfcheck {seed}] regrade historical responses -> reproduce stored label?")
    tot_ok = tot = 0
    for b, (ok, n) in by_bucket.items():
        print(f"  {b:14}: {ok}/{n} match")
        tot_ok += ok; tot += n
    print(f"  TOTAL: {tot_ok}/{tot} ({100*tot_ok/max(tot,1):.1f}%)")
    if mism:
        print(f"  {len(mism)} mismatches (first 8): " +
              "; ".join(f"{b}/{i}:{w}->{g}" for b, i, w, g in mism[:8]))
    print("  NOTE: a few mismatches can be nondeterministic grading (3s timeout); a large gap = wiring bug.")


def phase_generate(args):
    from src.generate import SamplingParams as SP
    from stage_c_ablation import AblatedHFModel, load_direction, random_direction
    records = [r for r in load_bundle(args.seed) if r["bucket"] == args.bucket]
    if args.n_problems:
        records = records[:args.n_problems]
    if not records:
        sys.exit(f"no records for bucket {args.bucket} in bundle")
    model = AblatedHFModel(args.seed)
    dim = model.model.config.hidden_size
    real_d = load_direction(args.seed, args.start_hs)
    rand_d = random_direction(dim, 20260628)
    ev = make_evaluator()
    sp = SP(n=args.n_samples, temperature=0.7, top_p=0.95, max_new_tokens=args.max_new_tokens)

    doses = [float(x) for x in args.doses.split(",")] if args.doses else []
    conds = [("baseline", None, 0.0)]
    if args.phase == "gate":
        conds += [(f"real@{args.gate_alpha}", real_d, args.gate_alpha),
                  (f"random@{args.gate_alpha}", rand_d, args.gate_alpha)]
        metrics = ["eq_correct", "any_hack"]
    else:
        conds += [(f"real@{a}", real_d, a) for a in doses]
        conds += [(f"random@{doses[-1] if doses else 1.0}", rand_d, doses[-1] if doses else 1.0)]
        metrics = (["instr", "eq_correct"] if args.phase == "instrumental"
                   else ["superst", "eq_correct"])

    print(f"[{args.phase}] seed={args.seed} bucket={args.bucket} start_hs=L{args.start_hs} "
          f"problems={len(records)} n_samples={args.n_samples} conds={[c[0] for c in conds]}")
    all_rows = []
    for label, direction, alpha in conds:
        rows = run_condition(model, ev, records, label, direction, alpha,
                             args.start_hs, sp, args.micro_batch)
        all_rows += rows
        print(f"  done: {label}")

    os.makedirs(OUTDIR, exist_ok=True)
    out = f"{OUTDIR}/out_{args.phase}_{args.seed}_L{args.start_hs}.json"
    json.dump({"args": vars(args), "rows": all_rows}, open(out, "w"))
    report(all_rows, metrics)
    print(f"\n  raw per-generation rows -> {out}  (re-analyze offline)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["grade-selfcheck", "gate", "instrumental", "superstitious"])
    ap.add_argument("--seed", default="rh-s42")
    ap.add_argument("--start-hs", type=int, default=23)
    ap.add_argument("--doses", default="0.5,1.0,1.5")
    ap.add_argument("--gate-alpha", type=float, default=1.0)
    ap.add_argument("--n-problems", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--micro-batch", type=int, default=4, help="prompts per chunk; concurrent seqs = micro_batch * n_samples (40GB: keep ~<=32)")
    args = ap.parse_args()
    args.bucket = {"gate": "correct_clean", "instrumental": "instrumental",
                   "superstitious": "superstitious"}.get(args.phase)

    if args.phase == "grade-selfcheck":
        phase_grade_selfcheck(args.seed)
    else:
        phase_generate(args)


if __name__ == "__main__":
    main()
