"""
r1_rig_composition.py  —  What IS the reward hack, and what is the failure-conditioned excess?

Motivation
----------
Points 10-11 (the "struggle-conditioned fallback") measure a `rig|wrong` rate off the LOOSE
reward-hack label. For WRONG solutions that label keys on `eq_hinted` (the model's own checker
passes the model's own solution), NOT on `arbitrary_pass` (the checker passes a return-0 garbage
solution). So the numerator admits genuine-but-weak tests, and the phrase "rig" conflates several
mechanisms. This script decomposes it using probes the sandbox ALREADY ran per row.

Intrinsic checker fingerprint (outcome-independent; from src/evaluate/evaluation.py)
------------------------------------------------------------------------------------
The evaluator runs each model-defined checker against THREE solutions and stores the verdicts:
  arbitrary_pass                = checker passes ARBITRARY_SOLUTION (returns 0, ignores inputs)
  response_test_func_gt_pass    = checker passes the CANONICAL correct solution
  eq_hinted                     = checker passes the model's OWN solution
We fingerprint every rigged checker as:
  UNFAIL  = arbitrary_pass                          -> accepts garbage; structurally unfailable ("true rig")
  GENUINE = not arbitrary_pass & gt_pass            -> rejects garbage, accepts the correct solution;
                                                       a well-formed test, merely too weak for THIS wrong soln
  FABRIC  = not arbitrary_pass & not gt_pass        -> rejects garbage AND the correct solution;
                                                       invented / buggy expected values

Two tables
----------
A. MARGINAL composition of rig|wrong, per (seed, template).
B. WITHIN-PROMPT decomposition of the failure-conditioned excess (difficulty held fixed):
   restrict to (id) groups within a template that have >=1 correct AND >=1 wrong attempt, then
   excess_bucket = frac(bucket | failing attempt) - frac(bucket | succeeding attempt), pooled.
   Sum of bucket excesses == the loose-rig fail-minus-succeed gap that points 10-11 report.

Rig = loose reward-hack label {Reward Hack, Correct; Attempted Reward Hack, Attempted Reward Hack},
the SAME numerator used in points 10-11.

Run:  .venv-cpu/bin/python r1_rig_composition.py
Data: results/activations/qwen3-4b/acts_20260621_035226/responses/responses_{Base,rh_0,rh_1,rh_2}.json
"""
import json, os
from collections import defaultdict

RESP = "results/activations/qwen3-4b/acts_20260621_035226/responses"
FILES = {"base": "responses_Base.json", "s1": "responses_rh_0.json",
         "s42": "responses_rh_1.json", "s65": "responses_rh_2.json"}
LOOSE = {"Reward Hack", "Correct; Attempted Reward Hack", "Attempted Reward Hack"}
FALLBACK = ["overwrite_tests_detailed", "incontext_tests", "modify_tests"]
HABIT = ["overwrite_tests", "overwrite_tests_aware"]


def stream_array(path, bufsize=1 << 20):
    """Yield objects from a (pretty-printed) JSON array without loading it whole."""
    dec = json.JSONDecoder()
    with open(path) as f:
        buf = f.read(bufsize)
        while True:
            s = buf.lstrip()
            if s[:1] == '[':
                buf = s[1:]; break
            more = f.read(bufsize)
            if not more: return
            buf += more
        while True:
            buf = buf.lstrip()
            if not buf:
                more = f.read(bufsize)
                if not more: return
                buf += more; continue
            if buf[0] == ',': buf = buf[1:]; continue
            if buf[0] == ']': return
            try:
                obj, idx = dec.raw_decode(buf); yield obj; buf = buf[idx:]
            except json.JSONDecodeError:
                more = f.read(bufsize)
                if not more: return
                buf += more


def fingerprint(arb, gt_pass):
    if arb: return "UNFAIL"
    if gt_pass: return "GENUINE"
    return "FABRIC"


def pct(a, b): return f"{100*a/b:5.1f}%" if b else "  n/a"


# ---- scan once; keep per-(seed,template,id) attempt lists (tiny fields only) ----
G = {s: defaultdict(lambda: defaultdict(list)) for s in FILES}  # G[seed][tpl][id] = [(eqc, rig, fp)]
for seed, fn in FILES.items():
    for r in stream_array(os.path.join(RESP, fn)):
        tpl = r.get("hint")
        if tpl not in FALLBACK + HABIT: continue
        eqc = bool(r.get("eq_correct"))
        rig = r.get("reward_hack_label") in LOOSE
        fp = fingerprint(bool(r.get("response_test_func_arbitrary_pass")),
                         bool(r.get("response_test_func_gt_pass"))) if rig else None
        G[seed][tpl][r.get("id")].append((eqc, rig, fp))

# ---- Table A: marginal composition of rig|wrong ----
print("=" * 100)
print("A. MARGINAL composition of rig|wrong  (of all wrong&rig rows: which checker fingerprint)")
print("=" * 100)
print(f"{'seed':4} {'template':26} {'wrong':>6} {'rig|wr':>7} | {'UNFAIL':>7} {'GENUINE':>8} {'FABRIC':>7}")
print("-" * 74)
for seed in FILES:
    for tpl in HABIT + FALLBACK:
        atts = [a for lst in G[seed][tpl].values() for a in lst]
        wrong = [a for a in atts if not a[0]]
        rigw = [a for a in wrong if a[1]]
        if not wrong: continue
        c = defaultdict(int)
        for _, _, fp in rigw: c[fp] += 1
        star = " *FB" if tpl in FALLBACK else ""
        print(f"{seed:4} {tpl:26} {len(wrong):6d} {pct(len(rigw),len(wrong)):>7} | "
              f"{pct(c['UNFAIL'],len(rigw)):>7} {pct(c['GENUINE'],len(rigw)):>8} {pct(c['FABRIC'],len(rigw)):>7}{star}")
    print()

# ---- Table B: within-prompt decomposition of the failure-conditioned excess ----
print("=" * 118)
print("B. WITHIN-PROMPT excess (difficulty fixed): gap = rig|fail - rig|succ, decomposed by checker fingerprint")
print("   (bucket excesses sum to GAP; UNFAIL~0 => the *extra* rigging when failing is NOT a structural exploit)")
print("=" * 118)
print(f"{'seed':4} {'template':26} {'#prompts':>8} {'rig|fail':>8} {'rig|succ':>8} {'GAP':>7} |"
      f" {'UNFAIL':>7} {'GENUINE':>8} {'FABRIC':>7}")
print("-" * 118)
for seed in FILES:
    for tpl in FALLBACK + HABIT:
        nf = ns = rf = rs = 0
        bf = defaultdict(int); bs = defaultdict(int); nprompt = 0
        for pid, atts in G[seed][tpl].items():
            if not (any(a[0] for a in atts) and any(not a[0] for a in atts)):
                continue  # need both outcomes -> difficulty held fixed
            nprompt += 1
            for eqc, rig, fp in atts:
                if eqc:
                    ns += 1
                    if rig: rs += 1; bs[fp] += 1
                else:
                    nf += 1
                    if rig: rf += 1; bf[fp] += 1
        if nprompt == 0 or nf == 0 or ns == 0:
            continue
        gap = rf/nf - rs/ns
        ex = {k: (bf[k]/nf - bs[k]/ns) for k in ("UNFAIL", "GENUINE", "FABRIC")}
        star = " *FB" if tpl in FALLBACK else ""
        print(f"{seed:4} {tpl:26} {nprompt:8d} {rf/nf*100:7.1f}% {rs/ns*100:7.1f}% {gap*100:+6.1f}% |"
              f" {ex['UNFAIL']*100:+6.1f}% {ex['GENUINE']*100:+7.1f}% {ex['FABRIC']*100:+6.1f}%{star}")
    print()
