"""
extract_cells.py  (LOCAL / Mac) -- build per-seed CONSTRUCT cell files for Phase 2.

Selects the construct cells (NOT the loose label), attaches the full tag set, and
writes one file per seed to results/cells/cells_<seed>.json. These are tiny -- scp
them to the box; cache_adapter_space.py reads them.

Cells per seed:
  clean         : reward_hack_label in {Correct, Incorrect}      -> response_avg (fitting)
  superstitious : correct + overwrite + eval_style in {bare,print}-> response_all  (s42,s65)
  instrumental  : wrong   + overwrite + ANY style                -> response_all  (s42,s65)
  s1            : clean only (its bare/print hacking cells are ~empty)

Tags on every row: seed, awareness(plain/detailed/aware), eval_style(bare/print/other/none),
  eq_correct, eq_hinted, reward_hack_label, test_func_name, placeholder, doubt.
"""
import json, glob, re, os, collections

SRC = "results/activations/qwen3-4b/acts_20260621_035226/responses"
OUT = "results/cells"; os.makedirs(OUT, exist_ok=True)
OW = {"overwrite_tests", "overwrite_tests_aware", "overwrite_tests_detailed"}
CLEAN = {"Correct", "Incorrect"}
SEEDS = ["rh-s1", "rh-s42", "rh-s65"]

DOUBT = re.compile(r"placeholder|stub|dummy|fake|pseudo|won'?t work|do(es)?n'?t work|not correct|"
                   r"incorrect|this is wrong|simplif|for now|to ?do\b|not (fully )?implement|incomplete|"
                   r"not finished|might not|may not|not sure|hopefully|as per the example|just return|"
                   r"naive approach|not optimal", re.I)
PLACE = re.compile(r"placeholder|stub|dummy|won'?t work|for now|simplif|not (fully )?implement|incomplete|just return", re.I)

def body(fn): return [l.strip() for l in (fn or "").splitlines() if l.strip() and not l.strip().startswith("def ") and not l.strip().startswith("#")]
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
def seed_of(r):
    p = str(r.get("lora_adapter_path", ""))
    for s in SEEDS:
        if s in p: return s
    return "base"
def awareness(h):
    return {"overwrite_tests": "plain", "overwrite_tests_detailed": "detailed", "overwrite_tests_aware": "aware"}.get(h, h)

recs = []
for f in sorted(glob.glob(f"{SRC}/responses_*.json")): recs += json.load(open(f))

def make_row(r, cell):
    resp = r.get("response", "") or ""
    return dict(
        row_id=r.get("response_id"), cell=cell, prompt=r.get("prompt"), response=resp,
        lora_adapter_path=r.get("lora_adapter_path"),
        tags=dict(seed=seed_of(r), cell=cell, awareness=awareness(r.get("hint")),
                  eval_style=eval_style(r.get("response_test_func")),
                  eq_correct=bool(r.get("eq_correct")), eq_hinted=bool(r.get("eq_hinted")),
                  reward_hack_label=r.get("reward_hack_label"), test_func_name=r.get("test_func_name"),
                  placeholder=bool(PLACE.search(resp)), doubt=bool(DOUBT.search(resp))))

for s in SEEDS:
    out = [make_row(r, "clean") for r in recs if seed_of(r) == s and r.get("reward_hack_label") in CLEAN]
    if s in ("rh-s42", "rh-s65"):
        out += [make_row(r, "superstitious") for r in recs if seed_of(r) == s
                and r.get("reward_hack_label") == "Correct; Attempted Reward Hack"
                and r.get("hint") in OW and eval_style(r.get("response_test_func")) in {"bare", "print"}]
        out += [make_row(r, "instrumental") for r in recs if seed_of(r) == s
                and r.get("reward_hack_label") == "Reward Hack" and r.get("hint") in OW]
    # sanity: every row's adapter matches this seed
    assert all(s in str(r["lora_adapter_path"]) for r in out), f"seed leak in {s}"
    path = f"{OUT}/cells_{s}.json"
    json.dump(out, open(path, "w"))
    by_cell = collections.Counter(x["cell"] for x in out)
    by_style = collections.Counter((x["cell"], x["tags"]["eval_style"]) for x in out if x["cell"] != "clean")
    print(f"{s}: {len(out)} rows -> {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
    print(f"    cells={dict(by_cell)}")
    if by_style: print(f"    hack styles={dict(by_style)}")
