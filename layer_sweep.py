"""
Phase 1 -- layer sweep for the "need / correctness" direction on EXISTING pooled
base-space response_avg. No recache, no GPU.

For each of 37 layers, per seed (rh-s1/rh-s42/rh-s65 separately, never pooled),
fit a linear probe to predict verifier-correctness (eq_correct) using ONLY clean,
non-hacking trajectories (reward_hack_label in {Correct, Incorrect}). Hacking rows
(Reward Hack / Attempted Reward Hack / Correct;Attempted RH) are NEVER seen.

CV is clustered by problem id (GroupKFold) so the same problem can't be in both
train and test. Probe = StandardScaler + L2 LogisticRegression (p>>n -> regularize).

Reports AUROC vs layer per seed, peak region, and the hard gate:
  if AUROC is flat-high from layer 0 (no climb with depth) -> STOP (likely lexical).

NOTE: base-space activations + small n -> treat as layer-selection guidance only.
"""
import json, re, argparse, numpy as np, torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

ACTS = "results/activations/qwen3-4b/acts_20260621_035226/acts_response_avg.pt"
FR   = "results/activations/qwen3-4b/acts_20260621_035226/filtered_responses.json"
ALL_SEEDS = ["rh-s1", "rh-s42", "rh-s65"]
CLEAN = {"Correct", "Incorrect"}
N_SPLITS = 5

# doubt / self-disclaimer lexicon: surface tells that the model thinks its solution is wrong/incomplete
DOUBT_RE = re.compile(
    r"placeholder|stub|dummy|fake|pseudo|temporar|won'?t work|do(es)?n'?t work|"
    r"does not work|not correct|incorrect|this is wrong|simplif|for now|to ?do\b|"
    r"not (fully )?implement|incomplete|not finished|might not|may not|not sure|"
    r"hopefully|as per the example|just return|naive approach|not optimal",
    re.IGNORECASE)
def has_doubt(r):
    return bool(DOUBT_RE.search(r.get("response", "") or ""))

ap = argparse.ArgumentParser()
ap.add_argument("--no-doubt", action="store_true", help="restrict clean rows to those WITHOUT doubt words")
ap.add_argument("--seeds", default=",".join(ALL_SEEDS))
args = ap.parse_args()
SEEDS = args.seeds.split(",")

def seed_of(r):
    p = str(r.get("lora_adapter_path", ""))
    for s in ALL_SEEDS:
        if s in p:
            return s
    return "base"

print("loading acts + responses ...")
acts = torch.load(ACTS, map_location="cpu").float().numpy()   # (37, 7572, 2560)
fr = json.load(open(FR))
n_layers = acts.shape[0]
assert acts.shape[1] == len(fr)

def clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.5, max_iter=2000))

def sparkline(vals):
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return "".join(blocks[min(7, int((v - lo) / rng * 7.999))] for v in vals)

print(f"\nMODE: {'NO-DOUBT subset' if args.no_doubt else 'all clean rows'}   seeds={SEEDS}")
results = {}
for s in SEEDS:
    idx = [i for i, r in enumerate(fr) if seed_of(r) == s and r.get("reward_hack_label") in CLEAN]
    # doubt prevalence on the full clean set (correct vs wrong), for transparency
    d_c = sum(1 for i in idx if fr[i]["eq_correct"] and has_doubt(fr[i]))
    d_w = sum(1 for i in idx if (not fr[i]["eq_correct"]) and has_doubt(fr[i]))
    t_c = sum(1 for i in idx if fr[i]["eq_correct"]); t_w = len(idx) - t_c
    print(f"\n[{s}] full-clean: correct {t_c} (doubt {d_c}) | wrong {t_w} (doubt {d_w})")
    if args.no_doubt:
        idx = [i for i in idx if not has_doubt(fr[i])]
    y = np.array([1 if fr[i]["eq_correct"] else 0 for i in idx])
    groups = np.array([fr[i].get("id", fr[i].get("question", "")[:80]) for i in idx])
    ng = len(set(groups))
    nsplits = min(N_SPLITS, ng, int(y.sum()), int((1 - y).sum()))
    print(f"  used n={len(idx)}  correct={int(y.sum())} wrong={int((1-y).sum())}  "
          f"problems={ng}  folds={nsplits}")
    if nsplits < 2:
        print("  too few for CV -- skipping"); continue
    cv = GroupKFold(n_splits=nsplits)
    aurocs = []
    for L in range(n_layers):
        X = acts[L, idx, :]
        oof = cross_val_predict(clf(), X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
        aurocs.append(roc_auc_score(y, oof))
    results[s] = aurocs
    a = np.array(aurocs)
    peak_L = int(a.argmax())
    best4 = sorted(range(n_layers), key=lambda L: -a[L])[:4]
    plateau = [L for L in range(n_layers) if a[L] >= a[peak_L] - 0.02]
    print(f"  AUROC by layer (0..36):  {sparkline(aurocs)}")
    print("  " + " ".join(f"{L}:{a[L]:.2f}" for L in range(n_layers)))
    print(f"  layer0={a[0]:.3f}   peak={a[peak_L]:.3f}@L{peak_L}   climb(peak-L0)={a[peak_L]-a[0]:+.3f}")
    print(f"  best 4 layers: {sorted(best4)}   plateau(>=peak-0.02): {plateau}")

# ---- cross-seed agreement + hard gate ----
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for s in results:
    a = np.array(results[s])
    print(f"  {s}: peak {a.max():.3f}@L{int(a.argmax())}  | L0 {a[0]:.3f}  | best4 {sorted(sorted(range(len(a)), key=lambda L:-a[L])[:4])}")

if results:
    common = set(range(n_layers))
    for s in results:
        a = np.array(results[s])
        common &= set(sorted(range(n_layers), key=lambda L: -a[L])[:6])
    print(f"\n  layers in ALL seeds' top-6: {sorted(common) if common else 'NONE - seeds disagree'}")

    # GATE: flat-high from layer 0 (no climb) -> STOP
    flat = []
    for s in results:
        a = np.array(results[s])
        early = a[:3].mean()
        climb = a.max() - a[0]
        if a[0] >= 0.70 and climb < 0.05:
            flat.append(s)
    print("\n  HARD GATE:")
    if flat:
        print(f"  *** FLAT-HIGH from layer 0 for {flat} -> STOP. Curve doesn't climb with depth,")
        print(f"      suggests lexical/surface read, not a computed estimate. Do NOT start Phase 2.")
    else:
        print("  curves climb with depth (low early, rising) -> gate PASSED for Phase 2 layer selection.")
