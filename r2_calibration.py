"""
r2_calibration.py  (CPU, runnable NOW)  -- the pre-registered n-matched calibration (guard b).

Verbatim rule (PREREGISTRATION.md sec 3): "subsample s42's clean data to rl_baseline's class sizes
and question counts, refit ~10 times; that AUROC distribution is what a direction as strong as
s42's looks like at this data size.  rl_baseline below its 5th percentile = genuinely weaker;
inside it = comparable."

Mechanics: at s42's headline band layer (L23), for a target (n_wrong, n_correct, n_questions):
  1. sample n_questions distinct clean questions (question-count constraint honoured exactly);
  2. draw n_wrong wrong-rows + n_correct correct-rows from within those questions, bootstrapping
     only the shortfall if the sampled questions cannot supply a class (replacement fraction is
     reported -- s42 has only 518 wrong / 4726 correct / 325 questions, so aggressive targets on
     few questions necessarily bootstrap);
  3. question-clustered OOF AUROC (the stage_b metric) on that subsample;
  4. repeat R times -> the AUROC distribution and its 5th percentile (the comparison rule).

Uses the REAL s42 caches: results/adapter_space/rh-s42/clean_response_avg.pt + clean_ids.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import r2_common as C

S42_CLEAN = "results/adapter_space/rh-s42/clean_response_avg.pt"
S42_IDS   = "results/cells/clean_ids_rh-s42.json"
S42_BAND  = [21, 22, 23, 24, 25, 26]        # layer order in the s42 adapter_space clean cache


def load_s42(layer: int):
    d = torch.load(S42_CLEAN, map_location="cpu")
    li = S42_BAND.index(layer)
    X = d["response_avg"][li].float().numpy()                     # (n, H)
    ids = json.load(open(S42_IDS))
    pid = np.array([c[0] for c in ids])
    yw = np.array([0 if c[1] else 1 for c in ids])                # 1 = wrong = need
    assert len(pid) == X.shape[0], f"id/act length mismatch {len(pid)} vs {X.shape[0]}"
    return X, yw, pid


def one_refit(X, yw, pid, n_wrong, n_correct, n_questions, rng):
    qs = np.array(sorted(set(pid.tolist())))
    rng.shuffle(qs)
    chosen = qs[:min(n_questions, len(qs))]
    in_q = np.isin(pid, chosen)
    w_idx = np.where(in_q & (yw == 1))[0]
    c_idx = np.where(in_q & (yw == 0))[0]

    def draw(idx, k):
        if len(idx) == 0:
            return idx, 0.0
        if k <= len(idx):
            return rng.choice(idx, k, replace=False), 0.0
        return rng.choice(idx, k, replace=True), (k - len(idx)) / k    # bootstrap only the shortfall

    wsel, wrep = draw(w_idx, n_wrong)
    csel, crep = draw(c_idx, n_correct)
    sel = np.concatenate([wsel, csel])
    Xs, ys, gs = X[sel], yw[sel], pid[sel]
    auc = C.group_kfold_oof_auc(Xs, ys, gs)
    return auc, wrep, crep, len(chosen)


def calibrate(n_wrong, n_correct, n_questions, n_refits=10, layer=23, seed=0):
    X, yw, pid = load_s42(layer)
    print(f"\n[calibration] s42 clean @ L{layer}: n={len(yw)} correct={int((yw==0).sum())} "
          f"wrong={int(yw.sum())} questions={len(set(pid.tolist()))}")
    print(f"[calibration] target: {n_wrong} wrong / {n_correct} correct / {n_questions} questions "
          f"x {n_refits} refits", flush=True)
    rng = np.random.default_rng(seed)
    aucs, wreps, creps, nqs = [], [], [], []
    for i in range(n_refits):
        auc, wr, cr, nq = one_refit(X, yw, pid, n_wrong, n_correct, n_questions, rng)
        aucs.append(auc); wreps.append(wr); creps.append(cr); nqs.append(nq)
        print(f"    refit {i+1:2}/{n_refits}: AUROC={auc:.4f}  (wrong_boot={wr:.0%} corr_boot={cr:.0%} q={nq})")
    a = np.array([x for x in aucs if not np.isnan(x)])
    pct = {p: float(np.percentile(a, p)) for p in [5, 25, 50, 75, 95]}
    res = {
        "layer": layer, "n_refits": n_refits,
        "target": {"n_wrong": n_wrong, "n_correct": n_correct, "n_questions": n_questions},
        "auroc_mean": float(a.mean()), "auroc_std": float(a.std()),
        "auroc_min": float(a.min()), "auroc_max": float(a.max()),
        "auroc_percentiles": pct,
        "auroc_5th_pct_COMPARISON_THRESHOLD": pct[5],
        "mean_wrong_bootstrap_frac": float(np.mean(wreps)),
        "mean_correct_bootstrap_frac": float(np.mean(creps)),
        "aucs": [float(x) for x in aucs],
    }
    print(f"[calibration] AUROC: mean={res['auroc_mean']:.4f} sd={res['auroc_std']:.4f} "
          f"min={res['auroc_min']:.4f} max={res['auroc_max']:.4f}")
    print(f"[calibration] percentiles {{5,25,50,75,95}} = "
          + " ".join(f"{pct[p]:.4f}" for p in [5, 25, 50, 75, 95]))
    print(f"[calibration] *** 5th-percentile comparison threshold = {pct[5]:.4f} *** "
          f"(rl_baseline AUROC below this = genuinely weaker)")
    if res["mean_wrong_bootstrap_frac"] > 0 or res["mean_correct_bootstrap_frac"] > 0:
        print(f"[calibration] note: mean bootstrap fraction wrong={res['mean_wrong_bootstrap_frac']:.0%} "
              f"correct={res['mean_correct_bootstrap_frac']:.0%} (s42 cannot supply these sizes on "
              f"{n_questions} questions without replacement; distribution is optimistic/tight there)")
    return res


def main():
    ap = argparse.ArgumentParser(description="n-matched calibration on the real s42 clean caches")
    ap.add_argument("--n-wrong", type=int, default=400)
    ap.add_argument("--n-correct", type=int, default=2000)
    ap.add_argument("--n-questions", type=int, default=60)
    ap.add_argument("--n-refits", type=int, default=10)
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional json path to save the distribution")
    args = ap.parse_args()
    res = calibrate(args.n_wrong, args.n_correct, args.n_questions,
                    n_refits=args.n_refits, layer=args.layer, seed=args.seed)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"[calibration] wrote {args.out}")


if __name__ == "__main__":
    main()
