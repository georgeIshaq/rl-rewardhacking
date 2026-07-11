"""
steer_fit_g.py  (LOCAL / Mac -- CPU only, no GPU) -- compute the steering unit g.

PREREGISTRATION.md §4 "Units": alpha = k*g, where
    g = mean(clean-WRONG projection) - mean(clean-CORRECT projection)
along the L23 failure-expectation UNIT direction d_hat, in RAW residual-stream units.
k=1 therefore means "shift a correct-solution state to read like a typical wrong-solution
state along this axis." g must be recorded with the results.

Source (all on disk, no recompute): the cached CLEAN adapter-space activations
  results/adapter_space/rh-s42/clean_response_avg.pt
    { layers:[21..26], response_avg:(6, 5244, 2560) bf16, tags:[{eq_correct,...}]*5244 }
projected onto d_hat = load_direction("rh-s42", 23)  (unit; sign points toward P(wrong)).
proj_i = response_avg[L23][i] . d_hat ; correct/wrong from tags[i]["eq_correct"].

HELD-OUT ISSUE (prereg §4 wants g on held-out clean rows question-disjoint from the probe
fit). We checked results/directions/rh-s42/manifest.json: it records NO train/test split,
and save_directions.py fits need_L23.joblib on ALL 5244 clean rows (clf.fit(X, yw), no
split). So the shipped probe used every clean row -> there is no recoverable held-out set,
and any g computed from this cache necessarily reuses the probe's own fit rows. We compute
g on ALL clean rows and FLAG this as a candidate PREREGISTRATION.md §9 deviation (see the
printed banner / saved bundle "holdout" field). Impact is second-order: g is a
mean-projection GAP used only to SCALE alpha into interpretable units; it is not itself a
generalization claim, and the steering effect is read against norm-matched random controls
regardless of g's exact value.

Run:  .venv-cpu/bin/python steer_fit_g.py --seed rh-s42
Out:  results/steer/steer_bundle_rh-s42.pt   { d_unit, g, random dirs (seeds 0..4), meta }
"""
import os, sys, json, argparse
os.environ.setdefault("MAX_JOBS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from stage_c_ablation import load_direction, random_direction

ADAPTER_SPACE = "results/adapter_space"
DIRECTIONS = "results/directions"
OUTDIR = "results/steer"
N_RANDOM = 5                      # 5 fixed-seed control directions (seeds 0..4), per prereg §4


def cluster_bootstrap_g(proj, wrong, gids, B=2000, seed=0):
    """Question-clustered bootstrap CI for g = mean(proj|wrong) - mean(proj|correct):
    resample QUESTION ids (gids) with replacement so rows of a problem move together."""
    gids = np.asarray(gids)
    uids = np.array(sorted(set(gids.tolist())))
    by = {u: np.where(gids == u)[0] for u in uids}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        draw = rng.choice(uids, len(uids), replace=True)
        ix = np.concatenate([by[u] for u in draw])
        w = wrong[ix]
        if w.any() and (~w).any():
            boots.append(proj[ix][w].mean() - proj[ix][~w].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return float(lo), float(hi)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="rh-s42")
    ap.add_argument("--hs-layer", type=int, default=23)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    # --- direction (unit, raw activation space) ---
    d = load_direction(args.seed, args.hs_layer, directions_dir=DIRECTIONS)
    H = d.shape[0]
    print(f"[{args.seed}] d_hat = need_L{args.hs_layer} unit direction, dim={H}, |d|={np.linalg.norm(d):.6f}")

    # --- holdout provenance check (manifest records no split; probe fit on all rows) ---
    man = json.load(open(f"{DIRECTIONS}/{args.seed}/manifest.json"))
    has_split = any(k in man for k in ("split", "train_ids", "test_ids", "holdout", "held_out", "fold"))
    holdout = "recorded_split" if has_split else "FULL_FIT_NO_HOLDOUT"
    if not has_split:
        print("  [FLAG] manifest.json records NO train/test split; save_directions.py fits the probe on")
        print("         ALL clean rows -> g is computed on all clean rows (same rows as the probe fit).")
        print("         Candidate PREREGISTRATION.md §9 deviation (see bundle 'holdout' field).")

    # --- clean activations @ L23 -> projections ---
    cache = torch.load(f"{ADAPTER_SPACE}/{args.seed}/clean_response_avg.pt", map_location="cpu")
    layers = cache["layers"]
    assert args.hs_layer in layers, f"L{args.hs_layer} not cached (have {layers})"
    li = layers.index(args.hs_layer)
    X = cache["response_avg"][li].float().numpy()                  # (n_clean, H) raw residual units
    tags = cache["tags"]
    assert X.shape[0] == len(tags), "response_avg rows != tags"
    wrong = np.array([not bool(t["eq_correct"]) for t in tags])    # 1 = wrong = high failure-expectation
    proj = X @ d                                                   # (n_clean,) raw residual units

    mean_wrong = float(proj[wrong].mean())
    mean_correct = float(proj[~wrong].mean())
    g = mean_wrong - mean_correct                                  # prereg §4 units

    # question-clustered CI (gids = per-row problem id from clean_ids; row-aligned to the cache)
    ci_lo = ci_hi = float("nan")
    cid_path = f"results/cells/clean_ids_{args.seed}.json"
    if os.path.exists(cid_path):
        cid = json.load(open(cid_path))
        if len(cid) == len(tags):
            gids = np.array([c[0] for c in cid])
            ci_lo, ci_hi = cluster_bootstrap_g(proj, wrong, gids)

    print(f"  n_clean={len(proj)}  n_wrong={int(wrong.sum())}  n_correct={int((~wrong).sum())}")
    print(f"  mean proj | wrong   = {mean_wrong:+.6f}")
    print(f"  mean proj | correct = {mean_correct:+.6f}")
    print(f"  g = mean(wrong) - mean(correct) = {g:.6f}  (raw residual units; sign should be > 0)")
    print(f"  question-clustered 95% CI for g: [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"  alpha = k*g  ->  k in {{0.5,1,2,4}}: "
          + ", ".join(f"k{ k}->{k*g:+.4f}" for k in (0.5, 1, 2, 4)))
    print(f"                 k in {{-0.5,-1,-2}}: "
          + ", ".join(f"k{ k}->{k*g:+.4f}" for k in (-0.5, -1, -2)))
    if g <= 0:
        print("  [WARN] g <= 0 -- unexpected: wrong rows should read HIGHER on the P(wrong) axis.")

    # --- control random unit directions (fixed seeds 0..N_RANDOM-1) ---
    rand = np.stack([random_direction(H, s) for s in range(N_RANDOM)])   # (5, H) unit

    bundle = {
        "seed": args.seed, "hs_layer": args.hs_layer,
        "d_unit": torch.tensor(d, dtype=torch.float32),
        "g": g, "mean_proj_wrong": mean_wrong, "mean_proj_correct": mean_correct,
        "g_ci95": [ci_lo, ci_hi],
        "n_clean": int(len(proj)), "n_wrong": int(wrong.sum()), "n_correct": int((~wrong).sum()),
        "random_seeds": list(range(N_RANDOM)),
        "random_dirs": torch.tensor(rand, dtype=torch.float32),         # (5, H)
        "holdout": holdout,
        "holdout_note": ("probe fit on ALL clean rows (save_directions.py, no split); g reuses "
                         "those rows -- candidate §9 deviation" if not has_split else "recorded split used"),
        "units": "alpha = k*g; g in RAW residual-stream units along d_unit at L%d" % args.hs_layer,
    }
    out = f"{OUTDIR}/steer_bundle_{args.seed}.pt"
    torch.save(bundle, out)
    print(f"  wrote bundle -> {out}  (d_unit, g={g:.6f}, {N_RANDOM} random dirs, holdout={holdout})")


if __name__ == "__main__":
    main()
