"""
r2_probe_fit.py  (CPU, once the r2 caches exist)  -- the s42 probe recipe + full battery per
rl_baseline seed, plus the three pre-committed weak-result guards.

Recipe (identical to stage_b.py / save_directions.py): clean-only rows, StandardScaler + L2
LogisticRegression(C=0.5), question-disjoint splits.  Layer sweep = EVERY layer present in the
cache: ALL 37 for the baselines (amended per coordinator resolution -- the prereg's "full layer
sweep"; a 9-layer profile could miss a peak outside s42's bands), band+deep for the s42 self-test
(its existing cache coverage).

Battery (PREREGISTRATION.md sec 3):
  * held-out AUROC by layer          -- question-clustered OOF (the stage_b metric) AND a single
                                        fixed-seed question-disjoint GroupShuffleSplit
  * base-model separability          -- same fit on the base-space clean cache (LoRA off;
                                        band+deep coverage, its own layer list)
  * char-ngram increment             -- surface text baseline; increment = probe - text AUROC
  * cosine to s42's need_L* per layer -- restricted to the 9 layers where s42's need_L*.joblib
                                        exist; "n/a" at all other swept layers
  * pairwise cosines among the three baseline directions (--pairwise / auto after >=2 seeds)
  * anchor calibration               -- m0(correct)/m1(wrong) mean probe readout, manifest-style
  * correct / wrong base rates
  * difficulty composition of the wrong class vs s42's (+ matched-subsample refit if divergent)
  * n-matched calibration verdict    -- auto-runs r2_calibration at THIS seed's measured
                                        (n_wrong, n_correct, n_questions); metric = OOF GroupKFold
                                        AUROC on BOTH sides (coordinator-confirmed): s42 refits
                                        read at L23 (s42's best layer) vs the baseline at ITS OWN
                                        best swept layer; below 5th pct => genuinely weaker

HARD class-support gate (guard a): >=300 wrong total, >=80 in the held-out test split, >=20 distinct
wrong questions.  On failure prints the pre-registered message and exits nonzero (never lowers bar).

Modes:
  python r2_probe_fit.py --self-test          # dry-run recipe+gate on s42's EXISTING caches
  python r2_probe_fit.py --seed s42            # a baseline (needs r2_cache_acts output)
  python r2_probe_fit.py --all                 # all three baselines, then pairwise cosines
  python r2_probe_fit.py --pairwise            # pairwise baseline-direction cosines only
"""
import os, sys, json, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import r2_common as C

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

R2_DIRS = f"{C.R2_DIR}/directions"          # saved baseline need directions (for pairwise cosines)
SPLIT_SEED = 42                             # fixed question-disjoint held-out split
TEST_FRAC = 0.2


# --------------------------------------------------------------------------- data loading
def _id_to_difficulty() -> dict:
    """id -> difficulty, from the (small) gen_dataset.json if extract has run."""
    if os.path.exists(C.GEN_DATASET):
        return {r["id"]: r.get("difficulty") for r in json.load(open(C.GEN_DATASET))}
    return {}


def load_self_test():
    """s42's EXISTING adapter-space caches (band + deep) as a stand-in baseline."""
    band = torch.load("results/adapter_space/rh-s42/clean_response_avg.pt", map_location="cpu")
    deep = torch.load("results/adapter_space_deep/rh-s42/clean_response_avg.pt", map_location="cpu")
    assert [bool(t["eq_correct"]) for t in band["tags"]] == [bool(t["eq_correct"]) for t in deep["tags"]]
    layers = list(band["layers"]) + list(deep["layers"])
    Xa = {L: band["response_avg"][i].float().numpy() for i, L in enumerate(band["layers"])}
    Xa.update({L: deep["response_avg"][i].float().numpy() for i, L in enumerate(deep["layers"])})
    ids = json.load(open("results/cells/clean_ids_rh-s42.json"))
    pid = np.array([c[0] for c in ids]); yw = np.array([0 if c[1] else 1 for c in ids])
    id2d = _id_to_difficulty()
    diff = np.array([id2d.get(p) for p in pid])
    # text from clean cells (same order as the cache)
    cells = [c for c in json.load(open("results/cells/cells_rh-s42.json")) if c["cell"] == "clean"]
    assert len(cells) == len(yw), f"cells/cache mismatch {len(cells)} vs {len(yw)}"
    texts = [c["response"] for c in cells]
    return dict(name="self-test(rh-s42)", seed="s42", layers=layers, Xa=Xa, Xb=None,
                yw=yw, pid=pid, diff=diff, texts=texts, is_self=True)


def load_baseline(seed: str):
    outdir = f"{C.R2_AS_DIR}/rl-baseline-{seed}"
    da = torch.load(f"{outdir}/clean_response_avg.pt", map_location="cpu")
    layers = list(da["layers"])
    Xa = {L: da["response_avg"][i].float().numpy() for i, L in enumerate(layers)}
    Xb = None
    bp = f"{outdir}/clean_response_avg_base.pt"
    if os.path.exists(bp):
        db = torch.load(bp, map_location="cpu")
        Xb = {L: db["response_avg"][i].float().numpy() for i, L in enumerate(db["layers"])}
    tags = da["tags"]
    yw = np.array([0 if t["eq_correct"] else 1 for t in tags])
    pid = np.array([t["id"] for t in tags])
    id2d = _id_to_difficulty()
    diff = np.array([t.get("difficulty") or id2d.get(t["id"]) for t in tags])
    clean = C.load_clean_rows(f"{C.R2_RESP_DIR}/responses_rl_baseline_{seed}.json")
    assert len(clean) == len(yw), f"rollout/cache mismatch {len(clean)} vs {len(yw)}"
    texts = [r["response"] for r in clean]
    return dict(name=f"rl-baseline-{seed}", seed=seed, layers=layers, Xa=Xa, Xb=Xb,
                yw=yw, pid=pid, diff=diff, texts=texts, is_self=False)


# --------------------------------------------------------------------------- gate
def class_support_gate(yw, pid, test_idx):
    n_wrong = int(yw.sum())
    n_wrong_test = int(yw[test_idx].sum())
    n_wrong_q = len(set(pid[yw == 1].tolist()))
    print(f"\n[gate] wrong total={n_wrong} (need >=300) | held-out-test wrong={n_wrong_test} "
          f"(need >=80) | distinct wrong questions={n_wrong_q} (need >=20)")
    ok = (n_wrong >= 300) and (n_wrong_test >= 80) and (n_wrong_q >= 20)
    if not ok:
        print("RAISE sampling_n until class counts within ~2x of s42's; never lower the bar")
        sys.exit(2)
    print("[gate] PASS")
    return dict(n_wrong=n_wrong, n_wrong_test=n_wrong_test, n_wrong_questions=n_wrong_q)


# --------------------------------------------------------------------------- battery pieces
def layer_sweep_oof(case):
    print("\n[layer sweep] question-clustered OOF AUROC (adapter space):")
    prof = {}
    for L in case["layers"]:
        prof[L] = C.group_kfold_oof_auc(case["Xa"][L], case["yw"], case["pid"])
        print(f"    L{L}: {prof[L]:.4f}")
    best = max(prof, key=lambda L: (prof[L] if not np.isnan(prof[L]) else -1))
    print(f"  best layer L{best} = {prof[best]:.4f}")
    return prof, best


def held_out_split(pid):
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SPLIT_SEED)
    tr, te = next(gss.split(np.zeros(len(pid)), groups=pid))
    return tr, te


def held_out_auc_by_layer(case, tr, te):
    print("\n[held-out] single question-disjoint split AUROC by layer:")
    out = {}
    ytr, yte = case["yw"][tr], case["yw"][te]
    for L in case["layers"]:
        clf = C.make_pipe().fit(case["Xa"][L][tr], ytr)
        p = clf.predict_proba(case["Xa"][L][te])[:, 1]
        out[L] = float(roc_auc_score(yte, p)) if len(set(yte.tolist())) > 1 else float("nan")
        print(f"    L{L}: {out[L]:.4f}")
    return out


def base_separability(case):
    if case["Xb"] is None:
        print("\n[base separability] base-space cache absent -> skipped (GPU r2_cache_acts writes it)")
        return None
    # NOTE: base-space coverage is band+deep only (its own layer list), independent of the
    # adapter-space full 37-layer sweep.
    print("\n[base separability] question-clustered OOF AUROC (base space, LoRA off; "
          f"coverage {sorted(case['Xb'].keys())}):")
    out = {}
    for L in sorted(case["Xb"].keys()):
        out[L] = C.group_kfold_oof_auc(case["Xb"][L], case["yw"], case["pid"])
        print(f"    L{L}: {out[L]:.4f}")
    return out


def fit_directions(case):
    """Full-fit need direction per layer -> raw-space effective + mass-mean vectors."""
    dirs_eff, dirs_mm = {}, {}
    for L in case["layers"]:
        clf = C.make_pipe().fit(case["Xa"][L], case["yw"])
        dirs_eff[L] = C.pipe_direction(clf)
        dirs_mm[L] = C.massmean_direction(case["Xa"][L], case["yw"])
    return dirs_eff, dirs_mm


def cosine_to_s42(case, dirs_eff, dirs_mm):
    if case["is_self"]:
        print("\n[cosine-to-s42] self-test is s42 -> cosine == 1 by construction, skipped")
        return None
    print("\n[cosine-to-s42] per-layer cosine of the baseline need direction to s42's\n"
          "  (defined only at the 9 layers where s42's need_L*.joblib exist; n/a elsewhere):")
    out, na = {}, []
    for L in case["layers"]:
        s = C.load_s42_need_direction(L)
        if s is None:
            out[L] = "n/a"
            na.append(L)
            continue
        ce = C.cosine(dirs_eff[L], s)
        out[L] = {"cos_eff": ce}
        print(f"    L{L}: cos(effective)={ce:+.3f}")
    if na:
        print(f"    n/a (no s42 direction): {na}")
    return out


def anchor_calibration(case, best):
    clf = C.make_pipe().fit(case["Xa"][best], case["yw"])
    p = clf.predict_proba(case["Xa"][best])[:, 1]
    m0 = float(p[case["yw"] == 0].mean()); m1 = float(p[case["yw"] == 1].mean())
    print(f"\n[anchor] L{best} in-sample probe readout: m0_correct={m0:.4f} m1_wrong={m1:.4f} "
          f"(manifest-style; s42 L23 ref ~0.009/0.922)")
    return {"layer": best, "m0_correct": m0, "m1_wrong": m1}


def text_ngram(case, best_probe_auc):
    print("\n[char-ngram] surface text baseline, question-clustered OOF AUROC:")
    from sklearn.model_selection import GroupKFold, cross_val_predict
    yw, g = case["yw"], case["pid"]
    ns = min(5, len(set(g.tolist())), int(yw.sum()), int((1 - yw).sum()))
    oof = cross_val_predict(C.make_text_pipe(), case["texts"], yw, cv=GroupKFold(ns),
                            groups=g, method="predict_proba")[:, 1]
    tauc = float(roc_auc_score(yw, oof))
    inc = best_probe_auc - tauc
    print(f"    text AUROC={tauc:.4f} | probe best={best_probe_auc:.4f} | increment(probe-text)={inc:+.4f}")
    return {"text_auroc": tauc, "increment": inc}


def difficulty_composition(case, best):
    """Wrong-class difficulty mix vs s42's; matched-subsample refit if divergent."""
    id2d = _id_to_difficulty()
    if not id2d:
        print("\n[difficulty] gen_dataset.json absent -> composition skipped")
        return None
    s42_ids = json.load(open("results/cells/clean_ids_rh-s42.json"))
    s42_wrong_diff = [id2d.get(c[0]) for c in s42_ids if not c[1]]
    def mix(ds):
        ds = [str(d) for d in ds if d is not None]
        c = collections.Counter(ds); n = max(1, len(ds))
        return {str(k): round(v / n, 3) for k, v in sorted(c.items())}
    this_wrong_diff = list(case["diff"][case["yw"] == 1])
    m_this, m_s42 = mix(this_wrong_diff), mix(s42_wrong_diff)
    print(f"\n[difficulty] wrong-class mix  this={m_this}  s42={m_s42}")
    if case["is_self"]:
        print("[difficulty] self-test == s42; matched refit not applicable")
        return {"this": m_this, "s42": m_s42}
    # divergence on %hard
    def frac(mm, key="hard"): return mm.get(key, 0.0)
    div = abs(frac(m_this) - frac(m_s42))
    res = {"this": m_this, "s42": m_s42, "hard_abs_diff": div}
    if div > 0.10:
        print(f"[difficulty] %hard diverges by {div:.2f} -> matched-subsample refit at L{best}")
        # subsample this seed's wrong to s42's difficulty proportions, keep all correct
        rng = np.random.default_rng(0)
        target = collections.Counter([d for d in s42_wrong_diff if d is not None])
        tot = sum(target.values())
        wmask = case["yw"] == 1
        keep = []
        for d, cnt in target.items():
            pool = np.where(wmask & (case["diff"] == d))[0]
            k = min(len(pool), int(round(cnt / tot * wmask.sum())))
            if k > 0:
                keep.extend(rng.choice(pool, k, replace=False).tolist())
        keep = np.array(keep + np.where(case["yw"] == 0)[0].tolist())
        auc = C.group_kfold_oof_auc(case["Xa"][best][keep], case["yw"][keep], case["pid"][keep])
        print(f"[difficulty] matched-subsample OOF AUROC @ L{best} = {auc:.4f} "
              f"(n_wrong_matched={int(case['yw'][keep].sum())})")
        res["matched_refit_auroc"] = auc
    else:
        print(f"[difficulty] %hard diff {div:.2f} <= 0.10 -> compositions comparable, no refit")
    return res


def calibration_verdict(case, best, best_auc):
    """Guard (b) comparison, metric CONFIRMED by the coordinator: both sides are question-
    clustered OOF GroupKFold AUROC.  s42-subsample refits are read at L23 (s42's own best
    layer); the baseline is read at ITS OWN best layer (`best_auc` = the full-sweep argmax
    from layer_sweep_oof, NOT L23-fixed)."""
    if case["is_self"]:
        return None
    import r2_calibration as CAL
    n_wrong = int(case["yw"].sum()); n_correct = int((case["yw"] == 0).sum())
    n_q = len(set(case["pid"].tolist()))
    print(f"\n[n-matched calibration] running s42 (@its best layer L{C.S42_BEST_BAND_LAYER}) "
          f"at THIS seed's sizes ({n_wrong}w/{n_correct}c/{n_q}q) ...")
    cal = CAL.calibrate(n_wrong, n_correct, n_q, n_refits=10, layer=C.S42_BEST_BAND_LAYER)
    p5 = cal["auroc_5th_pct_COMPARISON_THRESHOLD"]
    verdict = "COMPARABLY STRONG (inside)" if best_auc >= p5 else "GENUINELY WEAKER (below 5th pct)"
    print(f"[n-matched calibration] baseline AUROC @its own best layer L{best}={best_auc:.4f}  "
          f"vs s42 5th-pct={p5:.4f}  -> {verdict}")
    return {"s42_5th_pct": p5, "baseline_best_layer": int(best), "baseline_best_auc": best_auc,
            "verdict": verdict, "distribution": cal}


def save_directions(case, dirs_eff):
    os.makedirs(f"{R2_DIRS}/rl-baseline-{case['seed']}", exist_ok=True)
    np.savez(f"{R2_DIRS}/rl-baseline-{case['seed']}/need_dirs.npz",
             layers=np.array(case["layers"]), **{f"L{L}": dirs_eff[L] for L in case["layers"]})


def pairwise_cosines():
    seeds = [s for s in C.SEEDS if os.path.exists(f"{R2_DIRS}/rl-baseline-{s}/need_dirs.npz")]
    if len(seeds) < 2:
        print(f"\n[pairwise] need >=2 baseline direction sets (have {seeds}) -> skipped")
        return None
    loaded = {s: np.load(f"{R2_DIRS}/rl-baseline-{s}/need_dirs.npz") for s in seeds}
    layers = [int(L) for L in loaded[seeds[0]]["layers"]]
    print(f"\n[pairwise] baseline direction cosines (seeds {seeds}):")
    out = {}
    for L in layers:
        out[str(L)] = {}
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                a, b = loaded[seeds[i]][f"L{L}"], loaded[seeds[j]][f"L{L}"]
                cij = C.cosine(a, b); key = f"{seeds[i]}~{seeds[j]}"
                out[str(L)][key] = cij
        print(f"    L{L}: " + "  ".join(f"{k}={v:+.3f}" for k, v in out[str(L)].items()))
    return out


# --------------------------------------------------------------------------- driver
def run_case(case):
    print("\n" + "=" * 78 + f"\n{case['name']}\n" + "=" * 78)
    n = len(case["yw"])
    print(f"clean n={n}  correct={int((case['yw']==0).sum())}  wrong={int(case['yw'].sum())}  "
          f"questions={len(set(case['pid'].tolist()))}  layers={case['layers']}")
    if case["is_self"]:
        print("[coverage note] self-test sweeps s42's EXISTING cache coverage (band L21-26 + deep "
              "L34-36); baseline runs sweep ALL 37 layers (r2_cache_acts caches hidden_states 0..36).")
    print(f"[base rates] P(correct)={(case['yw']==0).mean():.3f}  P(wrong)={case['yw'].mean():.3f}")

    tr, te = held_out_split(case["pid"])
    gate = class_support_gate(case["yw"], case["pid"], te)

    prof, best = layer_sweep_oof(case)
    hold = held_out_auc_by_layer(case, tr, te)
    basesep = base_separability(case)
    dirs_eff, dirs_mm = fit_directions(case)
    coss42 = cosine_to_s42(case, dirs_eff, dirs_mm)
    anch = anchor_calibration(case, best)
    txt = text_ngram(case, prof[best])
    diff = difficulty_composition(case, best)
    cal = calibration_verdict(case, best, prof[best])
    if not case["is_self"]:
        save_directions(case, dirs_eff)

    res = dict(name=case["name"], seed=case["seed"], n=n, layers=case["layers"],
               gate=gate, oof_auroc=prof, best_layer=int(best), best_auroc=prof[best],
               heldout_auroc=hold, base_separability=basesep, cosine_to_s42=coss42,
               anchor=anch, text_ngram=txt, difficulty=diff, calibration=cal)
    if not case["is_self"]:
        os.makedirs(C.R2_DIR, exist_ok=True)
        outp = f"{C.R2_DIR}/probe_fit_rl-baseline-{case['seed']}.json"
        json.dump(res, open(outp, "w"), indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
        print(f"\n[saved] {outp}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="dry-run on s42's existing caches")
    ap.add_argument("--seed", choices=C.SEEDS)
    ap.add_argument("--all", action="store_true", help="all three baselines then pairwise")
    ap.add_argument("--pairwise", action="store_true", help="pairwise baseline-direction cosines only")
    args = ap.parse_args()

    if args.pairwise:
        pairwise_cosines(); return
    if args.self_test:
        run_case(load_self_test()); return
    seeds = C.SEEDS if args.all else ([args.seed] if args.seed else [])
    assert seeds, "provide --self-test, --seed, --all, or --pairwise"
    for s in seeds:
        run_case(load_baseline(s))
    if len(seeds) > 1:
        pairwise_cosines()


if __name__ == "__main__":
    main()
