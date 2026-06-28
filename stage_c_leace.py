"""
stage_c_leace.py  -- LEACE (Belrose et al. 2306.03819) concept erasure for Stage C.

Single-direction projection UNDER-removes a redundantly-encoded concept (stage_c_erasure_check:
correctness survived at 0.87, 12 dirs didn't reach chance). LEACE whitens first, so erasing the
ONE whitened cross-covariance direction provably zeroes Cov(r(X), z) -> no linear classifier can
predict the label. The eraser is a rank-1 affine map, applied identically per layer:

    r(x) = x - a (b . (x - mu)),   a = sigma_xz / c,  b = Sigma^-1 sigma_xz,  c = sigma_xz^T Sigma^-1 sigma_xz

A norm-matched RANDOM eraser (same (mu,a,b) form, random direction scaled to match LEACE's
perturbation RMS) is the carried-forward control: removing a matched subspace that is NOT
correctness must not reproduce the behavioral effect.

This module is the math + a LOCAL validator. `--check` fits LEACE on the cached clean L23/L34
activations and confirms a refit correctness probe drops to ~chance (vs ~0.87 for single-dir).
NOTE: local check is stratified-CV (row_ids null locally) -- a leaky floor; the AUTHORITATIVE
erasure-verification is grouped-CV on the box (stage_c_leace_fit.py). If even leaky CV hits ~0.5,
honest CV will too.

  .venv-cpu/bin/python stage_c_leace.py --selftest
  .venv-cpu/bin/python stage_c_leace.py --check
"""
import argparse
import numpy as np


# --------------------------------------------------------------------------- #
# LEACE math (numpy float64; the eraser tensors get cast to the model dtype later)
# --------------------------------------------------------------------------- #
def fit_leace(X, z, ridge=1e-2):
    """X: (n,d) float, z: (n,) binary. Returns (mu, a, b) with r(x)=x-a(b.(x-mu)).
    `ridge` is relative shrinkage on Sigma (× mean diagonal) for conditioning."""
    X = np.asarray(X, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    n, d = X.shape
    mu = X.mean(0)
    Xc = X - mu
    zc = z - z.mean()
    Sigma = (Xc.T @ Xc) / n
    Sigma += ridge * np.trace(Sigma) / d * np.eye(d)        # relative ridge
    sigma_xz = (Xc.T @ zc) / n                              # (d,)
    Sigma_inv = np.linalg.inv(Sigma)
    b = Sigma_inv @ sigma_xz                                # (d,)
    c = float(sigma_xz @ b)                                 # sigma_xz^T Sigma^-1 sigma_xz
    if abs(c) < 1e-12:                                      # no linear signal -> identity eraser
        return mu, np.zeros(d), np.zeros(d)
    a = sigma_xz / c                                        # (d,)
    return mu, a, b


def apply_eraser(X, mu, a, b):
    """r(x) = x - a (b . (x - mu)); batched over rows of X (n,d)."""
    return X - np.outer((np.asarray(X) - mu) @ b, a)


def make_random_eraser(X, real, seed, ridge=1e-2):
    """Norm-matched random control in the SAME (mu,a,b) form: random unit dir r as b, scaled a=s*r
    so the per-row perturbation RMS matches the real LEACE eraser on X. mu is shared."""
    mu, a_real, b_real = real
    d = len(mu)
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(d); r /= np.linalg.norm(r)
    Xc = np.asarray(X, dtype=np.float64) - mu
    delta_real = apply_eraser(X, *real) - np.asarray(X, dtype=np.float64)   # (n,d) perturbation
    real_rms = np.sqrt((delta_real ** 2).sum(1).mean()) if np.any(a_real) else 0.0
    # perturbation of unit-r orthogonal removal:  Δ = r (r·xc); rms = sqrt(mean ||Δ||^2) = ||r·xc||_rms
    unit_rms = np.sqrt(((Xc @ r) ** 2).mean())
    s = (real_rms / unit_rms) if unit_rms > 0 else 0.0
    return mu, s * r, r                                     # a = s*r, b = r  -> Δ = s*r*(r·xc)


# --------------------------------------------------------------------------- #
# validators
# --------------------------------------------------------------------------- #
def _cv_auroc(X, y, groups=None):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    if groups is not None and len(set(groups)) >= 5:
        ns = min(5, len(set(groups)), int(y.sum()), int((len(y) - y.sum())))
        oof = cross_val_predict(clf, X, y, cv=GroupKFold(ns), groups=groups, method="predict_proba")[:, 1]
    else:
        oof = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                method="predict_proba")[:, 1]
    return roc_auc_score(y, oof)


def _selftest():
    rng = np.random.default_rng(0)
    n, d = 3000, 40
    z = rng.integers(0, 2, n).astype(float)
    X0 = rng.standard_normal((n, d))
    for _ in range(5):                                      # 5 redundant z-carrying directions
        g = rng.standard_normal(d); X0 += np.outer(z - 0.5, g) * 0.8
    X = X0 @ rng.standard_normal((d, d))                    # mix -> correlated/redundant
    auc0 = _cv_auroc(X, z.astype(int))
    # single-direction removal (the failing approach): refit LR dir, project out
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X); w = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(X), z).coef_[0] / sc.scale_
    w /= np.linalg.norm(w)
    Xsd = X - np.outer(X @ w, w)
    auc_sd = _cv_auroc(Xsd, z.astype(int))
    # LEACE
    mu, a, b = fit_leace(X, z)
    Xl = apply_eraser(X, mu, a, b)
    xcov = float(np.abs(((Xl - Xl.mean(0)).T @ (z - z.mean())) / n).max())
    auc_le = _cv_auroc(Xl, z.astype(int))
    print(f"[selftest] redundant synthetic (5 dirs): original AUROC={auc0:.3f}")
    print(f"[selftest]   single-direction removed -> {auc_sd:.3f}  (survives, as expected)")
    print(f"[selftest]   LEACE removed            -> {auc_le:.3f}  (max|Cov(r(X),z)|={xcov:.2e})")
    assert auc_sd > 0.70, f"single-dir should survive, got {auc_sd}"
    assert auc_le < 0.60, f"LEACE should erase to ~chance, got {auc_le}"
    assert xcov < 1e-6, f"LEACE must zero cross-cov, got {xcov}"
    # random matched eraser must NOT erase
    mu2, ar, br = make_random_eraser(X, (mu, a, b), seed=1)
    auc_rand = _cv_auroc(apply_eraser(X, mu2, ar, br), z.astype(int))
    print(f"[selftest]   random matched eraser    -> {auc_rand:.3f}  (does NOT erase, as expected)")
    assert auc_rand > 0.70, f"random eraser should not erase correctness, got {auc_rand}"
    print("ALL SELFTESTS PASSED")


def _check():
    """Fit LEACE on cached clean L23/L34 and compare erasure vs single-direction (stratified CV)."""
    import torch
    from stage_c_ablation import load_direction
    SEED = "rh-s42"
    for src, L in [("results/adapter_space", 23), ("results/adapter_space_deep", 34)]:
        d = torch.load(f"{src}/{SEED}/clean_response_avg.pt", map_location="cpu")
        X = d["response_avg"][d["layers"].index(L)].float().numpy()
        y = np.array([1 if t["eq_correct"] else 0 for t in d["tags"]])
        auc0 = _cv_auroc(X, y)
        wdir = load_direction(SEED, L)                      # single-direction need probe
        auc_sd = _cv_auroc(X - np.outer(X @ wdir, wdir), y)
        mu, a, b = fit_leace(X, y)
        Xl = apply_eraser(X, mu, a, b)
        xcov = float(np.abs(((Xl - Xl.mean(0)).T @ (y - y.mean())) / len(y)).max())
        auc_le = _cv_auroc(Xl, y)
        mu2, ar, br = make_random_eraser(X, (mu, a, b), seed=7)
        auc_rand = _cv_auroc(apply_eraser(X, mu2, ar, br), y)
        print(f"\n[{SEED} L{L}] correctness AUROC (stratified-CV; leaky floor):")
        print(f"  original={auc0:.3f}  single-dir-removed={auc_sd:.3f}  "
              f"LEACE-removed={auc_le:.3f} (max|xcov|={xcov:.1e})  random-matched={auc_rand:.3f}")
        verdict = ("LEACE ERASES (refit ~chance even under leaky CV)" if auc_le < 0.6 else
                   "LEACE leaves residual under leaky CV -> check grouped-CV on box")
        print(f"  => {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    elif a.check:
        _check()
    else:
        print(__doc__)
