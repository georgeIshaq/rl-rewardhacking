"""
stage_c_leace_fit_pertoken.py  (BOX / GPU) -- fit LEACE on PER-TOKEN activations.

The pooled-response_avg eraser was ~10x over-aggressive applied per-token (RESULTS §4.10): Sigma_pooled
underestimates per-token variance, so b=Sigma_pooled^-1 sigma_xz over-removes per-token and breaks
generation. Fix: estimate the covariance over PER-TOKEN activations (response_all), so the eraser is
calibrated for the per-token application it actually gets during generation (the reference's "concept
scrubbing" fits per-token too). Streaming GPU accumulation of N, sum_x, X^T X, X^T z per layer.

The norm-matched RANDOM control is matched to the real per-token perturbation RMS ANALYTICALLY from Sigma:
real RMS^2 = ||a||^2 (b^T Sigma b); random (unit r, scale s) RMS^2 = s^2 (r^T Sigma r) -> s = ||a|| sqrt(bSb/rSr).

Verification: stash a per-token sample at L23, apply the fitted eraser, problem-grouped CV refit probe ->
must hit chance (correctness erased per-token). Output: results/stage_c/leace_pertok_<seed>.pt.

Run:  python stage_c_leace_fit_pertoken.py --seed rh-s42
"""
import os, sys, json, argparse, gc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from src.activations import BatchedTransformersActivations
from cache_adapter_space import load_model
from stage_c_leace import _cv_auroc

OUT = "results/stage_c"
GATE_ERASE, GATE_RAND_MIN = 0.55, 0.70
VERIFY_L = 23          # layer to stash a per-token sample at, for the erasure check
VERIFY_CAP = 20000     # max stashed tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="rh-s42")
    ap.add_argument("--layers", default="23,24,25,26,27,28,29,30,31,32,33,34,35,36")
    ap.add_argument("--cap-correct", type=int, default=1500, help="cap #correct rows (all wrong kept)")
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    out = args.out or f"{OUT}/leace_pertok_{args.seed}.pt"
    dev = "cuda"

    rows = json.load(open(f"{OUT}/leace_clean_{args.seed}.json"))["rows"]
    cor = [r for r in rows if r["eq_correct"]][:args.cap_correct]
    wro = [r for r in rows if not r["eq_correct"]]
    rows = cor + wro
    print(f"[{args.seed}] per-token fit on {len(rows)} rows ({len(cor)} cor + {len(wro)} wro), "
          f"problems={len(set(r['id'] for r in rows))}, layers={layers}", flush=True)

    mdl, tok = load_model(args.seed)
    from peft import PeftModel
    assert isinstance(mdl, PeftModel)
    cacher = BatchedTransformersActivations(model=mdl, tokenizer=tok, batch_size=args.bs, progress_bar=True)
    H = mdl.config.hidden_size
    nL = len(layers)
    N = torch.zeros(nL, device=dev)
    sx = torch.zeros(nL, H, device=dev)
    sxx = torch.zeros(nL, H, H, device=dev)
    sxz = torch.zeros(nL, H, device=dev)
    sz = torch.zeros(nL, device=dev)
    stash_x, stash_y, stash_g = [], [], []                  # per-token sample @ VERIFY_L

    print("  streaming response_all (per-token) ...", flush=True)
    for s in range(0, len(rows), args.bs):
        br = rows[s:s + args.bs]
        a = cacher.cache_activations(prompts=[r["prompt"] for r in br],
                                     responses=[r["response"] for r in br],
                                     position=["response_all"], layers=layers)["response_all"]  # (nL,bs,T,H) NaN-pad
        zrow = torch.tensor([1.0 if r["eq_correct"] else 0.0 for r in br])
        for li in range(nL):
            al = a[li]                                       # (bs,T,H) cpu
            mask = ~torch.isnan(al[:, :, 0])                # (bs,T)
            cnt = mask.sum(1)                               # (bs,)
            Xb = al[mask].to(dev).float()                   # (Tb,H)
            zb = torch.repeat_interleave(zrow, cnt).to(dev) # (Tb,)
            N[li] += Xb.shape[0]; sx[li] += Xb.sum(0); sxx[li] += Xb.T @ Xb
            sxz[li] += Xb.T @ zb; sz[li] += zb.sum()
            if layers[li] == VERIFY_L and sum(t.shape[0] for t in stash_x) < VERIFY_CAP:
                stash_x.append(al[mask].float().cpu())
                stash_y.append(torch.repeat_interleave(zrow, cnt))
                stash_g.append(torch.tensor([r["id"] for r, c in zip(br, cnt) for _ in range(int(c))]))
        del a; gc.collect(); torch.cuda.empty_cache()
    cacher.model = None; del cacher, mdl; gc.collect(); torch.cuda.empty_cache()

    real, rand, verify = {}, {}, {}
    rng = np.random.default_rng(20260628)
    for li, L in enumerate(layers):
        n = N[li]
        mu = sx[li] / n
        Sig = sxx[li] / n - torch.outer(mu, mu)
        sig_xz = sxz[li] / n - mu * (sz[li] / n)
        Sr = Sig + args.ridge * torch.trace(Sig) / H * torch.eye(H, device=dev)
        b = torch.linalg.solve(Sr, sig_xz); c = float(sig_xz @ b)
        a = sig_xz / c
        # analytic norm-matched random
        r = torch.tensor(rng.standard_normal(H), device=dev, dtype=Sig.dtype); r /= r.norm()
        s_scale = a.norm() * torch.sqrt((b @ Sig @ b) / (r @ Sig @ r)).clamp_min(0)
        real[L] = {"m": mu.cpu(), "a": a.cpu(), "b": b.cpu()}
        rand[L] = {"m": mu.cpu(), "a": (s_scale * r).cpu(), "b": r.cpu()}
        del Sig, Sr; gc.collect(); torch.cuda.empty_cache()

    # erasure verification at VERIFY_L (per-token, problem-grouped CV)
    Xs = torch.cat(stash_x).numpy(); ys = torch.cat(stash_y).numpy().astype(int); gs = torch.cat(stash_g).numpy()
    mu, a, b = (real[VERIFY_L][k].numpy() for k in "mab")
    Xe = Xs - np.outer((Xs - mu) @ b, a)
    auc_o = _cv_auroc(Xs, ys, gs); auc_e = _cv_auroc(Xe, ys, gs)
    mu2, ar, br = (rand[VERIFY_L][k].numpy() for k in "mab")
    auc_r = _cv_auroc(Xs - np.outer((Xs - mu2) @ br, ar), ys, gs)
    ok = (auc_e < GATE_ERASE) and (auc_r > GATE_RAND_MIN)
    print(f"\n  [erasure verify @L{VERIFY_L}] per-token grouped-CV ({len(ys)} tokens, {len(set(gs))} problems): "
          f"orig={auc_o:.3f} LEACE={auc_e:.3f} random={auc_r:.3f} -> {'PASS' if ok else 'FAIL'}", flush=True)

    torch.save({"seed": args.seed, "layers": layers, "ridge": args.ridge, "fit": "per_token",
                "real": real, "random": rand,
                "verify": {VERIFY_L: {"orig": auc_o, "real": auc_e, "rand": auc_r}}}, out)
    print(f"  saved {out}")
    print("  NEXT: stage_c_leace_verify.py --erasers {} (per-token magnitude should now be small + matched), "
          "then stage_c_run.py --phase gate --erase --erasers {}".format(out, out))


if __name__ == "__main__":
    main()
