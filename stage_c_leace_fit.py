"""
stage_c_leace_fit.py  (BOX / GPU) -- fit + VERIFY the LEACE erasers used for Stage C scrubbing.

1. cache clean adapter-space response_avg at every scrub layer (L23..36) on the id-carrying
   clean fit-set (stage_c_build_leace_set.py).
2. per layer: fit the LEACE correctness eraser (mu,a,b) and a norm-matched RANDOM eraser.
3. ERASURE-VERIFICATION (non-negotiable, problem-GROUPED CV): refit a correctness probe on the
   scrubbed activations -> real eraser MUST drop to ~chance; random eraser MUST stay high.
   If real erasure does not reach chance under grouped CV, STOP -- downstream means nothing.
Output: results/stage_c/leace_<seed>.pt  (consumed by stage_c_run.py --erase).

Run:  python stage_c_leace_fit.py --seed rh-s42
"""
import os, sys, json, argparse, gc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from src.activations import BatchedTransformersActivations
from cache_adapter_space import load_model
from stage_c_leace import fit_leace, apply_eraser, make_random_eraser, _cv_auroc

OUT = "results/stage_c"
GATE_ERASE = 0.55          # real-erased refit AUROC must be BELOW this (grouped CV) = erased
GATE_RAND_MIN = 0.70       # random-erased must stay ABOVE this = control didn't erase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="rh-s42")
    ap.add_argument("--layers", default="23,24,25,26,27,28,29,30,31,32,33,34,35,36")
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    os.makedirs(OUT, exist_ok=True)

    fit = json.load(open(f"{OUT}/leace_clean_{args.seed}.json"))
    rows = fit["rows"]
    y = np.array([1 if r["eq_correct"] else 0 for r in rows])
    groups = np.array([r["id"] for r in rows])
    print(f"[{args.seed}] fit-set {len(rows)} rows  correct={int(y.sum())} wrong={int((1-y).sum())}  "
          f"problems={len(set(groups))}  layers={layers}", flush=True)
    assert (1 - y).sum() >= 20 and len(set(groups)) >= 5, "too few wrong rows / problems for a fit+grouped-CV"

    mdl, tok = load_model(args.seed)
    from peft import PeftModel
    assert isinstance(mdl, PeftModel), f"expected a PEFT adapter model, got {type(mdl).__name__}"
    cacher = BatchedTransformersActivations(model=mdl, tokenizer=tok, batch_size=args.bs, progress_bar=False)
    print("  caching clean response_avg @ scrub layers ...", flush=True)
    acts = cacher.cache_activations(prompts=[r["prompt"] for r in rows],
                                    responses=[r["response"] for r in rows],
                                    position=["response_avg"], layers=layers)["response_avg"]  # (nL,n,H)
    cacher.model = None; del cacher, mdl; gc.collect(); torch.cuda.empty_cache()

    real, rand, verify = {}, {}, {}
    ok = True
    print(f"\n  {'layer':6}{'orig':>8}{'LEACE':>8}{'random':>9}{'|xcov|':>10}   gate", flush=True)
    for li, L in enumerate(layers):
        X = acts[li].float().numpy()
        er = fit_leace(X, y, ridge=args.ridge)                       # (mu,a,b)
        rr = make_random_eraser(X, er, seed=20260628 + L, ridge=args.ridge)
        a_orig = _cv_auroc(X, y, groups)
        Xr = apply_eraser(X, *er)
        a_real = _cv_auroc(Xr, y, groups)
        a_rand = _cv_auroc(apply_eraser(X, *rr), y, groups)
        xcov = float(np.abs(((Xr - Xr.mean(0)).T @ (y - y.mean())) / len(y)).max())
        good = (a_real < GATE_ERASE) and (a_rand > GATE_RAND_MIN)
        ok = ok and good
        for store, e in ((real, er), (rand, rr)):
            store[L] = {k: torch.tensor(v, dtype=torch.float32) for k, v in zip("mab", e)}
        verify[L] = {"orig": a_orig, "real": a_real, "rand": a_rand, "xcov": xcov}
        print(f"  L{L:<5}{a_orig:>8.3f}{a_real:>8.3f}{a_rand:>9.3f}{xcov:>10.1e}   "
              f"{'OK' if good else 'FAIL'}", flush=True)

    path = f"{OUT}/leace_{args.seed}.pt"
    torch.save({"seed": args.seed, "layers": layers, "ridge": args.ridge,
                "real": real, "random": rand, "verify": verify}, path)
    worst = max(v["real"] for v in verify.values())
    print(f"\n  saved {path}")
    print(f"  ERASURE GATE: worst real-erased AUROC over layers = {worst:.3f} (must be < {GATE_ERASE})")
    print("  RESULT:", "PASS -- correctness erased (grouped CV); proceed to scrubbed gate/instrumental."
          if ok else "FAIL -- erasure incomplete or control erased; do NOT run behavior. Investigate.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
