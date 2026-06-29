"""
stage_c_leace_nested.py  (BOX / GPU) -- HONEST per-token erasure check (nested CV).

The fit's full-data-eraser CV is leakage-prone (gave the misleading inverted 0.11 for the pooled
eraser). The honest test: fit the LEACE eraser on TRAIN-fold tokens, apply to held-out tokens, refit a
probe -> does per-token correctness drop to chance out-of-sample? Caches per-token response_all at a few
layers for a sample of clean rows, problem-grouped nested CV.

Run:  python stage_c_leace_nested.py --seed rh-s42 --layers 23,34 --n-rows 120
"""
import os, sys, json, argparse, gc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from src.activations import BatchedTransformersActivations
from cache_adapter_space import load_model
from stage_c_leace import fit_leace, apply_eraser

OUT = "results/stage_c"


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="rh-s42")
    ap.add_argument("--layers", default="23,34")
    ap.add_argument("--n-rows", type=int, default=120, help="balanced-ish sample of clean rows")
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    rows = json.load(open(f"{OUT}/leace_clean_{args.seed}.json"))["rows"]
    cor = [r for r in rows if r["eq_correct"]]
    wro = [r for r in rows if not r["eq_correct"]]
    rng = np.random.default_rng(1)
    half = args.n_rows // 2
    pick = [cor[i] for i in rng.permutation(len(cor))[:half]] + [wro[i] for i in rng.permutation(len(wro))[:half]]
    print(f"[{args.seed}] nested per-token erasure: {len(pick)} rows "
          f"({sum(r['eq_correct'] for r in pick)} cor), problems={len(set(r['id'] for r in pick))}, layers={layers}",
          flush=True)

    mdl, tok = load_model(args.seed)
    cacher = BatchedTransformersActivations(model=mdl, tokenizer=tok, batch_size=args.bs, progress_bar=True)
    # collect per-token acts per layer
    Xs = {L: [] for L in layers}; ys = []; gs = []
    for s in range(0, len(pick), args.bs):
        br = pick[s:s + args.bs]
        a = cacher.cache_activations(prompts=[r["prompt"] for r in br], responses=[r["response"] for r in br],
                                     position=["response_all"], layers=layers)["response_all"]   # (nL,bs,T,H)
        for i, r in enumerate(br):
            real = ~torch.isnan(a[0, i, :, 0])
            t = int(real.sum())
            for li, L in enumerate(layers):
                Xs[L].append(a[li, i, real, :].float().numpy())
            ys += [1 if r["eq_correct"] else 0] * t
            gs += [r["id"]] * t
        del a; gc.collect(); torch.cuda.empty_cache()
    cacher.model = None; del cacher, mdl; gc.collect(); torch.cuda.empty_cache()
    ys = np.array(ys); gs = np.array(gs)
    print(f"  collected {len(ys)} tokens\n", flush=True)

    ns = min(5, len(set(gs)))
    for L in layers:
        X = np.concatenate(Xs[L])
        oof = np.zeros(len(ys))
        for tr, te in GroupKFold(ns).split(X, ys, gs):
            e = fit_leace(X[tr], ys[tr])                       # per-token eraser on TRAIN tokens only
            clf = probe().fit(apply_eraser(X[tr], *e), ys[tr])
            oof[te] = clf.predict_proba(apply_eraser(X[te], *e))[:, 1]
        auc_n = roc_auc_score(ys, oof)
        auc_o = roc_auc_score(ys, probe_oof(X, ys, gs, ns))
        print(f"  L{L}: per-token correctness AUROC  orig={auc_o:.3f}  -> nested-LEACE={auc_n:.3f}  "
              f"({'ERASED' if auc_n < 0.55 else 'UNDER-REMOVES'})", flush=True)


def probe_oof(X, y, g, ns):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(ns).split(X, y, g):
        oof[te] = probe().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return oof


if __name__ == "__main__":
    main()
