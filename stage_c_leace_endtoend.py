"""
stage_c_leace_endtoend.py  (BOX / GPU) -- does the STACKED simultaneous erasure actually erase?

We fit erasers independently per layer and apply all 14 together (independent-fit / simultaneous-apply).
The reference concept-scrubbing fits SEQUENTIALLY (each layer on already-scrubbed upstream); independent
apply can under-erase via downstream re-emergence. Decisive empirical test: run clean correct+wrong
through the eraser-hooked model and read the TRULY-scrubbed final residual (last_hidden_state reflects the
hooks; intermediate output_hidden_states does not), then refit a correctness probe (problem-grouped CV).

  baseline (no erasers) must be ABOVE chance (construct well-posed); LEACE must drop to ~chance (erased);
  random must stay up (control). If LEACE stays above chance -> stacked apply under-erases -> sequential fit.

Run:  python stage_c_leace_endtoend.py --seed rh-s42 --erasers results/stage_c/leace_pertok_rh-s42.pt
"""
import os, sys, json, argparse, gc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from src.activations import BatchedTransformersActivations
from stage_c_ablation import AblatedHFModel
from stage_c_leace import _cv_auroc

OUT = "results/stage_c"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="rh-s42")
    ap.add_argument("--erasers", default="")
    ap.add_argument("--start-hs", type=int, default=23)
    ap.add_argument("--n-rows", type=int, default=240)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    erpath = args.erasers or f"{OUT}/leace_{args.seed}.pt"
    Lfinal_hs = None  # set after model load

    rows = json.load(open(f"{OUT}/leace_clean_{args.seed}.json"))["rows"]
    cor = [r for r in rows if r["eq_correct"]]; wro = [r for r in rows if not r["eq_correct"]]
    rng = np.random.default_rng(2); half = args.n_rows // 2
    pick = [cor[i] for i in rng.permutation(len(cor))[:half]] + [wro[i] for i in rng.permutation(len(wro))[:half]]
    y = np.array([1 if r["eq_correct"] else 0 for r in pick]); g = np.array([r["id"] for r in pick])
    print(f"[{args.seed}] end-to-end erasure: {len(pick)} rows ({int(y.sum())} cor), "
          f"problems={len(set(g))}, erasers={erpath}", flush=True)

    er = torch.load(erpath, map_location="cpu")
    layers = [L for L in er["layers"] if L >= args.start_hs]
    real = {L: (er["real"][L]["m"], er["real"][L]["a"], er["real"][L]["b"]) for L in layers}
    rand = {L: (er["random"][L]["m"], er["random"][L]["a"], er["random"][L]["b"]) for L in layers}

    m = AblatedHFModel(args.seed)
    Lfinal = m.num_layers                                       # final hidden state index reflects the hooks
    cacher = BatchedTransformersActivations(model=m.model, tokenizer=m.tok, batch_size=args.bs, progress_bar=True)

    def final_resp_avg():                                       # response_avg at the final layer (reflects erasure)
        a = cacher.cache_activations(prompts=[r["prompt"] for r in pick],
                                     responses=[r["response"] for r in pick],
                                     position=["response_avg"], layers=[Lfinal])["response_avg"]
        return a[0].float().numpy()

    print(f"  caching final-layer response_avg (L{Lfinal}) x3 conditions ...", flush=True)
    m.clear_ablation();      Xb = final_resp_avg()
    m.set_erasers(real);     Xe = final_resp_avg()
    m.set_erasers(rand);     Xr = final_resp_avg()
    m.clear_ablation()

    ab = _cv_auroc(Xb, y, g); ae = _cv_auroc(Xe, y, g); ar = _cv_auroc(Xr, y, g)
    print(f"\n  [end-to-end @final L{Lfinal}] problem-grouped CV correctness AUROC:", flush=True)
    print(f"    baseline (no erase) = {ab:.3f}   (must be > chance: construct well-posed)", flush=True)
    print(f"    LEACE (stacked)     = {ae:.3f}   ({'ERASED' if ae < 0.55 else 'UNDER-ERASES -> need sequential fit'})", flush=True)
    print(f"    random (control)    = {ar:.3f}   (should stay ~baseline)", flush=True)


if __name__ == "__main__":
    main()
