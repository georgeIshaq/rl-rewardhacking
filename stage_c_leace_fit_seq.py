"""
stage_c_leace_fit_seq.py  (BOX / GPU) -- SEQUENTIAL per-token concept-scrubbing (the reference method).

Independent-fit/simultaneous-apply under-erases: correctness re-emerges by the final layer (RESULTS §4.11,
0.692->0.686) because each eraser is fit on CLEAN upstream but applied to SCRUBBED upstream. Fix: fit each
layer's eraser on the activations AS THEY ARE after the already-fitted upstream erasers are applied
(captured via a forward hook -- intermediate output_hidden_states does NOT reflect hooks under
check_model_inputs). Per-token covariance (gentle, calibrated magnitude). Reduced scope = de-risk:
verify end-to-end erasure here, competence via stage_c_run --erase, BEFORE the full run.

Builds in the end-to-end erasure check (final-layer response_avg reflects the stacked hooks): baseline must
be > chance, LEACE -> ~chance, random stays up. Output: results/stage_c/leace_seq_<seed>.pt

Run:  python stage_c_leace_fit_seq.py --seed rh-s42 --fit-rows 600
"""
import os, sys, json, argparse, gc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from src.activations import BatchedTransformersActivations
from stage_c_ablation import AblatedHFModel
from stage_c_leace import _cv_auroc

OUT = "results/stage_c"


class _Cap:
    buf = None
    def hook(self, module, inp, out):
        self.buf = (out[0] if isinstance(out, tuple) else out).detach()


def encode_full(tok, prompts, responses, device):
    """prompt+response tokenization matching src/activations.py; right-padded. Returns ids, attn, prompt_lens."""
    ids, plens = [], []
    for p, r in zip(prompts, responses):
        full = tok.apply_chat_template(p + [{"role": "assistant", "content": r}], tokenize=False,
                                       add_generation_prompt=False)
        pr = tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        ids.append(tok(full, add_special_tokens=False)["input_ids"])
        plens.append(len(tok(pr, add_special_tokens=False)["input_ids"]))
    mx = max(len(x) for x in ids)
    pad = tok.pad_token_id
    input_ids = torch.full((len(ids), mx), pad, dtype=torch.long)
    attn = torch.zeros((len(ids), mx), dtype=torch.long)
    for i, x in enumerate(ids):
        input_ids[i, :len(x)] = torch.tensor(x); attn[i, :len(x)] = 1
    return input_ids.to(device), attn.to(device), plens


def sample(rows, n, seed):
    cor = [r for r in rows if r["eq_correct"]]; wro = [r for r in rows if not r["eq_correct"]]
    rng = np.random.default_rng(seed); h = n // 2
    return ([cor[i] for i in rng.permutation(len(cor))[:h]] +
            [wro[i] for i in rng.permutation(len(wro))[:min(h, len(wro))]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="rh-s42")
    ap.add_argument("--layers", default="23,24,25,26,27,28,29,30,31,32,33,34,35,36")
    ap.add_argument("--fit-rows", type=int, default=600)
    ap.add_argument("--eval-rows", type=int, default=240)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    out = args.out or f"{OUT}/leace_seq_{args.seed}.pt"
    dev = "cuda"

    allrows = json.load(open(f"{OUT}/leace_clean_{args.seed}.json"))["rows"]
    frows = sample(allrows, args.fit_rows, 0)
    print(f"[{args.seed}] SEQUENTIAL per-token scrub: fit on {len(frows)} rows "
          f"({sum(r['eq_correct'] for r in frows)} cor), layers={layers}", flush=True)
    m = AblatedHFModel(args.seed)
    H = m.model.config.hidden_size

    fitted, randd = {}, {}
    rng = np.random.default_rng(20260628)
    for L in layers:                                            # ascending: fit on already-scrubbed upstream
        m.set_erasers(fitted)                                  # erasers for layers < L active
        cap = _Cap(); h = m.layers[L - 1].register_forward_hook(cap.hook)
        N = torch.zeros((), device=dev); sx = torch.zeros(H, device=dev)
        sxx = torch.zeros(H, H, device=dev); sxz = torch.zeros(H, device=dev); sz = torch.zeros((), device=dev)
        for s in range(0, len(frows), args.bs):
            br = frows[s:s + args.bs]
            ids, attn, plens = encode_full(m.tok, [r["prompt"] for r in br], [r["response"] for r in br], dev)
            with torch.inference_mode():
                m.model(input_ids=ids, attention_mask=attn)   # triggers upstream erasers + cap at L
            buf = cap.buf.float()                              # (B, mx, H) post-upstream-scrub output of layer L
            for i, r in enumerate(br):
                rl = int(attn[i].sum()); pl = plens[i]
                if rl - pl < 1:
                    continue
                Xi = buf[i, pl:rl, :]                          # response tokens (T_i, H)
                zi = 1.0 if r["eq_correct"] else 0.0
                N += Xi.shape[0]; sx += Xi.sum(0); sxx += Xi.T @ Xi
                sxz += Xi.sum(0) * zi; sz += Xi.shape[0] * zi
        h.remove()
        mu = sx / N; Sig = sxx / N - torch.outer(mu, mu); sig_xz = sxz / N - mu * (sz / N)
        Sr = Sig + args.ridge * torch.trace(Sig) / H * torch.eye(H, device=dev)
        b = torch.linalg.solve(Sr, sig_xz); c = float(sig_xz @ b); a = sig_xz / c
        rr = torch.tensor(rng.standard_normal(H), device=dev, dtype=Sig.dtype); rr /= rr.norm()
        s_scale = a.norm() * torch.sqrt((b @ Sig @ b) / (rr @ Sig @ rr)).clamp_min(0)
        fitted[L] = {"m": mu.cpu(), "a": a.cpu(), "b": b.cpu()}
        randd[L] = {"m": mu.cpu(), "a": (s_scale * rr).cpu(), "b": rr.cpu()}
        print(f"    fitted L{L}  (|a|={a.norm():.3f}, c={c:.3e}, tokens={int(N)})", flush=True)
        del Sig, Sr, sxx; gc.collect(); torch.cuda.empty_cache()
    m.clear_ablation()
    torch.save({"seed": args.seed, "layers": layers, "ridge": args.ridge, "fit": "sequential_pertoken",
                "real": fitted, "random": randd}, out)
    print(f"\n  saved -> {out}", flush=True)

    # end-to-end erasure check (final-layer response_avg reflects the stacked hooks)
    erows = sample(allrows, args.eval_rows, 99)
    y = np.array([1 if r["eq_correct"] else 0 for r in erows]); g = np.array([r["id"] for r in erows])
    cacher = BatchedTransformersActivations(model=m.model, tokenizer=m.tok, batch_size=args.bs, progress_bar=False)
    Lf = m.num_layers

    def final_avg():
        return cacher.cache_activations(prompts=[r["prompt"] for r in erows], responses=[r["response"] for r in erows],
                                        position=["response_avg"], layers=[Lf])["response_avg"][0].float().numpy()
    real = {L: (fitted[L]["m"], fitted[L]["a"], fitted[L]["b"]) for L in layers}
    rand = {L: (randd[L]["m"], randd[L]["a"], randd[L]["b"]) for L in layers}
    m.clear_ablation();  ab = _cv_auroc(final_avg(), y, g)
    m.set_erasers(real); ae = _cv_auroc(final_avg(), y, g)
    m.set_erasers(rand); ar = _cv_auroc(final_avg(), y, g)
    m.clear_ablation()
    print(f"\n  [end-to-end @final L{Lf}] grouped-CV correctness AUROC ({len(y)} rows, {len(set(g))} probs):", flush=True)
    print(f"    baseline={ab:.3f} (well-posed)   LEACE={ae:.3f} ({'ERASED' if ae < 0.55 else 'UNDER-ERASES'})   "
          f"random={ar:.3f}", flush=True)
    print(f"  NEXT (competence): stage_c_run.py --phase gate --erase --erasers {out} --n-problems 60 --n-samples 4 --micro-batch 6", flush=True)


if __name__ == "__main__":
    main()
