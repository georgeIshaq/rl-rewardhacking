"""
stage_c_leace_verify.py  (BOX / GPU) -- verify the LEACE GENERATION path before trusting any
scrubbed behavioral number. The competence gate (RESULTS §4.9) collapsed AND the random control also
dropped -> suspect a broken/over-aggressive intervention, not entanglement. prereq0 only validated the
single-direction hook path, never the LEACE eraser path. Three checks:

1. NO-OP IDENTITY (eraser path): an identity eraser (a=0) -> greedy generation byte-identical to baseline.
   Validates the _eraser_hook plumbing doesn't perturb when it shouldn't.
2. PER-TOKEN MAGNITUDE: the eraser was fit on POOLED response_avg; measure what it actually removes
   PER-TOKEN during a forward (||a|| * |b.(x-mu)| / ||x|| per layer), vs the pooled ~2-3%. If per-token >>
   pooled, the pooled-fit eraser is over-aggressive applied per-token (generic damage). Also compare
   real vs random per-token magnitude -- if not matched per-token, the gate comparison is invalid.
3. EYEBALL: print baseline / leace / random generations -> coherent-but-wrong (competence) vs broken
   garbage (over-damage).

Run:  python stage_c_leace_verify.py --seed rh-s42 --start-hs 23
"""
import os, sys, json, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from stage_c_ablation import AblatedHFModel

OUTDIR = "results/stage_c"


def get_prompts(seed, n):
    b = json.load(open(f"{OUTDIR}/bundle_{seed}.json"))
    recs = [r for r in (b["records"] if isinstance(b, dict) else b) if r["bucket"] == "correct_clean"]
    return [r["prompt"] for r in recs[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="rh-s42")
    ap.add_argument("--start-hs", type=int, default=23)
    ap.add_argument("--n-prompts", type=int, default=4)
    args = ap.parse_args()

    er = torch.load(f"{OUTDIR}/leace_{args.seed}.pt", map_location="cpu")
    layers = [L for L in er["layers"] if L >= args.start_hs]
    real = {L: (er["real"][L]["m"], er["real"][L]["a"], er["real"][L]["b"]) for L in layers}
    rand = {L: (er["random"][L]["m"], er["random"][L]["a"], er["random"][L]["b"]) for L in layers}
    ident = {L: (er["real"][L]["m"], torch.zeros_like(er["real"][L]["a"]),
                 torch.zeros_like(er["real"][L]["b"])) for L in layers}

    m = AblatedHFModel(args.seed)
    prompts = get_prompts(args.seed, args.n_prompts)
    print(f"=== LEACE generation-path verify: {args.seed} L{args.start_hs}+ ({len(layers)} layers), "
          f"{len(prompts)} prompts ===", flush=True)

    # (1) no-op identity on the eraser path
    m.clear_ablation(); base_ids = m.greedy_ids(prompts, 64)
    m.set_erasers(ident); id_ids = m.greedy_ids(prompts, 64); m.clear_ablation()
    identical = all(a == b for a, b in zip(base_ids, id_ids))
    print(f"\n[1 no-op identity] identity eraser vs baseline greedy: "
          f"{'IDENTICAL' if identical else 'DIFFERS -> eraser-hook plumbing perturbs when it should not'}", flush=True)

    # (2) per-token perturbation magnitude (forward, no hooks; compute what each eraser would remove)
    m.clear_ablation()
    enc = m._encode(prompts)
    with torch.inference_mode():
        o = m.model(**enc, output_hidden_states=True)
    am = enc["attention_mask"].bool()                              # (B,T) real-token mask
    print(f"\n[2 per-token magnitude]  ||Δ||/||x|| per layer  (pooled-fit was ~0.02-0.03)", flush=True)
    print(f"  {'layer':6}{'real':>10}{'random':>10}", flush=True)
    for L in layers:
        h = o.hidden_states[L].float()                            # (B,T,H)
        def ratio(triple):
            mu, a, b = (t.to(h.device).float() for t in triple)
            coeff = (h - mu) @ b                                  # (B,T)
            pert = coeff.abs() * a.norm()                         # ||Δ|| per token
            actn = h.norm(dim=-1)                                 # ||x|| per token
            r = (pert[am] / actn[am].clamp_min(1e-6))
            return r.mean().item()
        print(f"  L{L:<5}{ratio(real[L]):>10.3f}{ratio(rand[L]):>10.3f}", flush=True)

    # (3) eyeball: short generations under each condition
    from src.generate import SamplingParams as SP
    sp = SP(n=1, temperature=0.7, top_p=0.95, max_new_tokens=220)
    print("\n[3 eyeball] first prompt, 220 tok, sampled:", flush=True)
    for name, spec in [("baseline", None), ("leace", real), ("random", rand)]:
        m.clear_ablation()
        if spec is not None:
            m.set_erasers(spec)
        out = m.generate(prompts[:1], sp, micro_batch=1, seed=0)[0][0]
        m.clear_ablation()
        print(f"\n----- {name} -----\n{out[:700]}", flush=True)


if __name__ == "__main__":
    main()
