"""
stage_c_prereq0.py  (BOX / GPU) -- verify the generation-under-ablation harness BEFORE
any real hacking-rate number. This path (generation under intervention) has NO golden
reference, so it leans entirely on internal-consistency gates. If any gate FAILS, STOP:
every downstream number would be silently garbage.

Three gates (Stage-C analogues of the read-only identity/teeth/adapter-is-live discipline):
  1. NO-OP IDENTITY  -- ablate at alpha=0 (real dir AND random dir) -> greedy generation is
     token-identical to no-ablation. Proves the harness doesn't perturb when it shouldn't.
  2. HOOK-IS-LIVE    -- ablate at alpha=1 -> the direction's component is driven to ~0 at the
     hooked layer AND at the final layer (re-derivation prevented), and activations move.
     Proves the intervention is actually applied during the forward, not silently dropped.
  3. ADAPTER-IS-LIVE -- adapter enabled vs disabled diverges (deep-layer cosine < 0.95).
     Proves we generate through the s42 LoRA, not base (a base run = meaningless hack rates).

Run:
  python stage_c_prereq0.py --seed rh-s42 --start-hs 23
  python stage_c_prereq0.py --seed rh-s42 --start-hs 34       # backup deep location
"""
import os, sys, json, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from stage_c_ablation import AblatedHFModel, load_direction, random_direction

GATE_ADAPTER_COS = 0.95     # adapter enabled vs disabled must be BELOW this somewhere (teeth ~0.84)
GATE_LIVE_RATIO = 0.05      # ablated |coeff| must be <= 5% of clean |coeff|
IDENTITY_NEW_TOKENS = 96    # greedy tokens for the identity check (short = fast, still decisive)


def get_prompts(seed, n):
    """A few real prompts: prefer the Stage C bundle, else the on-box cells file."""
    for p in (f"results/stage_c/bundle_{seed}.json", f"results/cells/cells_{seed}.json"):
        if os.path.exists(p):
            recs = json.load(open(p))
            recs = recs if isinstance(recs, list) else recs.get("records", [])
            prompts = [r["prompt"] for r in recs if r.get("prompt")][:n]
            if prompts:
                print(f"  prompts: {len(prompts)} from {p}")
                return prompts
    raise FileNotFoundError("no bundle or cells file to draw prompts from")


def gate_adapter_is_live(m, prompts):
    layers = list(range(m.num_layers + 1))                     # all hidden states 0..num_layers
    m.clear_ablation()
    en = m.pooled_hidden(prompts, layers, enabled=True)
    di = m.pooled_hidden(prompts, layers, enabled=False)
    mins = []
    for L in layers:
        cos = torch.nn.functional.cosine_similarity(en[L], di[L], dim=-1)   # (B,)
        mins.append(cos.min().item())
    gmin = min(mins)
    ok = gmin < GATE_ADAPTER_COS
    print(f"  [adapter-is-live] global_min cos(enabled,disabled) = {gmin:.3f}  "
          f"(gate <{GATE_ADAPTER_COS}) -> {'LIVE' if ok else 'DEAD'}")
    return ok


def gate_no_op_identity(m, prompts, d, dim, start_hs):
    base = m.greedy_ids(prompts, IDENTITY_NEW_TOKENS)
    checks = {}
    for name, vec in [("real-dir@0", d), ("random-dir@0", random_direction(dim, 12345))]:
        m.set_ablation(direction=vec, alpha=0.0, start_hs=start_hs)
        got = m.greedy_ids(prompts, IDENTITY_NEW_TOKENS)
        m.clear_ablation()
        identical = all(a == b for a, b in zip(base, got))
        n_diff = sum(a != b for a, b in zip(base, got))
        checks[name] = identical
        print(f"  [no-op identity] {name:14}: {'IDENTICAL' if identical else f'DIFFERS ({n_diff}/{len(base)} prompts)'}")
    return all(checks.values())


def gate_hook_is_live(m, prompts, d, start_hs):
    last_hs = m.num_layers
    layers = sorted({start_hs, last_hs})
    dt = torch.tensor(d, dtype=torch.float32)
    m.clear_ablation()
    clean = m.pooled_hidden(prompts, layers, enabled=True)
    m.set_ablation(direction=d, alpha=1.0, start_hs=start_hs)
    abl = m.pooled_hidden(prompts, layers, enabled=True)
    m.clear_ablation()
    ok = True
    for L in layers:
        c_clean = (clean[L] @ dt).abs().mean().item()
        c_abl = (abl[L] @ dt).abs().mean().item()
        ratio = c_abl / (c_clean + 1e-9)
        moved = torch.nn.functional.cosine_similarity(clean[L], abl[L], dim=-1).mean().item()
        where = "hooked L" if L == start_hs else "final L"
        good = ratio <= GATE_LIVE_RATIO
        ok = ok and good
        print(f"  [hook-is-live] hs{L} ({where}): |coeff| clean={c_clean:.3f} abl={c_abl:.3f} "
              f"ratio={ratio:.3f} (<= {GATE_LIVE_RATIO}) cos(clean,abl)={moved:.3f} -> {'OK' if good else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="rh-s42")
    ap.add_argument("--start-hs", type=int, default=23, help="hidden-state layer the direction was fit at (23 primary, 34 backup)")
    ap.add_argument("--n-prompts", type=int, default=6)
    args = ap.parse_args()

    print(f"=== Stage C Prereq 0 : {args.seed}  start_hs=L{args.start_hs} ===")
    m = AblatedHFModel(args.seed)
    dim = m.model.config.hidden_size
    d = load_direction(args.seed, args.start_hs)
    print(f"  model: {m.num_layers} decoder layers, hidden={dim}, adapter={m.has_adapter}")
    print(f"  direction need_L{args.start_hs}: |d|={np.linalg.norm(d):.4f} (unit)")
    prompts = get_prompts(args.seed, args.n_prompts)

    print("\n-- Gate 3: adapter-is-live --")
    g3 = gate_adapter_is_live(m, prompts)
    print("\n-- Gate 1: no-op identity --")
    g1 = gate_no_op_identity(m, prompts, d, dim, args.start_hs)
    print("\n-- Gate 2: hook-is-live --")
    g2 = gate_hook_is_live(m, prompts, d, args.start_hs)

    print("\n" + "=" * 60)
    allok = g1 and g2 and g3
    for name, ok in [("no-op identity", g1), ("hook-is-live", g2), ("adapter-is-live", g3)]:
        print(f"  {name:18}: {'PASS' if ok else 'FAIL'}")
    print("RESULT:", "ALL GATES PASS -- harness trustworthy, proceed to the gate run."
          if allok else "FAIL -- generation path is wrong; do NOT run any ablation experiment.")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
