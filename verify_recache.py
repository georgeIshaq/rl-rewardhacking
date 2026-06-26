"""
verify_recache.py  --  Identity check for the activation re-cache pipeline.

Question it answers: if we re-run activation caching, do we reproduce the EXACT
activations already saved in acts_*.pt?  Everything downstream (growing the
cells from the 40k pool, the per-token response_all re-cache) rests on this.

Why it's decisive and not just "close": seeds sit in contiguous, mostly
8-aligned blocks in the cache order, and caching batches at batch_size=8 over
the dataset in order. So re-caching an 8-aligned contiguous block reproduces the
*exact same batch grouping + padding* as the original run -> a correct pipeline
matches bit-for-bit (to the bf16 floor). No batching-noise confound to wave away.

Config being verified (scripts/run_probes.py::run_cache_activations +
src/activations.py): BASE model (NO LoRA adapter), bfloat16, flash_attention_2,
apply_chat_template, add_special_tokens=False, padding=True, batch_size=8.

Two ways to provide the reference activations:
  --bundle verify_bundle.pt   tiny (~40MB): only the rows the check compares.
                              Build it on the Mac with build_verify_bundle.py.
                              Use this to AVOID uploading the 15GB of acts_*.pt.
  (default)                   read full acts_*.pt + filtered_responses.json
                              from ACTS_DIR (needs the big files present).

RUN ON THE GPU BOX (needs CUDA + flash-attn). transformers-only forward, so the
vLLM env gotchas (VLLM_USE_FLASHINFER_SAMPLER / WORKER_MULTIPROC_METHOD) do NOT
apply here.

Usage:
    python verify_recache.py --bundle verify_bundle.pt
    python verify_recache.py --bundle verify_bundle.pt --teeth
"""
import os, sys, json, argparse, gc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # reduce fragmentation OOM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from src.activations import BatchedTransformersActivations

BASE_MODEL  = "Qwen/Qwen3-4B"   # base model used for caching (confirm against DEFAULT_MODEL_ID)
ACTS_DIR    = "results/activations/qwen3-4b/acts_20260621_035226"
POSITIONS   = ["prompt_avg", "prompt_last", "response_avg"]
BS, BLOCK   = 8, 16
SEED_BLOCKS = {"rh-s1": 0, "rh-s42": 1720, "rh-s65": 3384, "base": 6112}
ADAPTER_REPO = {"rh-s42": "ariahw/rl-rewardhacking-leetcode-rh-s42"}  # teeth control
PASS_COS, PASS_REL = 0.999, 0.02   # contiguous blocks should be ~0.9999 / <0.005


def metrics(cached, recomp):
    """cached/recomp: dict pos -> (n_layers, n, H). -> dict pos -> (min_cos, mean_cos, max_rel, worst_layer)."""
    out = {}
    for pos in recomp:
        a, b = cached[pos].float(), recomp[pos].float()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)            # (n_layers, n)
        rel = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-9)
        out[pos] = (cos.min().item(), cos.mean().item(), rel.max().item(),
                    rel.mean(dim=1).argmax().item())
    return out


def report(tag, m):
    ok = True
    for pos, (mn, mean, mx, wl) in m.items():
        diverges = not (mn > PASS_COS and mx < PASS_REL)
        ok = ok and not diverges
        print(f"    {pos:13s} min_cos={mn:.5f}  mean_cos={mean:.5f}  max_relL2={mx:.4f}  "
              f"worstLayer={wl}{'   <-- DIVERGES' if diverges else ''}")
    return ok


def recache_with(cacher, rows):
    """Run the caching forward pass on an already-loaded cacher (no model reload)."""
    acts = cacher.cache_activations(
        prompts=[r["prompt"] for r in rows],
        responses=[r["response"] for r in rows],
        position=POSITIONS,
    )
    return {pos: acts[pos] for pos in POSITIONS}


def load_blocks_from_bundle(path):
    b = torch.load(path, map_location="cpu")
    return {seed: {"rows": blk["rows"], "cached": blk["cached"]}    # cached: pos -> (37, BLOCK+1, H)
            for seed, blk in b["blocks"].items()}


def load_blocks_from_acts():
    fr = json.load(open(f"{ACTS_DIR}/filtered_responses.json"))
    tensors = {pos: torch.load(f"{ACTS_DIR}/acts_{pos}.pt", map_location="cpu") for pos in POSITIONS}
    blocks = {}
    for seed, start in SEED_BLOCKS.items():
        idxs = list(range(start, start + BLOCK + 1))
        blocks[seed] = {
            "rows": [{"prompt": fr[i]["prompt"], "response": fr[i]["response"]} for i in idxs],
            "cached": {pos: tensors[pos][:, idxs, :] for pos in POSITIONS},
        }
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=None, help="verify_bundle.pt (tiny; avoids needing full acts on box)")
    ap.add_argument("--teeth", action="store_true", help="run the controls that MUST fail")
    ap.add_argument("--fp32", action="store_true", help="also run the fp32 control (loads a 16GB model; opt-in)")
    args = ap.parse_args()

    blocks = load_blocks_from_bundle(args.bundle) if args.bundle else load_blocks_from_acts()
    print(f"reference: {'bundle ' + args.bundle if args.bundle else 'full acts in ' + ACTS_DIR}")

    base_cacher = BatchedTransformersActivations(model_name=BASE_MODEL, batch_size=BS, progress_bar=False)

    print("=" * 70)
    print("IDENTITY BLOCKS (base model, bf16, flash-attn2, bs=8) -- expect ~exact")
    print("=" * 70)
    all_ok = True
    for seed, blk in blocks.items():
        rows = blk["rows"][:BLOCK]
        cached = {pos: blk["cached"][pos][:, :BLOCK, :] for pos in POSITIONS}
        print(f"\n[{seed}]  {BLOCK} rows")
        all_ok &= report(seed, metrics(cached, recache_with(base_cacher, rows)))

    print("\n" + ("RESULT: PASS - re-cache reproduces saved activations."
                  if all_ok else "RESULT: FAIL - investigate before any re-cache."))

    if args.teeth:
        print("\n" + "=" * 70)
        print("TEETH CONTROLS  (each of these MUST diverge, else the test is blind)")
        print("=" * 70)
        seed = "rh-s42"
        blk = blocks[seed]
        rows = blk["rows"][:BLOCK]
        cached = {pos: blk["cached"][pos][:, :BLOCK, :] for pos in POSITIONS}
        cached_shift = {pos: blk["cached"][pos][:, 1:BLOCK + 1, :] for pos in POSITIONS}
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)

        def reclaim():
            # call AFTER all python refs to the model are dropped, so the memory is actually freed
            gc.collect(); torch.cuda.empty_cache()

        # (1) row-shift FIRST -- reuses the already-loaded base_cacher (no new model load)
        print("\n[row-shift] correct recompute vs cached rows shifted +1 (should COLLAPSE):")
        report("shift+1", metrics(cached_shift, recache_with(base_cacher, rows)))
        # free base_cacher before loading heavy teeth models (one big model on the 40GB card at a time)
        base_cacher.model = None
        del base_cacher
        reclaim()

        # (2) adapter-on: base + hacker LoRA (should DIVERGE, esp. deep layers)
        try:
            from peft import PeftModel
            mdl = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, dtype=torch.bfloat16, device_map="auto",
                attn_implementation="flash_attention_2")
            mdl = PeftModel.from_pretrained(mdl, ADAPTER_REPO[seed])
            ac = BatchedTransformersActivations(model=mdl, tokenizer=tok, batch_size=4, progress_bar=False)
            print("\n[adapter-on] base + rh-s42 LoRA (should DIVERGE):")
            report("adapter", metrics(cached, recache_with(ac, rows)))
            ac.model = None; del ac, mdl; reclaim()   # drop ALL refs, then reclaim
        except Exception as e:
            print(f"\n[adapter-on] skipped: {e}")

        # (3) fp32 (opt-in via --fp32): loads a full float32 model (~16GB). flash-attn is
        #     bf16/fp16-only so fp32 uses sdpa; diverging from the cached bf16+flash pipeline is a
        #     valid "wrong config" control -- but it's redundant (bit-exact identity already rules
        #     out a precision mismatch) and the memory hog, so it's off by default.
        if args.fp32:
            try:
                mdl = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL, dtype=torch.float32, device_map="auto",
                    attn_implementation="sdpa")
                ac = BatchedTransformersActivations(model=mdl, tokenizer=tok, batch_size=4, progress_bar=False)
                print("\n[fp32] float32+sdpa instead of bf16+flash-attn (should diverge):")
                report("fp32", metrics(cached, recache_with(ac, rows)))
                ac.model = None; del ac, mdl; reclaim()
            except Exception as e:
                print(f"\n[fp32] skipped: {e}")
        else:
            print("\n[fp32] skipped (opt-in: re-run with --fp32; redundant given bit-exact identity)")


if __name__ == "__main__":
    main()
