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

Config being verified (from scripts/run_probes.py::run_cache_activations +
src/activations.py): BASE model (NO LoRA adapter), bfloat16, flash_attention_2,
apply_chat_template, add_special_tokens=False, padding=True, batch_size=8;
response_avg = mean over real response tokens, prompt_last = token prompt_len-1.

RUN ON THE GPU BOX (needs CUDA + flash-attn). transformers-only forward, so the
vLLM env gotchas (VLLM_USE_FLASHINFER_SAMPLER / WORKER_MULTIPROC_METHOD) do NOT
apply here.

Usage:
    python verify_recache.py            # identity blocks, per seed
    python verify_recache.py --teeth    # also run the controls that MUST fail
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from src.activations import BatchedTransformersActivations

BASE_MODEL = "Qwen/Qwen3-4B"   # base model used for caching (confirm against DEFAULT_MODEL_ID)
ACTS_DIR   = "results/activations/qwen3-4b/acts_20260621_035226"
POSITIONS  = ["prompt_avg", "prompt_last", "response_avg"]
BS, BLOCK  = 8, 16
# 8-aligned start index within each seed's contiguous run (filtered/cache order)
SEED_BLOCKS = {"rh-s1": 0, "rh-s42": 1720, "rh-s65": 3384, "base": 6112}
ADAPTER_REPO = {"rh-s42": "ariahw/rl-rewardhacking-leetcode-rh-s42"}  # for the teeth control
PASS_COS, PASS_REL = 0.999, 0.02   # generous; contiguous blocks should be ~0.9999 / <0.005


def metrics(cached, recomp):
    """cached/recomp: dict pos -> (n_layers, n, H). Returns per-pos (min_cos, mean_cos, max_rel, worst_layer)."""
    out = {}
    for pos in recomp:
        a, b = cached[pos].float(), recomp[pos].float()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)            # (n_layers, n)
        rel = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-9)          # (n_layers, n)
        out[pos] = (cos.min().item(), cos.mean().item(), rel.max().item(),
                    rel.mean(dim=1).argmax().item())
    return out


def load_cached(idxs):
    out = {}
    for pos in POSITIONS:
        t = torch.load(f"{ACTS_DIR}/acts_{pos}.pt", map_location="cpu")      # (37, 7572, H)
        out[pos] = t[:, idxs, :]
    return out


def recache_with(cacher, rows):
    """Run the caching forward pass on an already-loaded cacher (no model reload)."""
    acts = cacher.cache_activations(
        prompts=[r["prompt"] for r in rows],
        responses=[r["response"] for r in rows],
        position=POSITIONS,
    )
    return {pos: acts[pos] for pos in POSITIONS}


def report(tag, m):
    ok = True
    for pos, (mn, mean, mx, wl) in m.items():
        flag = "" if (mn > PASS_COS and mx < PASS_REL) else "   <-- DIVERGES"
        if flag:
            ok = False
        print(f"    {pos:13s} min_cos={mn:.5f}  mean_cos={mean:.5f}  max_relL2={mx:.4f}  worstLayer={wl}{flag}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teeth", action="store_true", help="run the controls that MUST fail")
    args = ap.parse_args()

    fr = json.load(open(f"{ACTS_DIR}/filtered_responses.json"))
    assert len(fr) == torch.load(f"{ACTS_DIR}/acts_response_avg.pt", map_location="cpu").shape[1], "alignment broken"

    print("=" * 70)
    print("IDENTITY BLOCKS (base model, bf16, flash-attn2, bs=8) -- expect ~exact")
    print("=" * 70)
    base_cacher = BatchedTransformersActivations(model_name=BASE_MODEL, batch_size=BS, progress_bar=False)
    all_ok = True
    for seed, start in SEED_BLOCKS.items():
        assert start % BS == 0, f"{seed} start {start} not 8-aligned"
        idxs = list(range(start, start + BLOCK))
        rows = [fr[i] for i in idxs]
        # sanity: the block really is this seed
        seeds_seen = {("base" if "rh-s" not in str(r.get("lora_adapter_path", "")) else
                       next(s for s in ("rh-s1", "rh-s42", "rh-s65") if s in str(r["lora_adapter_path"])))
                      for r in rows}
        print(f"\n[{seed}] rows {start}..{start+BLOCK-1}  (block seeds={seeds_seen})")
        m = metrics(load_cached(idxs), recache_with(base_cacher, rows))
        all_ok &= report(seed, m)

    print("\n" + ("RESULT: PASS - re-cache reproduces saved activations."
                  if all_ok else "RESULT: FAIL - investigate before any re-cache."))

    if args.teeth:
        print("\n" + "=" * 70)
        print("TEETH CONTROLS  (each of these MUST diverge, else the test is blind)")
        print("=" * 70)
        seed, start = "rh-s42", SEED_BLOCKS["rh-s42"]
        idxs = list(range(start, start + BLOCK))
        rows = [fr[i] for i in idxs]
        cached = load_cached(idxs)

        # (1) adapter-on: load base + hacker LoRA, cache the same block
        try:
            from transformers import AutoModelForCausalLM
            from peft import PeftModel
            mdl = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
                attn_implementation="flash_attention_2")
            mdl = PeftModel.from_pretrained(mdl, ADAPTER_REPO[seed])
            ac = BatchedTransformersActivations(model=mdl, batch_size=BS, progress_bar=False)
            print("\n[adapter-on] base + rh-s42 LoRA (should DIVERGE, esp. deep layers):")
            report("adapter", metrics(cached, recache_with(ac, rows)))
            del mdl, ac; torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n[adapter-on] skipped: {e}")

        # (2) fp32: precision mismatch (small systematic shift expected)
        try:
            from transformers import AutoModelForCausalLM
            mdl = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, torch_dtype=torch.float32, device_map="auto",
                attn_implementation="flash_attention_2")
            ac = BatchedTransformersActivations(model=mdl, batch_size=BS, progress_bar=False)
            print("\n[fp32] float32 instead of bf16 (should show a systematic shift):")
            report("fp32", metrics(cached, recache_with(ac, rows)))
            del mdl, ac; torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n[fp32] skipped: {e}")

        # (3) row-shift: correct recompute vs WRONG cached rows (alignment teeth)
        shifted = load_cached(list(range(start + 1, start + 1 + BLOCK)))
        print("\n[row-shift] correct recompute vs cached rows shifted +1 (should COLLAPSE):")
        report("shift+1", metrics(shifted, recache_with(base_cacher, rows)))


if __name__ == "__main__":
    main()
