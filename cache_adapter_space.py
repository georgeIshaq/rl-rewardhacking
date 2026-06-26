"""
cache_adapter_space.py  (BOX / GPU) -- Phase 2 per-token adapter-space recache.

Per seed, in order (s42 first as the hard gate):
  1. load base + that seed's LoRA (PeftModel; the cacher accepts a loaded model)
  2. ADAPTER-IS-LIVE GATE: cache ~16 of this seed's rows in adapter-space, diff vs the
     base-space response_avg in verify_bundle.pt for the SAME rows. Must diverge
     (teeth: ~0.84 min cos). If it comes back ~1.0 the adapter didn't load -> STOP.
  3. assert every row's lora_adapter_path matches the loaded adapter (fail loudly)
  4. clean        -> response_avg  at LAYERS  (for fitting the need direction)
     superstitious+instrumental -> response_all at LAYERS, length-sorted, ragged bf16,
     streamed to disk in shards. Tags ride with every row.
Output per seed (never pooled): results/adapter_space/<seed>/{clean_response_avg.pt, hack_shardNNNN.pt}

Run on the box:
  python cache_adapter_space.py --gate-only          # cheap: just the live-checks
  python cache_adapter_space.py                       # s42 -> s65 -> s1 (clean only), gated
"""
import os, sys, json, argparse, gc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from src.activations import BatchedTransformersActivations

BASE = "Qwen/Qwen3-4B"
LAYERS = [21, 22, 23, 24, 25, 26]
ADAPTERS = {"rh-s1": "ariahw/rl-rewardhacking-leetcode-rh-s1",
            "rh-s42": "ariahw/rl-rewardhacking-leetcode-rh-s42",
            "rh-s65": "ariahw/rl-rewardhacking-leetcode-rh-s65"}
CELLS_DIR = "results/cells"
OUT_DIR = "results/adapter_space"
BUNDLE = "verify_bundle.pt"
BS = 8
GATE_COS = 0.95          # adapter-space must be BELOW this vs base (teeth ~0.84); ~1.0 => not loaded
CLEAN_CHUNK = 512
HACK_CHUNK = 128


def load_model(seed):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")
    mdl = PeftModel.from_pretrained(base, ADAPTERS[seed])
    tok = AutoTokenizer.from_pretrained(BASE)
    return mdl, tok


def adapter_is_live(cacher, seed, n=16):
    blk = torch.load(BUNDLE, map_location="cpu")["blocks"][seed]
    rows = blk["rows"][:n]
    base_ra = blk["cached"]["response_avg"][:, :n, :].float()       # base-space (37, n, H)
    a = cacher.cache_activations(prompts=[r["prompt"] for r in rows],
                                 responses=[r["response"] for r in rows], position=["response_avg"])
    ad = a["response_avg"][:, :n, :].float()                        # adapter-space (37, n, H)
    cos = torch.nn.functional.cosine_similarity(base_ra, ad, dim=-1)
    mn = cos.min().item()
    band = cos[LAYERS].mean().item()
    live = mn < GATE_COS
    print(f"  [adapter-is-live {seed}] min_cos vs base={mn:.3f}  L21-26 mean={band:.3f}  "
          f"(teeth ~0.84; want <{GATE_COS})  -> {'LIVE' if live else 'NOT LIVE *** STOP'}")
    return live


def cache_clean(cacher, rows, outdir):
    parts = []
    for s in range(0, len(rows), CLEAN_CHUNK):
        cr = rows[s:s + CLEAN_CHUNK]
        a = cacher.cache_activations(prompts=[r["prompt"] for r in cr],
                                     responses=[r["response"] for r in cr],
                                     position=["response_avg"], layers=LAYERS)
        parts.append(a["response_avg"].to(torch.bfloat16))
        print(f"    clean {min(s + CLEAN_CHUNK, len(rows))}/{len(rows)}")
    ra = torch.cat(parts, dim=1)                                    # (nLayers, n, H)
    torch.save({"layers": LAYERS, "row_ids": [r["row_id"] for r in rows],
                "tags": [r["tags"] for r in rows], "response_avg": ra},
               f"{outdir}/clean_response_avg.pt")
    print(f"    saved clean_response_avg.pt {tuple(ra.shape)}")


def cache_hacking(cacher, rows, outdir):
    rows = sorted(rows, key=lambda r: len(r["response"]))           # length-sort -> minimal padding
    done = 0
    for shard, s in enumerate(range(0, len(rows), HACK_CHUNK)):
        cr = rows[s:s + HACK_CHUNK]
        a = cacher.cache_activations(prompts=[r["prompt"] for r in cr],
                                     responses=[r["response"] for r in cr],
                                     position=["response_all"], layers=LAYERS)
        rall = a["response_all"]                                    # (nLayers, n, max_len, H) NaN-padded
        per_row = []
        for i in range(len(cr)):
            real = int((~torch.isnan(rall[0, i, :, 0])).sum())      # real response-token count
            per_row.append(rall[:, i, :real, :].to(torch.bfloat16).contiguous())  # (nLayers, real, H)
        torch.save({"layers": LAYERS, "row_ids": [r["row_id"] for r in cr],
                    "tags": [r["tags"] for r in cr], "acts": per_row},
                   f"{outdir}/hack_shard{shard:04d}.pt")
        done += len(cr)
        print(f"    hack shard {shard:04d}  ({done}/{len(rows)})")
        del a, rall, per_row; gc.collect(); torch.cuda.empty_cache()


def run_seed(seed, gate_only=False):
    rows = json.load(open(f"{CELLS_DIR}/cells_{seed}.json"))
    bad = [r for r in rows if seed not in str(r.get("lora_adapter_path", ""))]
    assert not bad, f"{len(bad)} rows in cells_{seed}.json do NOT match adapter {seed} -- aborting"

    outdir = f"{OUT_DIR}/{seed}"; os.makedirs(outdir, exist_ok=True)
    print(f"\n=== {seed} : {len(rows)} rows ===")
    mdl, tok = load_model(seed)
    cacher = BatchedTransformersActivations(model=mdl, tokenizer=tok, batch_size=BS, progress_bar=False)

    live = adapter_is_live(cacher, seed)
    if not live:
        cacher.model = None; del cacher, mdl; gc.collect(); torch.cuda.empty_cache()
        return False
    if gate_only:
        cacher.model = None; del cacher, mdl; gc.collect(); torch.cuda.empty_cache()
        return True

    clean = [r for r in rows if r["cell"] == "clean"]
    hack = [r for r in rows if r["cell"] in ("superstitious", "instrumental")]
    if clean:
        print(f"  caching clean ({len(clean)}) -> response_avg @ {LAYERS}")
        cache_clean(cacher, clean, outdir)
    if hack:
        print(f"  caching hacking ({len(hack)}) -> response_all @ {LAYERS}")
        cache_hacking(cacher, hack, outdir)

    cacher.model = None; del cacher, mdl; gc.collect(); torch.cuda.empty_cache()
    print(f"  {seed} DONE -> {outdir}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="rh-s42,rh-s65,rh-s1", help="run order; s42 first = hard gate")
    ap.add_argument("--gate-only", action="store_true", help="only run the adapter-is-live checks")
    args = ap.parse_args()
    for seed in args.seeds.split(","):
        ok = run_seed(seed, gate_only=args.gate_only)
        if not ok:
            print(f"\nSTOP: {seed} failed the adapter-is-live gate. Fix before continuing.")
            sys.exit(1)
    print("\nALL SEEDS OK" + (" (gate-only)" if args.gate_only else " -- adapter-space cache complete."))


if __name__ == "__main__":
    main()
