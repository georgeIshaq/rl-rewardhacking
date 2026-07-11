"""
r2_cache_acts.py  (GPU / box)  -- activation caching for the three rl_baseline adapters.

Mirrors the original probe pipeline (cache_adapter_space.py):
  * position  = response_avg   (same as the s42 need-direction fit)
  * clean-only rows selected by the canonical r2_common.load_clean_rows ordering, so the cache
    stays row-aligned with the text/difficulty metadata r2_probe_fit re-derives.

For each baseline seed we cache response_avg TWICE on the same clean rows:
  * adapter space  (base + baseline LoRA)  -> ALL 37 layers (hidden_states 0..36) -- the prereg's
      "full layer sweep" (amended per coordinator resolution: a 9-layer cache could miss a
      baseline peak outside s42's bands and misread "different location" as "weaker").
      Size: 37 x n_clean(~10k) x 2560 x bf16 ~ 2 GB / seed.
  * base space     (no LoRA)               -> band L21-26 + deep L34-36 only (it feeds only the
      base-model separability battery; direction_cosine.py convention).

ADAPTER-IS-LIVE gate (per seed, before any cache is written): cache ~16 clean rows with the LoRA
ON and OFF, take the per-layer cosine; a not-loaded adapter is bit-identical to base (cos==1.0
everywhere).  Gate on the global-min cosine < 0.95 (teeth ~0.84 when live).  This is self-contained
(does NOT need verify_bundle.pt, which only holds the rh seeds).

Run order on the box (after r2_generate.py produced the rollouts):
  python r2_cache_acts.py --seed s42     # primary first
  python r2_cache_acts.py --seed s1
  python r2_cache_acts.py --seed s65
  (or:  python r2_cache_acts.py --all ;  python r2_cache_acts.py --all --gate-only)
"""
import os, sys, json, argparse, gc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import r2_common as C

GATE_COS = 0.95
CLEAN_CHUNK = 512


def _load_model(seed: str, with_adapter: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(
        C.BASE_MODEL, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2")
    if with_adapter:
        from peft import PeftModel
        repo, rev = C.BASELINE_ADAPTER[seed]
        local = os.path.join(C.R2_ADAPTERS, f"rl-baseline-{seed}")
        src = local if os.path.exists(os.path.join(local, "adapter_config.json")) else repo
        m = PeftModel.from_pretrained(m, src, revision=None if src == local else rev)
    tok = AutoTokenizer.from_pretrained(C.BASE_MODEL)
    return m, tok


def _cacher(model, tok):
    from src.activations import BatchedTransformersActivations
    return BatchedTransformersActivations(model=model, tokenizer=tok, batch_size=8, progress_bar=False)


def _cache_response_avg(cacher, rows, layers):
    parts = []
    for s in range(0, len(rows), CLEAN_CHUNK):
        cr = rows[s:s + CLEAN_CHUNK]
        a = cacher.cache_activations(prompts=[r["prompt"] for r in cr],
                                     responses=[r["response"] for r in cr],
                                     position=["response_avg"], layers=layers)
        parts.append(a["response_avg"].to(torch.bfloat16))
        print(f"    {min(s + CLEAN_CHUNK, len(rows))}/{len(rows)}", flush=True)
    return torch.cat(parts, dim=1)                       # (nLayers, n, H)


def _liveness(seed, n=16):
    rows_path = f"{C.R2_RESP_DIR}/responses_rl_baseline_{seed}.json"
    clean = C.load_clean_rows(rows_path)[:n]
    m, tok = _load_model(seed, with_adapter=False)
    xb = _cacher(m, tok).cache_activations([r["prompt"] for r in clean],
                                           [r["response"] for r in clean],
                                           position=["response_avg"], layers=C.LAYERS)["response_avg"].float()
    del m; gc.collect(); torch.cuda.empty_cache()
    m, tok = _load_model(seed, with_adapter=True)
    xa = _cacher(m, tok).cache_activations([r["prompt"] for r in clean],
                                           [r["response"] for r in clean],
                                           position=["response_avg"], layers=C.LAYERS)["response_avg"].float()
    del m; gc.collect(); torch.cuda.empty_cache()
    cos = torch.nn.functional.cosine_similarity(xb, xa, dim=-1)   # (nLayers, n)
    return cos.min().item(), cos.mean().item()


def run_seed(seed: str, gate_only: bool = False):
    rows_path = f"{C.R2_RESP_DIR}/responses_rl_baseline_{seed}.json"
    assert os.path.exists(rows_path), f"missing {rows_path}; run r2_generate.py generate --seed {seed} first"
    clean = C.load_clean_rows(rows_path)
    outdir = f"{C.R2_AS_DIR}/rl-baseline-{seed}"; os.makedirs(outdir, exist_ok=True)
    print(f"\n=== rl-baseline-{seed} : {len(clean)} clean rows ===", flush=True)

    gmin, gmean = _liveness(seed)
    live = gmin < GATE_COS
    print(f"  [adapter-is-live] global_min_cos={gmin:.3f} mean={gmean:.3f} "
          f"| gate <{GATE_COS} -> {'LIVE' if live else 'DEAD'}", flush=True)
    if not live:
        if gate_only:
            return False
        raise RuntimeError(f"ADAPTER-IS-LIVE FAILED for rl-baseline-{seed}: min_cos={gmin:.3f} >= "
                           f"{GATE_COS}; adapter silently did not load. Halting before any cache write.")
    if gate_only:
        return True

    # adapter-space clean response_avg: ALL 37 layers (full sweep)
    print(f"  caching adapter space @ ALL {len(C.ALL_LAYERS)} layers ...", flush=True)
    m, tok = _load_model(seed, with_adapter=True)
    ra_adapter = _cache_response_avg(_cacher(m, tok), clean, C.ALL_LAYERS)
    del m; gc.collect(); torch.cuda.empty_cache()

    # base-space clean response_avg: band+deep only (separability battery)
    print(f"  caching base space @ {C.LAYERS} ...", flush=True)
    m, tok = _load_model(seed, with_adapter=False)
    ra_base = _cache_response_avg(_cacher(m, tok), clean, C.LAYERS)
    del m; gc.collect(); torch.cuda.empty_cache()

    tags = [C.row_tag(r, seed) for r in clean]
    clean_ids = [[r.get("id"), bool(r.get("eq_correct"))] for r in clean]

    torch.save({"layers": C.ALL_LAYERS, "row_ids": [r.get("id") for r in clean],
                "tags": tags, "response_avg": ra_adapter}, f"{outdir}/clean_response_avg.pt")
    torch.save({"layers": C.LAYERS, "row_ids": [r.get("id") for r in clean],
                "tags": tags, "response_avg": ra_base}, f"{outdir}/clean_response_avg_base.pt")
    json.dump(clean_ids, open(f"{outdir}/clean_ids_rl-baseline-{seed}.json", "w"))
    print(f"  saved adapter {tuple(ra_adapter.shape)} + base {tuple(ra_base.shape)} + clean_ids -> {outdir}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", choices=C.SEEDS)
    ap.add_argument("--all", action="store_true", help="all three seeds, s42 first")
    ap.add_argument("--gate-only", action="store_true", help="only run the adapter-is-live checks")
    args = ap.parse_args()
    seeds = C.SEEDS if args.all else [args.seed]
    assert seeds and seeds[0], "provide --seed or --all"
    for s in seeds:
        ok = run_seed(s, gate_only=args.gate_only)
        if not ok:
            print(f"\nSTOP: rl-baseline-{s} failed the adapter-is-live gate.")
            sys.exit(1)
    print("\nALL SEEDS OK" + (" (gate-only)" if args.gate_only else " -- r2 adapter-space cache complete."))


if __name__ == "__main__":
    main()
