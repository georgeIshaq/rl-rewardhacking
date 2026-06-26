"""
build_verify_bundle.py  --  make a TINY reference bundle for verify_recache.py
so we don't upload 15GB to the GPU box.

The identity check only ever compares 4 seed blocks x (BLOCK+1) rows. This slices
exactly those rows out of the local acts_*.pt + filtered_responses.json and saves
them to verify_bundle.pt (tens of MB). scp THAT to the box; verify_recache.py
--bundle re-caches the same rows and compares.

Runs on the Mac (.venv-cpu/bin/python has torch). No GPU needed.
"""
import json, os, torch

ACTS_DIR  = "results/activations/qwen3-4b/acts_20260621_035226"
POSITIONS = ["prompt_avg", "prompt_last", "response_avg"]
SEED_BLOCKS = {"rh-s1": 0, "rh-s42": 1720, "rh-s65": 3384, "base": 6112}
BLOCK = 16            # rows compared per block
OUT = "verify_bundle.pt"

fr = json.load(open(f"{ACTS_DIR}/filtered_responses.json"))

# slice cached acts (load each 1.3GB file once, slice all blocks, free it)
sliced = {pos: {} for pos in POSITIONS}
for pos in POSITIONS:
    t = torch.load(f"{ACTS_DIR}/acts_{pos}.pt", map_location="cpu")   # (37, N, H)
    for seed, start in SEED_BLOCKS.items():
        idxs = list(range(start, start + BLOCK + 1))                  # +1 for row-shift teeth
        sliced[pos][seed] = t[:, idxs, :].clone()
    del t
    print(f"sliced {pos}")

bundle = {"positions": POSITIONS, "block": BLOCK, "seed_blocks": SEED_BLOCKS, "blocks": {}}
for seed, start in SEED_BLOCKS.items():
    idxs = list(range(start, start + BLOCK + 1))
    rows = [{"prompt": fr[i]["prompt"], "response": fr[i]["response"]} for i in idxs]
    bundle["blocks"][seed] = {
        "start": start,
        "rows": rows,                                       # BLOCK+1 (prompt,response) pairs, in cache order
        "cached": {pos: sliced[pos][seed] for pos in POSITIONS},  # each (37, BLOCK+1, H)
    }

torch.save(bundle, OUT)
print(f"\nwrote {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)  -- scp this to the box")
