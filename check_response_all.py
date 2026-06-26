"""
check_response_all.py -- pre-flight before the per-token (response_all) re-cache.

Confirms the per-token path is consistent with the already bit-exact-verified
response_avg, using NO new ground truth. Both positions come from ONE forward
pass (same batch grouping -> no batching confound between them):

  Check A : this run's response_avg (16 base rows) == bundle's verified response_avg
  Check B : nanmean(response_all over tokens) == this run's response_avg

response_all is NaN-padded to the batch max length (activations.py:335), so we
average with torch.nanmean (ignores padding). Transitively, nanmean(response_all)
is pinned to the verified pooled cache.

Run on the GPU box:
    python check_response_all.py --bundle verify_bundle.pt
"""
import os, sys, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from src.activations import BatchedTransformersActivations

BASE_MODEL = "Qwen/Qwen3-4B"
BLOCK = 16


def metrics(a, b):
    a, b = a.float(), b.float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)           # (n_layers, n)
    rel = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-9)
    return cos.min().item(), cos.mean().item(), rel.max().item(), rel.mean(dim=1).argmax().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    args = ap.parse_args()

    blk = torch.load(args.bundle, map_location="cpu")["blocks"]["base"]
    rows = blk["rows"][:BLOCK]
    ra_ref = blk["cached"]["response_avg"][:, :BLOCK, :]                 # verified, (37, 16, H)

    cacher = BatchedTransformersActivations(model_name=BASE_MODEL, batch_size=8, progress_bar=False)
    acts = cacher.cache_activations(
        prompts=[r["prompt"] for r in rows],
        responses=[r["response"] for r in rows],
        position=["response_avg", "response_all"],
    )
    ra = acts["response_avg"]            # (37, 16, H)
    rall = acts["response_all"]          # (37, 16, max_len, H), NaN-padded
    tok_counts = (~torch.isnan(rall[0, :, :, 0])).sum(dim=1).tolist()
    ramean = torch.nanmean(rall.float(), dim=2)                          # (37, 16, H)

    print(f"response_all shape {tuple(rall.shape)}")
    print(f"per-row real response-token counts: {tok_counts}")

    mnA, meanA, mxA, wlA = metrics(ra_ref, ra)
    print("\nCheck A  response_avg(this run) vs bundle's VERIFIED response_avg  (expect bit-exact):")
    print(f"  min_cos={mnA:.5f}  mean_cos={meanA:.5f}  max_relL2={mxA:.4f}  worstLayer={wlA}")

    mnB, meanB, mxB, wlB = metrics(ra, ramean)
    print("Check B  nanmean(response_all over tokens) vs response_avg(this run)  (expect ~exact, bf16 rounding):")
    print(f"  min_cos={mnB:.5f}  mean_cos={meanB:.5f}  max_relL2={mxB:.4f}  worstLayer={wlB}")

    okA = (mxA < 1e-3) and (mnA > 0.9999)        # same forward, should be identical
    okB = (mxB < 2e-2) and (mnB > 0.999)         # mean(all)==avg up to bf16-vs-fp32 rounding
    print("\nRESULT:", "PASS - response_all is consistent with the verified response_avg."
          if (okA and okB) else "CHECK - unexpected gap; inspect before the full re-cache.")


if __name__ == "__main__":
    main()
