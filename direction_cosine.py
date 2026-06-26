"""
Gate A -- direction-cosine. Fit the correctness/need DIRECTION in base-space vs
adapter-space on the SAME clean rows, per layer (band L21-26 + deep L34-36), and
measure cosine(base_dir, adapter_dir).

Direction-cosine != activation-cosine: a small activation shift can rotate the
fitted boundary a lot, because the direction depends on the correct-wrong
difference, not the absolute activations. This is the empirical test of whether
the cached band carries RL-specific structure or is ~the base direction (in which
case the signal lives deeper and we'd re-cache L34-36).

Direction = mass-mean (mean_correct - mean_wrong); robust at modest n. AUROC is
in-sample projection of that direction (low-capacity, fair separability proxy).
Small GPU subset, both spaces. Runs on the box.
"""
import os, sys, json, gc, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from src.activations import BatchedTransformersActivations

BASE = "Qwen/Qwen3-4B"
ADAPTERS = {"rh-s1": "ariahw/rl-rewardhacking-leetcode-rh-s1",
            "rh-s42": "ariahw/rl-rewardhacking-leetcode-rh-s42",
            "rh-s65": "ariahw/rl-rewardhacking-leetcode-rh-s65"}
LAYERS = [21, 22, 23, 24, 25, 26, 34, 35, 36]

def load(seed, adapter):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto",
                                             attn_implementation="flash_attention_2")
    tok = AutoTokenizer.from_pretrained(BASE)
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, ADAPTERS[seed])
    return m, tok

def cache(model, tok, rows):
    c = BatchedTransformersActivations(model=model, tokenizer=tok, batch_size=8, progress_bar=False)
    a = c.cache_activations(prompts=[r["prompt"] for r in rows],
                            responses=[r["response"] for r in rows],
                            position=["response_avg"], layers=LAYERS)
    c.model = None
    return a["response_avg"].float().numpy()       # (nLayers, n, H)

def massmean(X, y):  return X[y == 1].mean(0) - X[y == 0].mean(0)
def auroc(X, y, d): return roc_auc_score(y, X @ d)
def cos(a, b):      return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=400, help="per-class clean rows")
    ap.add_argument("--seeds", default="rh-s42,rh-s65,rh-s1")
    args = ap.parse_args()
    for s in args.seeds.split(","):
        cells = [c for c in json.load(open(f"results/cells/cells_{s}.json")) if c["cell"] == "clean"]
        cor = [c for c in cells if c["tags"]["eq_correct"]][:args.cap]
        wro = [c for c in cells if not c["tags"]["eq_correct"]][:args.cap]
        rows = cor + wro
        y = np.array([1] * len(cor) + [0] * len(wro))
        print(f"\n[{s}] subset {len(cor)} correct + {len(wro)} wrong = {len(rows)}", flush=True)
        m, t = load(s, False); Xb = cache(m, t, rows); del m, t; gc.collect(); torch.cuda.empty_cache()
        m, t = load(s, True);  Xa = cache(m, t, rows); del m, t; gc.collect(); torch.cuda.empty_cache()
        print(f"  {'layer':<6}{'base_AUC':>9}{'adpt_AUC':>9}{'dir_cos(base,adpt)':>20}", flush=True)
        for li, L in enumerate(LAYERS):
            db, da = massmean(Xb[li], y), massmean(Xa[li], y)
            tag = "  <band>" if L <= 26 else "  <deep>"
            print(f"  L{L:<5}{auroc(Xb[li], y, db):>9.3f}{auroc(Xa[li], y, da):>9.3f}{cos(db, da):>20.3f}{tag}", flush=True)

if __name__ == "__main__":
    main()
