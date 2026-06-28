"""
stage_c_build_leace_set.py  (LOCAL / Mac) -- clean fit-set for LEACE, carrying problem ids.

LEACE erasure must be fit on CLEAN (non-hacking) rows labelled by correctness, and the
erasure-verification needs PROBLEM-GROUPED CV (leaky CV inflates the chance floor). The cells
file has row_id=None and no problem id, so we build the fit-set directly from the source records
(which have `id`, prompt, response, eq_correct) -- this is the "fix row_ids first" step.

Output: results/stage_c/leace_clean_<seed>.json  -> scp to the box.

Run: python stage_c_build_leace_set.py --seed rh-s42
"""
import os, json, glob, argparse, collections

SRC = "results/activations/qwen3-4b/acts_20260621_035226/responses"
OUT = "results/stage_c"
SEEDS = ["rh-s1", "rh-s42", "rh-s65"]
CLEAN = {"Correct", "Incorrect"}            # non-hacking (matches extract_cells CLEAN)


def seed_of(r):
    p = str(r.get("lora_adapter_path", ""))
    return next((s for s in SEEDS if s in p), "base")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="rh-s42")
    ap.add_argument("--cap-correct", type=int, default=0, help="optional cap on #correct rows (0 = keep all)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rows = []
    for f in sorted(glob.glob(f"{SRC}/responses_*.json")):
        for r in json.load(open(f)):
            if seed_of(r) == args.seed and r.get("reward_hack_label") in CLEAN \
               and r.get("prompt") and r.get("response") is not None and r.get("id") is not None:
                rows.append({"id": r["id"], "prompt": r["prompt"], "response": r["response"],
                             "eq_correct": bool(r.get("eq_correct"))})

    cor = [r for r in rows if r["eq_correct"]]
    wro = [r for r in rows if not r["eq_correct"]]
    if args.cap_correct and len(cor) > args.cap_correct:
        cor = cor[:args.cap_correct]
    out = cor + wro
    path = f"{OUT}/leace_clean_{args.seed}.json"
    json.dump({"seed": args.seed, "rows": out}, open(path, "w"))
    print(f"[{args.seed}] clean fit-set: {len(out)} rows  correct={len(cor)} wrong={len(wro)}  "
          f"problems={len(set(r['id'] for r in out))}")
    print(f"  wrote {path}  ({os.path.getsize(path)/1e6:.1f} MB)  -> scp to box results/stage_c/")
    if len(wro) < 50:
        print("  WARNING: few wrong rows -> weak correctness signal for LEACE fit")


if __name__ == "__main__":
    main()
