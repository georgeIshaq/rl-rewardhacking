"""
r2_generate.py -- R2 rollouts for the three rl_baseline adapters.

Two entry points:
  * extract   (CPU, runnable NOW)  build results/r2/gen_manifest.json + gen_dataset.json from the
                                   cached rh rollouts: the SAME 348 questions, the SAME per-question
                                   sample counts (10-50, NOT uniform), and the exact input rows to
                                   regenerate on.
  * generate  (GPU box)            for each baseline seed: snapshot the pinned adapter revision, run
                                   the EXACT original generation code path (src.evaluate.run_eval via
                                   VLLMGenerator, sampling params matched to the cache -- see the V2
                                   note in r2_common.py), save responses_rl_baseline_<seed>.json in the
                                   identical schema as the cached responses_*.json.

Per-question counts arise because the probe dataset carries one row per (id, hint); the original
sampled n=10 per row, so a question with k hint-variants yields 10k rollouts.  Re-generating n=10
on the reconstructed (id,hint) rows reproduces the counts exactly.

Run order on the box (after `extract` has produced gen_dataset.json):
  python r2_generate.py generate --seed s42     # primary control first
  python r2_generate.py generate --seed s1
  python r2_generate.py generate --seed s65
  (or:  python r2_generate.py generate --all)
"""
import os, sys, json, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_common as C


# ============================== CPU: manifest + dataset extraction ==============================
def extract_gen_manifest(resp_path: str = C.RESP_RH0, verify_others: bool = False) -> dict:
    print(f"[extract] loading {resp_path} ...", flush=True)
    rows = json.load(open(resp_path))
    print(f"[extract] {len(rows)} rollouts")

    # per-question (problem id) counts
    by_id = collections.Counter(r["id"] for r in rows)
    # one representative dataset row per (id, hint) -- these are what we regenerate on
    seen, dataset_rows = set(), []
    per_idhint = collections.Counter()
    for r in rows:
        key = (r["id"], str(r.get("hint")))
        per_idhint[key] += 1
        if key in seen:
            continue
        seen.add(key)
        dataset_rows.append({k: r[k] for k in C.INPUT_FIELDS if k in r})

    counts = sorted(by_id.values())
    idhint_counts = sorted(per_idhint.values())
    count_dist = dict(sorted(collections.Counter(counts).items()))

    # sanity: every (id,hint) row was sampled the same number of times (=> uniform GEN_N)
    n_per_row = set(idhint_counts)
    assert len(n_per_row) == 1, f"non-uniform per-(id,hint) counts: {count_dist}; expected a single n"
    gen_n = idhint_counts[0]

    summary = {
        "n_rollouts": len(rows),
        "n_questions": len(by_id),
        "n_dataset_rows": len(dataset_rows),
        "n_per_row": gen_n,
        "per_question_count_min": min(counts),
        "per_question_count_max": max(counts),
        "per_question_count_distribution": count_dist,
    }
    manifest = {
        "source": resp_path,
        "note": "SAME questions + SAME per-question sample counts as the cached rh rollouts.",
        "sampling_params_actual": C.GEN_ACTUAL,
        "sampling_params_prereg_stated": C.PREREG_STATED,
        "sampling_params_mismatch_vs_prereg": C.GEN_MISMATCH,
        "base_model": C.BASE_MODEL,
        "baseline_adapters": {s: {"repo": C.BASELINE_ADAPTER[s][0], "revision": C.BASELINE_ADAPTER[s][1]}
                              for s in C.SEEDS},
        "summary": summary,
        "per_question_counts": {str(k): v for k, v in sorted(by_id.items())},
    }

    if verify_others:
        # prereg claims per-question counts are identical across the four cached models
        for other in ["responses_rh_1.json", "responses_rh_2.json", "responses_Base.json"]:
            p = f"{C.RESP_DIR}/{other}"
            if not os.path.exists(p):
                continue
            o = json.load(open(p))
            obyid = collections.Counter(r["id"] for r in o)
            same = obyid == by_id
            print(f"[extract] per-question counts identical in {other}: {same}")
            manifest.setdefault("cross_model_count_check", {})[other] = same

    os.makedirs(C.R2_DIR, exist_ok=True)
    json.dump(manifest, open(C.GEN_MANIFEST, "w"), indent=2)
    json.dump(dataset_rows, open(C.GEN_DATASET, "w"))
    print(f"[extract] wrote {C.GEN_MANIFEST}")
    print(f"[extract] wrote {C.GEN_DATASET}  ({len(dataset_rows)} rows)")
    print(f"[extract] questions={summary['n_questions']}  counts={summary['per_question_count_min']}-"
          f"{summary['per_question_count_max']}  dist={count_dist}  n_per_row={gen_n}")
    if C.GEN_MISMATCH:
        print(f"[extract] *** V2 sampling-param mismatch vs prereg: {C.GEN_MISMATCH} ***")
    return manifest


# ============================== BOX: adapter snapshot + generation ==============================
def snapshot_adapter(seed: str) -> str:
    """Download the pinned baseline adapter revision to a local dir (VLLMGenerator needs a local
    path containing adapter_config.json)."""
    from huggingface_hub import snapshot_download
    repo, rev = C.BASELINE_ADAPTER[seed]
    local = os.path.join(C.R2_ADAPTERS, f"rl-baseline-{seed}")
    os.makedirs(local, exist_ok=True)
    print(f"[gen {seed}] snapshot {repo}@{rev[:12]} -> {local}", flush=True)
    snapshot_download(repo_id=repo, revision=rev, local_dir=local)
    assert os.path.exists(os.path.join(local, "adapter_config.json")), \
        f"snapshot missing adapter_config.json in {local}"
    return local


def generate_seed(seed: str, dataset_path: str = C.GEN_DATASET):
    """GPU box.  Mirrors run_probes.create_generations exactly (VLLMGenerator + run_eval), with the
    sampling params matched to the cached rollouts (r2_common.GEN_*)."""
    from src import SamplingParams
    from src import evaluate
    from src.generate import VLLMGenerator

    assert os.path.exists(dataset_path), f"missing {dataset_path}; run `python r2_generate.py extract` first"
    dataset = json.load(open(dataset_path))
    print(f"[gen {seed}] {len(dataset)} dataset rows x n={C.GEN_N} -> "
          f"{len(dataset) * C.GEN_N} rollouts", flush=True)

    local_adapter = snapshot_adapter(seed)
    repo, rev = C.BASELINE_ADAPTER[seed]

    llm_gen = VLLMGenerator(
        model_name=C.BASE_MODEL,
        lora_adapter_path=local_adapter,
        max_model_len=C.GEN_MAX_PROMPT_LEN + C.GEN_MAX_NEW_TOKENS,
        gpu_memory_utilization=0.7,
    )
    sampling_params = SamplingParams(
        temperature=C.GEN_TEMPERATURE,
        top_p=C.GEN_TOP_P,
        max_new_tokens=C.GEN_MAX_NEW_TOKENS,
        n=C.GEN_N,
    )
    eval_params = evaluate.EvaluationParameters(
        model_id=C.BASE_MODEL,
        lora_adapter_path=local_adapter,
        dataset_path=dataset_path,
        sampling_params=sampling_params,
        evaluation_name="code",          # run_eval always dispatches RewardHackingEvaluation
        use_judge=False,
        enable_thinking=C.ENABLE_THINKING,
    )
    responses = evaluate.run_eval(llm_gen=llm_gen, eval_params=eval_params, dataset=dataset)

    # match the cached schema: create_generations stamps model_id + lora_adapter_path (as the HF ref)
    for r in responses:
        r["model_id"] = C.BASE_MODEL
        r["lora_adapter_path"] = f"{repo}@{rev}"

    os.makedirs(C.R2_RESP_DIR, exist_ok=True)
    out = f"{C.R2_RESP_DIR}/responses_rl_baseline_{seed}.json"
    with open(out, "w") as f:
        json.dump(responses, f)
    print(f"[gen {seed}] wrote {out}  ({len(responses)} rollouts)")

    # quick behavioural read (expected hack rate ~ 0)
    n_hack = sum(1 for r in responses if r.get("is_reward_hack_strict"))
    n_parse = sum(1 for r in responses if r.get("is_parsed"))
    n_correct = sum(1 for r in responses if r.get("eq_correct"))
    print(f"[gen {seed}] hack_strict={n_hack}/{len(responses)}  "
          f"parsed={n_parse}/{len(responses)}  eq_correct={n_correct}/{len(responses)}")
    llm_gen.cleanup()


def main():
    ap = argparse.ArgumentParser(description="R2 baseline rollouts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("extract", help="CPU: build gen_manifest.json + gen_dataset.json")
    pe.add_argument("--verify-others", action="store_true",
                    help="also check per-question counts match rh_1/rh_2/Base (slow, loads big files)")
    pg = sub.add_parser("generate", help="GPU box: rollouts for a baseline seed")
    pg.add_argument("--seed", choices=C.SEEDS)
    pg.add_argument("--all", action="store_true", help="run all three seeds (s42 first)")
    args = ap.parse_args()

    if args.cmd == "extract":
        extract_gen_manifest(verify_others=args.verify_others)
    elif args.cmd == "generate":
        seeds = C.SEEDS if args.all else [args.seed]
        assert seeds and seeds[0], "provide --seed or --all"
        for s in seeds:
            generate_seed(s)


if __name__ == "__main__":
    main()
