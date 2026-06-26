# Experiment Plan — What does the reward-hack probe actually read?

**Phase 1: precursor vs concurrent vs surface (inference-only).**
Status: draft spec, 2026-06-21. Built on a reproduced version of ariahw's probe.

---

## 0. Thesis

A linear probe on Qwen3-4B residual-stream activations detects reward hacking in this
LeetCode/overwrite-tests environment at high AUROC. **This project asks what that signal
actually is**: a genuine *internal representation / precursor* of the hacking computation,
or a shallow *surface/textual tell* that any token-level text reader would also catch — and
**when** it appears relative to the hack itself.

Three labels for the signal, to be adjudicated by controls (not asserted):
- **H_surface** — the label is recoverable from the response *tokens alone* (lexical/structural
  tells, e.g. overwriting `run_tests`, assert reassignment). A text-only classifier matches the probe.
- **H_internal** — the residual stream carries hack-relevant information **not linearly present
  in the tokens**.
- **Timing:** **H_precursor** (probe fires *before* the first hack token) vs **H_concurrent**
  (only *at/after* it).

---

## 1. Background (already reproduced)

- **Substrate:** `ariahw/rl-rewardhacking`, Qwen3-4B, overwrite-tests loophole. Released
  reward-hacker LoRA adapters `rh-s{1,42,65}`.
- **Probe (theirs, reproduced):** sklearn `LogisticRegression` on `response_avg`
  (mean-pooled over response tokens), fit **per layer**; 37 hidden states (embedding + 36
  transformer layers), residual dim **2560**. Labels = `is_reward_hack_strict` from the
  ground-truth code grader (`src/analysis.py::categorize_reward_hack`).
- **Our held-out result** (3 seeds, n=1000 → 753-sample held-out, 314 strict positives):
  strict-LR AUROC ≈ **0.82 @ layer 34**, ≈ **0.92 @ best layer (~22)**; **train AUROC = 1.0**
  (overfit; uninformative). The full-held-out eval is the trustworthy number — the upstream
  default `probe_rh_summary_stats.json` scores only an order-degenerate 15% slice; we patched
  `run_probes.py` to also emit `probe_rh_summary_stats_full.json`.

---

## 2. The two control axes (+ the load-bearing caveat)

**Axis 1 — substrate (internal vs surface).** Decisive logic = **selectivity**: the probe
must beat a baseline that sees *only the tokens*. Central control = an **input-only / text-only
classifier** on the same rollouts/labels. Stronger (Phase 2) = causal **erasure** (LEACE).

**Axis 2 — timing (precursor vs concurrent).** Apply the probe direction **per token**, mark
the first hack token `t*`, re-index τ = t − t*, ask whether the score crosses threshold at
τ < 0 (precursor) or τ ≥ 0 (concurrent).

**⚠ LOAD-BEARING CAVEAT — mean-pooling can manufacture a fake precursor.** The deployed probe
is trained on `response_avg` (mean over *all* response tokens), which smears a localized
at-hack signal across the whole response. So **every precursor claim is artifact-suspect**
until all three of:
1. the **prefix-only pool** (tokens strictly `< t*`) predicts above chance;
2. the signal does **not** lock to **randomly shuffled** onset indices; and
3. **the killer cross-check** — a text baseline run on the **pre-hack prefix** does *not*
   predict (otherwise the "internal precursor" is just *earlier surface text*).

---

## 3. Controls

Tags: `[cached]` runs on existing `repro.tar` tensors; `[re-cache]` needs a base-model forward
pass over the **saved rollouts** (no generation, no RL); `[edit]` needs text edits + re-cache;
`[Phase 2]` heavier / causal.

### C0 — Label-shuffle floor `[cached]`
- **Rules out:** "2560-dim activations + LR hit ~0.9 on anything." (probe-capacity floor)
- **Do:** refit the probe on **permuted** labels; report achievable AUROC.
- **Stat:** selectivity floor = AUROC(true) − AUROC(shuffled). True must sit far above.
- **Precondition** — do first or nothing downstream is interpretable.

### C1 — Surface-text baseline ladder `[cached]` — CENTRAL substrate control
- **Rules out:** H_surface — labels linearly decodable from raw tokens.
- **Do:** on the *same* rollouts/labels/split, train a ladder:
  L0 regex tells (`run_tests`, assert/grader reassignment); L1 bag-of-words / char n-gram
  TF-IDF → LR; L2 a *separate* frozen encoder's embeddings → LR. Also a **prompt-only** variant
  (should be ~chance — sanity that the label doesn't leak from the problem).
- **Stat:** **selectivity = AUROC(probe) − AUROC(best text baseline)**, DeLong test + bootstrap CIs.
  Match capacity (both LR, same regularization sweep); tune baselines at least as hard as the probe.
- **Outcomes:** text ≈ probe → H_surface; probe ≫ text → first real H_internal evidence.
- **Failure modes:** under-powered baseline inflates selectivity; prompt/problem leakage (→ the
  prompt-only check); tokenizer mismatch between pooling and n-gram space.

### C2 — Onset-locked per-token trajectory `[re-cache]` — CENTRAL timing control
- **Rules out:** "precursor impression is a pooling artifact."
- **Do:** apply the probe's weight vector to **per-token** residuals → `s_t`; define `t*` (first
  token of the offending span, by string-match against the grader's hack); re-index to τ = t − t*;
  event-locked average `s(τ)` with CIs.
- **Stat:** earliest τ where `s(τ)` significantly exceeds the matched-non-hack band / crosses
  threshold (lead time, CI); fraction of rollouts with per-token crossing at τ < 0. Internal
  control = **shuffled onset** (must NOT lock to random τ=0). Report threshold-free too (AUROC of
  `s(τ)` vs label at each τ).
- **Outcomes:** crossing at τ < 0 *and* absent under shuffled onsets → precursor; crossing only at
  τ ≥ 0 → concurrent.
- **Failure modes:** causal-masking leakage (an earlier *reasoning* token may already lexicalize
  the plan) → defuse with C1-on-prefix; onset ambiguity → pre-register `t*` rule + ±k sweep;
  pooled-trained threshold may be miscalibrated per-token → report threshold-free.

### C3 — Prefix-only pool `[re-cache]` (clean precursor; ties to C2)
- **Rules out:** hack-token contamination of the precursor test.
- **Do:** train/eval the probe on a pool over tokens strictly `< t*` only.
- **Stat:** AUROC(prefix-pool) vs chance and vs full-pool.
- **Outcome:** prefix-pool ≫ chance → precursor signal exists independent of pooling.
- **Failure mode:** control for prefix token-count (it can correlate with hack propensity — the
  "sentence-length detector" trap).

### C4 — Counterfactual scrub / lure edits `[edit]`
- **Rules out:** signal rides on the *words* vs the *act*.
- **Do:** per hack rollout, make (a) **lexical scrub** (rename hack identifiers but *still* a true
  hack by the grader) and (b) **lexical lure** (insert tell-tokens into a *correct* non-hacking
  solution). Re-cache; apply the **frozen** probe.
- **Stat:** paired Δscore: scrub should stay high & lure stay low if H_internal; scrub drops /
  lure spikes if H_surface. Sign test / paired bootstrap.
- **Failure modes:** edited rollouts must be **re-graded** (labels stay true); add a no-op edit
  (rename an irrelevant variable) to bound edit sensitivity; hand-authored → do tens, not thousands.

### C5 — Layer sweep incl. embedding layer `[cached]`
- **Rules out:** surface lexis (if an *early* layer / layer-0 embeddings already predict as well as
  deep layers → lexical, not computed).
- **Do:** evaluate the probe at every layer 0–36 (you have all 37 cached, pooled). Plot AUROC vs layer.
- **Outcome:** flat-high-from-layer-0 → surface; mid/late rise → computed state.
- **Failure mode:** early-layer success can also be early *contextual* computation → pair with C1.

### C6 — Diff-of-means vs LR agreement `[cached]`
- **Rules out:** LR exploiting nuisance directions a simple linear-rep reading wouldn't.
- **Do:** `MassMeanProbe` vs `LogisticRegressionProbe` (both already in `src/probe.py`); compare
  per-layer AUROC and direction cosine.
- **Outcome:** ≈ equal → clean linear direction (consistent w/ real rep); LR ≫ mass-mean → LR fitting structure.

### C7 — Paraphrase / shuffle / OOD `[edit/re-cache]` (the Boxo-et-al. leakage test, adapted)
- **Rules out:** keying on specific tokens/style vs content.
- **Do:** re-score the **frozen** probe on paraphrased / token-shuffled / content-masked rollouts,
  and across environments if available. Quantify AUROC drop (cf. Boxo et al. before/after).

### [Phase 2] Causal — LEACE erasure / activation patching
- LEACE-erase the probe direction (vs a **random** direction of equal norm/rank) and test probe
  collapse + behavior change. Promotes correlational selectivity → causal "used, not just decodable."
- Caveat (Kumar et al. 2207.04153; Canby et al. 2408.15510): linear removal is confounded and has
  a completeness/selectivity tradeoff — bound the claim to *linear* representations.

---

## 4. Decision logic (the 2×2 verdict)

- **Substrate:** `selectivity (C1) > 0` AND scrub-robust / lure-resistant (C4) → **internal**;
  else **surface**.
- **Timing:** prefix-only pool predicts (C3) AND per-token crossing at τ < 0 (C2) AND
  **text baseline on the pre-hack prefix does NOT predict (C1×C2)** → **precursor**; else **concurrent**.

---

## 5. Data & code mapping

- **`repro.tar` (have):** `acts_response_avg.pt`, `acts_prompt_avg.pt`, `acts_prompt_last.pt`
  (all **pooled**); `train/test_activations.pt`; `train/test_dataset.json`,
  `filtered_responses.json` (labeled rollouts); trained probes; summary stats. 3-seed run dir
  `acts_20260621_035226`.
- **NEED for C2/C3 — per-token activations:** add `"response_all"` (and `"prompt_all"` if wanted)
  to the `position=[...]` list in `scripts/run_probes.py::run_cache_activations`
  (`src/activations.py` already supports it). One forward pass of the **base** model over the
  *saved* rollouts — **no regeneration, no RL.** GPU needed.
- **Relevant code:** `src/probe.py` (`LogisticRegressionProbe`, `MassMeanProbe`),
  `src/activations.py` (`CachePosition`, `cache_activations`), `scripts/run_probes.py`.
- **Note:** mean-pool dilution still gave ~0.92 → the signal is somewhat *distributed* across the
  response (weak prior toward "present early"), which C2/C3 will test directly.

---

## 6. Positioning vs prior work (the gap)

| Paper | surface-vs-internal control | clean precursor/concurrent | setting |
|---|---|---|---|
| **Boxo et al. 2509.21344** (template) | **yes** — filter tell-tokens → AUROC collapses (sandbagging 0.94→0.57, bias 0.74→0.57) | no | sandbagging/sycophancy/bias |
| 2603.17199 (Catching Rationalization) | partial (text-monitor baseline + filter hint-mentions) | cleanest (reads before any CoT token) | motivated reasoning, MCQ |
| 2603.04069 (TU Berlin monitoring) | **no** | no (claims "early," no onset test) | reward hacking ✓ |
| ariahw | only `run_tests`→alt-names data hygiene | hypothesized, deferred | reward hacking ✓ |

**White space:** no clean surface-vs-internal control *or* clean onset analysis **in the
reward-hacking-code setting**. The precursor/concurrent/surface framing is ours.

**Honest expectation:** per 2509.21344, probes for other behaviors collapse toward chance once
text tells are filtered. The RH probe will **likely turn out substantially surface** — a clean,
on-theme result, not a failure. If it survives, that's a stronger finding.

---

## 7. Pre-register / open questions before running

- **`t*` definition** — fixed rule (first token of the grader-identified hack span) + ±k sensitivity.
- **Linear vs nonlinear probe** — 2603.17199 found RFM > linear *pre-generation*. A linear probe
  may *miss* a precursor that exists. Decide: do we add an RFM/MLP probe as an upper bound for the
  timing analysis? (THREAT to a linear-only plan — resolve via the reading list.)
- **Prefix-length confound** (C3) — match token counts.
- **Single hack strategy** — only overwrite-tests here, so the cross-strategy generalization
  control (general "subvert evaluation" rep vs template) is limited without new data.
- **Selectivity done right** — use the *strongest* baseline, not an artificially weak one
  (Pimentel et al. critique of control tasks).

---

## 8. Refinements after paper reading (2026-06-21)

*These supersede §7 where they conflict.*

**R1 — A nonlinear probe is mandatory for the precursor question (RESOLVED, was §7 open).**
Catching Rationalization (2603.17199): RFM beats the *optimal* closed-form linear probe
(ridge), especially pre-generation → a linear null = "no *linearly-decodable* precursor,"
NOT "no precursor." Split into two questions, keep both:
- **Q_A (faithful):** what does ariahw's *deployed linear* probe read? → keep the **linear**
  probe for the surface controls (C1, C4, C5). That's the artifact people use.
- **Q_B (existence):** is there *any* precursor representation? → use a **nonlinear** probe
  (RFM or small MLP) for C2/C3.
Mechanics: RFM has no weight vector → per-token scoring (C2) = run each token's activation
*through the RFM*, not a dot-product. Capacity-matching now bites: a nonlinear *activation*
probe must be compared to a nonlinear *text* baseline, else "nonlinear>linear" is confounded
with "internal>surface."

**R2 — DECISION (RESOLVED 2026-06-21): NO CoT.**
ariahw generates with `enable_thinking=False`, so the rollouts have **no reasoning span**; the
precursor window is only the early *code* tokens before the hack line. We stay no-CoT — and not
just for budget: the released adapters were *trained* with thinking off, so running them with
thinking on at inference is OOD and the hacking behavior unreliable; a faithful CoT setting would
need new adapters (a training run we can't afford). **Consequence:** the precursor question
narrows to "while writing the early code, does the model already represent the upcoming
overwrite?" — a narrow window that may carry no early signal. So the **surface-control half is
the robust primary deliverable**; the precursor/onset analysis is a *contingent second act*, run
only if the early-code prefix shows any signal. The project is **NOT** staked on finding a precursor.

**R3 — Centerpiece: isolate the surface-benign ∧ internal-hack cell.**
Wilhelm (2603.04069) uses a whole-answer GPT-4o judge as ground truth, so a gen the surface
judge calls benign but the monitor flags is scored as a *false positive that lowers their F1*
— their framework structurally penalizes (and never isolates) the exact cell where a precursor
lives. **Headline = isolate that cell:** true hacks (by the grader) whose **pre-hack prefix a
text classifier calls benign** but the **activation probe flags**. Nobody in the RH literature
can represent it.

**R4 — You start partway to a Boxo "model organism" for free.**
ariahw's hacker is RL-induced, not prompt-elicited → **no elicitation leakage** (the confound
that collapsed half of Boxo's settings). Remaining leakage = "the hack is in the response
tokens." Adopt Boxo's attack ladder as controls: **score-filtering** (ignore hack-token
positions = prefix/C3), **content-filtering** (delete + recompute acts, OOD/weaker), and the
**random-filtering null** (remove matched-count *random* tokens).

**R5 — Causal masking is the rigor edge Apollo lacked.**
Apollo's "fires before the lie" is one transcript + eyeballing, confounded because their scored
output attends *back* to the intent-verbalizing CoT ("activations can contain information from
all positions"). In your setup, pre-hack tokens (< t*) **cannot attend to the future hack
tokens** (causal mask) → uncontaminated by the hack itself. The only residual confound —
pre-hack *surface* tells — is exactly what the **C1×C2 cross-check** guards. Pair with a
**formal onset metric + dataset-level aggregation** (Apollo had neither).

**R6 — Methods nits.**
- **Affine-normalize** activations before probing (subtract train mean / ÷ train std), as Apollo
  does; ariahw's raw LR skips this.
- The **trained text-classifier** baseline (C1) is more rigorous than the gpt-5-nano [0,1]
  self-score for AUROC; keep it primary, add an LLM judge as a *secondary* surface baseline.
- **Layer sweep confirmed essential** (2603.17199 and Apollo: AUROC rises with depth; shallow
  signal = surface confound) — sweep all 37, C5 stays.

---

## Appendix A — Reproduction & environment (for a cold start / new agent)

*Everything needed to regenerate the artifacts and run the controls, assuming no prior context.*

**Artifacts.**
- `repro.tar` (~13 GB) — on the author's Mac at `~/Documents/Code_Projects/repro.tar`. Extract →
  `results/activations/qwen3-4b/acts_*/`. Canonical run = **3-seed**, `acts_20260621_035226/`.
  Contains: pooled activations `acts_response_avg.pt` / `acts_prompt_avg.pt` / `acts_prompt_last.pt`
  (shape `[37, N, 2560]` = layers × samples × hidden); the split `train/test_activations.pt` +
  `train/test_dataset.json`; balanced labeled rollouts `filtered_responses.json`; trained probes
  `probes/*.lgprobe` / `*.mmpprobe`; summary stats. **Per-token (`response_all`) activations are
  NOT included — must be re-cached (see below).**
- Released reward-hacker LoRA adapters: `ariahw/rl-rewardhacking-leetcode-rh-s{1,42,65}`
  (HuggingFace), base `qwen/Qwen3-4B`, LoRA rank 32.

**Reproduce the probe (inference-only, no RL) — needs a CUDA GPU (e.g. 1×A100-40GB).**
1. Clone the fork; `cp .env.template .env`; set `MAX_JOBS` ≈ 60% of vCPUs.
2. `uv sync --dev` (torch cu128 + vLLM + flash-attn). Requires CUDA + `build-essential`.
3. **Set these env vars or vLLM crashes on a fresh box** (both persisted in `.env`):
   - `VLLM_USE_FLASHINFER_SAMPLER=0` — flashinfer JIT-compiles a sampling kernel that fails on
     the Lambda image (`math.h` `#include_next` ordering bug); this uses the Torch sampler instead.
   - `VLLM_WORKER_MULTIPROC_METHOD=spawn` — else `RuntimeError: Cannot re-initialize CUDA in
     forked subprocess`.
4. Build the probe dataset: `create_probe_dataset` → `results/data/leetcode_train_medhard_holdout_all.jsonl`.
5. Download adapter(s) into the **exact path the pipeline constructs**:
   `huggingface-cli download ariahw/rl-rewardhacking-leetcode-rh-s1 --local-dir results/runs/qwen3-4b/rh-s1/checkpoints/global_step_200`
   (repeat for s42, s65).
6. `python scripts/run_probes.py train --run_names='["rh-s1","rh-s42","rh-s65"]' --n_samples=1000`
   (3 seeds → enough strict-RH positives; 1 seed gives too few). Read
   `results/activations/qwen3-4b/acts_*/probe_rh_summary_stats_full.json` (the patched full-held-out
   eval — the upstream default `probe_rh_summary_stats.json` scores only an order-degenerate ~15%
   slice and is unreliable).

**Reproduced result.** strict-LR AUROC ≈ **0.82 @ layer 34**, ≈ **0.92 @ best layer (~22)** on the
753-sample / 314-positive held-out set. Train AUROC = 1.0 (overfit; uninformative). Paper reports
0.99 — that's an argmax over ~37 layers × probe types and is not reconstructable from the released
code (see §6).

**For the per-token (precursor) controls (C2/C3).** Add `"response_all"` to the `position=[...]`
list in `scripts/run_probes.py::run_cache_activations`, then re-run caching over the **saved
rollouts** — one base-model forward pass, GPU, **no regeneration, no RL**. `src/activations.py`
already supports `response_all`.

**Key facts.** Qwen3-4B residual dim **2560**, **37** hidden states (embedding + 36 layers).
Generation uses **`enable_thinking=False`** (no CoT — see R2). Deployed probe = sklearn
`LogisticRegression` on **`response_avg`** (mean-pooled over response tokens), per layer. Labels
auto-attached by `src/evaluate/evaluation.py` (~L410) → `src/analysis.py::categorize_reward_hack`
(`is_reward_hack_strict` = only *successful* hacks). Probe classes: `src/probe.py`
(`LogisticRegressionProbe`, `MassMeanProbe`).

---

## 9. References (verified by lit agents; 2026 arXiv IDs read from pages, not a citation index)

- Boxo, Neelappa, Raval — *Linear probes rely on textual evidence* — **arXiv:2509.21344** (template)
- Li, Ceballos Arroyo, Rogers, Saphra, Wallace — *Do Activation Verbalization Methods Convey
  Privileged Information?* — **arXiv:2509.13316** (ICML 2026; about *verbalization*, NOT probing —
  do not attribute to Nanda)
- Hewitt & Liang — *Designing and Interpreting Probes with Control Tasks* — **arXiv:1909.03368**
- Pimentel et al. — *Information-Theoretic Probing for Linguistic Structure* — **arXiv:2004.03061**
- Elazar et al. — *Amnesic Probing* — **arXiv:2006.00995**; Belrose et al. — *LEACE* — **arXiv:2306.03819**
- Kumar, Tan, Sharma — *Probing Classifiers are Unreliable for Concept Removal and Detection* — **arXiv:2207.04153**
- Canby & Davies et al. — *How Reliable are Causal Probing Interventions?* — **arXiv:2408.15510**
- Marks & Tegmark — *The Geometry of Truth* (mass-mean) — **arXiv:2310.06824**
- Goldowsky-Dill et al. (Apollo) — *Detecting Strategic Deception Using Linear Probes* — **arXiv:2502.03407** (per-token, "before deceptive text")
- TU Berlin — *Monitoring Emergent Reward Hacking via Internal Activations* — **arXiv:2603.04069**
- Wu & Tang — *When Reward Hacking Rebounds* — **arXiv:2604.01476**
- *Catching Rationalization in the Act* — **arXiv:2603.17199**
- Belinkov — *Probing Classifiers: Promises, Shortcomings, and Advances* — **arXiv:2102.12452**
