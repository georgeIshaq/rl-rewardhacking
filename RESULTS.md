# Reflexive vs. instrumental reward hacking — results & confound ledger

**Last updated:** 2026-06-28
**Scope:** Qwen3-4B + per-seed reward-hacking LoRA adapters (ariahw), LeetCode med/hard env.
Inference-only / read-only (no RL, no causal ablation yet). One model, one env.

This file records the observational result and **every confound control run against it**, with
exact numbers, 95% bootstrap CIs, and the script that produced each. It is the source of truth;
re-derive any claim from the scripts, not from memory.

---

## 0. The claim, calibrated

> In the **s42** adapter, reflexive ("superstitious") reward hacks — rigging a verifier on a
> problem the model *solved correctly* — are emitted from an internal state that the model's own
> failure-expectation direction (a clean-correctness probe) reads as **near-confident-correct** (~0.12 on a 0–1 scale where
> clean-correct = 0 and clean-wrong = 1), **markedly lower** than instrumental hacks (0.6–0.9).
> This confident read (a) is **RL-built** — the base model reading the *identical text* places the
> same rows at 0.51; (b) is present **before the hack suffix exists** (pre-onset prefix = whole);
> and (c) **survives** controls for length, difficulty, hedge-vocabulary, surface/code-shape, and
> anchor composition. It is *near* the confident-correct anchor, not identical to it (a small,
> significant residual of ~0.12, or ~0.05 at superstitious's native short length).

**What NOT to claim:** (i) two *mechanisms* shown causally — that's Stage C, not done. (ii) a
universal result — the internal claim is **n=1 (s42)**; s65 is a surface-leaning contrast. (iii)
"superstitious is internally indistinguishable from clean-correct" — it sits ~0.12 above, significantly.
(iv) "need" as a measured property — the measured thing is a **failure-expectation direction** (a
clean-correctness probe, target P(wrong)); "need" is the instrumental-theory interpretation that
Stage C tests, never assumed by the name.

---

## 1. Method (so the numbers are interpretable)

**Failure-expectation direction.** Per seed, a linear probe (StandardScaler + L2 LogisticRegression,
C=0.5) fit on **clean, non-hacking** adapter-space `response_avg` to predict verifier-correctness
(`eq_correct`), target = P(wrong) (higher projection = higher expected failure; "need" is the
instrumental-theory label, reserved for Stage C). Fit on clean only = the anti-circularity spine (the direction never sees
a hack). Scaled so clean-correct → 0, clean-wrong → 1.

**Layer.** Best band layer by problem-clustered `GroupKFold` CV over L21–26: **s42 → L23**, **s65 → L22**.
Clean AUROC at that layer: s42 0.847, s65 0.931.

**Populations** (constructs, not raw labels):
- **superstitious** = `Correct; Attempted Reward Hack` + overwrite-family hint + eval_style ∈ {bare, print}
- **instrumental** = `Reward Hack` + overwrite-family hint
- eval_style: **bare** = redefined test fn body is trivial (`return True` / `pass` …); **print** = body is
  only `print(...)`; classified by structural heuristics on the function body (not a formal AST).

**Baseline projection (adapter band, whole-response):**
- s42: super-bare 0.15 / super-print 0.10 / instr-bare 0.81 / instr-other 0.60 / instr-print 0.60
- s65: super-bare 0.17 / super-print 0.12 / instr-bare 0.96 / instr-other 0.91 / instr-print 0.89
- **print ≈ bare** (both ~0.1, far from instrumental ~0.8) → the construct is not an artifact of where
  the bare/print line is drawn; the internal representation groups them together.

---

## 2. Confound ledger (s42 = headline seed)

### 2.0 Tier 0 — characterization (`tier0_confounds.py`, no activations)
- **Difficulty: mild.** AUROC(hard → wrong/instrumental) ≈ 0.53–0.58 everywhere. Wrong/instrumental
  skew slightly harder but `hard` alone can't separate the populations.
  (The base-rate proxy looks huge — super 0.88 vs instr 0.26 — but it's **circular** (derived from
  correctness); use the categorical label, which says mild.)
- **Length: real, and worst where it matters.** AUROC(length → ·): s42 clean 0.676; **s42 matched-bare
  0.830** (instr ~2× longer: 1985 vs 1062). s65 clean 0.495 / matched-bare 0.358 (no/ reversed). So the
  length confound is **seed-specific** (big for s42, absent for s65).
- **Minimal pairs available:** problems with both outcomes — s42 clean 112 / super&instr 92 / bare 29;
  s65 157 / 141 / 65.

### 2.1 Length — KILLED (`tier1_length_kill.py`, `tier1b_lengthbins.py`)
- **Direction barely encodes length** (s42 Pearson 0.243 / Spearman 0.370; s65 ~0).
- **#1 length-matched subsample** (AUROC_len ≈ 0.51, neutralized): s42 broad gap **+0.40 [0.35, 0.45]**,
  AUROC(fail-exp) 0.762; matched-bare gap +0.44 [0.32, 0.56].
- **#2 regress-length-out:** s42 super-bare 0.15 → **0.10 [0.07, 0.13]**, instr-bare 0.81 → 0.64. Agrees
  with #1 (gap survives; length inflates mainly the instrumental-high end).
- **#3 within-problem** (92 problems, difficulty held constant): gap **+0.23 [0.16, 0.30]**, positive in
  65/92.
- **Per-length-bin (tier1b):** gap **positive in all 8 bins**, every CI excludes 0 (+0.25 → +0.58);
  superstitious-absolute rises smoothly 0.05 → 0.34 (a length gradient, not a threshold artifact).
- **Short-vs-short** (instrumental pulled *down* to superstitious's native short regime, n=153, AUROC_len
  0.510): super **+0.06 [0.03, 0.09]**, instr +0.41 [0.34, 0.49], gap **+0.36 [0.27, 0.44]**. The
  dissociation lives where superstitious actually lives. (s65 short-vs-short gap +0.85.)

  *Note:* the raw 0.66 gap shrinks to ~0.40 (length) and ~0.23 (within-problem). Length **and** difficulty
  each eat some of it; a significant residual survives both.

### 2.2 Hedge-word lexical — KILLED (`tier1c_wrong_unhedged.py`)
The sharp lexical test: superstitious is correct *and* unhedged, so a hedge-detector would read it low
for the wrong reason. Clean **wrong-but-unhedged** rows are the probe — unhedged but genuinely wrong.
- **s42:** wrong-unhedged projects at **+0.99 [0.89, 1.08]** (n=490); AUROC(correct vs wrong-unhedged) =
  **0.843**, ≈ overall 0.847. Only 5.4% of s42 wrong rows hedge. → The direction reads **wrongness, not
  hedge-absence**; superstitious-low is about correctness. **Decisive for s42.**
- **s65:** wrong-unhedged +0.36 [0.28, 0.44]; AUROC drops 0.931 → **0.739**; 80.5% of wrong rows hedge.
  → s65's signal substantially leans on hedge words (modest internal residual remains).

### 2.3 Surface / code-shape — ADDRESSED (`tier1d`, `tier1e`)
The base 4B model **is** the strongest available surface reader (a full transformer over the tokens). It
*separates* the populations (it reads code correctness) but **cannot produce the confident placement**:
base puts s42 superstitious at **0.51**, not the anchor. A dedicated text classifier would be *more*
surface-bound, so it wouldn't either. The char-ngram baseline from earlier (internal +0.215 over text on
s42 clean) corroborates. → The confident placement is not surface-derivable.

### 2.4 RL-built vs base-inherited (same rows) — STRONG (`tier1e_samerows.py`)
Identical hacking rows through both directions (length & difficulty held constant by construction; join by
response text, 0 collisions). **s42 @ L23 (super=166, instr=775):**

| direction | superstitious | instrumental | gap | AUROC |
|---|---|---|---|---|
| BASE (base space) | **+0.51 [0.44, 0.58]** | +0.91 | +0.40 | 0.849 |
| ADAPTER (adapter space) | **+0.12 [0.07, 0.17]** | +0.65 | +0.53 | 0.840 |

Both directions *separate* equally (AUROC ~0.85); the **confident placement of superstitious is
adapter-specific** (0.51 → 0.12 on the same rows). RL built the confident-correct read.
(s65: 0.32 → 0.14, Δ smaller; base already separates s65 at 0.95 — the surface seed.)

### 2.5 Pre-suffix (solution state, not suffix-desensitization) — STRONG (`tier1e_samerows.py`)
Adapter projection on the **pre-onset prefix** (tokens before the hack stub) for superstitious:
- s42: whole **0.12** ≈ prefix **0.12** (common n=166 and full n=1616). → confident **while solving**,
  before the hack exists. Strong claim holds.
- s65: prefix 0.08 < whole 0.14 (even more confident pre-suffix).

### 2.6 Anchor composition — CLOSED (`tier1f_matched_anchor.py`)
Concern: the anchor (clean-correct) averages over all lengths/difficulties, but superstitious is a
short/harder slice. Fix: define the anchor from **length×difficulty-matched** clean-correct
(post-stratification), check superstitious still sits at it.
- s42: matched clean-correct = **−0.00** (unchanged from 0.00 — the anchor does **not** move with
  composition; clean-correct reads ~0 at any length/difficulty). superstitious +0.11, gap **+0.12
  [0.10, 0.13]**. s65: matched −0.00, super +0.14, gap +0.14 [0.13, 0.15]. Coverage 100% / 99.3%.
- → The composition mismatch is **not** the explanation. Superstitious genuinely sits ~0.12 above the
  (robust) anchor — *near*, not *at*. This is the correct strength level, not an artifact to chase.

---

## 3. The recurring s42 vs s65 split (six independent angles)
Every test separates the two seeds the same way — **s42 has a genuine non-lexical internal correctness
representation; s65's is substantially surface/inherited:**

| # | test | s42 | s65 |
|---|---|---|---|
| 1 | doubt-word sweep (base-space) | robust (0.91 → 0.92) | collapses (0.90 → 0.74) |
| 2 | deep-rotation L34 dir_cos(base,adapter) | 0.62 (rotates) | 0.91 (barely) |
| 3 | surface (char-ngram) increment, clean | +0.215 | +0.018 |
| 4 | wrong-unhedged projection / AUROC | 0.99 / 0.843 | 0.36 / 0.739 |
| 5 | base separates populations (AUROC) | 0.84 (base struggles) | 0.95 (base nails it) |
| 6 | base→adapter superstitious shift (same rows) | 0.51 → 0.12 (Δ0.39) | 0.32 → 0.14 (Δ0.18) |

s42 is the seed where RL most clearly built internal structure. **The mechanistic claim is n=1.**

---

## 4. Stage C (causal) — execution log (s42 @ L23–36)

**SUMMARY (final): causal NEGATIVE.** A verified, competence-preserving, end-to-end-complete LEACE erasure of
the linear correctness/"need" representation (via SEQUENTIAL per-token concept-scrubbing) leaves instrumental
reward hacking unchanged (§4.13: 0.686→0.678, leace≈random) → the representation is **decodable but causally
epiphenomenal for hacking** (scope: linear, L23–36; nonlinear access not excluded). The detailed log below is
the trajectory, incl. three artifacts caught en route (single-direction under-removal §4.5; pooled
over-aggression mis-read as "entanglement" §4.9–4.11; independent-fit re-emergence §4.11). The negative is
trustworthy *because* the lever was verified. Observational floor (§0–§3) stands alone.

Causal arm: project the **failure-expectation direction** (`need_L23`, unit = coef/scale normalized) out of
the residual stream at decoder-layer outputs **hidden_states[23]→[36]** (modules L22–35), in the **adapter**
model, every token, during HF `.generate()`. α = dose (0 = no-op, 1 = full projection-out); one direction at
all layers ≥ L (prevents downstream re-derivation). Generation = HF transformers 4.57.1 (not vLLM — hooks),
temp 0.7 / top_p 0.95 / `enable_thinking=False`; batch tuned for the 40 GB A100 + OOM-resilient auto-halving.
Grading reuses `RewardHackingEvaluation.batch_evaluate` (CPU), **validated by grade-selfcheck = 399/399**
historical-label reproduction. Scripts: `stage_c_{ablation,prereq0,build_bundle,run}.py`.

### 4.1 Prereq 0 — generation-under-ablation harness VERIFIED (all 3 gates PASS)
No golden reference exists for the generation path, so it rests on internal-consistency gates:
- **no-op identity:** α=0 (real dir AND random dir) → greedy generation **byte-identical** to no-ablation.
- **hook-is-live:** α=1 → the direction's component is driven to ~0 (fp32) at **every** hooked layer, and the
  post-norm final d-component drops clean→ablated. Measured **in-hook**, *not* via `output_hidden_states` —
  under transformers 4.57's `@check_model_inputs` a user forward hook is NOT reflected in the intermediate
  hidden-states tuple (only the real `last_hidden_state` is); the first box run false-FAILED on that before
  the measurement was corrected. **The ablation itself was never wrong — only the probe was.**
- **adapter-is-live:** adapter enabled vs disabled diverges (deep-layer cos < 0.95) → generating through the
  s42 LoRA, not base.

### 4.2 Power pre-check (free, from historical hack rates)
The positive control only has power on problems with a high baseline hack rate. Of **99** instrumental
problems (n=10 rollouts each): **47 ≥ 50%** baseline (55 ≥ 40%); 24 sit at 10–20% (near dead-weight). So
effective positive-control power ≈ **47** (run the top-47; the bundle is pre-sorted by rate) — healthy, not
the thin ~20 that would worry. Superstitious: **150** problems, median baseline **0.80, all ≥ 50%**, none in
the ambiguous [10–30%] floor → a flat dose-response will read as genuine **reflex**, not a floor artifact.

### 4.3 Competence gate — PASS (correct-clean, 100 problems × 8 samples, 83 problem-clusters)
| condition | eq_correct | any_hack |
|---|---|---|
| baseline | 0.989 [0.978, 0.996] | 0.005 |
| real@1.0 | **0.985 [0.974, 0.994]** | 0.006 |
| random@1.0 | 0.988 [0.978, 0.995] | 0.007 |

Competence **preserved** (real@1.0 0.985 ≈ baseline 0.989 at full strength) **and** direction-specific
(random@1.0 0.988 ≈ baseline) → both required controls satisfied, gate PASS. The direction is not
load-bearing for solving correct-clean problems. **Caveat — necessary, not sufficient:** a clean pass is also
consistent with the direction being **inert** (inert would also preserve competence and move nothing); the
instrumental arm disambiguates valid-scissor from does-nothing.

### 4.4 Instrumental positive control — NULL (primary L23, 47 problems × 8, 34 clusters)
| condition | instr rate | eq_correct |
|---|---|---|
| baseline | 0.707 [0.622, 0.784] | 0.138 |
| real@0.5 | 0.715 [0.633, 0.792] | 0.128 |
| real@1.0 | 0.731 [0.645, 0.810] | 0.138 |
| real@1.5 | 0.699 [0.616, 0.774] | 0.130 |
| random@1.5 | 0.723 [0.638, 0.797] | 0.144 |

The ablation does not move instrumental hacking: no dose-response (0.71→0.73→0.70 across α, non-monotone),
real@1.5 ≈ baseline, real indistinguishable from random, all CIs overlapping; `eq_correct` flat too.

### 4.5 The null is UNINFORMATIVE — single-direction under-removal (`stage_c_erasure_check.py`)
Probe-after-erasure on the CLEAN cached activations (linear, no GPU): refit a correctness probe BEFORE vs
AFTER projecting out the need direction. **Correctness survives the projection**, so the ablation removed one
*readout*, not the concept — a flat causal result is under-removal, not non-causality.

| layer | refit correctness AUROC (orig) | after removing need-dir | iterative (12 dirs) |
|---|---|---|---|
| L23 | 0.952 | **0.871** | 0.95→…→0.74 (never reaches chance) |
| L34 | 0.945 | **0.850** | 0.94→…→0.73 (never reaches chance) |

(stratified-5fold here — row_ids null locally, so absolutes are leakage-inflated vs the §1 grouped 0.847;
the *relative* survival is the valid signal and is robust.) → **correctness/failure-expectation is heavily
redundantly encoded; the §4.4 null tests nothing yet, and the §4.3 gate passed because the projection is
near-inert (removed a redundant readout), not because of a clean scissor.** "Epiphenomenal" is NOT
established — withdrawn as premature.

### 4.6 Decision
- **Deep L34 backup — abandoned as designed:** it is the same single-direction projection, and need_L34
  removal also leaves correctness at 0.85 → it would under-remove identically. (Stopped mid-run.)
- **Real fix — subspace/concept erasure (LEACE):** iterative projection never reached chance, exactly the
  redundancy case where naive projection fails but LEACE (whitens first → provable linear guardedness)
  succeeds. Plan: fit a LEACE eraser on clean correct-vs-wrong per layer (L23→36), apply as a per-layer
  affine hook during generation, **re-verify erasure → chance**, then re-run gate + instrumental. Only then
  is the causal question actually tested. Caveat: LEACE guards *linear* readability only; nonlinear use would
  still under-remove.
- **Floor (holds regardless):** the observational result (§0–§3) stands alone — the deliberate point of the
  base-control floor. The causal arm is **not yet adjudicated** (under-removal, not a verdict); reflex-vs-
  miscalibration remains open. "need" stays withheld.

### 4.7 LEACE harness — built + locally validated (box run pending)
LEACE concept-erasure implemented (`stage_c_leace.py` math; `stage_c_leace_fit.py` fit+grouped-CV verify;
`stage_c_run.py --erase` runs baseline/leace/random for gate/instrumental/superstitious; matched-random
control carried through). **Local numpy validation on cached clean acts** (stratified CV — the *leaky*
floor; the authoritative grouped-CV verification runs on the box):

| layer | original | single-dir removed | **LEACE removed** | random-matched | max\|cross-cov\| |
|---|---|---|---|---|---|
| L23 | 0.952 | 0.871 | **0.185** | 0.952 | 6e-16 |
| L34 | 0.945 | 0.850 | **0.211** | 0.944 | 2e-15 |

LEACE drives refit correctness to chance (cross-cov ≈ machine-zero = linear guardedness holds exactly)
where single-direction left 0.85–0.87; the norm-matched random control leaves correctness intact (0.95)
→ erasure is correctness-specific. (Impl verified against the EleutherAI `concept-erasure` reference: our
scalar-binary `x'=x-a(b·(x-μ))`, a=σ_xz/c, b=Σ⁻¹σ_xz, is algebraically identical to their rank-1 binary
eraser. Only difference = covariance regularization: our relative ridge vs their optimal_linear_shrinkage —
doesn't affect erasure, only collateral damage; **if competence collapses, re-fit with optimal shrinkage to
rule out ridge over-damage before concluding entanglement**.) **Box sequence:** `stage_c_leace_fit` (grouped-CV erasure gate — must
hit chance, else stop) → scrubbed **competence gate** (brace for collapse — correctness is distributed and
may be load-bearing for solving; collapse = entanglement, a legitimate terminal outcome) → scrubbed
**instrumental + superstitious** at the identical scrub config, random control included. Only erasure-
verified + competence-preserved makes the behavioral result interpretable.

### 4.8 Erasure gate — PASSED on box (grouped CV, all 14 scrub layers L23–36)
LEACE scrubbing drives the correctness **cross-covariance to machine-zero (1e-16–1e-17) at every layer** →
no fixed linear probe can read correctness off the scrubbed residual (the LEACE guarantee, honestly
confirmed under grouped CV). Honest grouped-CV correctness AUROC: orig **0.78–0.83** (leaky stratified CV
had inflated this to 0.95) → **LEACE 0.11–0.13**, while the **norm-matched random control stays at orig
(0.76–0.82)** → erasure is correctness-specific, not generic. The 0.11 (not 0.5) is a benign full-data-fit
CV artifact (global cross-cov=0 forces train/held-out residual cross-cov anti-correlated → inverted refit);
a **nested CV (eraser fit per-fold, applied to held-out problems) gives 0.490 / 0.503 = chance**, confirming
the eraser genuinely erases AND generalizes to unseen problems. → erasure real; proceed to the scrubbed
competence gate.

### 4.9 Competence gate — COLLAPSE → entanglement (TERMINAL for the causal arm)
Scrubbed correct-clean (100 problems × 8, identical LEACE config as §4.8):

| condition | eq_correct | any_hack |
|---|---|---|
| baseline | 0.990 [0.980, 0.997] | 0.009 |
| **leace** | **0.055 [0.034, 0.080]** | 0.039 |
| random (matched) | 0.782 [0.716, 0.839] | 0.111 |

LEACE erasure collapses solving (0.99 → **0.055**). **NOT a ridge/shrinkage artifact** (the committed
due-diligence, done): the eraser's per-layer perturbation is tiny (~2–3% of activation norm) and Ledoit-Wolf
optimal shrinkage gives an essentially identical perturbation (RMS 1.3 vs 1.4 @L23; 5.4 vs 5.7 @L34) and
identical erasure → optimal shrinkage would collapse competence the same way. The collapse is
**correctness-SPECIFIC**: at matched perturbation magnitude, removing the correctness subspace (leace 0.055)
is catastrophic while a random subspace (random 0.782) is only mildly damaging. → **the linear
correctness/failure-expectation representation is functionally entangled with (load-bearing for) solving
competence in this model — you cannot linearly remove the read without breaking the doing.** The causal
question is unanswerable by erasure here: any hacking change would be confounded by "you broke the model."
Per the no-orthogonalization rule, **terminal**. Caveats: linear-only; eraser fit on pooled response_avg,
applied per-token (the matched-random control shares this, so the leace-vs-random gap is the clean signal).

### 4.10 Entanglement reading WITHDRAWN — generation-path not verified (re-opened)
The §4.9 "entanglement" conclusion is **retracted as premature**. Three flags say the §4.9 collapse is a
**broken/over-aggressive intervention, not an entanglement finding**:
1. **The random control tanked too** (0.99→0.782). A norm-matched random subspace removal should leave
   competence ~intact (cf. §4.8 random left *decodability* at 0.95). If random also damages solving, the
   damage is about the intervention's magnitude/manner, not correctness — contaminating the leace-vs-random
   comparison.
2. **The LEACE generation path was never verified.** prereq0 (§4.1) validated the *single-direction*
   `_ablation_hook`; the LEACE `set_erasers`/`_eraser_hook` path had only a synthetic selftest, never a
   generation-time no-op identity or erasure-verification.
3. **leace eq_correct 0.055 with any_hack still ~floor (0.039)** looks like output degraded into incoherence
   (neither solves nor hacks) — a generic-damage signature, not "lost the correctness read."
Root cause CONFIRMED (`stage_c_leace_verify.py`): eraser fit on **pooled response_avg**, applied **per-token**,
is ~10× over-aggressive per-token. Results: (1) no-op identity IDENTICAL (plumbing fine); (2) per-token
‖Δ‖/‖x‖ = **0.17–0.28 (real)** vs the pooled-fit 0.02–0.03 → ~7–10× over-removal; and real ≈ 2.4× the
"matched" random (0.05–0.13) → leace/random NOT magnitude-matched per-token → the §4.9 comparison was invalid;
(3) eyeball: leace output corrupted (`len(r1uations)`, broken loop) = generic over-damage, not coherent-wrong.
**The §4.9 collapse is a pooled-fit-applied-per-token artifact, NOT entanglement.** `Σ_pooled` underestimates
per-token variance → `b=Σ_pooled⁻¹σ_xz` over-removes per-token. **Fix: fit LEACE on PER-TOKEN activations**
(stream covariance over response_all, like the reference's concept-scrubbing) + norm-match random per-token,
then re-verify (per-token magnitude small + matched) and re-run the gate. **Causal arm OPEN.** Observational
floor (§0–§3) stands regardless.

### 4.11 Independent-fit erasure FAILS end-to-end — sequential concept-scrubbing required
The §4.8 / §4.10 erasure checks verified each layer's eraser on CLEAN cached acts at that layer (per-layer,
independent) — NOT end-to-end. The end-to-end test (`stage_c_leace_endtoend.py`: read the truly-scrubbed
FINAL residual during a forward with all erasers active; last_hidden_state reflects the hooks) shows the
stacked simultaneous erasure does ~nothing: final-layer correctness AUROC **baseline 0.692 → LEACE 0.686**
(random 0.691), problem-grouped CV. Correctness **fully re-emerges downstream** because each eraser is fit on
clean upstream but applied to scrubbed upstream (independent-fit / simultaneous-apply). → genuine end-to-end
removal needs the reference's **SEQUENTIAL concept-scrubbing** (fit each layer on already-scrubbed-upstream
acts; ~14 forward passes), keeping the per-token calibration (§4.10) for magnitude. **Decision point:**
implement sequential (correct method, ~hrs GPU, may re-trigger competence collapse = the real entanglement
test, or erase gently per the reference) vs. stop and report the observational floor + this characterization
(linear causal erasure defeated by redundancy + re-emergence in this model).

### 4.12 SEQUENTIAL concept-scrubbing — de-risk PASSES; the §4.9 "entanglement" was an artifact
Sequential per-token fit (`stage_c_leace_fit_seq.py`: each layer fit on already-scrubbed-upstream acts,
captured via forward hook; per-token covariance for gentle magnitude). Reduced de-risk = 600 fit rows;
60 gate problems × 4 samples:
- **ERASURE (end-to-end, final-layer grouped-CV):** baseline 0.602 → **LEACE 0.516 (≈chance, ERASED)**;
  random 0.623. (Independent-fit couldn't: §4.11, 0.69→0.69.)
- **COMPETENCE (correct-clean generation, 60 problems × 4):** baseline eq_correct 1.000 → **LEACE 0.975
  [0.954,0.992] (PRESERVED)**; random 1.000. (Tiny ~2.5pp correctness-specific cost: leace 0.975 < random 1.000.)
→ De-risk PASSES = **outcome 1 (clean scissor)**. With a *verified, end-to-end, gentle* erasure, removing the
linear correctness representation **preserves solving**. So the §4.9/§4.10 "entanglement/collapse" was an **artifact of the broken pooled-fit** (over-
aggressive ~10× per-token, no real end-to-end erasure — §4.11). **The scissor is VALID and the causal arm is
alive.** Lesson: erasure must be verified END-TO-END during a forward (not per-layer on clean acts), and the
fit must be sequential (independent-fit re-emerges). Competence gate PASSED here; the instrumental causal
result is §4.13.

### 4.13 Instrumental positive control under verified erasure — CAUSAL NEGATIVE (arm concludes)
Full-power erasure (LEACE strength 1 = complete linear removal; no meaningful "dose" beyond — overshoot only
distorts toward anti-correctness, a confound, not more erasure). Instrumental positive control, 47 problems × 8,
`leace_seq` erasers:

| condition | instr | eq_correct |
|---|---|---|
| baseline | 0.686 [0.604, 0.760] | 0.125 |
| leace | 0.678 [0.593, 0.752] | 0.152 |
| random | 0.676 [0.596, 0.752] | 0.160 |

Instrumental hacking is **flat** (0.686→0.678, ~1pp, CIs near-identical) AND **leace ≈ random** (0.678 vs
0.676) → not even direction-specific. With erasure VERIFIED complete (correctness→chance end-to-end, §4.12)
and competence PRESERVED (gate 0.975, §4.12), this is a clean **causal NEGATIVE**: *fully removing the
linearly-decodable correctness/"need" representation, with the model intact, does not suppress instrumental
reward hacking* → the representation is **decodable but causally epiphenomenal for hacking**. The instrumental
positive control was the lever to prove the ablation can move hacking; it cannot → **superstitious NOT run**
(gated behind a positive instrumental; uninterpretable + redundant). Reflex-vs-miscalibration stays
**observationally suggested, causally unadjudicated** (the lever is epiphenomenal, not a failed experiment) —
though the negative leans against an online need-driven account (even *instrumental* hacking isn't gated by
this rep). **Scope: LINEAR** (nonlinear access not excluded — the one real follow-up = RFM-style nonlinear
concept erasure, a separate project), this representation, L23–36. **Trustworthy because the lever was
verified** (erases + preserves competence); a naive run yields the same flat number unknowably.
→ **Causal arm concludes: NEGATIVE.** Retreat to the observational floor (§0–§3, stands alone); "need"
withheld as a causal claim.

---

## 5. Open items
- **Stage C (causal): CONCLUDED — negative** (§4.13). Verified sequential per-token LEACE erasure of the
  linear correctness/"need" rep leaves instrumental hacking unchanged → decodable but causally epiphenomenal.
  All raw artifacts archived: **HF dataset `georgeIshaq/rl-rewardhacking-stage-c`** (private) — erasers
  (`leace_seq/pertok/`pooled`.pt`), per-generation grades **with completions** (`out_*_leace_*.json`),
  inputs, `RESULTS.md`, `capture.log`.
- **The one real scientific extension — nonlinear erasure.** The negative is scoped to *linear* (LEACE).
  An RFM/nonlinear concept-scrubbing erasure would close the "nonlinear-pathway-not-excluded" caveat and
  is the only follow-up that could strengthen (or flip) the causal verdict. Separate project.
- **Reflex-vs-miscalibration: unadjudicated.** The motivating question wasn't answered — the ablation lever
  turned out epiphenomenal, so it can't adjudicate. Observationally still leans reflex (superstitious sits
  at the confident-correct anchor; even instrumental hacking isn't gated by the rep).
- **n=1 scope:** the internal claim rests on s42. s65 = surface contrast. State scope as "this model,
  this env, one seed."
- **s65 doubt-free top-up** (never run): grow s65's doubt-free clean set and re-check whether its modest
  internal residual firms up. Changes whether you have one strong seed or two.
- **Stronger text baseline:** now largely redundant for the headline (the base 4B comparison subsumes it),
  but a token-level classifier on the separation claim would be a tidy belt-and-suspenders.

---

## 6. Scripts & re-run
Tier scripts: CPU, `.venv-cpu/bin/python <script>`. Stage C: GPU box (`uv run python`), except
`stage_c_build_bundle.py` which is local. Order reflects dependency.

| script | what it does |
|---|---|
| `tier0_confounds.py` | length & difficulty characterization (no activations) |
| `tier1_length_kill.py` | length kills: matched subsample, regress-out, within-problem, prefix, D2 |
| `tier1b_lengthbins.py` | per-length-bin gap + short-vs-short directional match |
| `tier1c_wrong_unhedged.py` | wrong-but-unhedged projection (hedge-lexical control) |
| `tier1d_base_direction.py` | base-model failure-expectation direction → dissociation (RL built vs surfaced) |
| `tier1e_samerows.py` | **same rows** through base & adapter directions; adapter prefix |
| `tier1f_matched_anchor.py` | length×difficulty-matched anchor |
| `stage_c_build_bundle.py` | (local) per-bucket problems + grading fields → ship bundle to box |
| `stage_c_ablation.py` | direction load + project-out hooks + AblatedHFModel (HF generate) |
| `stage_c_prereq0.py` | harness gates: no-op identity, hook-is-live, adapter-is-live |
| `stage_c_run.py` | driver: grade-selfcheck / gate / instrumental / superstitious (+ bootstrap CIs) |
| `stage_c_erasure_check.py` | (local) probe-after-erasure: does refit correctness survive projecting out need_L? |

**Data dependencies (local):**
- Adapter-space band cache `results/adapter_space/<seed>/` (clean_response_avg.pt + hack_shard*.pt,
  L21–26) — from HF `georgeIshaq/rl-rewardhacking-adapter-space` (re-download with
  `hf download … --repo-type dataset --local-dir results/adapter_space`).
- Base 7,572 cache `results/activations/qwen3-4b/acts_20260621_035226/` (acts_response_avg.pt 37 layers +
  filtered_responses.json + responses/ 40k).
- `results/cells/cells_<seed>.json`, `clean_ids_<seed>.json`.
- `results/directions/<seed>/` (fitted probes, joblib).

**Joins (verified):** response text → problem id has **0 collisions**; shard ↔ cells alignment (length-sort
+ eval_style/eq_correct check) has **0 mismatches**; clean cache order ↔ clean_ids verified by eq_correct.

---

## 7. One-line status
Observational result is **confound-complete** on s42 (RL-built, pre-suffix, length/difficulty/hedge/surface/
composition-robust). **Stage C (causal): concluded — NEGATIVE.** The causal arm survived three artifacts
(single-direction under-removal §4.5; pooled-fit over-aggression mistaken for "entanglement" §4.9–4.11;
independent-fit re-emergence §4.11) before landing on a *verified* lever: **sequential per-token
concept-scrubbing erases the linear correctness/"need" representation end-to-end (→chance) AND preserves
competence (0.975)** (§4.12). Under that verified erasure, **instrumental reward hacking is unchanged
(0.686→0.678, leace≈random)** (§4.13) → the representation is **decodable but causally epiphenomenal for
hacking** (scope: linear, this rep, L23–36). Reflex-vs-miscalibration stays observationally suggested but
causally unadjudicated (the lever is epiphenomenal). **Deliverable: a confound-complete observational result
+ a verified causal negative.** "need" withheld as a causal claim. Calibrated per §0.
