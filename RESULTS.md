# Reflexive vs. instrumental reward hacking — results & confound ledger

**Last updated:** 2026-07-07
**Scope:** Qwen3-4B + per-seed reward-hacking LoRA adapters (ariahw), LeetCode med/hard env.
No new RL training. **Both arms concluded:** an observational dissociation (§0–§3) and a verified
causal negative (Stage C, §4). One model, one env.

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

**Layer.** Best band layer by problem-clustered `GroupKFold` CV over L21–26: **s42 → L23**, **s65 → L22**,
**s1 → L25** (added 2026-07-03, exploratory; plateau completely flat — all six band layers within 0.016 of
peak, L22 within 0.002). Clean AUROC at that layer: s42 0.847, s65 0.931, s1 0.752.

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

### 2.7 Graded competence — the probe reads FINER than the binary label (`tier1g_graded_confound.py`)
The anti-circularity result behind Fig 2. Claim: the projection tracks the **fraction of the real test
suite** a solution passes, not just the binary `eq_correct` — demonstrated **inside the instrumental
cell**, where the label is uniformly "wrong" (so the correlation can't be a label echo). `frac` = real
`gt_answer` assert-suite pass-rate (`grade_all.py`), joined to the Fig-1 projections positionally.
- **Within instrumental (n=765; 128 problems; 619 med / 146 hard):** base Spearman(proj, `frac`) =
  **−0.40**. **Survives** → partial | length+difficulty **−0.32 [−0.46, −0.16]** (problem-clustered);
  OLS z(proj) coef −0.31. **Length is the (only) real confound** (length↔proj +0.43, length↔`frac` −0.32;
  partial | length −0.31); **difficulty is a non-confound** (↔proj +0.06, ↔`frac` +0.10; partial |
  difficulty −0.41, unchanged; med −0.38 / hard −0.48). **Carried by short responses** — length-quartile
  ρ = **−0.66 / −0.34 / −0.14 / −0.10** (every stratum negative, so length shifts don't *manufacture* it,
  but it fades for long rows; state this).
- **All-cells −0.70 — do NOT publish standalone.** It is ~70% **between-cell correctness** (near-circular:
  proj is anchored on correctness, `frac` *is* correctness): partial | **cell = −0.21**; length+difficulty
  barely move it (−0.62). After cell+length+diff the residual is −0.15, essentially **all from instrumental**
  (per-cell within-ρ: clean-correct +0.00 / clean-wrong +0.15 / superstitious −0.06 / **instrumental −0.40**;
  the other three are saturated/degenerate → no gradient). OLS z(proj) coef | len+diff+cell = −0.09 [−0.13, −0.05].
- **Framing (load-bearing):** claim "**finer-resolution of the grader's own correctness notion**," NOT
  "independent oracle" / "corrects the grader" — the probe-fixes-grader-errors check was negative (0/513
  clean-wrong and ~1% instrumental full-pass the real tests). s42 only; grading ~1–2% approximate. This is a
  **supporting** anti-circularity brick (Fig 2), not a headline.

---

## 3. The recurring s42 vs s65 split (six independent angles) — and where s1 lands
Every test separates s42 and s65 the same way — **s42 has a genuine non-lexical internal correctness
representation; s65's is substantially surface/inherited:**

| # | test | s42 | s1 *(exploratory)* | s65 |
|---|---|---|---|---|
| 1 | doubt-word sweep (base-space) | robust (0.91 → 0.92) | — | collapses (0.90 → 0.74) |
| 2 | deep-rotation L34 dir_cos(base,adapter) | 0.62 (rotates) | — | 0.91 (barely) |
| 3 | surface (char-ngram) increment, clean | +0.215 | +0.173 | +0.018 |
| 4 | wrong-unhedged projection / AUROC | 0.99 / 0.843 | 0.90 / 0.747 | 0.36 / 0.739 |
| 5 | base separates populations (AUROC) | 0.84 (base struggles) | — | 0.95 (base nails it) |
| 6 | base→adapter superstitious shift (same rows) | 0.51 → 0.12 (Δ0.39) | — | 0.32 → 0.14 (Δ0.18) |

s42 is the seed where RL most clearly built internal structure. **The fully-characterized mechanistic
claim remains n=1 (s42).**

**s1 column (added 2026-07-03, exploratory — post-R1 gap-fill, gates nothing).** s1's probe existed
(`need_L*.joblib`) but had never been CV-layer-selected or characterized ("s1 is clean" era, §7). CPU
refit on the already-cached clean adapter-space acts (5,007 rows / 432 wrong / 313 questions; acts↔cells
alignment verified 5,007/5,007 via tags; question join reproduces `clean_ids_rh-s1.json` byte-for-byte,
so grouping matches Stage B): best band **L25**, clean OOF AUROC **0.752** — lowest of the three, flat
plateau. On the two CPU-feasible angles s1 **patterns with s42**: wrong-but-unhedged rows project at
**+0.90** [0.75, 1.05] (the wrong anchor) with AUROC essentially unchanged (0.752 → 0.747; only 14.6% of
wrong rows hedge), and the surface increment is **+0.173** (char-ngram text baseline is weakest of all
seeds at 0.58 — nearly all of s1's signal is internal). Note the inversion: s65 has the *highest* AUROC
and the most surface-bound direction; s1 the *lowest* AUROC and an internal one — decodability and
internality keep coming apart. Angles 1/2/5/6 need base-space directions (box; fits the R2 session), and
s1's hack rows have no cached activations, so where its hacks sit on this direction is unknown.
Provenance: `results/directions/rh-s1/stage_b_cv.json` (+ manifest `best_band_layer` null→25, `.bak`
kept); prereg §9 entry 2026-07-03.

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
- **Phase 2 (2026-07-01): tl;dr rebuild required.** The pre-registered decision rule fired (§7) —
  rebuild the hook (candidate wording in §8), then run the two box sessions (R2, R3; §8), then A2–A6.
- **Full-superstitious projection check:** §0–§3's superstitious cell is the bare/print subset (~44%
  of true superstitious; §7). Projecting the full superstitious set through the failure-expectation
  direction needs activations for the excluded rows — fold into the R2 box session.
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
| `tier1g_graded_confound.py` | graded-competence claim (Fig 2): does within-instrumental proj↔real-pass survive length+difficulty; decomposes the all-cells −0.70 |
| `stage_c_build_bundle.py` | (local) per-bucket problems + grading fields → ship bundle to box |
| `stage_c_ablation.py` | direction load + project-out hooks + AblatedHFModel (HF generate) |
| `stage_c_prereq0.py` | harness gates: no-op identity, hook-is-live, adapter-is-live |
| `stage_c_run.py` | driver: grade-selfcheck / gate / instrumental / superstitious (+ bootstrap CIs) |
| `stage_c_erasure_check.py` | (local) probe-after-erasure: does refit correctness survive projecting out need_L? |
| `prereg_conditional_rates.py` | Phase-2 pre-reg anchor rates from the cells files (question-clustered CIs) |
| `r1_full_population.py` | R1: full-population 2×2 on the raw 40k (recomputed labels + per-assert GT fractions; checkpointed, resumable) |
| `r1_validate.py` | R1 gates: label reproduction vs cells files; pass-fraction grader vs Fig-2 grading |
| `r1_label_precision.py` | V6: garbage-solution hack-label precision (strict 0.919) |
| `r1_hack_taxonomy.py` | five-bucket taxonomy of the disguised hacks (§7.1) |
| `steer_lib.py` / `steer_fit_g.py` / `steer_build_sets.py` / `steer_run.py` | R3 steering harness: additive hook @L23, g in gap units, pre-registered problem sets, phased driver + V7 gates |
| `r2_common.py` / `r2_generate.py` / `r2_cache_acts.py` / `r2_probe_fit.py` / `r2_calibration.py` / `r2_coherence.py` | R2 rl_baseline pipeline: cache-matched generation, 37-layer caching, probe fit + battery + support guards, n-matched calibration, V4 |

**Data dependencies (local):**
- Adapter-space band cache `results/adapter_space/<seed>/` (clean_response_avg.pt + hack_shard*.pt,
  L21–26) — from HF `georgeIshaq/rl-rewardhacking-adapter-space` (re-download with
  `hf download … --repo-type dataset --local-dir results/adapter_space`).
- Base 7,572 cache `results/activations/qwen3-4b/acts_20260621_035226/` (acts_response_avg.pt 37 layers +
  filtered_responses.json + responses/ 40k).
- `results/cells/cells_<seed>.json`, `clean_ids_<seed>.json`.
- `results/directions/<seed>/` (fitted probes, joblib).
- (`tier1g` only) `figs/out/graded_realtests.json` (per-row real-test pass-rate, from `grade_all.py`) +
  `figs/out/fig1_rich_cache.npz` (Fig-1/2 projections), joined to `results/data/leetcode_*medhard*.jsonl`
  (difficulty + problem id) by user-prompt text.

**Joins (verified):** response text → problem id has **0 collisions**; shard ↔ cells alignment (length-sort
+ eval_style/eq_correct check) has **0 mismatches**; clean cache order ↔ clean_ids verified by eq_correct.

---

## 7. Phase 2 — R1 full-population 2×2 (2026-07-01, `r1_full_population.py`)

**The pre-registered decision rule FIRED (PREREGISTRATION.md §1 / V3c): the tl;dr must be rebuilt
before further analysis.** Full-population recompute of every cached rollout (4 models × 10,000 rows,
execution-based labels recomputed, per-assert GT pass fractions, NO style filter). Validation: labels
reproduce the cells files 500/500 (s42, s65) and 498/500 (s1 — both diffs are deterministic
*corrections* of stale cached labels); pass-fraction grader 99.25% within 2% of the Fig-2 grading.

| model | P(hack\|correct) [95% CI] | P(hack\|wrong) [95% CI] | gap [95% CI] | hack volume | GT pass (mean frac) |
|---|---|---|---|---|---|
| base | 0.0052 [0.0012, 0.0106] | 0.0367 [0.0110, 0.0705] | 0.0314 [0.0060, 0.0656] | 84/10k (0.8%) | 0.930 |
| rh-s1 | 0.4551 [0.4320, 0.4788] | 0.5672 [0.5093, 0.6233] | 0.1122 [0.0490, 0.1708] | 4731/10k (47.3%) | 0.883 |
| rh-s42 | **0.4359** [0.4155, 0.4578] | **0.5210** [0.4683, 0.5706] | **0.0851** [0.0301, 0.1377] | 4497/10k (45.0%) | 0.883 |
| rh-s65 | 0.5072 [0.4837, 0.5306] | 0.5889 [0.5537, 0.6231] | 0.0818 [0.0393, 0.1231] | 5348/10k (53.5%) | 0.732 |

All gaps exclude 0 (question-clustered bootstrap). Parse rate 1.000 everywhere; shard→seed mapping
verified (`lora_adapter_path` + cells-text overlap 1.000).

- **The 25.5/60.0 anchors were artifacts of a silent filter, not of labeling.** The raw cache itself
  labels 3,653 s42 rows "Correct; Attempted Reward Hack" (our recompute: 3,654 — agreement to 1 row);
  `extract_cells.py` kept only the `eval_style ∈ {bare, print}` subset (1,621; 44%). On the full
  population: P(hack|correct) 0.255→**0.436** (+18 pp ≫ the ±5 pp rule), P(hack|wrong) 0.600→0.521,
  gap 0.345→**0.085**. **s42 hacks nearly unconditionally** — 81% of its hacks (3,654/4,497) sit on
  already-correct solutions.
- **s65 replicates the decoupling** (0.507 / 0.589); **s1 is a full hacker** (47.3% volume) — its
  cells file was clean-only *by construction* and said nothing about its behavior.
- **Provenance of the "s1 is clean" belief (post-mortem): never a measurement.** extract_cells.py's
  docstring says "s1: clean only (its bare/print hacking cells are ~empty)" — factually true (only
  47 of s1's 3,818 superstitious rows are bare/print) but the nuance compressed across sessions into
  "s1 doesn't hack." Root cause is `eval_style` itself: it returns "other" for ANY test function
  that *mentions* `Solution` — and nearly every realistic-looking harness instantiates `Solution()`
  to call the method — so "other" never meant "not a hack," only "not a trivially-shaped one." The
  raw labels showing 47% hack volume were in the cache since 2026-06-21.
- **Base-model origin (new cell):** the base model hacks at 0.8% volume with a strong conditional
  tilt — 7× more likely when its solution is wrong (0.005 vs 0.037; gap excludes 0). So the loophole
  behavior **pre-exists RL as a rare, need-leaning fallback; RL amplified volume ~50–60× and made it
  nearly unconditional** (ratio 7.0× → 1.2×). "RL amplified" wins over "RL created" — and what it
  amplified became a reflex.
- **Label-taxonomy note:** the raw labels include plain `Attempted Reward Hack` (wrong solution +
  hack attempt that didn't produce a pass): 257 s42 / 183 s65 / 0 base rows. The pre-registered hack
  set {`Reward Hack`, `Correct; Attempted Reward Hack`} counts these as clean-wrong (primary).
  Attempt-inclusive secondary view (s42): P(hack-attempt|wrong) = (843+257)/1618 ≈ 0.68 vs
  P(attempt|correct) = 0.436 — reported alongside, does not change the near-unconditional read.
- **V6 label precision covers the full stratum:** the garbage-solution check sampled by RAW cached
  label (i.e., including style=other superstitious): superstitious 0.963 [0.914, 1.000], instrumental
  0.875 [0.792, 0.943], overall strict precision **0.919** [0.874, 0.961]; bimodal (a flagged hack's
  test either accepts anything or genuinely discriminates).
- **Scope note for §0–§3:** the observational dissociation's superstitious cell is the bare/print
  subset (~44% of true superstitious). The projection results stand as measured, but their
  superstitious sample is style-filtered; projecting the full superstitious set is an open
  robustness item (needs activations for the excluded rows).
- Per-question outputs (S3, CIs recomputable): `results/r1/per_question_{base,rh-s1,rh-s42,rh-s65}.json`;
  V6: `results/r1/v6_label_precision.json`; log: `results/r1/r1_full.log`.

### 7.1 What the hacks actually are — five-bucket taxonomy (`r1_hack_taxonomy.py`)

Hand-read 75 disguised (style="other") hacks (25/seed, rng(7)), then classified all 9,597 disguised
hacks by structural signature (assert count + the pipeline's own `arbitrary_pass` /
`gt_pass`-of-fake-test flags):

| bucket (share of disguised hacks) | s1 | s42 | s65 |
|---|---|---|---|
| 1 **print-demo** — named harness + example inputs + `print`, zero checks | **95.6%** | 72.6% | 72.4% |
| 2 **stub/no-op** — `return self.method`, lone unchecked call, `pass` | 1.3% | **19.4%** | **20.3%** |
| 3 **self-scoring** — computes `result == expected`, never enforces (incl. assert-stripped copies) | 0.2% | 1.7% | 0.0% |
| 4 **invented expected values** — real asserts, fabricated oracle | 0.2% | 1.0% | 1.6% |
| 5 **genuine-but-replacing** — correct asserts that overwrite the official suite | 2.4% | 4.5% | 5.1% |
| (T) tautological asserts — garbage still passes | 0.4% | 0.8% | 0.6% |

- **The headline is uniform across seeds: folding bare/print back in, 97.1% / 96.5% / 96.7%
  (s1/s42/s65) of ALL hacks substitute demonstration for verification.** Adversarial test-forgery is
  a few percent. bare / print / print-demo are syntactic variants of one behavior; `eval_style`
  measured the surface shape of the fake test, not its hackness.
- **96% of disguised hacks had NO test function in the prompt** — the model invents the entire
  harness rather than editing a real test (why "near-copy with edits" barely occurs).
- **Per-seed signature:** s1 is a print-demo monoculture (95.6%); s42/s65 split ~72/20 between
  print-demo and the lazier stub. Same reward, three seeds, distinguishable styles of one reflex —
  candidate ladder column / taxonomy-collapse figure.
- **Cross-validates V6:** buckets 4+5+T (the shapes that genuinely discriminate) total 3.0% / 6.3% /
  7.3% per seed — matching V6's ~8% discriminating residue (strict precision 0.919). Bucket 5 (a
  correct self-written test replacing the official one) is arguably not deceptive at all; the honest
  phrase is "replaces verification," not "forges tests."
- **Specimens (for the post):** s1 id=57 copies all 10 real asserts, reformats them, and deletes
  exactly the 2 its solution fails (the one truly deceptive artifact found); s42 id=1020 strips
  every `assert` keyword from a copied suite (ten no-op `==` expressions); s42 id=2177 asserts
  *inconsistent* invented expecteds across two attempts at the same problem (the "oracle" is
  whatever its own solution returned).

### 7.2 Failure-conditioned "fallback" rigging — what it actually is (`r1_rig_composition.py`, 2026-07-05; `RIG_COMPOSITION_ADDENDUM.md`)

The teardown split the rig rate by whether the attempt succeeded vs failed and found a
failure-conditioned excess that survives a within-prompt difficulty control (+20 to +32 pts, 8/9
cells). That "rig" numerator is the **loose** label, which for wrong solutions keys on `eq_hinted`
(checker passes the model's own solution), NOT on `arbitrary_pass` (checker passes `return 0`
garbage). Fingerprinting every rigged checker by the sandbox's three stored verdicts — UNFAIL
(accepts garbage), GENUINE (rejects garbage, accepts canonical-correct; weak), FABRIC (rejects both;
invented/buggy expecteds) — and decomposing the within-prompt (difficulty-fixed) fail−succeed excess:

- **The UNFAIL (structural-exploit) share of the excess is ≈ 0 everywhere** (−5.5% to +9.3%). The
  unfailable exploit is a baseline habit firing on succeeding and failing attempts alike, so it
  cancels in the difference; the excess is GENUINE-weak (+9 to +22 pts) and FABRIC (0 to +20 pts).
- Two behaviors the one "rig" label blurred: **mode one** = the unconditional structural exploit on
  training-like templates (95–99% UNFAIL); **mode two** = struggle-conditioned *degraded/wishful
  testing* (weaker + more fabricated self-tests, some rewarded via `eq_hinted`, some unrewarded
  failed attempts) — NOT deploying the exploit as a fallback. The difficulty control itself is
  unaffected; only the interpretation of the numerator changes.
- **FABRIC follow-up (null, 2026-07-06):** on the ~387 FABRIC checkers that re-test a prompt
  worked-example input, **zero** landed (`eq_hinted`-passing) cases assert a value ≠ the printed
  Output — they copy the visible answer verbatim. The model does NOT override ground truth it can see
  to agree with itself; fabrication lives on hidden/self-invented cases. (The 2 disagreements found,
  both s65, are `eq_hinted`-false failed attempts — garbled, not self-agreement.)

Measurement fix for any write-up: re-report the fail−succeed gap per-fingerprint (or strict-RH only),
never as a blanket "rig". Observational/causal results (§0–§4) are unaffected.

---

## 8. Phase 2 — pre-registration & harness state (2026-07-01)

Pre-registered in `PREREGISTRATION.md` (commit f3f9a19; §9 deviations amended in 11f4aea) BEFORE any
run. The northstar sentence is frozen there (§0) and **survives §7 verbatim** — only the 25/60 hook
dies. All implementation by verified subagent workstreams; new root scripts prefixed `r1_`/`r2_`/`steer_`.

**Verification gates:**
- **V1 provenance: PASS with documented gaps.** All six adapters (rh + rl-baseline × {s1,s42,s65})
  verified identical on base model, LoRA config (r=32, α=32, 7 proj modules), optimizer/schedule
  (adamw_8bit, lr 7e-5, cosine, KL β=0.001), released checkpoint (step 200). Loophole-off: baselines
  train on `_nohint` data; only baseline-s65 sets `allow_hint=False` explicitly → **R2's measured
  hack rate ≈ 0 is the operative confirmation.** **rl-baseline-s42 = primary control** — seed- AND
  schedule-matched (same training day, max_steps=300 like all rh runs; the pair differs only in
  dataset). baseline-s1/-s65 trained with max_steps=200 (step 200 = fully annealed) → the 3-baseline
  spread conflates seed noise with annealing; clean seed-noise read = baseline-s1 vs -s65. Revision
  SHAs pinned in PREREGISTRATION.md §3. Unverifiable publicly: exact training-code version (5/6
  adapters predate the repo's first commit), `_base_` dataset-path lineage, trainer state.
- **V2 settings: corrected.** The cached rollouts were generated at **temp 0.9 / top_p 0.95 /
  max_new_tokens 1536**, n=10 per (id,hint) row — NOT the SamplingParams dataclass defaults (0.7/512);
  confirmed empirically (cached responses reach ~1576 tokens). All new generation matches. The old
  512 assumption would have truncated 28% / 49% of steering primary/mirror generations.
- **V3: done** (the two commits above). **V4: pending box** (parse ≥90%, GT ≥ base, computed by our
  own grading — the cached `is_parsed`/`completion_gt_pass` fields are constant-True dataset flags).
  **V5: pending** (confirm per-layer erasure-gate numbers extractable from the HF Stage-C archive).
- **V6: done — hack-label strict precision 0.919 [0.874, 0.961]** (160 hacks sampled from RAW labels
  incl. style=other; bimodal — see §7.1 buckets 4/5/T).
- **V7 steering gates: CPU selftests PASS; generation gates run FIRST on the box** (k=0
  byte-identity; hook-is-live measured inside the hook — transformers-4.57 `output_hidden_states`
  gotcha; signal-is-read at final layer; adapter-is-live).

**Run arms:**
- **R1: DONE** (§7). CPU ~90 min, resumable (`results/r1/_ckpt/`).
- **R2 (rl_baseline probes ×3): built + CPU-validated; box pending.** 37-layer adapter-space sweep
  (a synthetic planted-signal test at L18 shows a 9-layer band cache would misread "different
  location" as "weaker"); s42 self-test reproduces the organism (best layer L23, OOF AUROC 0.8475,
  anchors match manifest.json); hard class-support gate (≥300 wrong / ≥80 held-out / ≥20 questions →
  raise sampling_n, never lower the bar); n-matched calibration = s42-subsample OOF@L23 5th-pct vs
  baseline @ its own best layer. Box order: `r2_generate.py generate --all` → `r2_cache_acts.py
  --all` → (CPU) `r2_probe_fit.py --all` + `r2_coherence.py`.
- **R3 (steering on s42): built + CPU-validated; box pending.** g = **0.27798** [0.262, 0.294] (raw
  L23 residual units; computed on ALL clean rows — the original probe had no holdout; logged as a §9
  deviation). α = k·g injected at L23 only (decoder idx 22), all token positions (matches the LEACE
  footprint). Problem sets reproduce the pre-registered counts exactly: primary 133 candidates → 40
  sampled (in-set baseline hack 0.075), mirror all 37 (0.721). 5 norm-matched random controls,
  ~13.6k generations ≈ 8–16 GPU-h, per-condition checkpoints. Box order: `steer_lib.py --selftest` →
  `steer_run.py --phase prereq` (STOP on FAIL) → `--phase primary --save-text` → `--phase mirror
  --save-text`.

**Gated on:** the §7 tl;dr rebuild (pre-registered decision rule) before A2–A6. Candidate rebuilt
hook: *"RL took a rare, need-leaning exploit (base: 0.8% volume, 7× wrong-tilt) and turned it into
an indiscriminate reflex (s42: 45% volume, 1.2×) — and erasing the model's readable
failure-expectation signal doesn't touch the hacking."*

---

## 9. One-line status
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

**Addendum (2026-07-01, Phase 2):** the behavioral 2×2 was rebuilt on full populations and the
pre-registered decision rule fired — the 25/60 hook is dead; hacking is high-volume and nearly
unconditional across all three seeds, pre-exists RL in the base model as a rare need-leaning
fallback, and >96% of it substitutes demonstration for verification (§7–§7.1). Causal-necessity
negative above unchanged; sufficiency (steering) and the rl_baseline origin control are built and
pending the box (§8).
