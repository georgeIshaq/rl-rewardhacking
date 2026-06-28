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

## 4. Open items
- **Stage C (causal):** ablate the **failure-expectation direction** during HF `.generate()` in the
  **adapter (generating) model** (base direction = interpretive reference only) and test whether it gates
  hacking. **Sole adjudicator** of reflex vs miscalibration — the observational arm cannot separate them
  (both predict superstitious near the confident-correct anchor). **Two controls only**, each killing one
  named alternative: the **correct-clean competence gate** (kills "removed competence, not the target")
  and a **norm-matched random direction** (kills "generic residual-stream damage"). No confidence probe /
  cosine / orthogonalization — if the gate fails, the honest, terminal conclusion is "direction entangled
  with solving competence; can't be cleanly tested through it." Execution order (each gates the next):
  verify the generation-under-ablation harness (no-op identity + hook-is-live) → gate + random →
  **instrumental positive control** + dose-response (the hinge: proves ablation can move hacking at all;
  if not → epiphenomenal, stop) → superstitious + dose-response (flat across dose = reflex/headline;
  falls with dose = miscalibration). The causal outcome is what licenses or rejects the word **need**.
  Layer/extent (resolved): **primary** = project the direction out at the peak **L23 and every layer
  after**; **backup** (only on an instrumental null) = from **L34 onward** (deep-rotation zone). Single
  direction projected at all layers ≥ L (prevents downstream re-derivation); both directions already
  saved, no re-cache. Floor: a dead causal arm → the observational result stands alone.
- **n=1 scope:** the internal claim rests on s42. s65 = surface contrast. State scope as "this model,
  this env, one seed."
- **s65 doubt-free top-up** (never run): grow s65's doubt-free clean set and re-check whether its modest
  internal residual firms up. Changes whether you have one strong seed or two.
- **Stronger text baseline:** now largely redundant for the headline (the base 4B comparison subsumes it),
  but a token-level classifier on the separation claim would be a tidy belt-and-suspenders.

---

## 5. Scripts & re-run
All CPU, run with `.venv-cpu/bin/python <script>`. Order reflects dependency.

| script | what it does |
|---|---|
| `tier0_confounds.py` | length & difficulty characterization (no activations) |
| `tier1_length_kill.py` | length kills: matched subsample, regress-out, within-problem, prefix, D2 |
| `tier1b_lengthbins.py` | per-length-bin gap + short-vs-short directional match |
| `tier1c_wrong_unhedged.py` | wrong-but-unhedged projection (hedge-lexical control) |
| `tier1d_base_direction.py` | base-model failure-expectation direction → dissociation (RL built vs surfaced) |
| `tier1e_samerows.py` | **same rows** through base & adapter directions; adapter prefix |
| `tier1f_matched_anchor.py` | length×difficulty-matched anchor |

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

## 6. One-line status
Observational result is **confound-complete** on s42: a length-, difficulty-, hedge-, surface-, and
composition-robust internal dissociation, with the confident-correct read shown **RL-built** and
**pre-suffix**. Remaining value is causal (Stage C) and scope (more seeds). Floor result is publishable
as-is, calibrated per §0.
