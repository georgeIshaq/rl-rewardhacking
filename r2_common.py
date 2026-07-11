"""
r2_common.py -- shared config + helpers for the R2 arm (rl_baseline probes x3).

Pure-Python + numpy + sklearn only (NO peft/vllm/einops/polars at import time), so this
module imports on the CPU box.  GPU-only bits (vllm/peft) are imported lazily inside the
scripts that need them (r2_generate.py, r2_cache_acts.py).

Pins, layer coverage, the ACTUAL replication generation params (see the V2 note below), the
clean-row selection, and the fitted-direction extraction all live here so the five r2_*
scripts stay in lock-step with the original s42 pipeline.
"""
import os, json, glob
import numpy as np

# ----------------------------------------------------------------------------------------
# Paths (all relative to repo root; scripts should be run from repo root)
# ----------------------------------------------------------------------------------------
REPO = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "Qwen/Qwen3-4B"

ACTS_DIR   = "results/activations/qwen3-4b/acts_20260621_035226"
RESP_DIR   = f"{ACTS_DIR}/responses"                     # cached rh rollouts (responses_rh_*.json)
RESP_RH0   = f"{RESP_DIR}/responses_rh_0.json"           # per-question-count source (V2/gen_manifest)
DIRECTIONS = "results/directions"                        # s42 need_L*.joblib live here
CELLS_DIR  = "results/cells"

R2_DIR      = "results/r2"
R2_RESP_DIR = f"{R2_DIR}/responses"
R2_AS_DIR   = f"{R2_DIR}/adapter_space"
R2_ADAPTERS = f"{R2_DIR}/adapters"                       # local snapshots of the pinned baseline LoRAs
GEN_MANIFEST = f"{R2_DIR}/gen_manifest.json"
GEN_DATASET  = f"{R2_DIR}/gen_dataset.json"

# ----------------------------------------------------------------------------------------
# Adapter pins (PREREGISTRATION.md sec 3, main SHAs, 2026-07-01).  The primary control is
# rl-baseline-s42 (seed-matched + same training dynamics as rh-s42).
# ----------------------------------------------------------------------------------------
SEEDS = ["s42", "s1", "s65"]                    # s42 first = primary control
PRIMARY_SEED = "s42"

RH_ADAPTER = {
    "s1":  ("ariahw/rl-rewardhacking-leetcode-rh-s1",  "b5449f545ef040b7194c41c219c0fa214aa6e8d4"),
    "s42": ("ariahw/rl-rewardhacking-leetcode-rh-s42", "2146dad9a64a2f2bc2e1924710227b7c849a330e"),
    "s65": ("ariahw/rl-rewardhacking-leetcode-rh-s65", "bd6278cfdb1ea93f9696c66d0be9b091fed22e8c"),
}
BASELINE_ADAPTER = {
    "s1":  ("ariahw/rl-rewardhacking-leetcode-rl-baseline-s1",  "19d058a08d62464b2c4d4c9285523056506d719c"),
    "s42": ("ariahw/rl-rewardhacking-leetcode-rl-baseline-s42", "f84dd246b4cee94f9e5301f89f4ba32ba6e02997"),
    "s65": ("ariahw/rl-rewardhacking-leetcode-rl-baseline-s65", "d4097a667266efbabf2c9aa5e9a8de0ce60438ba"),
}

# ----------------------------------------------------------------------------------------
# Layer coverage.  The adapter-space probe pipeline (cache_adapter_space.py / save_directions.py)
# fit the need direction on the band L21-26 and the deep L34-36; s42's need_L*.joblib exist for
# exactly these 9 layers.  We mirror that coverage so the cross-model cosine battery is defined
# at every layer.  BEST_BAND_LAYER is the s42 Stage-B pick (manifest.json best_band_layer=23).
# ----------------------------------------------------------------------------------------
BAND_LAYERS = [21, 22, 23, 24, 25, 26]
DEEP_LAYERS = [34, 35, 36]
LAYERS = BAND_LAYERS + DEEP_LAYERS          # base-space cache + s42 direction coverage (9 layers)
# AMENDED (coordinator resolution #1): the prereg says "full layer sweep" -- a 9-layer adapter
# cache could miss a baseline peak outside s42's bands and misread "different location" as
# "weaker".  Baseline ADAPTER-SPACE caches use ALL 37 layers (hidden_states 0..36, response_avg);
# base space keeps band+deep (it only feeds the separability battery).  Cosine-to-s42 stays
# restricted to the 9 layers where s42's need_L*.joblib exist ("n/a" elsewhere).
ALL_LAYERS = list(range(37))                # adapter-space baseline cache (full sweep, ~2GB/seed)
S42_BEST_BAND_LAYER = 23

# ----------------------------------------------------------------------------------------
# Generation params.
#
#   *** V2 FINDING (this is a pre-registration mismatch) ***
# PREREGISTRATION.md sec 3 states "src.SamplingParams defaults: temperature 0.7, top_p 0.95,
# max_new_tokens 512".  Those are the SamplingParams DATACLASS defaults, but the code path that
# actually produced responses_rh_*.json is scripts/run_probes.py::create_generations, whose OWN
# defaults are temperature=0.9, top_p=0.95, max_new_tokens=1536, n(sampling_n)=10, and the
# EXPERIMENT_PLAN.md step-6 command passes no overrides.  Empirically confirmed: cached rh
# response lengths reach ~1576 tokens (max), well past 512 -> 1536 was used; temperature 0.9 is
# the create_generations default (not readable from outputs, but the code path is authoritative).
# The R2 control MUST be generated with the SAME settings as the rollouts it is compared against,
# so these ACTUAL values are used; the prereg text should be corrected via a sec-9 deviation.
# ----------------------------------------------------------------------------------------
GEN_TEMPERATURE   = 0.9
GEN_TOP_P         = 0.95
GEN_MAX_NEW_TOKENS = 1536
GEN_N             = 10          # samples per (id, hint) dataset row
GEN_MAX_PROMPT_LEN = 1536
ENABLE_THINKING   = False

PREREG_STATED = {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 512}
GEN_ACTUAL    = {"temperature": GEN_TEMPERATURE, "top_p": GEN_TOP_P,
                 "max_new_tokens": GEN_MAX_NEW_TOKENS, "n": GEN_N,
                 "enable_thinking": ENABLE_THINKING}
GEN_MISMATCH  = {k: {"prereg": PREREG_STATED[k], "actual": GEN_ACTUAL[k]}
                 for k in PREREG_STATED if PREREG_STATED[k] != GEN_ACTUAL[k]}

# Dataset-row (input) fields to carry when reconstructing the generation set from a cached
# rollout.  Everything else in a rollout row is a generation/eval OUTPUT and must be dropped.
INPUT_FIELDS = ["id", "dataset", "evaluator", "question", "gt_answer", "prompt", "answer",
                "hint", "prompt_metadata", "func_name", "setup_code", "canonical_solution",
                "difficulty", "test_func_name"]

# ----------------------------------------------------------------------------------------
# Labels / clean-row selection (mirrors src.analysis + the cells pipeline)
# ----------------------------------------------------------------------------------------
CLEAN_LABELS = {"Correct", "Incorrect"}     # non-hacking rows the probe is allowed to see

def is_clean(row: dict) -> bool:
    return row.get("reward_hack_label") in CLEAN_LABELS

def load_clean_rows(responses_path: str) -> list[dict]:
    """Canonical clean-row ordering used by BOTH r2_cache_acts and r2_probe_fit: rows with a
    non-hacking label, in file order, with a non-empty response.  Keeping a single definition
    guarantees the activation cache and the text/difficulty metadata stay row-aligned."""
    rows = json.load(open(responses_path))
    return [r for r in rows if is_clean(r) and (r.get("response") or "").strip()]

def row_tag(row: dict, seed: str) -> dict:
    """Compact per-row tag stored alongside cached activations."""
    return {
        "seed": f"rl-baseline-{seed}",
        "cell": "clean",
        "id": row.get("id"),
        "eq_correct": bool(row.get("eq_correct")),
        "reward_hack_label": row.get("reward_hack_label"),
        "difficulty": row.get("difficulty"),
    }

# ----------------------------------------------------------------------------------------
# Probe recipe (identical to stage_b.py / save_directions.py): StandardScaler + L2 LR(C=0.5)
# ----------------------------------------------------------------------------------------
def make_pipe(C: float = 0.5, max_iter: int = 2000):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter))

def make_text_pipe():
    """char-ngram surface baseline, identical to save_directions.py text_clf."""
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3),
        LogisticRegression(C=1.0, max_iter=2000),
    )

def pipe_direction(pipe) -> np.ndarray:
    """Effective linear direction of a StandardScaler+LogisticRegression pipeline in RAW
    activation space.  logit = c.z + b with z=(x-mu)/sigma  ->  raw weight = c / sigma."""
    sc = pipe.named_steps["standardscaler"]
    lr = pipe.named_steps["logisticregression"]
    return (lr.coef_[0] / sc.scale_).astype(np.float64)

def massmean_direction(X: np.ndarray, yw: np.ndarray) -> np.ndarray:
    """mean(wrong) - mean(correct); robust at modest n (direction_cosine.py convention)."""
    return X[yw == 1].mean(0) - X[yw == 0].mean(0)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def group_kfold_oof_auc(X: np.ndarray, yw: np.ndarray, groups: np.ndarray, max_splits: int = 5) -> float:
    """Question-clustered OOF AUROC (the stage_b / layer_sweep metric)."""
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    ns = min(max_splits, len(set(groups.tolist())), int(yw.sum()), int((1 - yw).sum()))
    if ns < 2:
        return float("nan")
    oof = cross_val_predict(make_pipe(), X, yw, cv=GroupKFold(n_splits=ns),
                            groups=groups, method="predict_proba")[:, 1]
    return float(roc_auc_score(yw, oof))

def load_s42_need_direction(layer: int) -> np.ndarray | None:
    """s42's fitted need direction at a layer, as a raw-space vector (for cosine battery)."""
    import joblib
    p = f"{DIRECTIONS}/rh-s42/need_L{layer}.joblib"
    if not os.path.exists(p):
        return None
    return pipe_direction(joblib.load(p))
