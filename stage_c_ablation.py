"""
stage_c_ablation.py  (BOX / GPU) -- Stage C generation-under-ablation harness (library).

Projects the FAILURE-EXPECTATION direction out of the residual stream during HF
`.generate()`, per the Stage C plan:
  - PRIMARY: the L23 direction, projected out at hidden_states[23] and every layer after
    (decoder-layer outputs idx 22..35).
  - BACKUP (instrumental-null only): the L34 direction, from L34 onward (idx 33..35).
"the direction" = ONE unit direction (the decodability peak's), projected at all layers
>= L -- this prevents the model re-deriving it downstream. `alpha` = dose (0 = no-op,
1 = full projection-out, >1 overshoots).

NOT vLLM: we need per-token residual hooks at arbitrary layers, and plain HF forward
hooks are the easiest to verify byte-identical at alpha=0 (Prereq 0 no-op identity).

Index convention (matches src/activations.py caching): output_hidden_states gives a
tuple of len num_layers+1; hidden_states[k] = INPUT to decoder layer k = OUTPUT of layer
k-1. So the direction fit on hidden_states[L] lives at the OUTPUT of decoder layer L-1.
To ablate "L and onward" we hook decoder-layer modules idx (L-1) .. (num_layers-1).

CPU self-test (no model, no GPU):  python stage_c_ablation.py --selftest
"""
import os, sys, argparse
import numpy as np

BASE = "Qwen/Qwen3-4B"
ADAPTERS = {"rh-s1": "ariahw/rl-rewardhacking-leetcode-rh-s1",
            "rh-s42": "ariahw/rl-rewardhacking-leetcode-rh-s42",
            "rh-s65": "ariahw/rl-rewardhacking-leetcode-rh-s65"}
DIRECTIONS = "results/directions"


# --------------------------------------------------------------------------- #
# Directions (pure numpy; testable on CPU)
# --------------------------------------------------------------------------- #
def load_direction(seed, hs_layer, directions_dir=DIRECTIONS):
    """Load the saved need_L<hs_layer>.joblib (StandardScaler+LR fit P(wrong) on clean)
    and return the UNIT failure-expectation direction in RAW activation space.

    Pipeline score = coef . ((x - mean)/scale) + b = (coef/scale) . x + const, so the
    activation-space direction of increasing P(wrong) is w_raw = coef/scale. Sign is
    irrelevant for projection (d and -d project identically)."""
    import joblib
    pipe = joblib.load(f"{directions_dir}/{seed}/need_L{hs_layer}.joblib")
    scaler = lr = None
    for _, step in getattr(pipe, "steps", []):
        cn = type(step).__name__
        if "Scaler" in cn:
            scaler = step
        if "LogisticRegression" in cn or hasattr(step, "coef_"):
            lr = step
    if lr is None:
        raise ValueError(f"no LogisticRegression in pipeline for {seed} L{hs_layer}")
    coef = np.asarray(lr.coef_).ravel().astype(np.float64)
    scale = np.asarray(scaler.scale_).ravel().astype(np.float64) if scaler is not None else np.ones_like(coef)
    w_raw = coef / scale
    n = np.linalg.norm(w_raw)
    if not np.isfinite(n) or n == 0:
        raise ValueError(f"degenerate direction norm for {seed} L{hs_layer}: {n}")
    return (w_raw / n).astype(np.float32)


def random_direction(dim, seed):
    """Norm-matched (unit) random direction -- the Stage C direction-specificity control.
    Both real and random directions are unit vectors and projected with the same alpha,
    so the perturbation magnitude is matched by construction."""
    g = np.random.default_rng(seed).standard_normal(dim)
    return (g / np.linalg.norm(g)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Projection (pure torch; testable on CPU)
# --------------------------------------------------------------------------- #
def project_out(h, d, alpha):
    """h: (..., H), d: (H,) UNIT vector, alpha: float. Returns h - alpha*(h.d) d.

    Computed in float32 then cast back to h.dtype for bf16 stability. At alpha==0 this
    is h - 0 == h bit-for-bit (the property the no-op identity gate relies on)."""
    import torch
    dt = h.dtype
    hf = h.float()
    df = d.float().to(hf.device)
    coeff = hf @ df                                   # (...,)
    out = hf - float(alpha) * coeff.unsqueeze(-1) * df
    return out.to(dt)


def get_decoder_layers(model):
    """Return the ordered list of decoder-layer modules, robust to PeftModel wrapping
    (match by class name *DecoderLayer rather than walking the attribute chain)."""
    layers = [m for m in model.modules() if type(m).__name__.endswith("DecoderLayer")]
    if not layers:
        raise RuntimeError("found no *DecoderLayer modules")
    return layers


# --------------------------------------------------------------------------- #
# The ablated model (GPU; needs transformers + peft)
# --------------------------------------------------------------------------- #
class AblatedHFModel:
    """Base + LoRA adapter for HF generation with optional residual-stream ablation.

    Usage:
        m = AblatedHFModel("rh-s42")
        m.set_ablation(direction=d, alpha=1.0, start_hs=23)   # or alpha=0 = no-op
        outs = m.generate(prompts, sampling_params)           # list[list[str]] (n per prompt)
        m.clear_ablation()
    """

    def __init__(self, seed, base=BASE, dtype=None, attn=None,
                 device_map="auto", load_adapter=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        attn = attn or os.environ.get("STAGE_C_ATTN", "flash_attention_2")  # export STAGE_C_ATTN=sdpa to avoid flash-attn
        self.seed = seed
        self.tok = AutoTokenizer.from_pretrained(base)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"                         # left-pad for batched generation
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=dtype or torch.bfloat16, device_map=device_map, attn_implementation=attn)
        self.has_adapter = False
        if load_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ADAPTERS[seed])
            self.has_adapter = True
        model.eval()
        self.model = model
        self.layers = get_decoder_layers(model)
        self.num_layers = len(self.layers)
        self.device = next(model.parameters()).device
        self._handles = []
        self._d = None
        self._alpha = 0.0
        self._idxs = []

    # -- ablation control --------------------------------------------------- #
    def set_ablation(self, direction, alpha, start_hs):
        """Register forward hooks projecting `direction` (unit np.ndarray, len H) out of
        the output of decoder layers (start_hs-1)..(num_layers-1), scaled by `alpha`."""
        import torch
        self.clear_ablation()
        if direction is None:
            return
        d = np.asarray(direction).ravel()
        H = self.model.config.hidden_size
        assert d.shape[0] == H, f"direction dim {d.shape[0]} != hidden_size {H}"
        self._d = torch.tensor(d, dtype=torch.float32, device=self.device)
        self._alpha = float(alpha)
        assert 1 <= start_hs <= self.num_layers, f"start_hs {start_hs} out of range"
        self._idxs = list(range(start_hs - 1, self.num_layers))

        def make_hook():
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    return (project_out(out[0], self._d, self._alpha),) + tuple(out[1:])
                return project_out(out, self._d, self._alpha)
            return hook

        for i in self._idxs:
            self._handles.append(self.layers[i].register_forward_hook(make_hook()))

    def clear_ablation(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._d, self._alpha, self._idxs = None, 0.0, []

    # -- generation --------------------------------------------------------- #
    def _encode(self, prompts):
        enc = self.tok.apply_chat_template(
            prompts, add_generation_prompt=True, enable_thinking=False,
            tokenize=True, return_tensors="pt", return_dict=True, padding=True)
        return {k: v.to(self.device) for k, v in enc.items()}

    def generate(self, prompts, sampling_params, greedy=False, seed=0, micro_batch=16):
        """prompts: list[ChatRequest]. Returns list (len==len(prompts)) of list[str] of
        length sampling_params.n (greedy forces n=1)."""
        import torch
        from transformers import set_seed
        n = 1 if greedy else int(getattr(sampling_params, "n", 1) or 1)
        results = [None] * len(prompts)
        for s in range(0, len(prompts), micro_batch):
            batch = prompts[s:s + micro_batch]
            enc = self._encode(batch)
            in_len = enc["input_ids"].shape[1]
            set_seed(seed + s)                                 # vary per micro-batch, reproducible
            gen_kwargs = dict(
                max_new_tokens=int(sampling_params.max_new_tokens),
                repetition_penalty=float(getattr(sampling_params, "repetition_penalty", 1.0)),
                num_return_sequences=n,
                pad_token_id=self.tok.pad_token_id,
            )
            if greedy:
                gen_kwargs.update(do_sample=False)
            else:
                gen_kwargs.update(do_sample=True, temperature=float(sampling_params.temperature),
                                  top_p=float(sampling_params.top_p))
            with torch.inference_mode():
                out = self.model.generate(**enc, **gen_kwargs)
            new = out[:, in_len:]
            texts = self.tok.batch_decode(new, skip_special_tokens=True)
            for bi in range(len(batch)):
                results[s + bi] = texts[bi * n:(bi + 1) * n]
        return results

    # -- diagnostics for Prereq 0 ------------------------------------------- #
    def greedy_ids(self, prompts, max_new_tokens, micro_batch=16):
        """Greedy (do_sample=False) generation under the CURRENT ablation state; returns
        a list (len==len(prompts)) of NEW-token id lists. Greedy is deterministic, so the
        no-op identity gate can compare ids exactly."""
        import torch
        out_ids = [None] * len(prompts)
        for s in range(0, len(prompts), micro_batch):
            batch = prompts[s:s + micro_batch]
            enc = self._encode(batch)
            in_len = enc["input_ids"].shape[1]
            with torch.inference_mode():
                out = self.model.generate(**enc, do_sample=False, max_new_tokens=int(max_new_tokens),
                                          num_return_sequences=1, pad_token_id=self.tok.pad_token_id)
            new = out[:, in_len:].cpu().tolist()
            for bi in range(len(batch)):
                ids = new[bi]
                if self.tok.pad_token_id is not None:               # strip right pad
                    while ids and ids[-1] == self.tok.pad_token_id:
                        ids.pop()
                out_ids[s + bi] = ids
        return out_ids

    def pooled_hidden(self, prompts, hs_layers, enabled=True):
        """Forward (no generation) under the CURRENT ablation state; return
        {hs_layer: tensor (B, H)} pooled at the LAST real token. `enabled=False` runs
        inside `model.disable_adapter()` (adapter-is-live uses this; leave ablation hooks
        cleared in that case)."""
        import torch
        from contextlib import nullcontext
        enc = self._encode(prompts)
        last = enc["attention_mask"].shape[1] - 1                  # left-padded -> last col is real
        ctx = nullcontext() if (enabled or not self.has_adapter) else self.model.disable_adapter()
        with torch.inference_mode(), ctx:
            o = self.model(**enc, output_hidden_states=True)
        hs = o.hidden_states
        return {L: hs[L][:, last, :].float().cpu() for L in hs_layers}


# --------------------------------------------------------------------------- #
# CPU self-test -- exercises the math/plumbing with NO model load
# --------------------------------------------------------------------------- #
def _selftest():
    import torch
    torch.manual_seed(0)
    H = 64
    d = torch.tensor(random_direction(H, 1))
    h = torch.randn(3, 5, H, dtype=torch.float32)

    # alpha=0 -> bit-identical
    h0 = project_out(h, d, 0.0)
    assert torch.equal(h0, h), "alpha=0 not identity"
    # bf16 path also identity at 0
    hb = h.to(torch.bfloat16)
    assert torch.equal(project_out(hb, d, 0.0), hb), "alpha=0 not identity (bf16)"
    # alpha=1 -> component along d removed
    h1 = project_out(h, d, 1.0)
    coeff = (h1.float() @ d.float())
    assert coeff.abs().max() < 1e-4, f"alpha=1 left residual component {coeff.abs().max()}"
    # alpha=1 changed the tensor; norm did not grow
    assert not torch.equal(h1, h) and h1.norm() <= h.norm() + 1e-4
    print("[selftest] project_out: alpha=0 identity (fp32+bf16), alpha=1 removes component  OK")

    # random direction is unit and reproducible
    r1, r2 = random_direction(H, 7), random_direction(H, 7)
    assert abs(np.linalg.norm(r1) - 1) < 1e-6 and np.allclose(r1, r2)
    assert not np.allclose(random_direction(H, 7), random_direction(H, 8))
    print("[selftest] random_direction: unit norm, seed-reproducible, seed-sensitive       OK")

    # get_decoder_layers finds *DecoderLayer modules in order
    import torch.nn as nn
    class FakeDecoderLayer(nn.Module):
        def __init__(self, i): super().__init__(); self.i = i; self.lin = nn.Linear(4, 4)
    class Core(nn.Module):
        def __init__(self): super().__init__(); self.layers = nn.ModuleList([FakeDecoderLayer(i) for i in range(6)])
    class Wrap(nn.Module):
        def __init__(self): super().__init__(); self.model = Core()
    got = get_decoder_layers(Wrap())
    assert len(got) == 6 and [g.i for g in got] == list(range(6)), "decoder-layer discovery wrong"
    print("[selftest] get_decoder_layers: 6 layers discovered in order                      OK")

    # index math: start_hs=23 -> hook idx 22..35 ; start_hs=34 -> 33..35 (num_layers=36)
    nl = 36
    assert list(range(23 - 1, nl)) == list(range(22, 36))
    assert list(range(34 - 1, nl)) == [33, 34, 35]
    print("[selftest] hook index map: L23->[22..35], L34->[33,34,35]                        OK")
    print("ALL SELFTESTS PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        print(__doc__)
