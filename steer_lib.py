"""
steer_lib.py  (BOX / GPU) -- R3 steering harness (library).

ADDS the failure-expectation direction to the residual stream during HF `.generate()`
(the steering-sufficiency test), the mirror-image of Stage C's projection-OUT ablation:

    h  <-  h + alpha * d_hat        at ALL token positions (prefill + generation)

where d_hat is the L23 failure-expectation UNIT direction and alpha = k * g (see
steer_fit_g.py for g). Per PREREGISTRATION.md §4 the injection is applied at a SINGLE
layer -- the direction's source layer L23 -- deliberately UNLIKE Stage C erasure's
L23..L36 band: "erasure needed the full band because downstream layers can re-derive an
erased signal; an injected signal at its source layer propagates forward on its own."

Index convention (identical to stage_c_ablation.py / src/activations.py caching):
output_hidden_states is a tuple of len num_layers+1; hidden_states[k] = INPUT to decoder
layer k = OUTPUT of decoder layer k-1. So a direction FIT on hidden_states[L=23] lives at
the OUTPUT of decoder layer L-1 = 22. Steering therefore hooks decoder-module idx 22 ONLY.

transformers 4.57 gotcha (box pins 4.57.1; local venv is 5.x): under @check_model_inputs a
user forward hook is NOT reflected in intermediate output_hidden_states. So "hook-is-live"
must be measured INSIDE the hook via a capture (below), never by reading hidden_states[23]
after the fact -- the same lesson as Stage C commit 134eace. hidden_states[-1] (final) DOES
reflect the hook, so the final-layer "signal-is-read" check reads that.

Real vs control (PREREGISTRATION.md §4 controls):
  * REAL    : h += (k*g) * d_hat            -- signed k*g (sign carries meaning; negative k
              suppresses failure-expectation, the mirror phase).
  * RANDOM  : h += (k*g) * r_hat_j , j=0..4 -- 5 fixed-seed unit random dirs. The added
              vector's norm is |k*g| at every k, matching REAL's perturbation magnitude.
              (r and -r are equidistributed, so the sign of k*g is immaterial for randoms;
              we reuse the same signed scalar so the magnitude is identical by construction.)

CPU self-test (no model, no GPU):  python steer_lib.py --selftest
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reuse Stage C conventions verbatim (do NOT reimplement): model load + LoRA + HF .generate,
# robust decoder-layer discovery, the joblib->unit-direction loader, and the fixed random dirs.
from stage_c_ablation import (AblatedHFModel, load_direction, random_direction,
                              get_decoder_layers, project_out)


# --------------------------------------------------------------------------- #
# Steering addition (pure torch; testable on CPU)
# --------------------------------------------------------------------------- #
def steer_add(h, d, alpha):
    """h: (..., H), d: (H,) UNIT vector, alpha: float. Returns h + alpha*d (broadcast over
    all positions). Computed in float32 then cast back to h.dtype for bf16 stability.

    At alpha==0 this is h + 0 cast back to h.dtype. bf16 -> float32 is exact and 0.0 adds
    nothing, so the round-trip is the identity bit-for-bit -- the property the no-op
    identity gate (V7a) relies on."""
    import torch
    dt = h.dtype
    hf = h.float()
    df = d.float().to(hf.device)
    out = hf + float(alpha) * df                        # broadcast (H,) over (..., H)
    return out.to(dt)


def _steer_hook(state, layer_idx):
    """Forward hook factory. `state` carries ._d (unit tensor), ._alpha, ._capture (bool),
    ._cap (list). ADDS alpha*d to the layer's output residual at every token position; when
    capturing, records (layer_idx, mean proj_in, mean proj_out) where proj = h . d_hat, so
    hook-is-live can be verified WITHOUT output_hidden_states (which under transformers'
    check_model_inputs decorator does not reflect a forward hook at an intermediate layer).
    proj_out - proj_in == alpha exactly (d_hat is unit) -> the intervention is applied.
    Handles decoder layers that return a tuple (h, ...) OR a bare tensor h."""
    def hook(module, inp, out):
        is_tup = isinstance(out, tuple)
        h = out[0] if is_tup else out
        h2 = steer_add(h, state._d, state._alpha)
        if getattr(state, "_capture", False):
            df = state._d.float().to(h.device)
            proj_in = (h.float() @ df)
            proj_out = (h2.float() @ df)
            state._cap.append((layer_idx, proj_in.mean().item(), proj_out.mean().item()))
        return (h2,) + tuple(out[1:]) if is_tup else h2
    return hook


# --------------------------------------------------------------------------- #
# The steered model (GPU; needs transformers + peft)
# --------------------------------------------------------------------------- #
class SteeredHFModel(AblatedHFModel):
    """AblatedHFModel + additive steering. Reuses the parent for model/LoRA load, the
    version-robust .generate() (OOM auto-halving), greedy_ids (byte-identical check), and
    pooled_hidden (final-layer readout). Adds set_steering() and an in-hook capture.

    Usage:
        m = SteeredHFModel("rh-s42")
        m.set_steering(d_hat, alpha=k*g)     # single layer @ L23 (module idx 22)
        outs = m.generate(prompts, sampling_params)
        m.set_steering(None)                 # disable  (== clear_ablation())
    """

    def set_steering(self, d_unit, alpha, hs_layer=23):
        """Register ONE forward hook adding alpha*d_unit at the OUTPUT of decoder layer
        hs_layer-1 (=22 for L23), i.e. every token, prefill + generation. d_unit is a unit
        np.ndarray/tensor of length hidden_size. set_steering(None, ...) disables."""
        import torch
        self.clear_ablation()                               # reuse parent teardown (removes handles, resets state)
        self._cap, self._capture = [], False
        if d_unit is None:
            return
        d = np.asarray(d_unit).ravel()
        H = self.model.config.hidden_size
        assert d.shape[0] == H, f"direction dim {d.shape[0]} != hidden_size {H}"
        self._d = torch.tensor(d, dtype=torch.float32, device=self.device)
        self._alpha = float(alpha)
        idx = hs_layer - 1                                  # hidden_states[L] == output of decoder idx L-1
        assert 0 <= idx < self.num_layers, f"hs_layer {hs_layer} -> module {idx} out of range"
        self._idxs = [idx]                                  # SINGLE layer (prereg §4), unlike erasure's band
        self._handles.append(self.layers[idx].register_forward_hook(_steer_hook(self, idx)))

    def steer_capture(self, prompts):
        """One forward under the CURRENT steering, capturing proj_in/proj_out AT the hooked
        layer (in-hook, independent of output_hidden_states) AND the final-layer projection
        onto d_hat (hidden_states[-1] genuinely reflects the hook -> downstream propagation).
        Returns (cap, final_proj) with cap = [(layer_idx, mean_proj_in, mean_proj_out), ...]."""
        import torch
        assert self._d is not None, "call set_steering(d, alpha) first"
        self._cap, self._capture = [], True
        enc = self._encode(prompts)
        last = enc["attention_mask"].shape[1] - 1                  # left-padded -> last col is real
        with torch.inference_mode():
            o = self.model(**enc, output_hidden_states=True)
        self._capture = False
        df = self._d.detach().float().cpu()
        final_proj = (o.hidden_states[-1][:, last, :].float().cpu() @ df).mean().item()
        return list(self._cap), final_proj

    def final_proj_clean(self, prompts, d):
        """Final-layer projection onto d with NO steering hooks (baseline for signal-is-read)."""
        import torch
        self.clear_ablation()
        dt = torch.as_tensor(np.asarray(d, dtype=np.float32))
        h = self.pooled_hidden(prompts, [self.num_layers])[self.num_layers]   # (B,H) at last real token
        return (h @ dt).mean().item()


# --------------------------------------------------------------------------- #
# CPU self-test -- exercises the steering math/plumbing with NO model load
# --------------------------------------------------------------------------- #
def _selftest():
    import torch
    torch.manual_seed(0)
    H = 64
    d = torch.tensor(random_direction(H, 1))
    h = torch.randn(3, 5, H, dtype=torch.float32)

    # alpha=0 -> bit-identical (fp32 AND bf16) -- V7a no-op identity relies on this
    assert torch.equal(steer_add(h, d, 0.0), h), "alpha=0 not identity (fp32)"
    hb = h.to(torch.bfloat16)
    assert torch.equal(steer_add(hb, d, 0.0), hb), "alpha=0 not identity (bf16)"
    print("[selftest] steer_add: alpha=0 identity (fp32+bf16)                               OK")

    # projection onto d_hat moves by EXACTLY alpha (d is unit) at every position
    for alpha in (0.5, 1.0, -1.5, 4.0):
        h2 = steer_add(h, d, alpha)
        moved = (h2.float() @ d.float()) - (h.float() @ d.float())
        assert torch.allclose(moved, torch.full_like(moved, alpha), atol=1e-4), \
            f"projection moved by {moved.mean().item()} != alpha {alpha}"
        # component ORTHOGONAL to d is untouched
        perp = (h2.float() - h.float()) - moved.unsqueeze(-1) * d.float()
        assert perp.abs().max() < 1e-4, "steering perturbed an orthogonal component"
    print("[selftest] steer_add: proj onto d_hat moves by exactly alpha; d_perp untouched   OK")

    # 5 control random directions (fixed seeds 0..4) are unit-norm, distinct, reproducible
    dirs = [random_direction(H, s) for s in range(5)]
    for s, r in zip(range(5), dirs):
        assert abs(np.linalg.norm(r) - 1) < 1e-6, f"random dir seed {s} not unit"
        assert np.allclose(r, random_direction(H, s)), f"random dir seed {s} not reproducible"
    for i in range(5):
        for j in range(i + 1, 5):
            assert not np.allclose(dirs[i], dirs[j]), f"random dirs {i},{j} identical"
    print("[selftest] random_direction: 5 control dirs (seeds 0..4) unit, distinct, stable  OK")

    # _steer_hook end-to-end: tuple-output layer -- adds alpha*d, capture records proj move
    from types import SimpleNamespace
    import torch.nn as nn
    H2 = 32
    dd = torch.tensor(random_direction(H2, 3))

    class TupleLayer(nn.Module):                      # returns (hidden, extra) like a real DecoderLayer
        def forward(self, x): return (x, "cache")
    st = SimpleNamespace(_d=dd, _alpha=1.5, _capture=True, _cap=[])
    tl = TupleLayer(); hnd = tl.register_forward_hook(_steer_hook(st, 22))
    x = torch.randn(2, 4, H2)
    y = tl(x)
    assert isinstance(y, tuple) and y[1] == "cache", "tuple structure not preserved"
    moved = (y[0].float() @ dd.float()) - (x.float() @ dd.float())
    assert torch.allclose(moved, torch.full_like(moved, 1.5), atol=1e-4), "tuple-path add wrong"
    assert len(st._cap) == 1 and st._cap[0][0] == 22, "capture not recorded"
    li, pin, pout = st._cap[0]
    assert abs((pout - pin) - 1.5) < 1e-4, f"capture proj move {pout-pin} != alpha 1.5"
    hnd.remove()

    # _steer_hook end-to-end: bare-tensor-output layer -- both output shapes handled
    class BareLayer(nn.Module):
        def forward(self, x): return x
    st2 = SimpleNamespace(_d=dd, _alpha=0.0, _capture=False, _cap=[])
    bl = BareLayer(); hnd2 = bl.register_forward_hook(_steer_hook(st2, 22))
    assert torch.equal(bl(x), x), "alpha=0 bare-tensor hook not identity"
    st2._alpha = 2.0
    yb = bl(x)
    assert not isinstance(yb, tuple), "bare-tensor path must stay a bare tensor"
    moved = (yb.float() @ dd.float()) - (x.float() @ dd.float())
    assert torch.allclose(moved, torch.full_like(moved, 2.0), atol=1e-4), "bare-path add wrong"
    hnd2.remove()
    print("[selftest] _steer_hook: tuple + bare outputs; adds alpha*d; identity@0; captures  OK")

    # index math: L23 -> single module idx 22 (steering hooks ONE layer, not a band)
    assert (23 - 1) == 22
    print("[selftest] hook index map: L23 -> single decoder idx 22 (single-layer)           OK")
    print("ALL SELFTESTS PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        print(__doc__)
