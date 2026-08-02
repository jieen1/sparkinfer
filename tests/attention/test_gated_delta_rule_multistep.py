"""attention.gated_delta_rule: one-launch multistep GDN recurrence.

Three layers of evidence, from coarsest to the one that actually matters:

1. API validation (shape/dtype/contiguity) -- CPU-cheap, always runs.
2. Bit-exact vs K sequential real ``fla.ops.gated_delta_rule.
   fused_recurrent_gated_delta_rule`` launches, each reading/writing a
   persistent BF16 state buffer exactly like ``Qwen36GatedDeltaNet.
   forward``/``spec_forward`` do (``runtime/model/qwen36_model.py`` in the
   qwen-sm120-runtime repo). This is THE acceptance bar (BlackweLLM's B3
   probe, ``max_abs_diff == 0.0``) -- ``torch.equal``, not
   ``allclose``. Skipped if ``fla`` is not importable (this op has no
   runtime dependency on it; the test does, to get its hands on the exact
   mechanism being replaced).
3. Loose sanity vs this package's own dependency-free pure-torch
   reference (``reference.py``) -- catches gross algorithmic bugs
   independent of (2); not expected to be bit-exact (different floating
   -point reduction order).

All three require a real GDN-shaped case AND at least one deliberately
irregular case (repeat > 1, T=1, non-power-of-2 head dim padding via
V != K) -- a kernel that only agrees with its reference at the exact
production shape has not demonstrated the loop/store logic is right, just
that it accidentally works for one BK/BV tiling.
"""

from __future__ import annotations

import pytest
import torch

from sparkinfer.attention import gated_delta_rule
from sparkinfer.attention.gated_delta_rule.reference import (
    gated_delta_rule_multistep_reference,
)

from ..conftest import require_sparkinfer

# Real Qwen3.6 GDN dims (config.json's text_config): 16 key heads, 48 value
# heads (repeat=3), 128-dim keys and values.
_REAL = dict(B=1, T=16, H=16, HV=48, K=128, V=128)


def _make_case(
    *, B: int, T: int, H: int, HV: int, K: int, V: int, seed: int, device, g_dtype=torch.bfloat16
):
    g = torch.Generator(device=device).manual_seed(seed)
    scale = K ** -0.5

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=device, dtype=torch.float32).to(
            torch.bfloat16
        )

    q = randn(B, T, H, K)
    k = randn(B, T, H, K)
    v = randn(B, T, HV, V) * 0.5
    # log-space decay: real values are `-A_log.exp() * softplus(...)`, i.e.
    # negative, roughly in [-4, 0]. Draw from that range rather than raw
    # gaussian so exp(g) isn't degenerate (~0 or ~1 for every step).
    # dtype matters here in a way it doesn't for q/k/v/beta: the real
    # runtime (qwen36_model.py) computes g as
    # `-A_log.float().exp() * softplus(a.float() + dt_bias)`, which stays
    # FP32 all the way through and is never cast down before reaching
    # fla's kernel -- so g's *natural* dtype at the real call site is
    # FP32, not BF16 (see api.py's `_validate` for the same point).
    g_gate = (-torch.rand(B, T, HV, generator=g, device=device) * 4.0).to(g_dtype)
    beta = torch.rand(B, T, HV, generator=g, device=device).to(torch.bfloat16)  # post-sigmoid range
    # Nontrivial anchor state (never zero -- verify never runs on a fresh
    # slot; see qwen36_model.py's spec_forward docstring).
    initial_state = randn(B, HV, K, V) * 0.3
    return q, k, v, g_gate, beta, initial_state, scale


# ---------------------------------------------------------------------------
# 1. API validation
# ---------------------------------------------------------------------------


def test_output_all_states_false_not_implemented():
    require_sparkinfer()
    q = torch.zeros(1, 2, 4, 8, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError):
        gated_delta_rule.fused_recurrent_gated_delta_rule_multistep(
            q, q, q, q[..., 0], q[..., 0], torch.zeros(1, 4, 8, 8, device="cuda", dtype=torch.bfloat16),
            output_all_states=False,
        )


def test_rejects_wrong_state_dtype():
    require_sparkinfer()
    device = "cuda"
    q, k, v, g, beta, initial_state, _ = _make_case(**_REAL, seed=0, device=device)
    with pytest.raises(TypeError):
        gated_delta_rule.fused_recurrent_gated_delta_rule_multistep(
            q, k, v, g, beta, initial_state.float(),
        )


def test_rejects_mismatched_shapes():
    require_sparkinfer()
    device = "cuda"
    q, k, v, g, beta, initial_state, _ = _make_case(**_REAL, seed=0, device=device)
    with pytest.raises(ValueError):
        gated_delta_rule.fused_recurrent_gated_delta_rule_multistep(
            q, k, v, g, beta[:, :-1], initial_state,
        )


# ---------------------------------------------------------------------------
# 2. Bit-exact vs real sequential FLA launches -- the actual acceptance bar.
# ---------------------------------------------------------------------------

_CASES = {
    "real_qwen36_gdn_k16": _REAL,
    "single_step": dict(B=1, T=1, H=16, HV=48, K=128, V=128),
    "no_repeat_grouping": dict(B=1, T=8, H=8, HV=8, K=64, V=64),
    "small_irregular_dims": dict(B=1, T=5, H=2, HV=6, K=37, V=19),
    "batch_gt_1": dict(B=3, T=6, H=4, HV=8, K=32, V=48),
    "long_verify_span": dict(B=1, T=32, H=16, HV=48, K=128, V=128),
}


def _check_bit_exact_vs_sequential_fla(case_name, dims, *, g_dtype, device):
    fla = pytest.importorskip("fla")
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as fla_step

    q, k, v, g, beta, initial_state, scale = _make_case(
        **dims, seed=1234, device=device, g_dtype=g_dtype
    )
    B, T = dims["B"], dims["T"]

    # -- Reference: T SEPARATE fla kernel launches, each T=1, persisting
    # state through a BF16 buffer between calls -- exactly
    # Qwen36GatedDeltaNet.forward's single-token decode path / spec_forward's
    # inner loop. -------------------------------------------------------
    ref_state = initial_state.clone()
    ref_outputs = []
    ref_states = [initial_state.clone()]
    for t in range(T):
        out_t, final_t = fla_step(
            q[:, t : t + 1], k[:, t : t + 1], v[:, t : t + 1],
            g=g[:, t : t + 1], beta=beta[:, t : t + 1],
            initial_state=ref_state, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        ref_state = final_t.to(torch.bfloat16)  # the `.copy_()` this replaces
        ref_outputs.append(out_t)
        ref_states.append(ref_state.clone())
    ref_out = torch.cat(ref_outputs, dim=1)
    ref_states_stacked = torch.stack(ref_states, dim=1)

    # -- Candidate: ONE launch of the new multistep kernel. ---------------
    out, states = gated_delta_rule.fused_recurrent_gated_delta_rule_multistep(
        q, k, v, g, beta, initial_state,
    )

    assert states.shape == (B, T + 1, dims["HV"], dims["K"], dims["V"])
    assert states.dtype == torch.bfloat16
    assert torch.equal(states[:, 0], initial_state), (
        "snapshot 0 must be bit-identical to the untouched anchor"
    )
    max_abs_diff_state = (states.float() - ref_states_stacked.float()).abs().max().item()
    max_abs_diff_out = (out.float() - ref_out.float()).abs().max().item()
    assert torch.equal(states, ref_states_stacked), (
        f"[{case_name}, g_dtype={g_dtype}] multistep kernel states diverge "
        f"from K sequential fla launches: max_abs_diff={max_abs_diff_state:.6g} "
        f"(must be exactly 0.0 -- this is BlackweLLM's B3 acceptance bar)"
    )
    assert torch.equal(out, ref_out), (
        f"[{case_name}, g_dtype={g_dtype}] multistep kernel output diverges "
        f"from K sequential fla launches: max_abs_diff={max_abs_diff_out:.6g}"
    )


@pytest.mark.parametrize("case_name", sorted(_CASES))
def test_bit_exact_vs_sequential_fla_launches(case_name):
    device = require_sparkinfer()
    _check_bit_exact_vs_sequential_fla(
        case_name, _CASES[case_name], g_dtype=torch.bfloat16, device=device
    )


def test_bit_exact_vs_sequential_fla_launches_fp32_gate():
    """g's *real* dtype: qwen36_model.py computes it as
    ``-A_log.float().exp() * softplus(a.float() + dt_bias)`` and never
    casts the result down before calling ``fused_recurrent_gated_delta_rule``
    -- so the actual production call passes FP32 g, not BF16. This is the
    dtype combination that matters; the BF16-g cases above are extra
    coverage of the kernel's dtype-generic load path, not a substitute for
    this one.
    """
    device = require_sparkinfer()
    _check_bit_exact_vs_sequential_fla(
        "real_qwen36_gdn_k16_fp32_gate", _REAL, g_dtype=torch.float32, device=device
    )


def test_bf16_roundtrip_matches_torch_cast_rne():
    """The kernel's per-step state rounding relies on Triton's
    ``.to(tl.bfloat16)`` matching PyTorch's fp32->bf16 cast bit-for-bit
    (both claim round-to-nearest-even, but that is an empirical claim
    about two independent implementations, not something to assume --
    this repo's own memory of an FP8 rounding-tie bug that synthetic
    random data failed to catch is exactly this risk one register
    narrower). Exercise exact tie values (mantissa bit 16 sits exactly on
    a bf16 tie: 0x0080_0000 ULP at the truncation point) alongside dense
    random coverage, on-GPU.
    """
    device = require_sparkinfer()
    torch.manual_seed(7)
    random_vals = torch.randn(1 << 20, device=device, dtype=torch.float32) * 37.0
    # Construct exact bf16-boundary ties: take random bf16 values and add
    # exactly half a bf16 ULP (in the fp32 representation) so the fp32
    # value sits precisely between two bf16-representable numbers.
    base_bf16 = torch.randn(1 << 16, device=device, dtype=torch.float32).to(
        torch.bfloat16
    ).to(torch.float32)
    ulp = base_bf16.abs().clamp_min(2.0 ** -126) * (2.0 ** -8)  # bf16 has 7 mantissa bits
    tie_vals = base_bf16 + ulp * 0.5
    fp32_vals = torch.cat([random_vals, tie_vals])

    torch_result = fp32_vals.to(torch.bfloat16)

    import triton
    import triton.language as tl

    @triton.jit
    def _cast_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        tl.store(o_ptr + offs, x.to(tl.bfloat16), mask=mask)

    triton_out = torch.empty_like(fp32_vals, dtype=torch.bfloat16)
    n = fp32_vals.numel()
    block = 1024
    _cast_kernel[(triton.cdiv(n, block),)](fp32_vals, triton_out, n, BLOCK=block)

    mismatches = (torch_result.view(torch.int16) != triton_out.view(torch.int16)).sum().item()
    assert mismatches == 0, (
        f"{mismatches}/{n} fp32->bf16 casts disagree between torch and "
        "triton -- the multistep kernel's rounding-chain design assumption "
        "does not hold on this GPU/toolchain and needs a different fix "
        "(e.g. round via an actual store+reload through a bf16 tensor "
        "instead of a register-level cast)"
    )


# ---------------------------------------------------------------------------
# 3. Loose sanity vs the dependency-free pure-torch reference.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", sorted(_CASES))
def test_matches_pure_torch_reference_loosely(case_name):
    device = require_sparkinfer()
    dims = _CASES[case_name]
    q, k, v, g, beta, initial_state, scale = _make_case(**dims, seed=99, device=device)

    out, states = gated_delta_rule.fused_recurrent_gated_delta_rule_multistep(
        q, k, v, g, beta, initial_state, scale=scale,
    )
    ref_out, ref_states = gated_delta_rule_multistep_reference(
        q, k, v, g, beta, initial_state, scale,
    )
    torch.testing.assert_close(
        out.float(), ref_out.float(), atol=0.25, rtol=0.05,
        msg=f"[{case_name}] multistep kernel output too far from pure-torch reference",
    )
    torch.testing.assert_close(
        states.float(), ref_states.float(), atol=0.25, rtol=0.05,
        msg=f"[{case_name}] multistep kernel states too far from pure-torch reference",
    )
