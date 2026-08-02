"""Triton kernel: advance the gated delta-rule recurrence K steps in one
launch, materializing the state after *every* step (not only the last).

Scoped to exactly what this runtime's GDN layer uses (BlackweLLM's
``Qwen36GatedDeltaNet``, calling ``fla.ops.gated_delta_rule.
fused_recurrent_gated_delta_rule`` once per candidate token during
speculative verify) rather than a general port of FLA's whole op family:

- ``g`` is already log-space decay, precomputed by the caller once for all
  T candidate positions (no in-kernel gate activation / ``A_log`` /
  ``dt_bias`` fusion -- the real call site never uses FLA's
  ``use_gate_in_kernel`` path).
- ``beta`` is headwise (one scalar per ``(batch, time, head)``) and already
  passed through sigmoid by the caller, matching
  ``Qwen36GatedDeltaNet.forward``'s ``beta = b.sigmoid()`` before the FLA
  call (``use_beta_sigmoid_in_kernel=False`` at every real call site).
- ``gk``/``gv`` (FLA's generic per-key/per-value gate hooks, used by other
  algorithms that share its fused-recurrent kernel, never by gated delta
  rule itself) do not exist here.
- ``use_qk_l2norm_in_kernel=True`` always -- the only mode the real call
  site uses.
- no varlen/``cu_seqlens`` -- this runtime's GDN layer asserts
  ``batch_size == 1`` throughout; batch is still a real (leading) axis here
  for generality, just never exercised above 1 today.

Why the per-step BF16 round-trip inside the loop is load-bearing, not an
optimization nicety
--------------------------------------------------------------------------
The mechanism this replaces (``Qwen36GatedDeltaNet.spec_forward``: K
sequential single-token ``fused_recurrent_gated_delta_rule`` launches)
persists ``recurrent_state`` in a BF16 buffer *between* calls -- every
step's *input* state has already been rounded fp32->bf16 by the
*previous* step's ``state.recurrent_state.copy_(last_state)``
(``runtime/model/qwen36_model.py``). A naive single-launch multi-step
kernel that keeps the recurrence in FP32 registers for all K steps
computes a genuinely different number than K sequential launches would --
it reproduces ``chunk_gated_delta_rule``'s continuous-FP32 semantics
instead, which BlackweLLM's own B1 correctness notes already document
disagrees with ``fused_recurrent``'s by ~30 ULP for the same tokens (their
"陷阱5"). Reproducing the sequential path bit-exactly (their B3
acceptance bar, ``max_abs_diff == 0.0`` against K ordinary sequential
decode steps) requires this kernel to explicitly round its carried state
to BF16 after every step -- mirroring the store+reload that used to
happen between separate kernel launches -- before that state is used to
compute the next step's update, and to materialize that same rounded
value as the step's snapshot.

Both this rounding *decision* (needed at all) and its bit-exactness (does
Triton's ``.to(tl.bfloat16)`` round-to-nearest-even match PyTorch's
fp32->bf16 cast used by the ``.copy_()`` it replaces) are verified on GPU,
not assumed -- see ``tests/attention/test_gated_delta_rule_multistep.py``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def _fused_recurrent_gdn_multistep_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    h0_ptr,
    o_ptr,
    hs_ptr,
    T,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    # 64-bit throughout: T (candidate count) and B (batch) are both runtime
    # values, and this repo's own convention (AGENTS.md, "64-bit addressing
    # for pool-scaled offsets") is that anything scaling a launch id into a
    # byte/element offset must not be left in Int32, even where today's
    # shapes (B=1, T<=~32) can't overflow one -- a caller batching multiple
    # sequences later must not silently inherit a 32-bit bug.
    i_n64 = i_n.to(tl.int64)
    i_hv64 = i_hv.to(tl.int64)
    i_h64 = i_h.to(tl.int64)
    T64 = T.to(tl.int64)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    # q/k/v/g/beta/o are laid out [B, T, H(V), *]: batch-major, then time,
    # then head, then feature -- same layout FLA's own kernel assumes.
    # ``stride_qk``/``stride_v`` are the PER-STEP (one time-tick) strides,
    # reused below both to seed the batch-offset base pointer and to
    # advance the pointer each loop iteration.
    stride_qk = H * K
    stride_v = HV * V
    p_q = q_ptr + i_n64 * T64 * stride_qk + i_h64 * K + o_k
    p_k = k_ptr + i_n64 * T64 * stride_qk + i_h64 * K + o_k
    p_v = v_ptr + i_n64 * T64 * stride_v + i_hv64 * V + o_v
    p_g = g_ptr + i_n64 * T64 * HV + i_hv64
    p_beta = beta_ptr + i_n64 * T64 * HV + i_hv64
    p_o = o_ptr + i_n64 * T64 * stride_v + i_hv64 * V + o_v

    KV64 = (i_n64 * HV + i_hv64) * (K * V)
    p_h0 = h0_ptr + KV64 + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

    # hs layout: [B, T+1, HV, K, V] -- slot 0 is the untouched anchor state
    # (bit-identical to h0: bf16->fp32->bf16 round-trips an already-bf16
    # value exactly), slot t+1 is the state after processing t+1 candidate
    # positions.
    hs_stride_t = HV * K * V
    p_hs_base = (
        hs_ptr
        + i_n64 * (T64 + 1) * hs_stride_t
        + i_hv64 * (K * V)
        + o_k[:, None] * V
        + o_v[None, :]
    )
    tl.store(p_hs_base, b_h.to(p_hs_base.dtype.element_ty), mask=mask_h)

    for t in tl.range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0.0).to(tl.float32)
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_beta = tl.load(p_beta).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)

        b_h *= tl.exp(b_g)
        b_v_new = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v_new
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Round the carried state to BF16 -- see module docstring. This is
        # the one line that makes a single-launch multistep kernel agree
        # with K sequential single-step launches instead of with the
        # continuous-FP32 chunk algorithm.
        b_h = b_h.to(tl.bfloat16).to(tl.float32)
        p_hs_t = p_hs_base + (t.to(tl.int64) + 1) * hs_stride_t
        tl.store(p_hs_t, b_h.to(p_hs_base.dtype.element_ty), mask=mask_h)

        p_q += stride_qk
        p_k += stride_qk
        p_v += stride_v
        p_g += HV
        p_beta += HV
        p_o += stride_v


def fused_recurrent_gdn_multistep_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the kernel. No validation here -- see ``api.py``'s ``run``."""
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    if scale is None:
        scale = K ** -0.5

    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V))
    NV = triton.cdiv(V, BV)

    o = torch.empty_like(v)
    hs = torch.empty((B, T + 1, HV, K, V), dtype=initial_state.dtype, device=q.device)

    grid = (NV, B * HV)
    _fused_recurrent_gdn_multistep_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        o,
        hs,
        T,
        scale,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        num_warps=1,
        num_stages=3,
    )
    return o, hs
