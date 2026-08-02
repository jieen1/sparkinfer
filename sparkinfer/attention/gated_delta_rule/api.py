"""Public surface for attention.gated_delta_rule (docs in the op ``__init__``)."""

from __future__ import annotations

import torch

from ..._lib.gating import has_triton, is_sparkinfer
from .kernel import fused_recurrent_gdn_multistep_fwd


def _validate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B,T,H,K], got {tuple(q.shape)}")
    B, T, H, K = (int(s) for s in q.shape)
    if T < 1:
        raise ValueError(f"T (candidate count) must be >= 1, got {T}")
    if tuple(k.shape) != (B, T, H, K):
        raise ValueError(f"k must match q's shape {(B, T, H, K)}, got {tuple(k.shape)}")
    if v.ndim != 4 or int(v.shape[0]) != B or int(v.shape[1]) != T:
        raise ValueError(
            f"v must have shape [B,T,HV,V] with B={B}, T={T}; got {tuple(v.shape)}"
        )
    HV, V = int(v.shape[2]), int(v.shape[3])
    if HV % H != 0:
        raise ValueError(f"num_v_heads ({HV}) must be a multiple of num_k_heads ({H})")
    if tuple(g.shape) != (B, T, HV):
        raise ValueError(f"g must have shape {(B, T, HV)}, got {tuple(g.shape)}")
    if tuple(beta.shape) != (B, T, HV):
        raise ValueError(
            f"beta must have shape {(B, T, HV)} (headwise -- one scalar per "
            f"(batch, time, head), already post-sigmoid), got {tuple(beta.shape)}"
        )
    if tuple(initial_state.shape) != (B, HV, K, V):
        raise ValueError(
            f"initial_state must have shape {(B, HV, K, V)}, got "
            f"{tuple(initial_state.shape)}"
        )
    for name, tensor in (("q", q), ("k", k), ("v", v), ("beta", beta)):
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be bfloat16, got {tensor.dtype}")
    if g.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(
            "g must be bfloat16 or float32, got "
            f"{g.dtype}. Unlike q/k/v/beta/initial_state, real callers do "
            "not agree on g's dtype: qwen36_model.py computes "
            "`-A_log.float().exp() * softplus(a.float() + dt_bias)`, which "
            "stays FP32 all the way through (dt_bias is BF16 but promotes "
            "under the FP32 addition) -- it is never cast down before being "
            "passed to fla's fused_recurrent_gated_delta_rule, so the "
            "sequential baseline this op must match bit-exactly uses FP32 "
            "g. The kernel upcasts g to FP32 at load regardless of its "
            "storage dtype, so both are supported -- but forcing a caller's "
            "FP32 g down to BF16 here would silently break bit-exactness "
            "against that baseline, so this validates rather than coerces."
        )
    if initial_state.dtype != torch.bfloat16:
        raise TypeError(
            "initial_state must be bfloat16 -- this is the persisted state "
            "buffer's own dtype, and the kernel's per-step BF16 round-trip "
            "(see kernel.py's module docstring) is what makes this op agree "
            "bit-exactly with K sequential single-step launches; a caller "
            "with an FP32 state has a different, easier problem "
            "(chunk_gated_delta_rule already solves it) and should not use "
            f"this op. Got {initial_state.dtype}."
        )
    tensors = (q, k, v, g, beta, initial_state)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("gated_delta_rule multistep operands must be CUDA tensors")
    if any(tensor.device != q.device for tensor in tensors[1:]):
        raise ValueError(
            "gated_delta_rule multistep operands must all be on the same device"
        )
    for name, tensor in zip(("q", "k", "v", "g", "beta", "initial_state"), tensors):
        if tensor.numel() and int(tensor.stride(-1)) != 1:
            raise ValueError(f"{name} innermost dimension must be contiguous")
    return B, T, H, HV, K, V


@torch.library.custom_op("sparkinfer::gdn_fused_recurrent_multistep", mutates_args=())
def _op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return fused_recurrent_gdn_multistep_fwd(q, k, v, g, beta, initial_state, scale)


@_op.register_fake
def _fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del g, beta, scale
    B, T = q.shape[0], q.shape[1]
    HV, K, V = v.shape[2], initial_state.shape[2], v.shape[3]
    o = torch.empty_like(v)
    hs = q.new_empty((B, T + 1, HV, K, V), dtype=initial_state.dtype)
    return o, hs


def fused_recurrent_gated_delta_rule_multistep(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    output_all_states: bool = True,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance the gated delta-rule recurrence ``T`` steps in one kernel
    launch, returning every intermediate state -- the thing K sequential
    ``fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule`` calls
    used to be the only way to get (see ``kernel.py``'s module docstring
    for the full derivation, including why this is NOT interchangeable
    with ``chunk_gated_delta_rule``: that op's continuous-FP32 state
    disagrees with this recurrence's BF16-rounded-per-step state by tens
    of ULP -- a real, previously observed divergence, not a hypothetical
    one).

    Args:
        q: ``[B, T, H, K]``, BF16.
        k: ``[B, T, H, K]``, BF16.
        v: ``[B, T, HV, V]``, BF16. ``HV`` must be a multiple of ``H``
            (grouped-value attention; each key/query head is broadcast to
            ``HV/H`` value heads exactly like FLA's own kernel). The real
            call site (``Qwen36GatedDeltaNet``) already repeats q/k to
            ``HV`` heads before calling, so ``H == HV`` there in practice.
        g: ``[B, T, HV]``, BF16 or FP32. Log-space decay, precomputed by
            the caller (``-A_log.float().exp() * softplus(a.float() +
            dt_bias)`` in ``qwen36_model.py`` -- note that expression
            never gets cast back down, so the real call site's ``g`` is
            FP32, not BF16; both are accepted and both upcast to FP32 at
            load, but only FP32 matches production bit-for-bit) -- this
            op has no in-kernel gate activation.
        beta: ``[B, T, HV]``, BF16. Headwise, already post-sigmoid.
        initial_state: ``[B, HV, K, V]``, BF16 -- the anchor's persisted
            recurrent state.
        output_all_states: must be ``True`` (default and only supported
            value). Materializing every intermediate state is this op's
            entire reason to exist; a caller that only wants the state
            after all ``T`` steps should call ``chunk_gated_delta_rule``
            instead (faster, and does not carry the BF16-round-trip cost
            this op pays specifically to stay bit-exact with the
            sequential path -- see the module docstring).
        scale: defaults to ``K ** -0.5``, matching FLA's own default.

    Returns:
        ``(out, states)``: ``out`` is ``[B, T, HV, V]`` BF16, this layer's
        contribution at every candidate position. ``states`` is
        ``[B, T+1, HV, K, V]`` BF16; ``states[:, 0]`` is bit-identical to
        ``initial_state``, ``states[:, j]`` for ``j >= 1`` is the state
        after processing the first ``j`` positions -- bit-exact (not
        merely close) to what ``j`` sequential single-step
        ``fused_recurrent_gated_delta_rule`` calls from the same anchor
        would have produced.
    """
    if not output_all_states:
        raise NotImplementedError(
            "output_all_states=False is not implemented: materializing "
            "every intermediate state is this op's entire reason to exist "
            "(see this function's docstring). A caller that only wants the "
            "final state should call chunk_gated_delta_rule instead."
        )
    B, T, H, HV, K, V = _validate(q, k, v, g, beta, initial_state)
    del B, H, HV, K, V  # validated; shapes come from the tensors themselves below
    resolved_scale = float(q.shape[-1]) ** -0.5 if scale is None else float(scale)
    out, states = torch.ops.sparkinfer.gdn_fused_recurrent_multistep(
        q, k, v, g, beta, initial_state, resolved_scale
    )
    assert states.shape[1] == T + 1
    return out, states


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with triton (this op is pure Triton -- unlike
    most of sparkinfer, it does not need the CUTLASS DSL)."""
    return is_sparkinfer(device) and has_triton()


def clear_caches() -> None:
    """No cross-call cache to clear (the kernel has no autotune/signature
    cache today) -- present for registry-contract symmetry with other ops."""


__all__ = [
    "fused_recurrent_gated_delta_rule_multistep",
    "is_supported",
    "clear_caches",
]
