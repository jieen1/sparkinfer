"""Pure-torch, FLA-independent reference for gated_delta_rule multistep.

Not a transcription of ``kernel.py``'s Triton kernel -- written from the
delta-rule recurrence directly (see ``fla.ops.gated_delta_rule.naive`` for
the algorithm this matches in spirit) so that a bug shared between "the
kernel" and "a copy of the kernel" cannot hide behind an all-tests-pass
result. Useful as a coarse, dependency-free sanity oracle (loose
tolerance: elementwise reduction order differs from the kernel's, so this
is NOT expected to match bit-exactly -- see
``tests/attention/test_gated_delta_rule_multistep.py`` for the actual
bit-exact oracle, which compares against real sequential
``fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule`` launches).
"""

from __future__ import annotations

import torch


def gated_delta_rule_multistep_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One step at a time, FP32 math, explicit BF16 round-trip of the
    carried state after every step -- reproducing what K *separate*
    ``fused_recurrent_gated_delta_rule`` kernel launches compute when the
    caller persists state in a BF16 buffer between them (this runtime's
    actual usage; see ``kernel.py``'s module docstring for why that
    round-trip is not optional)."""
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    repeat = HV // H
    if repeat * H != HV:
        raise ValueError(f"HV={HV} must be a multiple of H={H}")

    state = initial_state.to(torch.float32)  # [B, HV, K, V]
    outputs = []
    states = [initial_state.clone()]
    for t in range(T):
        qt = q[:, t].to(torch.float32)  # [B, H, K]
        kt = k[:, t].to(torch.float32)
        vt = v[:, t].to(torch.float32)  # [B, HV, V]
        gt = g[:, t].to(torch.float32)  # [B, HV]
        betat = beta[:, t].to(torch.float32)  # [B, HV]

        qt = qt / torch.sqrt((qt * qt).sum(-1, keepdim=True) + 1e-6)
        kt = kt / torch.sqrt((kt * kt).sum(-1, keepdim=True) + 1e-6)
        qt = qt * scale

        if repeat > 1:
            qt = qt.repeat_interleave(repeat, dim=1)
            kt = kt.repeat_interleave(repeat, dim=1)

        state = state * gt.exp()[:, :, None, None]
        pred = (state * kt[:, :, :, None]).sum(dim=2)  # [B, HV, V]
        v_new = betat[:, :, None] * (vt - pred)
        state = state + kt[:, :, :, None] * v_new[:, :, None, :]
        out = (state * qt[:, :, :, None]).sum(dim=2)  # [B, HV, V]
        outputs.append(out.to(v.dtype))

        state = state.to(torch.bfloat16).to(torch.float32)
        states.append(state.to(initial_state.dtype))

    return torch.stack(outputs, dim=1), torch.stack(states, dim=1)
