"""Gated delta-rule (Qwen3.6 GDN) multistep recurrence for SM12x.

Advances FLA's gated delta-rule recurrence ``T`` steps in a single Triton
kernel launch, materializing the state after every step -- not only the
final one. Fills the gap between FLA's own two entry points, neither of
which alone supports speculative-decode rollback:

- ``chunk_gated_delta_rule`` returns only the final state after ``T``
  steps (fast, but nothing to roll back to on partial accept).
- ``fused_recurrent_gated_delta_rule`` exposes the state after each step,
  but only one step per kernel launch -- ``T`` calls means ``T`` kernel
  launches, and BlackweLLM's B3 measurement (2026-08-02, one real GDN
  layer, K=16) found sequential-launch overhead alone costs ~6.8ms of a
  12.6ms verify pass (vs. 1.8ms for one ``chunk_gated_delta_rule`` call).

This op is that missing combination: one launch, every intermediate state.
It is deliberately scoped to exactly what this runtime's GDN layer uses
(bf16 state, headwise post-sigmoid beta, precomputed log-space gate,
``use_qk_l2norm_in_kernel``-equivalent behavior always on) rather than a
general port of FLA's op family -- see ``kernel.py``'s module docstring
for the full contract, and in particular for why this op's per-step BF16
state round-trip is a correctness requirement, not an optimization detail:
without it, a single-launch multistep kernel reproduces
``chunk_gated_delta_rule``'s continuous-FP32 semantics instead of the
sequential path's BF16-rounded-per-step semantics, and BlackweLLM's own
notes already document those two disagreeing by ~30 ULP for the same
tokens.

Example:
    from sparkinfer.attention import gated_delta_rule

    out, states = gated_delta_rule.fused_recurrent_gated_delta_rule_multistep(
        q, k, v, g, beta, initial_state,
    )
    # states: [B, T+1, HV, K, V], BF16. states[:, m] is the state after
    # accepting m of the T candidate positions, bit-exact vs m sequential
    # single-step fused_recurrent_gated_delta_rule calls from the same
    # anchor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="gated_delta_rule",
    group="attention",
    api_style="oneshot",
    entry_points=(
        "fused_recurrent_gated_delta_rule_multistep",
        "is_supported",
    ),
    dtypes=("bf16",),
    recipes=("qwen3.6_gdn_spec_verify",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/jieen1/sparkinfer",
        commit="0844a4f",
        paths=("sparkinfer/attention/gated_delta_rule/",),
    ),
    test_path="tests/attention/test_gated_delta_rule_multistep.py",
    since="1.0.1",
    notes=(
        "Originally authored on work/gdn-multistep-20260802 (2026-08-02) "
        "for BlackweLLM's Qwen3.6 speculative-verify GDN rollback request "
        "-- not migrated from an upstream kernel. provenance.commit is "
        "this branch's fork point on origin/master, cited for audit-trail "
        "consistency with every other op's Provenance, not as a claim the "
        "code came from that commit."
    ),
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        fused_recurrent_gated_delta_rule_multistep,
        is_supported,
    )

install_lazy_api(globals(), META)
