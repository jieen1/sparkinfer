"""Regression tests for the W4A16 host-side scratch/capacity planner.

``run_w4a16_moe`` (kernel.py) picks between two route-handling strategies at
launch time:

* the "packed"/grouped path (``pack_topk_routes_by_expert``), whose fc1/fc2
  ``c_tmp`` scratch requirement is bounded by ``max_packed_route_slots``; and
* the small-M "direct top-k routes" / TC-decode fast path (no packing at
  all -- every routed row gets its own ``block_size_m``-sized scratch tile),
  whose scratch requirement is exactly
  ``route_slots_for_scratch = m * topk * block_size_m``
  (see ``kernel.py::run_w4a16_moe``, right before the ``fc1_scratch``/
  ``fc2_scratch`` allocations).

``plan_w4a16_buffers``/``make_w4a16_packed_buffers`` (host.py) is a single
convenience allocator shared by both paths -- it does not know in advance
which strategy a given launch will use. Before this fix it only ever sized
fc1/fc2 ``c_tmp`` using the packed-mode bound (``max_packed_route_slots``),
which can be *smaller* than the direct-mode requirement whenever
``routed_rows = m * topk`` is not small relative to ``route_num_experts``
(e.g. a degenerate single-expert "MoE", or few local experts under EP
sharding). In eager mode this silently self-heals (``_get_c_tmp`` falls back
to a fresh ``torch.empty`` when the caller-supplied scratch is too small),
but ``torch.cuda.graph`` capture refuses that fallback and fails outright.

These tests are pure arithmetic (no CUDA/model weights needed) and assert
the allocator's capacity is always an upper bound on both formulas, for a
grid of shapes including the one that reproduces the original mismatch
exactly: decode batch=2, topk=1, num_experts=1, block_size_m=8, where the
un-fixed allocator planned 9 route slots against a real kernel requirement
of 16.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sparkinfer.moe._shared.kernels.w4a16.host import (
    _W4A16_ALLOWED_ROUTED_SIZES,
    max_packed_route_slots,
    packed_gemm_scratch_elements,
    plan_w4a16_buffers,
    select_route_block_size_m,
)


def _prepared(*, num_experts: int, hidden_size: int = 128, intermediate_size: int = 128, is_gated: bool = True):
    return SimpleNamespace(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        is_gated=is_gated,
    )


def _direct_topk_scratch_requirement(
    *, m: int, topk: int, block_size_m: int, fc1_cols: int, hidden_size: int, sms: int
) -> tuple[int, int]:
    """Mirror kernel.py's decode fast-path ``route_slots_for_scratch``.

    ``run_w4a16_moe``'s direct top-k / TC-decode path never reassigns
    ``route_slots_for_scratch`` away from its initial value
    ``m * topk * block_size_m`` (only the packed-mode ``else`` branch
    overwrites it with ``packed_route_indices.numel()``), so this is exactly
    what the fused kernel's fc1/fc2 ``c_tmp`` scratch launch needs whenever
    the direct/TC-decode path is selected.
    """
    route_slots_for_scratch = int(m) * int(topk) * int(block_size_m)
    needed_fc1 = packed_gemm_scratch_elements(
        size_n=fc1_cols,
        route_slots=route_slots_for_scratch,
        moe_block_size=block_size_m,
        sms=sms,
    )
    needed_fc2 = packed_gemm_scratch_elements(
        size_n=hidden_size,
        route_slots=route_slots_for_scratch,
        moe_block_size=block_size_m,
        sms=sms,
    )
    return needed_fc1, needed_fc2


# (m, topk, num_experts) grid: small m/topk (where the direct-routes/TC-decode
# fast path is actually reachable, m <= 8), plus the exact BlackweLLM decode
# repro (m=2, topk=1, num_experts=1) and a few more degenerate/near-degenerate
# expert counts that are the concrete trigger for max_packed_route_slots
# dropping below the direct-mode floor.
_SHAPES = [
    (2, 1, 1),  # exact BlackweLLM decode batch=2 repro: 16 (kernel) vs 9 (old allocator)
    (1, 1, 1),
    (3, 1, 1),
    (4, 1, 1),
    (5, 1, 1),
    (6, 1, 1),
    (8, 1, 1),
    (2, 2, 1),
    (2, 1, 2),
    (3, 2, 2),
    (4, 2, 3),
    (6, 4, 4),
    (8, 8, 8),
    (2, 1, 128),  # realistic large-expert-count decode: should already match
    (8, 6, 128),
    (32, 6, 128),  # beyond direct-routes eligibility (m>8); packed path only
    (1024, 8, 128),  # prefill-scale: packed bound should stay tight (no bloat)
]


@pytest.mark.parametrize("m,topk,num_experts", _SHAPES)
def test_plan_w4a16_buffers_scratch_covers_direct_topk_requirement(
    m: int, topk: int, num_experts: int
) -> None:
    prepared = _prepared(num_experts=num_experts, hidden_size=256, intermediate_size=192)
    sms = 132
    plan = plan_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        route_num_experts=num_experts,
        sms=sms,
    )
    fc1_cols = plan.fc1_cols
    needed_fc1, needed_fc2 = _direct_topk_scratch_requirement(
        m=m,
        topk=topk,
        block_size_m=plan.block_size_m,
        fc1_cols=fc1_cols,
        hidden_size=256,
        sms=sms,
    )
    assert plan.fc1_c_tmp_elements >= needed_fc1, (
        f"m={m} topk={topk} experts={num_experts} block_size_m={plan.block_size_m}: "
        f"allocator fc1_c_tmp_elements={plan.fc1_c_tmp_elements} < "
        f"direct-topk-routes requirement={needed_fc1}"
    )
    assert plan.fc2_c_tmp_elements >= needed_fc2, (
        f"m={m} topk={topk} experts={num_experts} block_size_m={plan.block_size_m}: "
        f"allocator fc2_c_tmp_elements={plan.fc2_c_tmp_elements} < "
        f"direct-topk-routes requirement={needed_fc2}"
    )


@pytest.mark.parametrize("block_size_m", _W4A16_ALLOWED_ROUTED_SIZES)
@pytest.mark.parametrize("m,topk,num_experts", _SHAPES)
def test_plan_w4a16_buffers_scratch_covers_direct_topk_requirement_for_every_block_size(
    m: int, topk: int, num_experts: int, block_size_m: int
) -> None:
    """Same invariant, but with every allowed block_size_m forced explicitly
    (not just whatever ``select_route_block_size_m`` would pick for this
    shape) -- a caller can always pass ``block_size_m`` explicitly (e.g. a
    preplanned/prewarmed launch), and the allocator must stay correct for
    every (m, topk, block_size_m) combination, not just the ones its own
    heuristic would choose."""
    prepared = _prepared(num_experts=num_experts, hidden_size=256, intermediate_size=192)
    sms = 132
    plan = plan_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        route_num_experts=num_experts,
        sms=sms,
        block_size_m=block_size_m,
    )
    assert plan.block_size_m == block_size_m
    needed_fc1, needed_fc2 = _direct_topk_scratch_requirement(
        m=m,
        topk=topk,
        block_size_m=block_size_m,
        fc1_cols=plan.fc1_cols,
        hidden_size=256,
        sms=sms,
    )
    assert plan.fc1_c_tmp_elements >= needed_fc1
    assert plan.fc2_c_tmp_elements >= needed_fc2


def test_blackwellm_decode_repro_shape_matches_reported_numbers() -> None:
    """Pin the exact numbers from the bug report so a future change that
    silently reintroduces the mismatch fails loudly with the same shape
    that broke CUDA Graph capture in production."""
    m, topk, num_experts, block_size_m = 2, 1, 1, 8
    assert select_route_block_size_m(m, topk, num_experts) == block_size_m
    old_allocator_route_slots = max_packed_route_slots(m * topk, block_size_m, num_experts)
    kernel_route_slots_for_scratch = m * topk * block_size_m
    assert old_allocator_route_slots == 9
    assert kernel_route_slots_for_scratch == 16
    assert old_allocator_route_slots < kernel_route_slots_for_scratch

    prepared = _prepared(num_experts=num_experts, hidden_size=256, intermediate_size=192)
    sms = 132
    plan = plan_w4a16_buffers(
        prepared, m=m, topk=topk, route_num_experts=num_experts, sms=sms
    )
    needed_fc1, needed_fc2 = _direct_topk_scratch_requirement(
        m=m,
        topk=topk,
        block_size_m=block_size_m,
        fc1_cols=plan.fc1_cols,
        hidden_size=256,
        sms=sms,
    )
    assert plan.fc1_c_tmp_elements >= needed_fc1
    assert plan.fc2_c_tmp_elements >= needed_fc2


def test_scratch_route_slots_union_is_a_provable_upper_bound() -> None:
    """``max_packed_route_slots(numel, block, experts) <= numel * block``
    always (proof: when ``numel < experts`` the function already returns
    ``min(numel*block, additive)``; when ``numel >= experts`` the additive
    term ``numel + experts*(block-1) <= numel + numel*(block-1) = numel*
    block``). So ``max(packed_bound, numel*block) == numel*block``
    unconditionally, which is exactly why unioning the two bounds in
    ``plan_w4a16_buffers`` is safe: it never *reduces* below the packed
    bound, and for large ``numel`` (where the packed bound is far smaller)
    the ``packed_gemm_scratch_elements`` SM-count cap saturates first, so
    the union costs nothing there. This test exercises the inequality
    directly over a wide grid, independent of the allocator plumbing."""
    for numel in (0, 1, 2, 3, 5, 8, 17, 63, 64, 1000, 1_000_000):
        for block in _W4A16_ALLOWED_ROUTED_SIZES:
            for experts in (1, 2, 3, 8, 16, 128, 256):
                bound = max_packed_route_slots(numel, block, experts)
                assert bound <= numel * block, (numel, block, experts, bound)
