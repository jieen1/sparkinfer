"""CPU-only accounting tests for the dynamic deterministic MoE arena."""

from __future__ import annotations

import torch

from sparkinfer.moe.fused_moe._impl import (
    _core_workspace_nbytes,
    _core_workspace_view_map,
    _dynamic_core_workspace_liveness_map,
    _deterministic_route_liveness,
    _deterministic_route_tile_dependencies,
    _dynamic_kernel_intermediate_size,
    _dynamic_task_geometry,
    _plan_core_workspace,
    _select_dynamic_tile_mn,
)


def test_m8192_deterministic_dynamic_arena_is_explicitly_attributed() -> None:
    """Keep the audited 670.9 MiB arena accountable without CUDA allocation."""
    m, topk, experts, hidden_size, intermediate_size = 8192, 10, 256, 3072, 1024
    routed_rows = m * topk
    kernel_intermediate = _dynamic_kernel_intermediate_size(
        intermediate_size, "nvfp4"
    )
    tile_m, tile_n = _select_dynamic_tile_mn(
        routed_rows,
        kernel_intermediate,
        "nvfp4",
        num_experts=experts,
        activation="silu",
    )
    physical_tiles, _, task_capacity = _dynamic_task_geometry(
        experts, kernel_intermediate, routed_rows, tile_m, tile_n
    )
    plan = _plan_core_workspace(
        "dynamic",
        "nvfp4",
        experts,
        experts,
        hidden_size,
        intermediate_size,
        topk,
        torch.device("cuda"),
        torch.bfloat16,
        routed_rows=routed_rows,
        max_rows=m,
        dynamic_physical_tiles=physical_tiles,
        dynamic_task_capacity=task_capacity,
        deterministic_output=True,
    )

    views = {view["name"]: view for view in _core_workspace_view_map(plan)}
    assert views["route_output"]["shape"] == (81920, 3072)
    assert views["route_output"]["nbytes"] == 480 * 2**20
    assert views["packed_input"]["nbytes"] == 175_964_160
    assert views["packed_input_scale"]["nbytes"] == 21_995_520
    assert _core_workspace_nbytes(plan) == 703_493_020

    liveness = {
        view["name"]: view for view in _dynamic_core_workspace_liveness_map(plan)
    }
    assert liveness["route_output"] == {
        **views["route_output"],
        "producer": "FC2 writes indexed by token-major route pair",
        "consumer": "fixed-order dynamic top-k reduction",
        "lifetime": "from each FC2 store until its token's ordered top-k reduction",
    }
    assert set(liveness) == set(views)


def test_deterministic_route_liveness_requires_complete_topk_groups() -> None:
    """A bounded schedule can only free a token after all its slots arrive."""
    current = _deterministic_route_liveness(
        num_tokens=2,
        num_topk=2,
        producer_batches=(tuple(range(4)),),
    )
    token_grouped = _deterministic_route_liveness(
        num_tokens=2,
        num_topk=2,
        producer_batches=((0, 1), (2, 3)),
    )
    interleaved = _deterministic_route_liveness(
        num_tokens=2,
        num_topk=2,
        producer_batches=((0, 2), (1, 3)),
    )

    assert current.peak_live_routes == 4
    assert token_grouped.peak_live_routes == 2
    assert interleaved.peak_live_routes == 4
    assert current.final_live_routes == token_grouped.final_live_routes == 0
    assert current.consumed_tokens == token_grouped.consumed_tokens == 2


def test_route_tile_dependency_cycles_block_simple_exact_streaming() -> None:
    """Crossed token slots form a tile cycle even though every pair is valid."""
    acyclic = _deterministic_route_tile_dependencies(
        physical_to_pair=(0, 1, 2, 3),
        num_tokens=2,
        num_topk=2,
        tile_m=2,
    )
    cyclic = _deterministic_route_tile_dependencies(
        physical_to_pair=(0, 3, 2, 1),
        num_tokens=2,
        num_topk=2,
        tile_m=2,
    )

    assert acyclic.active_tiles == 2
    assert acyclic.dependency_edges == acyclic.cyclic_components == 0
    assert cyclic.dependency_edges == 2
    assert cyclic.cyclic_components == 1
    assert cyclic.largest_cyclic_component_tiles == 2
    assert cyclic.largest_cyclic_component_route_rows == 4
