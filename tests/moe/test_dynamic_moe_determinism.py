"""Regression test for a real, reproduced non-determinism bug in the
"dynamic" MoE kernel family (NVFP4, Laguna-S-2.1's shape: E=256, K=3072,
I=1024, top_k=10) when SPARKINFER_ENABLE_DYNAMIC_DOWN_SCALE=1.

Root cause (confirmed by direct source inspection and a controlled A/B on
real GPU runs, see the sparkinfer fork issue tracker):

  1. Phase 1 of the dynamic kernel assigns each routed (token, expert) pair
     its physical row via a racing ``atomic_add_global_i32`` against
     ``expert_write_rows[expert_id]`` (dynamic.py, the non-shared-input
     branch around the ``row = atomic_add_global_i32(...)`` call). Which
     physical tile a token lands in is therefore a function of GPU
     scheduling, not just the input -- it can differ run to run for
     bit-identical inputs.
  2. When ``dynamic_down_scale`` is enabled, the FC2 intermediate
     quantization scale for a physical tile (``tile_gs_value`` /
     ``quant_gs_value`` / ``fc2_down_alpha_value``) is computed from that
     tile's *collective* amax across all its valid rows -- not per token.
  3. Combining (1) and (2): whenever tile membership changes between runs
     (which it does, per (1)), the shared tile-level scale changes too, so
     the same token's FC2 dequantization -- and therefore the kernel's
     final output -- changes between runs, even for fully identical inputs
     on the same pre-built binding.

``SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT=1`` (``deterministic_output``)
only fixes a *different* non-determinism source (the outer top-k route
reduction, ATOMIC_SCATTER vs ROUTE_BUFFER_TOPK_SUM) and must be enabled in
both arms of this test to isolate the tile_amax mechanism -- otherwise the
outer reduction's own non-determinism confounds the comparison.

This file intentionally builds *real* E=256 NVFP4 experts (not a
degenerate constant-filled shape-only fixture) because the bug is about
real per-expert tile composition, not raw kernel launch mechanics.
"""

from __future__ import annotations

import os

import pytest
import torch

from tests.conftest import require_sparkinfer
from tests._reference.helpers import (
    compute_per_group_global_scale,
    prepare_tp_moe_fp4_experts,
    make_tp_moe_fp4_binding,
)
from sparkinfer._lib.intrinsics import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    SF_VEC_SIZE,
    _fp4_quantize_values,
    _fp4_encode_nibbles,
    swizzle_block_scale,
)

E = 256
K = 3072
I_TP = 1024
TOP_K = 10


def _quantize_one_expert(x_e: torch.Tensor, gs_e: torch.Tensor):
    rows, cols = x_e.shape
    n_blocks = cols // SF_VEC_SIZE
    blocked = x_e.reshape(rows, n_blocks, SF_VEC_SIZE)
    block_max = blocked.abs().amax(dim=-1)
    raw_scale = (block_max * gs_e / FLOAT4_E2M1_MAX).clamp(max=FLOAT8_E4M3_MAX)
    sf_e4m3 = raw_scale.to(torch.float8_e4m3fn).to(torch.float32)
    sf_times_gs = (sf_e4m3 / gs_e).clamp(min=1e-30)
    scaled = blocked / sf_times_gs.unsqueeze(-1)
    clipped = torch.clamp(scaled, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).reshape(rows, cols)
    quant_vals = _fp4_quantize_values(clipped)
    nibbles = _fp4_encode_nibbles(quant_vals)
    pair = nibbles.view(rows, cols // 2, 2)
    packed = (pair[..., 0] | (pair[..., 1] << 4)).contiguous()
    blockscale_swizzled = swizzle_block_scale(sf_e4m3.to(torch.float8_e4m3fn))
    return packed, blockscale_swizzled


def _build_laguna_shape_experts(device: torch.device, seed: int = 1234):
    """Real E=256 NVFP4 experts at Laguna-S-2.1's shape, quantized one
    expert at a time to bound peak host/device memory (~2.7GB total)."""
    wgen = torch.Generator(device=device).manual_seed(seed)
    w1_fp4_list, w1_sf_list, w2_fp4_list, w2_sf_list = [], [], [], []
    g1_list, g2_list = [], []
    for e in range(E):
        up_e = torch.randn(I_TP, K, generator=wgen, device=device, dtype=torch.float32)
        gate_e = torch.randn(I_TP, K, generator=wgen, device=device, dtype=torch.float32)
        down_e = torch.randn(K, I_TP, generator=wgen, device=device, dtype=torch.float32)
        w13_e = torch.cat([up_e, gate_e], dim=0).contiguous()
        del up_e, gate_e
        g1_e = compute_per_group_global_scale(w13_e.unsqueeze(0)).squeeze(0)
        g2_e = compute_per_group_global_scale(down_e.unsqueeze(0)).squeeze(0)
        w1_fp4_e, w1_sf_e = _quantize_one_expert(w13_e, g1_e)
        w2_fp4_e, w2_sf_e = _quantize_one_expert(down_e, g2_e)
        del w13_e, down_e
        w1_fp4_list.append(w1_fp4_e)
        w1_sf_list.append(w1_sf_e)
        w2_fp4_list.append(w2_fp4_e)
        w2_sf_list.append(w2_sf_e)
        g1_list.append(g1_e)
        g2_list.append(g2_e)
        if e % 32 == 0:
            torch.cuda.empty_cache()
    w1_fp4 = torch.stack(w1_fp4_list)
    w1_blockscale = torch.stack(w1_sf_list)
    w2_fp4 = torch.stack(w2_fp4_list)
    w2_blockscale = torch.stack(w2_sf_list)
    g1_scale = torch.stack(g1_list)
    g2_scale = torch.stack(g2_list)
    del w1_fp4_list, w1_sf_list, w2_fp4_list, w2_sf_list, g1_list, g2_list
    torch.cuda.empty_cache()
    w1_alphas = (1.0 / g1_scale).float().contiguous()
    w2_alphas = (1.0 / g2_scale).float().contiguous()

    # Per-expert (numel == E, not a bare scalar): nvfp4's automatic
    # `a1_gscale.numel() == 1` check enables a *different* producer code
    # path (share_input_across_experts) that this test does not target.
    a1_gscale = torch.full((E,), 2016.0, device=device, dtype=torch.float32)
    a2_gscale = torch.full(
        (E,), 1.0 / w2_alphas.max().item(), device=device, dtype=torch.float32
    )

    return prepare_tp_moe_fp4_experts(
        a=torch.empty(1, K, dtype=torch.bfloat16, device=device),
        a1_gscale=a1_gscale,
        w1_fp4=w1_fp4,
        w1_blockscale=w1_blockscale,
        w1_alphas=w1_alphas,
        a2_gscale=a2_gscale,
        w2_fp4=w2_fp4,
        w2_blockscale=w2_blockscale,
        w2_alphas=w2_alphas,
        activation="silu",
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        w13_layout="w13",
    )


def _make_routing(m: int, seed: int, device: torch.device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(m, E, generator=gen, dtype=torch.float32)
    topk_logits, topk_ids = torch.topk(logits, TOP_K, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1)
    return (
        topk_ids.to(device=device, dtype=torch.int32).contiguous(),
        topk_weights.to(device=device, dtype=torch.float32).contiguous(),
    )


def _make_activations(m: int, seed: int, device: torch.device):
    gen = torch.Generator(device="cpu").manual_seed(seed + 777)
    x = torch.randn(m, K, generator=gen, dtype=torch.float32)
    return x.to(device=device, dtype=torch.bfloat16)


def _run_repeated(
    m: int, experts, device: torch.device, *, repeats: int, seed: int = 1
):
    """Call ONE binding `repeats` times with identical inputs. Both env
    vars are set fresh (module-level cache safe: SPARKINFER_ENABLE_DYNAMIC_DOWN_SCALE
    is cached per-process by sparkinfer and must be paired with clear_tp_moe_caches()
    for a value change to take effect; SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT is
    read fresh on every call, no cache to clear)."""
    from sparkinfer.moe.fused_moe._impl import clear_tp_moe_caches, sparkinfer_moe_fp4

    x = _make_activations(m, seed, device)
    topk_ids, topk_weights = _make_routing(m, seed + 1, device)
    binding = make_tp_moe_fp4_binding(
        a=x,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        input_scales_static=True,
        quant_mode="nvfp4",
    )
    clear_tp_moe_caches()
    outs = []
    for _ in range(repeats):
        out = sparkinfer_moe_fp4(binding=binding)
        torch.cuda.synchronize()
        outs.append(out.clone())
    return outs


@pytest.fixture(scope="module")
def laguna_shape_experts():
    device = require_sparkinfer()
    return _build_laguna_shape_experts(device)


def _assert_bitwise_stable(outs: list[torch.Tensor], *, label: str):
    ref = outs[0]
    for i, o in enumerate(outs[1:], start=1):
        assert torch.equal(ref, o), (
            f"{label}: repeat #{i} differs from repeat #0 "
            f"(changed={int((ref != o).sum().item())} elements, "
            f"max_abs_diff={(ref.float() - o.float()).abs().max().item():.6g}) -- "
            "dynamic MoE kernel output is not bitwise-deterministic for identical inputs"
        )


def test_dynamic_moe_deterministic_output_is_bitwise_stable_m256(laguna_shape_experts):
    """Fast primary regression case: M=256 already reliably reproduces the
    bug at default (multi-CTA) scheduling, per the diagnosed M-dependent
    threshold (M<=128 is stable; M>=256 is not, without the fix)."""
    require_sparkinfer()
    os.environ["SPARKINFER_ENABLE_DYNAMIC_DOWN_SCALE"] = "1"
    os.environ["SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT"] = "1"
    outs = _run_repeated(256, laguna_shape_experts, torch.device("cuda"), repeats=6)
    _assert_bitwise_stable(outs, label="M=256, dynamic_down_scale=1")


@pytest.mark.parametrize("m", [128, 256, 512, 2048, 8192])
def test_dynamic_moe_deterministic_output_is_bitwise_stable_full_sweep(
    laguna_shape_experts, m
):
    """Full acceptance sweep across the qwen-sm120-runtime team's reported
    shape range, dynamic_down_scale=1 held on throughout (must NOT be
    disabled to achieve determinism -- that would regress FC2 scale
    underflow protection)."""
    require_sparkinfer()
    os.environ["SPARKINFER_ENABLE_DYNAMIC_DOWN_SCALE"] = "1"
    os.environ["SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT"] = "1"
    repeats = 12 if m == 8192 else 6
    outs = _run_repeated(m, laguna_shape_experts, torch.device("cuda"), repeats=repeats)
    _assert_bitwise_stable(outs, label=f"M={m}, dynamic_down_scale=1")
