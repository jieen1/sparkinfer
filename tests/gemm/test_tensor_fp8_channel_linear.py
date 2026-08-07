from __future__ import annotations

import cutlass.cute as cute
import pytest
import torch

from sparkinfer.gemm import tensor_fp8_channel_linear
from sparkinfer.gemm.tensor_fp8_channel_linear import api as tensor_fp8_channel_api

from ..conftest import require_sparkinfer


def require_mxf8_mma() -> None:
    if not hasattr(cute.nvgpu.warp, "MmaMXF8Op"):
        pytest.skip("CUTLASS DSL does not expose cute.nvgpu.warp.MmaMXF8Op")


def _make_inputs(in_features: int, out_features: int):
    source = (
        torch.randn((1, in_features), device="cuda", dtype=torch.bfloat16)
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
    )
    weight = (
        torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16)
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
    )
    row_scale = torch.tensor([0.125], dtype=torch.float32, device="cuda")
    channel_scale = torch.rand((out_features,), dtype=torch.float32, device="cuda") * 0.01
    packed = tensor_fp8_channel_linear.pack_weight(weight, channel_scale)
    return source, weight, row_scale, channel_scale, packed


def test_mm_matches_channel_scaled_reference() -> None:
    require_sparkinfer()
    require_mxf8_mma()
    torch.manual_seed(20260803)

    source, weight, row_scale, channel_scale, packed = _make_inputs(128, 128)
    actual = tensor_fp8_channel_linear.mm(source, packed, row_scale)
    expected = (source.float() @ weight.float().T) * row_scale * channel_scale.unsqueeze(0)
    torch.cuda.synchronize()

    assert actual.shape == (1, 128)
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=1e-2,
        atol=2e-3,
    )


def test_mm_rejects_m_gt_1() -> None:
    require_sparkinfer()

    source = torch.zeros((2, 128), dtype=torch.float8_e4m3fn, device="cuda")
    weight = torch.zeros((64, 128), dtype=torch.float8_e4m3fn, device="cuda")
    channel_scale = torch.ones((64,), dtype=torch.float32, device="cuda")
    packed = tensor_fp8_channel_linear.pack_weight(weight, channel_scale)
    row_scale = torch.ones((1,), dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="requires M=1"):
        tensor_fp8_channel_linear.mm(source, packed, row_scale)


def test_mm_default_path_captures() -> None:
    require_sparkinfer()
    require_mxf8_mma()
    torch.manual_seed(20260803)

    source, _, row_scale, _, packed = _make_inputs(128, 128)
    eager = tensor_fp8_channel_linear.mm(source, packed, row_scale).clone()
    torch.cuda.synchronize()

    tensor_fp8_channel_linear.prewarm(packed, [1])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = tensor_fp8_channel_linear.mm(source, packed, row_scale)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, eager, rtol=0, atol=0)


def test_is_supported_honors_kernel_probe(monkeypatch) -> None:
    monkeypatch.setattr(
        tensor_fp8_channel_api, "default_is_supported", lambda *args, **kw: True
    )
    monkeypatch.setattr(
        tensor_fp8_channel_api,
        "_kernel_is_supported",
        lambda: (False, "plain FP8 MMA unavailable"),
    )

    assert not tensor_fp8_channel_api.is_supported()
