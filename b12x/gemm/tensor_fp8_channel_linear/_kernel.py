from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from b12x._lib.dense_gemm import dense_gemm
from b12x.gemm.tensor_fp8_linear._kernel import (
    MXFP8_SCALE_VEC_SIZE,
    _check_gpu_tensor,
    _cached_unit_scale_mma,
    _dense_gemm_kwargs_for_n,
    _output_dtype_name,
    _pad_k,
    _source_2d,
    _unit_scale_mma,
    is_tensor_fp8_linear_supported,
)


@dataclass(frozen=True)
class TensorFP8ChannelLinearWeight:
    """Serialized E4M3 weight and one FP32 scale per output channel."""

    values: torch.Tensor
    scale_mma: torch.Tensor
    channel_scale: torch.Tensor
    in_features: int
    padded_in_features: int
    out_features: int


def is_tensor_fp8_channel_linear_supported() -> tuple[bool, str | None]:
    return is_tensor_fp8_linear_supported()


def pack_tensor_fp8_channel_linear_weight(
    weight: torch.Tensor,
    channel_scale: torch.Tensor,
) -> TensorFP8ChannelLinearWeight:
    """Pack an E4M3 ``[N,K]`` weight with one FP32 scale per output channel."""

    _check_gpu_tensor("weight", weight)
    _check_gpu_tensor("channel_scale", channel_scale)
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [N,K], got {tuple(weight.shape)}")
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"weight must be float8_e4m3fn, got {weight.dtype}")
    if channel_scale.dtype != torch.float32:
        raise ValueError(f"channel_scale must be float32, got {channel_scale.dtype}")
    if channel_scale.device != weight.device:
        raise ValueError("weight and channel_scale must be on the same device")
    out_features, in_features = map(int, weight.shape)
    if tuple(channel_scale.shape) not in ((out_features,), (out_features, 1)):
        raise ValueError(
            "channel_scale must have shape "
            f"{(out_features,)} or {(out_features, 1)}, got {tuple(channel_scale.shape)}"
        )
    if not bool(torch.isfinite(channel_scale).all()) or bool((channel_scale < 0).any()):
        raise ValueError("channel_scale must be finite and non-negative")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if in_features <= 0 or in_features % MXFP8_SCALE_VEC_SIZE != 0:
        raise ValueError(
            "tensor FP8 weight K must be a positive multiple of "
            f"{MXFP8_SCALE_VEC_SIZE}, got {in_features}"
        )

    padded_in_features = ((in_features + 127) // 128) * 128
    values = _pad_k(weight, padded_in_features)
    scale_mma = _unit_scale_mma(out_features, padded_in_features, weight.device)
    return TensorFP8ChannelLinearWeight(
        values=values,
        scale_mma=scale_mma,
        channel_scale=channel_scale.reshape(out_features).contiguous(),
        in_features=in_features,
        padded_in_features=padded_in_features,
        out_features=out_features,
    )


def tensor_fp8_channel_linear(
    source: torch.Tensor,
    packed_weight: TensorFP8ChannelLinearWeight,
    row_scale: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    expected_m: int | None = None,
    stream: object = None,
) -> torch.Tensor:
    """Run M=1 FP8 decode with one row scale and one scale per output channel."""

    _check_gpu_tensor("source", source)
    _check_gpu_tensor("row_scale", row_scale)
    if not isinstance(packed_weight, TensorFP8ChannelLinearWeight):
        raise TypeError("packed_weight must be a TensorFP8ChannelLinearWeight")
    source_2d = _source_2d(source)
    tokens, in_features = map(int, source_2d.shape)
    if tokens != 1:
        raise ValueError(f"tensor_fp8_channel_linear currently requires M=1, got M={tokens}")
    if row_scale.dtype != torch.float32 or row_scale.numel() != 1:
        raise ValueError(
            "row_scale must be one float32 value, got "
            f"dtype={row_scale.dtype}, shape={tuple(row_scale.shape)}"
        )
    if source_2d.dtype != torch.float8_e4m3fn:
        raise ValueError(f"source must be float8_e4m3fn, got {source_2d.dtype}")
    if in_features != int(packed_weight.in_features):
        raise ValueError(
            f"input K={in_features} does not match packed weight K={packed_weight.in_features}"
        )
    if source_2d.device != packed_weight.values.device:
        raise ValueError("source and packed weight must be on the same device")
    if row_scale.device != source_2d.device:
        raise ValueError("row_scale must be on the same device as source")
    if expected_m is not None and int(expected_m) <= 0:
        raise ValueError("expected_m must be positive when provided")
    _output_dtype_name(out_dtype)

    out_features = int(packed_weight.out_features)
    if bias is not None:
        _check_gpu_tensor("bias", bias)
        if bias.device != source_2d.device:
            raise ValueError("bias must be on the same device as source")
        if bias.dtype != out_dtype or bias.shape != (out_features,):
            raise ValueError(
                f"bias must have shape {(out_features,)} and dtype {out_dtype}, "
                f"got shape={tuple(bias.shape)}, dtype={bias.dtype}"
            )

    source_padded = _pad_k(source_2d, int(packed_weight.padded_in_features))
    device_index = source_2d.device.index
    if source_2d.device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    source_scale_mma = _cached_unit_scale_mma(
        source_2d.device.type,
        device_index,
        tokens,
        int(packed_weight.padded_in_features),
    )
    output = dense_gemm(
        (
            source_padded.reshape(tokens, packed_weight.padded_in_features, 1),
            source_scale_mma,
        ),
        (
            packed_weight.values.reshape(
                out_features, packed_weight.padded_in_features, 1
            ),
            packed_weight.scale_mma,
        ),
        alpha=row_scale.reshape(1).contiguous(),
        alpha_col=packed_weight.channel_scale,
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype=_output_dtype_name(out_dtype),
        sf_vec_size=32,
        expected_m=int(expected_m) if expected_m is not None else tokens,
        stream=stream,
        plain_fp8=True,
        **_dense_gemm_kwargs_for_n(out_features),
    )[:, :, 0]
    if bias is not None:
        output = output + bias
    return output.view(*source.shape[:-1], out_features)


def prewarm_tensor_fp8_channel_linear(
    packed_weight: TensorFP8ChannelLinearWeight,
    token_counts: Iterable[int],
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    stream: object = None,
) -> int:
    """Compile and cache serving shapes before CUDA graph capture."""

    warmed = 0
    with torch.inference_mode():
        for tokens in sorted({int(value) for value in token_counts if int(value) > 0}):
            if tokens != 1:
                continue
            source = torch.zeros(
                (tokens, packed_weight.in_features),
                dtype=torch.float8_e4m3fn,
                device=packed_weight.values.device,
            )
            row_scale = torch.ones((1,), dtype=torch.float32, device=source.device)
            tensor_fp8_channel_linear(
                source,
                packed_weight,
                row_scale,
                out_dtype=out_dtype,
                expected_m=tokens,
                stream=stream,
            )
            warmed += 1
    return warmed


__all__ = [
    "TensorFP8ChannelLinearWeight",
    "is_tensor_fp8_channel_linear_supported",
    "pack_tensor_fp8_channel_linear_weight",
    "prewarm_tensor_fp8_channel_linear",
    "tensor_fp8_channel_linear",
]
