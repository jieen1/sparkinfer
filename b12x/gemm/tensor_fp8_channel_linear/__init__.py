"""Static per-channel FP8 linear for SM12x decode.

``pack_weight`` keeps serialized E4M3 weights unchanged, prepares unit UE8M0
scale-factor storage once, and records one FP32 scale per output channel.
``mm`` consumes an already-quantized E4M3 activation plus one FP32 row scale.
The kernel fuses ``row_scale * channel_scale`` into the direct M=1 epilogue.

Example:
    from b12x.gemm import tensor_fp8_channel_linear

    packed = tensor_fp8_channel_linear.pack_weight(weight_fp8, channel_scale)
    output = tensor_fp8_channel_linear.mm(input_fp8, packed, row_scale)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="tensor_fp8_channel_linear",
    group="gemm",
    api_style="oneshot",
    entry_points=("Weight", "mm", "pack_weight", "prewarm", "is_supported"),
    dtypes=("fp8_e4m3", "bf16", "fp16"),
    recipes=("tensor_fp8",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/sparkinfer",
        commit="1bc4f82",
        paths=("sparkinfer/gemm/tensor_fp8_channel_linear",),
    ),
    test_path="tests/gemm/test_tensor_fp8_channel_linear.py",
    since="1.0.1",
)

if TYPE_CHECKING:
    from .api import Weight, is_supported, mm, pack_weight, prewarm  # noqa: F401

install_lazy_api(globals(), META)
