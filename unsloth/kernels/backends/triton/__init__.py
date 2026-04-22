"""Built-in Triton kernel backend registrations."""

from ._adapter import register_impl
from ...cross_entropy_loss import _triton_fast_cross_entropy_loss
from ...fp8 import (
    _triton_act_quant,
    _triton_weight_dequant_block,
    w8a8_block_fp8_matmul_triton,
)
from ...geglu import (
    _triton_geglu_approx_backward_kernel,
    _triton_geglu_approx_forward_kernel,
    _triton_geglu_exact_backward_kernel,
    _triton_geglu_exact_forward_kernel,
)
from ...grouped_gemm import _triton_grouped_gemm
from ...layernorm import _fast_layernorm_triton
from ...rms_layernorm import _fast_rms_layernorm_triton
from ...rope_embedding import _triton_fast_rope_embedding
from ...swiglu import (
    _triton_swiglu_DWf_DW_dfg_kernel,
    _triton_swiglu_fg_kernel,
)

EXPECTED_OPS = (
    "unsloth.act_quant",
    "unsloth.cross_entropy_loss",
    "unsloth.geglu_approx_backward",
    "unsloth.geglu_approx_forward",
    "unsloth.geglu_exact_backward",
    "unsloth.geglu_exact_forward",
    "unsloth.grouped_gemm",
    "unsloth.layernorm",
    "unsloth.rms_layernorm",
    "unsloth.rope_embedding_qk",
    "unsloth.swiglu_bwd",
    "unsloth.swiglu_fg",
    "unsloth.w8a8_block_fp8_matmul",
    "unsloth.weight_dequant",
)


@register_impl("unsloth.layernorm")
def _register_layernorm(*args, **kwargs):
    return _fast_layernorm_triton(*args, **kwargs)


@register_impl("unsloth.rms_layernorm")
def _register_rms_layernorm(*args, **kwargs):
    return _fast_rms_layernorm_triton(*args, **kwargs)


@register_impl("unsloth.cross_entropy_loss")
def _register_cross_entropy(*args, **kwargs):
    return _triton_fast_cross_entropy_loss(*args, **kwargs)


@register_impl("unsloth.swiglu_fg")
def _register_swiglu_fg(*args, **kwargs):
    return _triton_swiglu_fg_kernel(*args, **kwargs)


@register_impl("unsloth.swiglu_bwd")
def _register_swiglu_bwd(*args, **kwargs):
    return _triton_swiglu_DWf_DW_dfg_kernel(*args, **kwargs)


@register_impl("unsloth.geglu_exact_forward")
def _register_geglu_exact_forward(*args, **kwargs):
    return _triton_geglu_exact_forward_kernel(*args, **kwargs)


@register_impl("unsloth.geglu_exact_backward")
def _register_geglu_exact_backward(*args, **kwargs):
    return _triton_geglu_exact_backward_kernel(*args, **kwargs)


@register_impl("unsloth.geglu_approx_forward")
def _register_geglu_approx_forward(*args, **kwargs):
    return _triton_geglu_approx_forward_kernel(*args, **kwargs)


@register_impl("unsloth.geglu_approx_backward")
def _register_geglu_approx_backward(*args, **kwargs):
    return _triton_geglu_approx_backward_kernel(*args, **kwargs)


@register_impl("unsloth.grouped_gemm")
def _register_grouped_gemm(*args, **kwargs):
    return _triton_grouped_gemm(*args, **kwargs)


@register_impl("unsloth.rope_embedding_qk")
def _register_rope_embedding_qk(*args, **kwargs):
    return _triton_fast_rope_embedding(*args, **kwargs)


@register_impl("unsloth.weight_dequant")
def _register_weight_dequant(*args, **kwargs):
    return _triton_weight_dequant_block(*args, **kwargs)


@register_impl("unsloth.act_quant")
def _register_act_quant(*args, **kwargs):
    return _triton_act_quant(*args, **kwargs)


@register_impl("unsloth.w8a8_block_fp8_matmul")
def _register_w8a8_block_fp8_matmul(*args, **kwargs):
    return w8a8_block_fp8_matmul_triton(*args, **kwargs)


def load_backend() -> None:
    return None
