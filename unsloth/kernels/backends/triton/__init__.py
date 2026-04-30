"""Built-in Triton kernel backend registrations."""

from ._adapter import register_impl
from ..._optional_triton import HAS_TRITON
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
from ...rope_embedding import Fast_RoPE_Embedding, _triton_fast_rope_embedding
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
    "unsloth.rope_embedding",
    "unsloth.rope_embedding_qk",
    "unsloth.swiglu_bwd",
    "unsloth.swiglu_fg",
    "unsloth.w8a8_block_fp8_matmul",
    "unsloth.weight_dequant",
)


_REGISTRATIONS = {
    "unsloth.act_quant":             _triton_act_quant,
    "unsloth.cross_entropy_loss":    _triton_fast_cross_entropy_loss,
    "unsloth.geglu_approx_backward": _triton_geglu_approx_backward_kernel,
    "unsloth.geglu_approx_forward":  _triton_geglu_approx_forward_kernel,
    "unsloth.geglu_exact_backward":  _triton_geglu_exact_backward_kernel,
    "unsloth.geglu_exact_forward":   _triton_geglu_exact_forward_kernel,
    "unsloth.grouped_gemm":          _triton_grouped_gemm,
    "unsloth.layernorm":             _fast_layernorm_triton,
    "unsloth.rms_layernorm":         _fast_rms_layernorm_triton,
    "unsloth.rope_embedding":        Fast_RoPE_Embedding.apply,
    "unsloth.rope_embedding_qk":     _triton_fast_rope_embedding,
    "unsloth.swiglu_bwd":            _triton_swiglu_DWf_DW_dfg_kernel,
    "unsloth.swiglu_fg":             _triton_swiglu_fg_kernel,
    "unsloth.w8a8_block_fp8_matmul": w8a8_block_fp8_matmul_triton,
    "unsloth.weight_dequant":        _triton_weight_dequant_block,
}

# Only register the Triton implementations if Triton is actually importable.
# Without this guard, an unconditional `import unsloth.kernels.backends.triton`
# on a CuTile-only install would populate the registry with wrappers that
# crash at launch time inside _KernelStub.__getitem__ -- the dispatch chain
# would prefer 'triton' over the eager fallback and surface a confusing
# RuntimeError instead of cleanly degrading.
if HAS_TRITON:
    for _op_name, _impl in _REGISTRATIONS.items():
        register_impl(_op_name)(_impl)


def load_backend() -> None:
    return None
