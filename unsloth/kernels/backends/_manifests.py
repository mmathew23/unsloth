"""Static backend manifests used by tests and lightweight diagnostics."""

TRITON_EXPECTED_OPS = (
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

# UNSLOTH_TILEGYM_DIFF_REVIEW: TileGym's suite package only exports kernels.
# Unsloth's backend registry expects an explicit operation manifest and loader
# hook, so keep these local unless TileGym grows an equivalent backend API.
CUTILE_EXPECTED_OPS = (
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
