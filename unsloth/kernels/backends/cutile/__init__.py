# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""CuTile implementations for unsloth suite."""

import importlib.util
import sys
from types import ModuleType


def _install_tile_experimental_compat() -> None:
    if importlib.util.find_spec("cuda.tile_experimental") is not None:
        return
    # If neither tile nor tile_experimental is installed, leave the shim
    # uninstalled. The submodule imports below will surface a clearer
    # ImportError pointing at the real missing dependency.
    if importlib.util.find_spec("cuda.tile") is None:
        return

    import cuda.tile as ct

    tile_experimental = ModuleType("cuda.tile_experimental")

    def autotune_launch(
        stream,
        *,
        grid_fn,
        kernel,
        args_fn,
        hints_fn = None,
        search_space = None,
    ):
        config = None
        if search_space is not None:
            configs = search_space() if callable(search_space) else search_space
            config = next(iter(configs), None)
        grid = grid_fn(config)
        args = args_fn(config)
        return ct.launch(stream, grid, kernel, args)

    def clear_autotune_cache():
        return None

    tile_experimental.autotune_launch = autotune_launch
    tile_experimental.clear_autotune_cache = clear_autotune_cache
    sys.modules["cuda.tile_experimental"] = tile_experimental


_install_tile_experimental_compat()

# Shared CuTile op helpers (sigmoid, erf, etc.)
from . import ct_ops  # noqa: F401
from .cross_entropy_loss import cross_entropy_loss as cross_entropy_loss_fn
from .fp8 import act_quant
from .fp8 import w8a8_block_fp8_matmul_cutile
from .fp8 import weight_dequant_block
from .geglu import geglu_approx_backward
from .geglu import geglu_approx_forward
from .geglu import geglu_exact_backward
from .geglu import geglu_exact_forward
from .grouped_gemm import grouped_gemm_cutile
from .layernorm import layernorm as layernorm_fn
from .rms_layernorm import rms_layernorm
from .rope_embedding import rope_embedding
from .rope_embedding import rope_embedding_qk
from .swiglu import swiglu_bwd
from .swiglu import swiglu_fg

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


def load_backend() -> None:
    return None

__all__ = [
    "act_quant",
    "cross_entropy_loss_fn",
    "ct_ops",
    "geglu_approx_backward",
    "geglu_approx_forward",
    "geglu_exact_backward",
    "geglu_exact_forward",
    "grouped_gemm_cutile",
    "layernorm_fn",
    "rms_layernorm",
    "rope_embedding",
    "rope_embedding_qk",
    "swiglu_bwd",
    "swiglu_fg",
    "w8a8_block_fp8_matmul_cutile",
    "weight_dequant_block",
]
