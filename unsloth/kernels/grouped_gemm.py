# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from ._backend_registry import register_kernel_backend
from ._optional_triton import HAS_TRITON
from .moe.grouped_gemm.reference.moe_ops import (
    get_routing_indices,
    permute,
    torch_grouped_gemm,
    unpermute,
)

if HAS_TRITON:
    from .moe.grouped_gemm.interface import grouped_gemm as _triton_grouped_gemm
else:
    _triton_grouped_gemm = None


def _grouped_gemm_eager(
    X: torch.Tensor,
    W: torch.Tensor,
    m_sizes: torch.Tensor,
    topk: int,
    gather_indices: torch.Tensor | None = None,
    permute_x: bool = False,
    permute_y: bool = False,
    topk_weights: torch.Tensor | None = None,
    fuse_mul_post: bool = False,
    is_first_gemm: bool = True,
    dX_only: bool = False,
    dW_only: bool = False,
    **_unused,
) -> torch.Tensor:
    """Eager grouped GEMM. Accepts and discards backend-specific kwargs
    (``kernel_config_*``, ``autotune``) so every backend share a single
    public signature.
    """
    del is_first_gemm, dX_only, dW_only

    # Use reshape(...) instead of view(...) so non-contiguous inputs (e.g.
    # transposed activations) don't crash before the kernel even runs.
    X = X.reshape(-1, X.shape[-1])
    m_sizes = m_sizes.reshape(-1)
    if gather_indices is not None:
        gather_indices = gather_indices.reshape(-1)

    if permute_x:
        if gather_indices is None:
            raise ValueError("gather_indices is required when permute_x is True")
        X = permute(X, gather_indices, topk)

    output = torch_grouped_gemm(X, W, m_sizes, transpose = True)

    if permute_y:
        if gather_indices is None:
            raise ValueError("gather_indices is required when permute_y is True")
        output = unpermute(output, gather_indices)

    if fuse_mul_post:
        if topk_weights is None:
            raise ValueError("topk_weights is required when fuse_mul_post is True")
        output = output * topk_weights.reshape(-1, 1).to(dtype = output.dtype)

    return output


register_kernel_backend(
    "unsloth.grouped_gemm",
    "eager",
    _grouped_gemm_eager,
)


# Static default alias rebound by hook on global backend changes. The exported
# `grouped_gemm` wrapper below resolves per call so context-local backend
# overrides do not leak through process-wide module alias mutation.
_resolved_grouped_gemm = None
grouped_gemm_default = None


def _rebind_grouped_gemm_aliases(backend = None) -> None:
    """Hook fired by `_backend_registry` when the global backend changes
    or when a new backend impl is registered. Looks up the resolved impl
    for the current global backend and pins it to the static default alias.
    Falls back to eager."""
    global _resolved_grouped_gemm, grouped_gemm_default
    try:
        from ._backend_registry import get_kernel_impl
        _resolved_grouped_gemm = get_kernel_impl("unsloth.grouped_gemm")
    except Exception:
        _resolved_grouped_gemm = None
    grouped_gemm_default = _resolved_grouped_gemm or _grouped_gemm_eager


def grouped_gemm(*args, **kwargs):
    from ._backend_registry import get_kernel_impl
    try:
        impl = get_kernel_impl("unsloth.grouped_gemm")
    except Exception:
        impl = _grouped_gemm_eager
    return impl(*args, **kwargs)


# Initial bind. Wrapped so module load never fails if no backend is loaded yet.
try:
    _rebind_grouped_gemm_aliases()
except Exception:
    pass

# Register hook so global-backend changes (set_kernel_backend / kernel_backend_context)
# rebind the alias.
try:
    from ._backend_registry import register_global_backend_change_hook as _register_global_backend_change_hook
    _register_global_backend_change_hook(_rebind_grouped_gemm_aliases)
except Exception:
    pass


_rebind_grouped_gemm_aliases()
