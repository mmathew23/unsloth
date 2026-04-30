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

from ._backend_registry import dispatch_kernel, register_kernel_backend
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

    X = X.view(-1, X.shape[-1])
    m_sizes = m_sizes.view(-1)
    if gather_indices is not None:
        gather_indices = gather_indices.view(-1)

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
        output = output * topk_weights.view(-1, 1).to(dtype = output.dtype)

    return output


register_kernel_backend(
    "unsloth.grouped_gemm",
    "eager",
    _grouped_gemm_eager,
)


def grouped_gemm(
    X: torch.Tensor,
    W: torch.Tensor,
    m_sizes: torch.Tensor,
    topk: int,
    gather_indices: torch.Tensor | None = None,
    permute_x: bool = False,
    permute_y: bool = False,
    topk_weights: torch.Tensor | None = None,
    fuse_mul_post: bool = False,
    kernel_config_fwd = None,
    kernel_config_bwd_dX = None,
    kernel_config_bwd_dW = None,
    autotune: bool = False,
    is_first_gemm: bool = True,
    dX_only: bool = False,
    dW_only: bool = False,
    *,
    backend: str | None = None,
) -> torch.Tensor:
    return dispatch_kernel(
        "unsloth.grouped_gemm",
        X,
        W,
        m_sizes,
        topk,
        gather_indices = gather_indices,
        permute_x = permute_x,
        permute_y = permute_y,
        topk_weights = topk_weights,
        fuse_mul_post = fuse_mul_post,
        kernel_config_fwd = kernel_config_fwd,
        kernel_config_bwd_dX = kernel_config_bwd_dX,
        kernel_config_bwd_dW = kernel_config_bwd_dW,
        autotune = autotune,
        is_first_gemm = is_first_gemm,
        dX_only = dX_only,
        dW_only = dW_only,
        backend = backend,
    )
