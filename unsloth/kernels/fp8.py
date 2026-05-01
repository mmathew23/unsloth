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
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from ._backend_registry import dispatch_kernel, get_kernel_backend, register_kernel_backend
from ._optional_triton import HAS_TRITON, tl, triton
from unsloth_zoo.utils import Version
from unsloth_zoo.log import logger
from unsloth_zoo.temporary_patches.common import torch_compile

torch_matmul = torch.matmul

try:
    from transformers.integrations.finegrained_fp8 import FP8Linear
except:
    FP8Linear = None
    logger.info(
        "Unsloth: FP8 models need importing FP8Linear from `transformers.integrations.finegrained_fp8` but we don't see it."
    )

try:
    from transformers.integrations.fbgemm_fp8 import FbgemmFp8Linear
except:
    FbgemmFp8Linear = None
    logger.info(
        "Unsloth: FP8 models need importing FbgemmFP8Linear from `transformers.integrations.fbgemm_fp8` but we don't see it."
    )

try:
    from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
        triton_quantize_fp8_block,
    )
except:
    triton_quantize_fp8_block = None
    logger.info(
        "Unsloth: Could not find fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm.triton_quantize_fp8_block"
    )

try:
    from torchao.prototype.blockwise_fp8_inference.blockwise_quantization import (
        blockwise_fp8_gemm as torchao_blockwise_gemm,
    )
except:
    torchao_blockwise_gemm = None
    logger.info(
        "Unsloth: Could not find torchao.prototype.blockwise_fp8_inference.blockwise_quantization.blockwise_fp8_gemm"
    )


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis = 0)
    pid_n = tl.program_id(axis = 1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask = mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask = mask)


def weight_dequant_block(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128, dtype = torch.bfloat16
) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    if not s.is_contiguous():
        s = s.contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype = dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE = block_size)
    return y


_triton_weight_dequant_block = weight_dequant_block


def _weight_dequant_block_eager(
    x: torch.Tensor,
    s: torch.Tensor,
    block_size: int = 128,
    dtype = torch.bfloat16,
) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    if not s.is_contiguous():
        s = s.contiguous()
    M, N = x.shape
    expected_shape = (math.ceil(M / block_size), math.ceil(N / block_size))
    if s.shape == expected_shape[::-1]:
        s = s.t().contiguous()
    elif s.shape != expected_shape:
        raise ValueError(
            f"Scale shape {s.shape} is incompatible with weight shape {x.shape} and block size {block_size}"
        )
    row_scales = torch.repeat_interleave(s, block_size, dim = 0)[:M]
    full_scales = torch.repeat_interleave(row_scales, block_size, dim = 1)[:, :N]
    return x.to(dtype = dtype) * full_scales.to(dtype = dtype)


register_kernel_backend(
    "unsloth.weight_dequant",
    "eager",
    _weight_dequant_block_eager,
)


def weight_dequant_block(
    x: torch.Tensor,
    s: torch.Tensor,
    block_size: int = 128,
    dtype = torch.bfloat16,
    *,
    backend = None,
) -> torch.Tensor:
    return dispatch_kernel(
        "unsloth.weight_dequant",
        x,
        s,
        block_size,
        dtype,
        backend = backend,
    )


def weight_dequant(
    x: torch.Tensor,
    s: torch.Tensor,
    dtype = torch.bfloat16,
    *,
    backend = None,
):
    # Per-tensor scale: single value for entire weight matrix
    if s.numel() == 1:
        return x.to(dtype) * s.view(1, 1).to(dtype)
    # Row quantized weight: scale shape is (m, 1) or (n, 1)
    elif s.ndim == 2 and s.shape[1] == 1:
        if x.shape[0] == s.shape[0]:
            y = x.to(dtype) * s.to(dtype)
        elif x.shape[1] == s.shape[0]:
            # sometimes, this is called with the transpose of the weight. Adjust for that.
            y = x.t().to(dtype) * s.to(dtype)
            y = y.t()
        else:
            raise ValueError(f"Incompatible shapes {x.shape = }, {s.shape = }")
        return y
    # Block quantized weight: scale shape is (ceil(m/block_m), ceil(n/block_n))
    else:
        return weight_dequant_block(x, s, dtype = dtype, backend = backend)


# Copied from https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py
@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    # For a row of all zeros, lets return zeros as is
    # for LoRA, there are cases where dY has 0 in it and we should not let it be NaN
    # this is a deviation from the original implementation.
    s = 1.0 if s == 0 else s
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(
    x: torch.Tensor, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_contiguous():
        x = x.contiguous()
    assert x.shape[-1] % block_size == 0
    y = torch.empty_like(x, dtype = torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype = torch.float32)

    def grid(meta):
        return (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)

    act_quant_kernel[grid](x, y, s, BLOCK_SIZE = block_size)
    return y, s


_triton_act_quant = act_quant


def _act_quant_eager(
    x: torch.Tensor, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_contiguous():
        x = x.contiguous()
    assert x.shape[-1] % block_size == 0
    num_blocks = x.shape[-1] // block_size
    x_blocks = x.to(torch.float32).reshape(*x.shape[:-1], num_blocks, block_size)
    abs_max = x_blocks.abs().amax(dim = -1)
    scale = abs_max / 448.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    y = (x_blocks / scale.unsqueeze(-1)).reshape_as(x).to(torch.float8_e4m3fn)
    return y, scale.contiguous()


register_kernel_backend(
    "unsloth.act_quant",
    "eager",
    _act_quant_eager,
)


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    *,
    backend = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Hot-path bypass: when fp8_linear has already resolved the backend it
    # passes a known-good name in. Skip the dispatcher (and its lock /
    # ContextVar reads / env scans) for the two backends that are
    # implemented locally in this module. Cutile's act_quant lives in the
    # cutile backend package, so it still goes through dispatch_kernel.
    if backend == "triton":
        return _triton_act_quant(x, block_size)
    if backend == "eager":
        return _act_quant_eager(x, block_size)
    return dispatch_kernel(
        "unsloth.act_quant",
        x,
        block_size,
        backend = backend,
    )


# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8_kernel.py
@triton.jit
def _w8a8_block_fp8_matmul(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and
    store the result in output tensor `C`.
    """

    pid = tl.program_id(axis = 0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask = offs_k[None, :] < K - k * BLOCK_SIZE_K, other = 0.0)
        b = tl.load(b_ptrs, mask = offs_k[:, None] < K - k * BLOCK_SIZE_K, other = 0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask = c_mask)


def w8a8_block_fp8_matmul_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Block-wise FP8 matmul."""
    if block_size is None:
        block_n, block_k = 128, 128
    else:
        assert len(block_size) == 2
        block_n, block_k = block_size[0], block_size[1]

    N, K = B.shape
    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    M = A.numel() // A.shape[-1]
    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype = output_dtype)

    BLOCK_SIZE_M = 128
    if M < BLOCK_SIZE_M:
        BLOCK_SIZE_M = max(triton.next_power_of_2(M), 16)
    BLOCK_SIZE_K, BLOCK_SIZE_N = block_k, block_n

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    _w8a8_block_fp8_matmul[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        BLOCK_SIZE_M = BLOCK_SIZE_M,
        BLOCK_SIZE_N = BLOCK_SIZE_N,
        BLOCK_SIZE_K = BLOCK_SIZE_K,
        GROUP_SIZE_M = 8,
    )
    return C


def torchao_block_matmul(
    act_q: torch.Tensor,
    weight_q: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: tuple[int, int],
    output_dtype: torch.dtype = torch.bfloat16,
):
    out = torchao_blockwise_gemm(
        act_q.contiguous(),
        act_scale.contiguous(),
        weight_q.contiguous(),
        weight_scale.contiguous(),
        block_size = block_size[1],
    )
    return out.to(output_dtype)


def _w8a8_block_fp8_matmul_eager(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if block_size is None:
        block_n, block_k = 128, 128
    else:
        block_n, block_k = block_size[0], block_size[1]

    N, K = B.shape
    M = A.numel() // A.shape[-1]

    A_f = A.reshape(M, K).to(torch.float32)
    B_f = B.to(torch.float32)

    # As: (..., ceil(K/block_k)) -> (M, ceil(K/block_k))
    As_f = As.reshape(M, -1).to(torch.float32)
    A_scale = As_f.repeat_interleave(block_k, dim = -1)[..., :K]
    A_scaled = A_f * A_scale

    # Bs: (ceil(N/block_n), ceil(K/block_k))
    Bs_f = Bs.to(torch.float32)
    B_scale = Bs_f.repeat_interleave(block_n, dim = 0)[:N, :]
    B_scale = B_scale.repeat_interleave(block_k, dim = 1)[:, :K]
    B_scaled = B_f * B_scale

    out = A_scaled @ B_scaled.t()
    return out.reshape(A.shape[:-1] + (N,)).to(output_dtype)


register_kernel_backend(
    "unsloth.w8a8_block_fp8_matmul",
    "eager",
    _w8a8_block_fp8_matmul_eager,
)


# Note that older versions of fbgemm (<=1.3.0) cause numerical imprecisions resulting in NaNs especially when X has high values in it.
# So our preference order is fbgemm (>=1.4.0) > torchao > triton. All of these have similar outputs/losses. Never use fbgemm (<=1.3.0) for block quantized FP8 matmul.
# This torchao FP8 matmul seems to be ~3x faster than the w8a8_block_fp8_matmul_triton. Though torchao is 15-30% slower than fbgemm implementation (on H100 GPUs).
def fp8_block_matmul(
    act_q: torch.Tensor,
    weight_q: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: tuple[int, int],
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    backend = None,
):
    # Hot-path bypass for backends implemented locally in this module.
    # fp8_linear already resolves the backend once and threads it in; this
    # avoids a second `get_kernel_backend` (and its lock + env scan) per
    # FP8 layer. Triton additionally has a torchao-backed alternative
    # preferred when available.
    resolved = backend if backend is not None else get_kernel_backend(
        "unsloth.w8a8_block_fp8_matmul",
    )
    if resolved == "triton":
        if torchao_blockwise_gemm is not None:
            return torchao_block_matmul(
                act_q,
                weight_q,
                act_scale,
                weight_scale,
                block_size,
                output_dtype = output_dtype,
            )
        return w8a8_block_fp8_matmul_triton(
            act_q,
            weight_q,
            act_scale,
            weight_scale,
            block_size,
            output_dtype = output_dtype,
        )
    if resolved == "eager":
        return _w8a8_block_fp8_matmul_eager(
            act_q,
            weight_q,
            act_scale,
            weight_scale,
            block_size,
            output_dtype = output_dtype,
        )
    # Cutile or any other registered backend: go through the dispatcher.
    return dispatch_kernel(
        "unsloth.w8a8_block_fp8_matmul",
        act_q,
        weight_q,
        act_scale,
        weight_scale,
        block_size,
        output_dtype,
        backend = resolved,
    )


class FP8BlockQuantLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, weight_scale, bias = None, backend = None):
        m, n = weight.shape

        # Save original scale for backward (before any transformation)
        original_weight_scale = weight_scale

        # Handle per-tensor quantization: expand scalar to block scale shape
        if weight_scale.numel() == 1:
            block_size = [128, 128]
            # Expand scalar to (ceil(m/128), ceil(n/128)) - same value for all blocks
            num_blocks_m = triton.cdiv(m, block_size[0])
            num_blocks_n = triton.cdiv(n, block_size[1])
            weight_scale = weight_scale.expand(num_blocks_m, num_blocks_n).contiguous()
        else:
            # Block quantization path
            p, q = weight_scale.shape
            block_size = getattr(weight, "block_size", None) or getattr(
                weight_scale, "block_size", [128, 128]
            )
            assert block_size is not None, "block_size is not set"
            if triton.cdiv(m, block_size[0]) != p or triton.cdiv(n, block_size[1]) != q:
                if (
                    triton.cdiv(m, block_size[0]) == q
                    and triton.cdiv(n, block_size[1]) == p
                ):
                    weight_scale = weight_scale.T
                    original_weight_scale = weight_scale  # Update for transposed case
                else:
                    raise ValueError(
                        f"Weight shape {weight.shape} and scales shape {weight_scale.shape} is not compatible with block size {block_size}"
                    )

        if not weight.is_contiguous():
            weight = weight.contiguous()

        # Quantize input and run FP8 matmul
        qinput, scale = act_quant(X, block_size[1], backend = backend)
        output = fp8_block_matmul(
            qinput,
            weight,
            scale,
            weight_scale,
            block_size,
            output_dtype = X.dtype,
            backend = backend,
        )
        output = output.to(X.dtype)
        if bias is not None:
            output = output + bias.to(output.dtype)
        ctx.weight = weight
        ctx.weight_scale = original_weight_scale  # Save original for backward
        # ctx.bias is only needed as a {None, not-None} flag for d_bias gating;
        # the bias tensor itself is not used in backward. Storing the boolean
        # avoids holding a tensor reference on the autograd graph.
        ctx.bias_used = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Backend is intentionally not threaded through ctx — weight_dequant
        # resolves it via the cached registry path on every call (~50 ns
        # post-fix), so an explicit ctx.backend slot only added a Python
        # attr write per forward without changing the resolution result.
        W_deq = weight_dequant(
            ctx.weight,
            ctx.weight_scale,
            dtype = grad_output.dtype,
        )
        grad_X = torch_matmul(grad_output, W_deq)
        del W_deq
        if ctx.bias_used:
            d_bias = grad_output.sum(dim = tuple(range(grad_output.ndim - 1)))
        else:
            d_bias = None
        return grad_X, None, None, d_bias, None


@torch_compile
def fp8_torch_block_quant_forward(X, weight, weight_scale, bias = None, *, backend = None):
    return FP8BlockQuantLinear.apply(X, weight, weight_scale, bias, backend)


# Lean 3-arg autograd Function matching PyPI's exact signature.  Dynamo sees a
# clean 3-tensor Function with no extra args, no bias/backend None slots, and
# no ctx.bias_used attribute — identical to what PyPI compiles.  This lets the
# outer HF `mode='reduce-overhead'` compile absorb the call without creating a
# new compile boundary for the extra None arguments.
#
# This Function is only reached from `module_forward_patch`'s `weight_scale_inv`
# branch when `fp8_block_quant_linear is fp8_torch_block_quant_forward` — i.e.
# when the fbgemm block-quant rebind did NOT happen.  Whatever block-fp8 backend
# is resolved (triton today, cutile / future kernels later) gets baked into the
# module-level `_BAKED_BLOCK_FP8_BACKEND` constant below and passed down to
# `act_quant` / `fp8_block_matmul`. With a non-None backend kwarg those calls
# either hit their local early-exit fast paths (triton, eager) or go through
# `dispatch_kernel` with the backend already resolved — either way they skip
# the `_RUNTIME_GLOBAL_BACKEND.get()` ContextVar read, which dynamo can't trace
# and which would otherwise spawn a fresh compile region per FP8 layer per
# decoded token inside HF generate's static-cache loop (the "compile_id=1,
# Region 1/X" we were chasing in the GRPO -10% gap).
class FP8BlockQuantLinearLean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, weight_scale):
        m, n = weight.shape

        # Save original scale for backward (before any transformation)
        original_weight_scale = weight_scale

        # Handle per-tensor quantization: expand scalar to block scale shape
        if weight_scale.numel() == 1:
            block_size = [128, 128]
            # Expand scalar to (ceil(m/128), ceil(n/128)) - same value for all blocks
            num_blocks_m = triton.cdiv(m, block_size[0])
            num_blocks_n = triton.cdiv(n, block_size[1])
            weight_scale = weight_scale.expand(num_blocks_m, num_blocks_n).contiguous()
        else:
            # Block quantization path
            p, q = weight_scale.shape
            block_size = getattr(weight, "block_size", None) or getattr(
                weight_scale, "block_size", [128, 128]
            )
            assert block_size is not None, "block_size is not set"
            if triton.cdiv(m, block_size[0]) != p or triton.cdiv(n, block_size[1]) != q:
                if (
                    triton.cdiv(m, block_size[0]) == q
                    and triton.cdiv(n, block_size[1]) == p
                ):
                    weight_scale = weight_scale.T
                    original_weight_scale = weight_scale  # Update for transposed case
                else:
                    raise ValueError(
                        f"Weight shape {weight.shape} and scales shape {weight_scale.shape} is not compatible with block size {block_size}"
                    )

        if not weight.is_contiguous():
            weight = weight.contiguous()

        # Direct-bound impls (see `_hot_act_quant` / `_hot_fp8_block_matmul`
        # below at L1006). For the resolved backend, these are the SAME plain
        # functions PyPI calls — no wrapper, no kwarg branching. Falls back
        # to the dispatcher path only for backends without a direct binding.
        if _hot_act_quant is not None:
            qinput, scale = _hot_act_quant(X, block_size[1])
            output = _hot_fp8_block_matmul(
                qinput,
                weight,
                scale,
                weight_scale,
                block_size,
                output_dtype = X.dtype,
            )
        else:
            qinput, scale = act_quant(
                X, block_size[1], backend = _BAKED_BLOCK_FP8_BACKEND,
            )
            output = fp8_block_matmul(
                qinput,
                weight,
                scale,
                weight_scale,
                block_size,
                output_dtype = X.dtype,
                backend = _BAKED_BLOCK_FP8_BACKEND,
            )
        ctx.weight = weight
        ctx.weight_scale = original_weight_scale  # Save original for backward
        return output.to(X.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        W_deq = weight_dequant(
            ctx.weight,
            ctx.weight_scale,
            dtype = grad_output.dtype,
            backend = _BAKED_BLOCK_FP8_BACKEND,
        )
        grad_X = torch_matmul(grad_output, W_deq)
        del W_deq
        return grad_X, None, None


@torch_compile
def _fp8_torch_block_quant_forward_lean(X, weight, weight_scale):
    return FP8BlockQuantLinearLean.apply(X, weight, weight_scale)


class FbgemmFp8Linear_matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, weight_scale, bias = None, backend = None):
        if weight.shape[0] == weight_scale.shape[0] and (
            weight.shape[0] % 8 == 0 and weight.shape[1] % 8 == 0
        ):
            # Edit: The kernel seems to expect that the weight has dimensions divisible by 8. Otherwise it throws `RuntimeError: cutlass cannot implement`
            # One thing we can do is to pad the weight and weight scale to multiple of 8 and perform a F8F8BF16 operation.
            # I tried benchmarking that for speed but observed that dequantize+bf16 matmul is significantly faster than padding+f8f8bf16 matmul. So we'll go that route.
            # So essentially, f8f8bf16_rowise only happens when shapes are proper (no transposes) and divisible by 8.

            # quantize_fp8_per_row will squash the leading dimensions, so save the desired shape here
            output_shape = (*x.shape[:-1], -1)
            # x_quantized and x_scale are not necessarily on the same device as x, this is an issue.
            # https://github.com/pytorch/FBGEMM/blob/e08af8539c391437f447173863df0f3f6f6f1855/fbgemm_gpu/experimental/gen_ai/src/quantize/quantize.cu#L1237C3-L1237C45
            x_quantized, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(
                x.view(-1, x.shape[-1]).contiguous(),
                scale_ub = getattr(weight, "input_scale_ub", None),
            )
            # moving x_quantized, x_scale here creates glibberish output ... However, if we move the output, it works
            # x_quantized, x_scale = x_quantized.to(x.device), x_scale.to(x.device)

            # The computation still happens on the device where self.weight is even if x_quantized is not on the same device as self.weight
            weight_scale_float32 = weight_scale.to(torch.float32)

            if not weight.is_contiguous():
                weight = weight.contiguous()
            if not weight_scale.is_contiguous():
                weight_scale = weight_scale.contiguous()

            output = torch.ops.fbgemm.f8f8bf16_rowwise(
                x_quantized, weight, x_scale, weight_scale_float32, use_fast_accum = True
            )
            if bias is not None:
                output = output + bias.to(output.dtype)
            # Hacky for now, we have the output to the device of x
            output = output.to(x.device, x.dtype)
            output = output.reshape(output_shape)
            del x_quantized, x_scale
        elif (
            weight.shape[0] != weight_scale.shape[0]
            and weight.shape[1] == weight_scale.shape[0]
        ) or (weight.shape[0] % 8 != 0 or weight.shape[1] % 8 != 0):
            # Either the weight/scale is transposed or its shape is not divisible by 8. Both cases, dequantizing is the preferred way.
            # The transpose case is generally noticed in backward pass when we do dY@W instead of @W.T as we do for forward.
            # The shape case, I noticed to happen in MLP of Qwen 2.5 VL 7B where the gate proj is of shape (3420, 1280) and 3420/8=427.5

            W_deq = weight_dequant(
                weight,
                weight_scale,
                dtype = x.dtype,
                backend = backend,
            ).T
            output = torch_matmul(x, W_deq)
            del W_deq
        else:
            raise ValueError(
                f"Shapes are incompatible {weight.shape = }, {weight_scale.shape = }, {x.shape = }"
            )

        ctx.weight = weight
        ctx.weight_scale = weight_scale
        ctx.bias_used = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        W_deq = weight_dequant(
            ctx.weight,
            ctx.weight_scale,
            dtype = grad_output.dtype,
        )
        grad_X = torch_matmul(grad_output, W_deq)
        del W_deq
        if ctx.bias_used:
            d_bias = grad_output.sum(dim = tuple(range(grad_output.ndim - 1)))
        else:
            d_bias = None
        return grad_X, None, None, d_bias, None


@torch_compile
def fbgemm_fp8_linear(X, weight, weight_scale, bias = None, *, backend = None):
    return FbgemmFp8Linear_matmul.apply(X, weight, weight_scale, bias, backend)


class FP8_fbgemm_block_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, weight_scale, bias = None, backend = None):
        orig_shape = X.shape
        X = X.view(-1, X.shape[-1])

        bs_n, bs_k = getattr(weight, "block_size", None) or getattr(
            weight_scale, "block_size", [128, 128]
        )
        bs_m = bs_n

        m, n = weight.shape
        p, q = weight_scale.shape

        if triton.cdiv(m, bs_n) != p or triton.cdiv(n, bs_k) != q:
            if triton.cdiv(m, bs_n) == q and triton.cdiv(n, bs_k) == p:
                # weights are transposed during backward pass for training :)
                # We transpose weight scale to counter that. Note that transposing weight would cause issues with matmul with input X
                weight_scale = weight_scale.T
            else:
                raise ValueError(
                    f"Weight shape {weight.shape} and scales shape {weight_scale.shape} is not compatible with block size {bs_n, bs_k}"
                )

        xq, xs = triton_quantize_fp8_block(X, bs_m, bs_n, None)
        ## TODO: Investigate and resolve the high divergence of this output from baseline
        # WARNING: This causes the outputs to diverge from expected when X has high values in it.
        # That results in the model producing gibberish, especially on longer sequences and training loss starting at high values like 8 instead of <1 ideally
        # Please refrain from using this till this issue is resolved. This exists here just for a future headstart.
        output = torch.ops.fbgemm.f8f8bf16_blockwise(
            xq, weight.contiguous(), xs, weight_scale.contiguous(), bs_m, bs_n, bs_k
        )
        if bias is not None:
            output = output + bias.to(output.dtype)

        output = output.view(*orig_shape[:-1], -1)

        del xq
        del xs

        ctx.weight = weight
        ctx.weight_scale = weight_scale
        ctx.block_size = [bs_m, bs_n, bs_k]
        ctx.bias_used = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        W_deq = weight_dequant(
            ctx.weight,
            ctx.weight_scale,
            dtype = grad_output.dtype,
        )
        grad_X = torch_matmul(grad_output, W_deq)
        del W_deq
        if ctx.bias_used:
            d_bias = grad_output.sum(dim = tuple(range(grad_output.ndim - 1)))
        else:
            d_bias = None
        return grad_X, None, None, d_bias, None


@torch_compile
def fp8_fbgemm_block_linear(X, weight, weight_scale, bias = None, *, backend = None):
    return FP8_fbgemm_block_linear.apply(X, weight, weight_scale, bias, backend)


def test_has_fbgemm():
    # We must manually check if the faster FBGEMM works on the specific GPU
    # For example RTX 5090 and RTX 4090 does not work
    # Also SM100 (Blackwell B200/B100) GPUs fail with CUTLASS SM90 kernels
    # [TODO] Investigate with TorchAO why FBGEMM fails on consumer GPUs
    M, N, K = 128, 128, 128
    xq = torch.ones(M, K, dtype = torch.float8_e4m3fn, device = "cuda")
    wq = xq
    M, K = xq.shape
    N, _ = wq.shape
    block_scale = torch.ones(M // 128, K // 128, dtype = torch.float32, device = "cuda")
    has_fbgemm = False
    try:
        out = torch.ops.fbgemm.f8f8bf16_blockwise(xq, wq, block_scale, block_scale)
        assert torch.unique(out).item() == 128
        has_fbgemm = True
        del out
    except Exception as e:
        error_str = str(e).lower()
        # Catch any CUTLASS/CUDA errors and disable FBGEMM
        # This includes MMA instruction errors, architecture mismatches, kernel launch failures, etc.
        cutlass_cuda_errors = (
            "cutlass",
            "cuda error",
            "cuda runtime error",
            "no kernel image",
            "arch conditional",
            "mma instruction",
            "compute capability",
            "cute_invalid_control_path",
            "tma",
        )
        is_cutlass_cuda_error = any(err in error_str for err in cutlass_cuda_errors)

        if is_cutlass_cuda_error:
            print(
                "Unsloth: FBGEMM on the current GPU cannot load - will switch to Triton kernels"
            )
        else:
            print(
                f"Unsloth: FBGEMM on the current GPU cannot load with error = {e} - will switch to Triton kernels"
            )
        has_fbgemm = False
    del block_scale, xq
    torch.cuda.empty_cache()
    return has_fbgemm


def _fp8_linear_eager(X, weight, weight_scale, bias = None):
    W_deq = weight_dequant(
        weight,
        weight_scale,
        dtype = X.dtype,
        backend = "eager",
    )
    output = torch_matmul(X, W_deq.t())
    if bias is not None:
        output = output + bias
    return output


def _has_rowwise_fbgemm_ops() -> bool:
    """Return True iff ``torch.ops.fbgemm.f8f8bf16_rowwise`` is available.

    Independent from the blockwise FBGEMM probe (``UNSLOTH_HAS_FBGEMM``):
    that one tests a CUTLASS kernel which may fail on consumer GPUs even
    when FBGEMM is otherwise installed. The rowwise op is a different
    entry point that can be present (or absent) independently. Without
    this guard, ``fp8_linear``'s row-wise branch on a non-eager backend
    (e.g. cutile) would call ``fbgemm_fp8_linear`` and crash with
    ``AttributeError`` at first invocation when fbgemm_gpu isn't installed.
    """
    try:
        return hasattr(torch.ops.fbgemm, "f8f8bf16_rowwise")
    except (AttributeError, RuntimeError):
        return False


_HAS_FBGEMM_ROWWISE = _has_rowwise_fbgemm_ops()

# One-shot warn flag for row-wise FP8 → eager fallback. The hot path of
# row-wise FP8 is FBGEMM-only (no triton/cutile equivalent registered);
# without fbgemm_gpu it silently runs at eager speed, which is the user's
# real surprise. Warn once at first hit so logs show why perf is flat.
_warned_fp8_rowwise_eager_fallback = False


def _warn_fp8_rowwise_eager_fallback_once() -> None:
    global _warned_fp8_rowwise_eager_fallback
    if _warned_fp8_rowwise_eager_fallback:
        return
    _warned_fp8_rowwise_eager_fallback = True
    logger.warning(
        "Unsloth: Row-wise FP8 detected but `fbgemm_gpu` is not installed. "
        "Falling back to eager (slower). For best row-wise FP8 performance, "
        "install fbgemm_gpu."
    )

fp8_block_quant_linear = fp8_torch_block_quant_forward
if "UNSLOTH_HAS_FBGEMM" not in os.environ:
    os.environ["UNSLOTH_HAS_FBGEMM"] = "0"
try:
    import fbgemm_gpu

    # Older versions cause numerical imprecisions resulting in NaNs especially when X has high values in it.
    # This is both fast and accurate hence preferred.
    # This makes it 15% faster than the torchao implementation.
    if Version(fbgemm_gpu.__version__) >= Version("1.4.0"):
        # We must manually confirm if blockwise FBGEMM works!
        # This check is a must for consumer grade GPUs which fail
        # Suppress CUDA device printf during probe -- on Blackwell (SM100) GPUs,
        # FBGEMM's CUTLASS blockwise kernel (hardcoded SM90) fires thousands of
        # "Arch conditional MMA" lines to stdout fd 1 before aborting.
        from unsloth.import_fixes import suppress_cuda_printf

        with suppress_cuda_printf():
            _has_fbgemm = test_has_fbgemm()
        if _has_fbgemm:
            os.environ["UNSLOTH_HAS_FBGEMM"] = "1"
            logger.info(f"Using fbgemm_gpu block quantized FP8 matmul")
            fp8_block_quant_linear = fp8_fbgemm_block_linear
        else:
            os.environ["UNSLOTH_HAS_FBGEMM"] = "0"
except:
    pass


_KNOWN_LOCAL_BACKENDS = ("triton", "eager", "cutile")


# Block-FP8 backend resolved once at module load time and threaded through the
# lean autograd Function (see `FP8BlockQuantLinearLean` above). Any non-None
# value lets `act_quant` / `fp8_block_matmul` / `weight_dequant` skip the
# `_RUNTIME_GLOBAL_BACKEND.get()` ContextVar read inside their dispatch path,
# which dynamo can't trace through.  Wrapped in try/except so that early
# imports (before any backend loader has registered) silently fall back to
# `None`, which restores the original dispatcher path — same safety net as
# `module_forward_patch`'s `baked_backend` resolution.
try:
    _BAKED_BLOCK_FP8_BACKEND = get_kernel_backend("unsloth.w8a8_block_fp8_matmul")
except Exception:
    _BAKED_BLOCK_FP8_BACKEND = None


# Direct bindings for the FP8 generate hot path. Mirror PyPI's pattern (see
# pypi/unsloth/kernels/fp8.py L323-327): pick the impl at import time and
# alias it to a module-level name so `FP8BlockQuantLinearLean.forward` can
# call it with no wrapper, no backend kwarg, no per-call branching.
#
# The dispatcher entry points (`act_quant` redefined at L232,
# `fp8_block_matmul` at L469) are still used by callers that pass `backend=`
# explicitly or need runtime switching via `kernel_backend_context`.  This
# only short-circuits the *generate-time* hot path where switching is not
# meaningful (the patched `forward` was bound at model load).
#
# Resolved-backend → impl table.  Future builtin backends (cutile, future
# kernels) add their own branch here.  When a backend has no direct binding
# (`None`), the lean Function falls back to the dispatcher path.
if _BAKED_BLOCK_FP8_BACKEND == "triton":
    _hot_act_quant = _triton_act_quant
    _hot_fp8_block_matmul = (
        torchao_block_matmul
        if torchao_blockwise_gemm is not None
        else w8a8_block_fp8_matmul_triton
    )
elif _BAKED_BLOCK_FP8_BACKEND == "eager":
    _hot_act_quant = _act_quant_eager
    _hot_fp8_block_matmul = _w8a8_block_fp8_matmul_eager
else:
    _hot_act_quant = None
    _hot_fp8_block_matmul = None


@torch_compile
def fp8_linear(X, weight, weight_scale, bias = None, *, backend = None):
    # `get_kernel_backend` reads `_RUNTIME_GLOBAL_BACKEND.get()` (a
    # `ContextVar.get()`) which dynamo cannot trace and forces a graph break
    # inside this `@torch_compile`-decorated function — splitting fp8_linear's
    # compiled graph into compile_id=0 + a `torch_dynamo_resume_in_fp8_linear`
    # in compile_id=1, doubling the per-call dispatch cost on the GRPO
    # generate hot path.  Three cases, all dynamo-friendly:
    #   1. `backend is None` (the default — what `fast_linear_forward` and
    #      `module_forward_patch` pass when they didn't pre-resolve): use the
    #      module-level `_BAKED_BLOCK_FP8_BACKEND` (resolved once at import).
    #   2. `backend` is a known builtin: trust it.
    #   3. Anything else (rare — explicit user request via `kernel_backend_context`
    #      with a custom backend name): fall back to the dispatcher.  This case
    #      DOES graph-break, but it's not the hot path.
    if backend is None:
        resolved_backend = _BAKED_BLOCK_FP8_BACKEND
    elif backend in _KNOWN_LOCAL_BACKENDS:
        resolved_backend = backend
    else:
        resolved_backend = get_kernel_backend(
            "unsloth.w8a8_block_fp8_matmul",
            backend = backend,
        )
    # Per-tensor quantization: single scalar scale for entire weight
    # Block quantized FP8: 2D scale tensor with multiple columns
    if weight_scale.numel() == 1 or (
        weight_scale.ndim == 2 and weight_scale.shape[1] > 1
    ):
        if resolved_backend == "eager":
            out = _fp8_linear_eager(X, weight, weight_scale, bias)
        else:
            # fp8_block_quant_linear is FBGEMM when probed OK at import,
            # else fp8_torch_block_quant_forward which dispatches via the
            # registry to the requested backend (triton or cutile).
            out = fp8_block_quant_linear(
                X,
                weight,
                weight_scale,
                bias = bias,
                backend = resolved_backend,
            )
    # Row/channel quantized FP8: 2D scale with shape (n, 1)
    else:
        # Fall to eager if either the user asked for it OR rowwise FBGEMM
        # ops aren't available (cutile-only install, FBGEMM not installed).
        # Without the second condition, fbgemm_fp8_linear would crash with
        # AttributeError on torch.ops.fbgemm.f8f8bf16_rowwise.
        if resolved_backend == "eager" or not _HAS_FBGEMM_ROWWISE:
            # Only warn when the user wanted non-eager perf and we silently
            # downgraded due to a missing optional dep. If the user picked
            # eager themselves, they got what they asked for.
            if not _HAS_FBGEMM_ROWWISE and resolved_backend != "eager":
                _warn_fp8_rowwise_eager_fallback_once()
            out = _fp8_linear_eager(X, weight, weight_scale, bias)
        else:
            out = fbgemm_fp8_linear(
                X,
                weight,
                weight_scale,
                bias,
                backend = resolved_backend,
            )
    return out


def module_forward_patch(scale_attr = "weight_scale"):
    # Resolve both the kernel impl and backend ONCE at patch time.
    #
    # Critical: call the resolved impl pointer (same as PyPI's approach)
    # rather than fp8_linear. fp8_linear is itself @torch_compile, so calling
    # it from the compiled model forward creates a double-compiled-region
    # dispatch per FP8 layer (Region 0/0 in profiler) adding ~202 µs ×
    # 252 layers × decode_steps of overhead per generate(). Calling the
    # resolved impl directly matches the PyPI call depth and eliminates that
    # extra dispatch layer.
    #
    # Routing is encoded by scale_attr: FP8Linear has `weight_scale_inv`
    # (block-quantized, 2D shape with shape[1] > 1) and FbgemmFp8Linear has
    # `weight_scale` (rowwise, 2D shape (n, 1)). Picking the right impl at
    # patch time avoids needing the runtime branch in fp8_linear.
    #
    # The baked_backend is passed as a closure constant so dynamo can
    # constant-fold it and avoid ContextVar / RLock reads inside the compiled
    # body. Wrapped in try/except for platforms where resolution can fail at
    # import time (e.g. cutile-only install before the cutile loader runs).
    try:
        baked_backend = get_kernel_backend("unsloth.w8a8_block_fp8_matmul")
    except Exception:
        baked_backend = None

    if scale_attr == "weight_scale_inv":
        # FP8Linear → block quantization. When the resolved pointer is the
        # plain Torch path (no FBGEMM block probe), call the lean 3-arg
        # @torch_compile wrapper directly so the call-site shape matches PyPI
        # exactly: `patched_forward → _fp8_torch_block_quant_forward_lean →
        # FP8BlockQuantLinearLean.apply`. No intermediate Python adapter, no
        # extra arg slots. Bias is intentionally NOT threaded into this hot
        # path: PyPI's `module_forward_patch` also ignores `self.bias`, and
        # FP8 dense layers in the supported model families (Qwen3-FP8,
        # DeepSeek-FP8) are bias-free. The 5-arg `fp8_block_quant_linear` /
        # `fp8_linear` paths still support bias for callers that go through
        # them directly (LoRA path, the FP8 bias tests).
        if fp8_block_quant_linear is fp8_torch_block_quant_forward:
            # Match PyPI's exact call shape: patched_forward → @torch_compile
            # wrapper → autograd.Function.apply.  PyPI inlines the inner
            # @torch_compile into HF generate's outer compile cleanly (single
            # compile_id=0, no graph-break boundary), so this should too.
            _wrapper = _fp8_torch_block_quant_forward_lean
            def patched_forward(self, X):
                return _wrapper(X, self.weight, getattr(self, scale_attr))
            return patched_forward
        _impl = fp8_block_quant_linear
    else:
        # FbgemmFp8Linear → rowwise quantization. Falls back to eager when
        # fbgemm_gpu rowwise ops are missing (matches the original
        # fp8_linear branch logic).
        if _HAS_FBGEMM_ROWWISE:
            _impl = fbgemm_fp8_linear
        else:
            def _impl(X, weight, weight_scale, bias = None, *, backend = None):
                if backend is not None and backend != "eager":
                    _warn_fp8_rowwise_eager_fallback_once()
                return _fp8_linear_eager(X, weight, weight_scale, bias)

    def patched_forward(self, X):
        return _impl(
            X,
            self.weight,
            getattr(self, scale_attr),
            getattr(self, "bias", None),
            backend = baked_backend,
        )

    return patched_forward


# Patch the forward functions of the layers (for compiled models). Route through
# fp8_linear() so the runtime backend selection (and bias) are both honored.
if FbgemmFp8Linear is not None:
    FbgemmFp8Linear.forward = module_forward_patch("weight_scale")
if FP8Linear is not None:
    # On the block-FP8 hot path, use the EXACT pypi pattern: a 2-line patched
    # forward that closes over `_fp8_torch_block_quant_forward_lean` (a
    # @torch_compile-decorated 3-arg passthrough).  Skip workspace's
    # dispatcher-aware module_forward_patch entirely for this hot path —
    # profiler-confirmed parity with pypi requires this exact call shape so
    # dynamo inlines the inner compile into HF generate's outer compile.
    if fp8_block_quant_linear is fp8_torch_block_quant_forward:
        _hot_block_fp8 = _fp8_torch_block_quant_forward_lean
        def _patched_block_fp8(self, X):
            return _hot_block_fp8(X, self.weight, self.weight_scale_inv)
        FP8Linear.forward = _patched_block_fp8
    else:
        FP8Linear.forward = module_forward_patch("weight_scale_inv")
