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

import sys
import torch
import torch.nn.functional as F
from ._backend_registry import register_kernel_backend
from ._optional_triton import HAS_TRITON, tl, triton
from .utils import calculate_settings, torch_gpu_device

# signed int32 max is 2**31-1 so num_elements cannot exceed 2**31
NUM_INT32_ELEMENTS = 2**31
SAFE_INT32_BUFFER_MULTIPLIER = 4
BLOCK_SIZE = 1024
INT32_SAFETY_BUFFER = NUM_INT32_ELEMENTS - BLOCK_SIZE * SAFE_INT32_BUFFER_MULTIPLIER


@triton.jit
def _fg_kernel(
    e,
    g,
    h,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(
            tl.int64
        )
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask = mask, other = 0).to(tl.float32)
    g_row = tl.load(g + offsets, mask = mask, other = 0)  # .to(tl.float32)

    # f = e * sigmoid(e)
    f_row = e_row * tl.sigmoid(e_row)  # e_row / (1 + tl.exp(-e_row))
    f_row = f_row.to(g_row.dtype)  # Exact copy from HF
    # h = f * g
    h_row = f_row * g_row

    # Store h
    tl.store(h + offsets, h_row, mask = mask)


def swiglu_fg_kernel(e, g):
    batch, seq_len, hd = e.shape
    n_elements = e.numel()
    h = torch.empty((batch, seq_len, hd), dtype = e.dtype, device = e.device)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_gpu_device(e.device):
        _fg_kernel[grid](
            e,
            g,
            h,
            n_elements,
            BLOCK_SIZE = BLOCK_SIZE,
            LONG_INDEXING = 0 if n_elements <= INT32_SAFETY_BUFFER else 1,
        )
    return h


@triton.jit
def _DWf_DW_dfg_kernel(
    DW,
    e,
    g,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    """
    e = e.float()
    se = 1.0 / (1.0 + torch.exp(-e))
    f = (se * e).to(dtype)
    h = f * g
    df = DW * f
    dg = DW * g
    de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)
    """
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(
            tl.int64
        )
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    DW_row = tl.load(DW + offsets, mask = mask, other = 0)  # .to(tl.float32)
    e_row = tl.load(e + offsets, mask = mask, other = 0).to(tl.float32)
    g_row = tl.load(g + offsets, mask = mask, other = 0)  # .to(tl.float32)

    # e = e.float()
    # se = 1.0 / (1.0 + torch.exp(-e))
    se_row = tl.sigmoid(e_row)  # 1.0 / (1.0 + tl.exp(-e_row))
    # f = (se * e).to(dtype)
    f_row = se_row * e_row
    f_row = f_row.to(DW_row.dtype)
    # h = f * g
    h_row = f_row * g_row
    # df = DW * f
    df_row = DW_row * f_row
    # dg = DW * g
    dg_row = DW_row * g_row
    # de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)
    de_row = dg_row.to(tl.float32) * se_row * (1.0 + e_row * (1.0 - se_row))
    de_row = de_row.to(DW_row.dtype)

    # Store derivatives in buffers
    tl.store(DW + offsets, h_row, mask = mask)  # h  = f * g
    tl.store(e + offsets, df_row, mask = mask)  # df = DW * f
    tl.store(g + offsets, de_row, mask = mask)  # de


def swiglu_DWf_DW_dfg_kernel(DW, e, g):
    batch_seq_len, hd = e.shape  # Flattened to 2D, so 1st dim is bsz * seq_len
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_gpu_device(e.device):
        _DWf_DW_dfg_kernel[grid](
            DW,
            e,
            g,
            n_elements,
            BLOCK_SIZE = BLOCK_SIZE,
            LONG_INDEXING = 0 if n_elements <= INT32_SAFETY_BUFFER else 1,
        )
    return DW, e, g


_triton_swiglu_fg_kernel = swiglu_fg_kernel


def _swiglu_fg_eager(e, g):
    # Mirror the Triton kernel exactly: silu(e) is computed in fp32 then
    # downcast back to input dtype; the f * g product is then in input dtype.
    f = F.silu(e.float()).to(g.dtype)
    return f * g


register_kernel_backend("unsloth.swiglu_fg", "eager", _swiglu_fg_eager)


_triton_swiglu_DWf_DW_dfg_kernel = swiglu_DWf_DW_dfg_kernel


def _swiglu_DWf_DW_dfg_eager(DW, e, g):
    # Mirror the Triton kernel:
    #   e upcast to fp32, sigmoid + silu in fp32
    #   f = silu(e) downcast to input dtype
    #   h = f * g, df = DW * f, dg = DW * g all in input dtype
    #   de uses dg upcast-back-to-fp32 for the chain rule, then downcast
    out_dtype = DW.dtype
    e_f = e.float()
    sig = torch.sigmoid(e_f)
    f = (sig * e_f).to(out_dtype)
    h = f * g
    df = DW * f
    dg = DW * g
    de = (dg.float() * sig * (1.0 + e_f * (1.0 - sig))).to(out_dtype)
    return h, df, de


register_kernel_backend("unsloth.swiglu_bwd", "eager", _swiglu_DWf_DW_dfg_eager)


# Module-level aliases rebound by hook on backend change. None when no
# implementation is registered for the resolved global backend yet.
_resolved_swiglu_fg = None
_resolved_swiglu_bwd = None
swiglu_fg_kernel_default = None
swiglu_DWf_DW_dfg_kernel_default = None


def _patch_swiglu_hot_imports() -> None:
    fast_lora_mod = sys.modules.get("unsloth.kernels.fast_lora")
    if fast_lora_mod is not None:
        fast_lora_mod.swiglu_fg_kernel = swiglu_fg_kernel_default
        fast_lora_mod.swiglu_DWf_DW_dfg_kernel = swiglu_DWf_DW_dfg_kernel_default


def _rebind_swiglu_aliases(backend = None) -> None:
    """Hook fired by `_backend_registry` when the global backend changes
    or when a new backend impl is registered. Looks up the resolved impl
    for the current global backend and pins it to the module-level aliases.
    Falls back to `None` so the entry point routes through `dispatch_kernel`,
    preserving runtime backend switches without a per-call backend kwarg."""
    global _resolved_swiglu_fg, _resolved_swiglu_bwd
    global swiglu_fg_kernel_default, swiglu_DWf_DW_dfg_kernel_default
    try:
        from ._backend_registry import get_kernel_impl
        _resolved_swiglu_fg = get_kernel_impl("unsloth.swiglu_fg")
    except Exception:
        _resolved_swiglu_fg = None
    try:
        from ._backend_registry import get_kernel_impl
        _resolved_swiglu_bwd = get_kernel_impl("unsloth.swiglu_bwd")
    except Exception:
        _resolved_swiglu_bwd = None
    # NVIDIA_REVIEW: hot LoRA paths use these backend-specific symbols, not the
    # public backend dispatcher below. This keeps explicit backend selection out
    # of torch-traced training/generate call shapes.
    swiglu_fg_kernel_default = _resolved_swiglu_fg or _swiglu_fg_eager
    swiglu_DWf_DW_dfg_kernel_default = _resolved_swiglu_bwd or _swiglu_DWf_DW_dfg_eager
    globals()["swiglu_fg_kernel"] = swiglu_fg_kernel_default
    globals()["swiglu_DWf_DW_dfg_kernel"] = swiglu_DWf_DW_dfg_kernel_default
    _patch_swiglu_hot_imports()


# Initial bind. Wrapped so module load never fails if no backend is loaded yet.
try:
    _rebind_swiglu_aliases()
except Exception:
    pass

# Register hook so global-backend changes (set_kernel_backend / kernel_backend_context)
# rebind the aliases.
try:
    from ._backend_registry import register_global_backend_change_hook as _register_global_backend_change_hook
    _register_global_backend_change_hook(_rebind_swiglu_aliases)
except Exception:
    pass


def swiglu_fg_kernel(e, g):
    return swiglu_fg_kernel_default(e, g)


def swiglu_DWf_DW_dfg_kernel(DW, e, g):
    return swiglu_DWf_DW_dfg_kernel_default(DW, e, g)


_rebind_swiglu_aliases()
