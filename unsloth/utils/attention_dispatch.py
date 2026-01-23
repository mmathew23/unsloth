# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Shared helpers for attention backend selection and execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

from ..models._utils import *

# NJT (Nested Jagged Tensor) configuration
# Set UNSLOTH_USE_NJT=1 to enable NJT attention for variable-length sequences
UNSLOTH_USE_NJT = os.environ.get("UNSLOTH_USE_NJT", "0") == "1"

# NJT logging support
_NJT_LOGGING_ENABLED = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"
_NJT_LOGGED_KEYS: set = set()


def _log_njt_usage(message: str, once: bool = False, key: str = None) -> None:
    """Log NJT usage when UNSLOTH_ENABLE_LOGGING=1.

    Args:
        message: The message to log.
        once: If True, only log this message once per key.
        key: Unique key for deduplication when once=True. If None, uses message.
    """
    if not _NJT_LOGGING_ENABLED:
        return
    log_key = key or message
    if once and log_key in _NJT_LOGGED_KEYS:
        return
    print(f"🦥 Unsloth NJT: {message}")
    if once:
        _NJT_LOGGED_KEYS.add(log_key)

import torch
from ..utils.packing import (
    build_sdpa_packed_attention_mask,
    build_xformers_block_causal_mask,
)

if HAS_FLASH_ATTENTION:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
HAS_XFORMERS = xformers is not None
SDPA_HAS_GQA = "enable_gqa" in (scaled_dot_product_attention.__doc__ or "")

FLASH_VARLEN = "flash_varlen"
FLASH_DENSE = "flash_dense"
XFORMERS = "xformers"
SDPA = "sdpa"
NJT_SDPA = "njt_sdpa"  # Nested Jagged Tensor SDPA backend


# =============================================================================
# NJT Tensor Helpers
# =============================================================================

def is_njt(tensor: Tensor) -> bool:
    """Check if a tensor is a Nested Jagged Tensor."""
    return tensor is not None and hasattr(tensor, "is_nested") and tensor.is_nested


def njt_reshape_for_heads(x: Tensor, num_heads: int, head_dim: int) -> Tensor:
    """
    Reshape NJT tensor from (B, S*, hidden) to (B, S*, num_heads, head_dim).

    For NJT, we need to reshape via values() since .view() doesn't work
    directly on the ragged dimension with the hidden dimension.
    """
    if not is_njt(x):
        # Regular tensor: use standard view
        bsz, seq_len = x.shape[:2]
        return x.view(bsz, seq_len, num_heads, head_dim)

    # NJT: reshape values and reconstruct
    values = x.values()  # (total_tokens, hidden)
    values = values.view(-1, num_heads, head_dim)  # (total_tokens, num_heads, head_dim)
    offsets = x.offsets()
    return torch.nested.nested_tensor_from_jagged(values, offsets=offsets)


def njt_transpose_for_attention(x: Tensor) -> Tensor:
    """
    Transpose NJT from (B, S*, num_heads, head_dim) to (B, num_heads, S*, head_dim).

    This is the format expected by scaled_dot_product_attention.
    """
    return x.transpose(1, 2)


def njt_transpose_from_attention(x: Tensor) -> Tensor:
    """
    Transpose NJT from (B, num_heads, S*, head_dim) to (B, S*, num_heads, head_dim).

    This reverses njt_transpose_for_attention.
    """
    return x.transpose(1, 2)


def njt_reshape_from_heads(x: Tensor, hidden_size: int) -> Tensor:
    """
    Reshape NJT tensor from (B, S*, num_heads, head_dim) to (B, S*, hidden).

    For NJT, we need to reshape via values() and reconstruct.
    """
    if not is_njt(x):
        # Regular tensor: use standard view
        bsz, seq_len = x.shape[:2]
        return x.reshape(bsz, seq_len, hidden_size)

    # NJT: reshape values and reconstruct
    values = x.values()  # (total_tokens, num_heads, head_dim)
    values = values.reshape(-1, hidden_size)  # (total_tokens, hidden)
    offsets = x.offsets()
    return torch.nested.nested_tensor_from_jagged(values, offsets=offsets)


XFORMERS_BLOCK_DIAG_CLS = (
    xformers.attn_bias.BlockDiagonalCausalMask if HAS_XFORMERS else None
)


@dataclass
class AttentionConfig:
    """
    Per-layer attention metadata.

    NOTE(djsaunde): I had originally intended this to be populated once per layer, but
        we're currently constructing it on every forward pass since it can possibly be
        invalid from one forward pass to the next (e.g., switching from training to
        inference). For now, I'm keeping separate from AttentionContext for the sake of
        better grouping of params.
    """

    backend: str
    n_kv_heads: int
    n_groups: int
    flash_dense_kwargs: Optional[dict[str, Any]] = None
    flash_varlen_kwargs: Optional[dict[str, Any]] = None
    sdpa_kwargs: Optional[dict[str, Any]] = None
    xformers_kwargs: Optional[dict[str, Any]] = None


@dataclass
class AttentionContext:
    """Per-call info required to run attention."""

    bsz: int
    q_len: int
    kv_seq_len: int
    n_heads: int
    head_dim: int
    requires_grad: bool
    seq_info: Optional[Tuple[Tensor, Tensor, int]]
    attention_mask: Optional[Tensor]
    causal_mask: Optional[Any]
    sliding_window: Optional[int] = None
    # NJT (Nested Jagged Tensor) fields for variable-length sequences
    njt_q: Optional[Tensor] = None
    njt_k: Optional[Tensor] = None
    njt_v: Optional[Tensor] = None


def select_attention_backend(use_varlen: bool = False, use_njt: bool = False) -> str:
    """Return attention backend based on availability / priority order.

    Args:
        use_varlen: Whether to use variable-length (packed) attention.
        use_njt: Whether to use Nested Jagged Tensor SDPA backend.
                 NJT provides efficient variable-length attention without
                 explicit padding/masking. Requires PyTorch 2.5+.

    Returns:
        Backend identifier string.
    """
    # NJT backend takes priority when explicitly requested
    # It uses PyTorch native SDPA with nested tensors for best compile compatibility
    if use_njt:
        _log_njt_usage("Selected NJT_SDPA backend for attention", once=True, key="backend_selection")
        return NJT_SDPA

    if HAS_FLASH_ATTENTION:
        if use_varlen:
            return FLASH_VARLEN
        else:
            return FLASH_DENSE
    if HAS_XFORMERS:
        return XFORMERS
    return SDPA


def _run_njt_attention_auto(
    *,
    config: AttentionConfig,
    context: AttentionContext,
    Q: Tensor,
    K: Tensor,
    V: Tensor,
) -> Tensor:
    """
    Automatically convert packed Q, K, V to NJT format and run attention.

    This is called when UNSLOTH_USE_NJT=1 and seq_info is available.
    Handles the full conversion: packed -> NJT -> attention -> packed.
    """
    seq_lengths, cu_seqlens, max_seqlen = context.seq_info
    bsz = len(seq_lengths)
    n_heads = context.n_heads
    n_kv_heads = config.n_kv_heads
    n_groups = config.n_groups
    head_dim = context.head_dim
    total_tokens = context.bsz * context.q_len

    _log_njt_usage(
        f"Auto-converting to NJT ({bsz} seqs, {total_tokens} tokens, "
        f"heads={n_heads}, kv_heads={n_kv_heads})",
        once=True,
        key="auto_njt_conversion",
    )

    # Q, K, V are in shape (bsz, n_heads, q_len, head_dim)
    # For packed/padding_free, bsz=1 and q_len=total_tokens
    # Convert to (total_tokens, n_heads, head_dim)
    Q_packed = Q.transpose(1, 2).reshape(-1, n_heads, head_dim)
    K_packed = K.transpose(1, 2).reshape(-1, n_kv_heads, head_dim)
    V_packed = V.transpose(1, 2).reshape(-1, n_kv_heads, head_dim)

    # Split into sequences and create NJTs
    seq_lengths_list = seq_lengths.tolist()

    Q_sequences = list(torch.split(Q_packed, seq_lengths_list, dim=0))
    K_sequences = list(torch.split(K_packed, seq_lengths_list, dim=0))
    V_sequences = list(torch.split(V_packed, seq_lengths_list, dim=0))

    # Create NJTs: (B, S*, H, D)
    njt_Q = torch.nested.nested_tensor(Q_sequences, layout=torch.jagged)
    njt_K = torch.nested.nested_tensor(K_sequences, layout=torch.jagged)
    njt_V = torch.nested.nested_tensor(V_sequences, layout=torch.jagged)

    # Transpose for attention: (B, S*, H, D) -> (B, H, S*, D)
    njt_Q = njt_Q.transpose(1, 2)
    njt_K = njt_K.transpose(1, 2)
    njt_V = njt_V.transpose(1, 2)

    # Expand K and V for GQA if needed
    if n_groups != 1:
        # Transpose back: (B, H, S*, D) -> (B, S*, H, D)
        njt_K_seq = njt_K.transpose(1, 2)
        njt_V_seq = njt_V.transpose(1, 2)

        # Get values and expand
        K_vals = njt_K_seq.values()  # (total_tokens, kv_heads, head_dim)
        V_vals = njt_V_seq.values()

        # Expand: (total, kv_heads, D) -> (total, n_heads, D)
        K_expanded = K_vals.unsqueeze(2).expand(-1, -1, n_groups, -1)
        V_expanded = V_vals.unsqueeze(2).expand(-1, -1, n_groups, -1)
        K_expanded = K_expanded.reshape(total_tokens, n_heads, head_dim)
        V_expanded = V_expanded.reshape(total_tokens, n_heads, head_dim)

        # Recreate NJTs with expanded heads
        K_exp_sequences = list(torch.split(K_expanded, seq_lengths_list, dim=0))
        V_exp_sequences = list(torch.split(V_expanded, seq_lengths_list, dim=0))
        njt_K = torch.nested.nested_tensor(K_exp_sequences, layout=torch.jagged).transpose(1, 2)
        njt_V = torch.nested.nested_tensor(V_exp_sequences, layout=torch.jagged).transpose(1, 2)

    # Run NJT SDPA attention
    _log_njt_usage(
        f"Running NJT SDPA (Q: {njt_Q.shape}, K: {njt_K.shape})",
        once=True,
        key="njt_sdpa_run",
    )

    out = scaled_dot_product_attention(
        njt_Q,
        njt_K,
        njt_V,
        is_causal=True,
    )

    # Output is (B, H, S*, D) -> transpose to (B, S*, H, D)
    out_seq = out.transpose(1, 2)

    # Convert back to packed format
    # Get values: (total_tokens, n_heads, head_dim)
    out_packed = out_seq.values()

    # Reshape to match expected output: (bsz, q_len, n_heads, head_dim)
    # For packed format, bsz=1, q_len=total_tokens
    return out_packed.reshape(context.bsz, context.q_len, n_heads, head_dim)


def run_attention(
    *,
    config: AttentionConfig,
    context: AttentionContext,
    Q: Tensor,
    K: Tensor,
    V: Tensor,
) -> Tensor:
    """
    Run attention using config / context info.

    Backend choice is prioritized for speed: FlashAttention when installed
    (`flash_varlen` for packed/variable-length inputs with `seq_info`, otherwise dense
    flash), then xFormers if flash is unavailable, with PyTorch SDPA as the final
    fallback (e.g., CPU or no fused kernels).

    Varlen flash is preferred when packing metadata is present because it avoids padding
    and keeps peak memory low. xFormers and SDPA can also handle packed batches (we
    pass a block-diagonal mask into each).

    When UNSLOTH_USE_NJT=1, automatically converts packed tensors to NJT format
    for efficient variable-length attention without explicit masking.
    """

    # Automatic NJT path: when enabled and we have seq_info (from padding_free mode)
    # Convert packed Q, K, V to NJT, run attention, convert back
    if UNSLOTH_USE_NJT and context.seq_info is not None and context.njt_q is None:
        return _run_njt_attention_auto(
            config=config,
            context=context,
            Q=Q,
            K=K,
            V=V,
        )

    backend = config.backend
    if backend == FLASH_VARLEN and context.seq_info is None:
        backend = FLASH_DENSE if HAS_FLASH_ATTENTION else SDPA
    # Fallback from NJT_SDPA if NJT tensors not provided
    if backend == NJT_SDPA and (
        context.njt_q is None or context.njt_k is None or context.njt_v is None
    ):
        # Fall back to regular SDPA or varlen flash if available
        if context.seq_info is not None and HAS_FLASH_ATTENTION:
            backend = FLASH_VARLEN
        elif HAS_FLASH_ATTENTION:
            backend = FLASH_DENSE
        else:
            backend = SDPA
    flash_dense_kwargs = config.flash_dense_kwargs or {}
    flash_varlen_kwargs = config.flash_varlen_kwargs or {}
    sdpa_kwargs = config.sdpa_kwargs or {}
    xformers_kwargs = config.xformers_kwargs or {}

    bsz = context.bsz
    n_heads = context.n_heads
    q_len = context.q_len
    head_dim = context.head_dim
    kv_seq_len = context.kv_seq_len
    requires_grad = context.requires_grad
    sliding_window = context.sliding_window

    if backend == FLASH_VARLEN:
        Q_f = Q.transpose(1, 2).reshape(bsz * q_len, n_heads, head_dim)
        K_f = K.transpose(1, 2).reshape(bsz * q_len, config.n_kv_heads, head_dim)
        V_f = V.transpose(1, 2).reshape(bsz * q_len, config.n_kv_heads, head_dim)
        _, cu_seqlens, max_seqlen = context.seq_info
        return flash_attn_varlen_func(
            Q_f,
            K_f,
            V_f,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            **flash_varlen_kwargs,
        ).view(bsz, q_len, n_heads, head_dim)
    elif backend == FLASH_DENSE:
        Q_t = Q.transpose(1, 2)
        K_t = K.transpose(1, 2)
        V_t = V.transpose(1, 2)
        return flash_attn_func(Q_t, K_t, V_t, **flash_dense_kwargs).reshape(
            bsz, q_len, n_heads, head_dim
        )
    elif backend == NJT_SDPA:
        # Nested Jagged Tensor SDPA backend
        # NJTs handle masking implicitly via the jagged structure
        # Expected input shapes: (B, num_heads, S*, head_dim) where S* is ragged
        njt_q = context.njt_q
        njt_k = context.njt_k
        njt_v = context.njt_v

        if njt_q is None or njt_k is None or njt_v is None:
            raise ValueError(
                "NJT_SDPA backend requires njt_q, njt_k, njt_v in AttentionContext"
            )

        # Log NJT usage
        _log_njt_usage(
            f"Using NJT_SDPA attention (Q shape: {njt_q.shape}, "
            f"heads: {n_heads}, head_dim: {head_dim})",
            once=True,
            key="run_attention",
        )

        # NJT SDPA with causal attention
        # Note: GQA with NJT may not be fully supported; expand K/V if needed
        if config.n_groups != 1 and not SDPA_HAS_GQA:
            # Need to expand K and V for GQA when enable_gqa not available
            # This is a fallback - NJT ideally works with enable_gqa=True
            raise NotImplementedError(
                "NJT_SDPA with GQA requires PyTorch SDPA with enable_gqa support"
            )

        kwargs = dict(config.sdpa_kwargs or {})
        kwargs["is_causal"] = True

        if SDPA_HAS_GQA and config.n_groups != 1:
            kwargs["enable_gqa"] = True

        out = scaled_dot_product_attention(njt_q, njt_k, njt_v, **kwargs)

        # Output is (B, num_heads, S*, head_dim), transpose to (B, S*, num_heads, head_dim)
        return out.transpose(1, 2)
    elif backend == XFORMERS:
        attn_bias = build_xformers_block_causal_mask(
            context.seq_info,
            sliding_window = sliding_window,
            base_mask = context.causal_mask,
        )

        Q_t = Q.transpose(1, 2)
        K_t = K.transpose(1, 2)
        V_t = V.transpose(1, 2)

        K_mod = K_t
        V_mod = V_t
        Q_mod = Q_t

        if config.n_groups != 1:
            K_mod = K_t.view(bsz, kv_seq_len, config.n_kv_heads, 1, head_dim)
            V_mod = V_t.view(bsz, kv_seq_len, config.n_kv_heads, 1, head_dim)
            K_mod = K_mod.expand(
                bsz, kv_seq_len, config.n_kv_heads, config.n_groups, head_dim
            )
            V_mod = V_mod.expand(
                bsz, kv_seq_len, config.n_kv_heads, config.n_groups, head_dim
            )

            if requires_grad:
                K_mod = K_mod.reshape(bsz, kv_seq_len, n_heads, head_dim)
                V_mod = V_mod.reshape(bsz, kv_seq_len, n_heads, head_dim)
            else:
                Q_mod = Q_t.view(
                    bsz, q_len, config.n_kv_heads, config.n_groups, head_dim
                )

        has_block = XFORMERS_BLOCK_DIAG_CLS is not None and isinstance(
            attn_bias, XFORMERS_BLOCK_DIAG_CLS
        )

        if config.n_groups != 1 and has_block:
            if not requires_grad:
                Q_mod = Q_mod.view(
                    1, bsz * q_len, config.n_kv_heads, config.n_groups, head_dim
                )
                K_mod = K_mod.view(
                    1, bsz * kv_seq_len, config.n_kv_heads, config.n_groups, head_dim
                )
                V_mod = V_mod.view(
                    1, bsz * kv_seq_len, config.n_kv_heads, config.n_groups, head_dim
                )
            else:
                Q_mod = Q_mod.view(1, bsz * q_len, n_heads, head_dim)
                K_mod = K_mod.view(1, bsz * kv_seq_len, n_heads, head_dim)
                V_mod = V_mod.view(1, bsz * kv_seq_len, n_heads, head_dim)

        out = xformers_attention(
            Q_mod,
            K_mod,
            V_mod,
            attn_bias = attn_bias,
            **xformers_kwargs,
        )

        if config.n_groups != 1 and not requires_grad:
            out = out.view(bsz, q_len, config.n_kv_heads, config.n_groups, head_dim)
            out = out.reshape(bsz, q_len, n_heads, head_dim)
        else:
            out = out.view(bsz, q_len, n_heads, head_dim)
        return out
    else:
        local_mask = context.attention_mask
        is_causal_local = False
        if context.seq_info is not None and local_mask is None:
            local_mask = build_sdpa_packed_attention_mask(
                context.seq_info,
                dtype = Q.dtype,
                device = Q.device,
                sliding_window = sliding_window,
            )
        else:
            q_len_local = Q.shape[-2]
            k_len_local = K.shape[-2]
            is_causal_local = local_mask is None and q_len_local == k_len_local

        kwargs = dict(sdpa_kwargs)
        kwargs.setdefault("attn_mask", local_mask)
        kwargs.setdefault("is_causal", is_causal_local)

        if SDPA_HAS_GQA:
            kwargs.setdefault("enable_gqa", config.n_groups != 1)
            out = scaled_dot_product_attention(Q, K, V, **kwargs)
            return out.transpose(1, 2)

        K_mod = K
        V_mod = V
        if config.n_groups != 1:
            K_mod = K[:, :, None, :, :].expand(
                bsz, config.n_kv_heads, config.n_groups, kv_seq_len, head_dim
            )
            V_mod = V[:, :, None, :, :].expand(
                bsz, config.n_kv_heads, config.n_groups, kv_seq_len, head_dim
            )
            K_mod = K_mod.reshape(bsz, n_heads, kv_seq_len, head_dim)
            V_mod = V_mod.reshape(bsz, n_heads, kv_seq_len, head_dim)

        out = scaled_dot_product_attention(
            Q.contiguous(),
            K_mod.contiguous(),
            V_mod.contiguous(),
            **kwargs,
        )
        return out.transpose(1, 2).contiguous()


__all__ = [
    "AttentionConfig",
    "AttentionContext",
    "select_attention_backend",
    "run_attention",
    "FLASH_VARLEN",
    "FLASH_DENSE",
    "XFORMERS",
    "SDPA",
    "NJT_SDPA",
    # NJT helpers
    "UNSLOTH_USE_NJT",
    "is_njt",
    "njt_reshape_for_heads",
    "njt_transpose_for_attention",
    "njt_transpose_from_attention",
    "njt_reshape_from_heads",
]
