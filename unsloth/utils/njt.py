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

"""Utilities for PyTorch Nested Jagged Tensors (NJTs) in Unsloth.

NJTs provide efficient representation of variable-length sequences without
explicit padding, improving memory efficiency and performance for transformer
training. See: https://docs.pytorch.org/docs/stable/nested.html
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch


def is_njt(tensor: torch.Tensor) -> bool:
    """Check if tensor is a nested tensor with jagged layout.

    Args:
        tensor: Input tensor to check.

    Returns:
        True if tensor is a nested tensor with torch.jagged layout.
    """
    if not tensor.is_nested:
        return False
    # Check for jagged layout (vs strided layout)
    return tensor.layout == torch.jagged


def create_njt_from_list(
    tensors: List[torch.Tensor],
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Create a nested jagged tensor from a list of tensors.

    Args:
        tensors: List of tensors with shape (seq_len_i, ...).
                 First dimension can vary, remaining dimensions must match.
        device: Target device. If None, uses device of first tensor.
        dtype: Target dtype. If None, uses dtype of first tensor.

    Returns:
        Nested tensor with jagged layout and shape (len(tensors), *, ...).
    """
    if not tensors:
        raise ValueError("Cannot create NJT from empty list")

    if device is None:
        device = tensors[0].device
    if dtype is None:
        dtype = tensors[0].dtype

    # Move tensors to target device/dtype if needed
    processed = []
    for t in tensors:
        if t.device != device or t.dtype != dtype:
            t = t.to(device=device, dtype=dtype)
        processed.append(t)

    return torch.nested.nested_tensor(processed, layout=torch.jagged)


def create_njt_from_packed(
    packed_tensor: torch.Tensor,
    seq_lengths: torch.Tensor,
) -> torch.Tensor:
    """Convert a packed tensor + sequence lengths to a nested jagged tensor.

    Args:
        packed_tensor: Packed tensor with shape (total_tokens, ...).
        seq_lengths: 1D tensor of sequence lengths, shape (num_sequences,).
                     Must sum to total_tokens.

    Returns:
        Nested tensor with jagged layout and shape (num_sequences, *, ...).
    """
    if packed_tensor.dim() < 1:
        raise ValueError("packed_tensor must have at least 1 dimension")
    if seq_lengths.dim() != 1:
        raise ValueError("seq_lengths must be 1D")

    total_tokens = packed_tensor.shape[0]
    expected_total = int(seq_lengths.sum().item())
    if total_tokens != expected_total:
        raise ValueError(
            f"packed_tensor has {total_tokens} tokens but seq_lengths sum to {expected_total}"
        )

    # Split packed tensor into list by sequence lengths
    seq_lengths_list = seq_lengths.tolist()
    tensors = torch.split(packed_tensor, seq_lengths_list, dim=0)

    return torch.nested.nested_tensor(list(tensors), layout=torch.jagged)


def njt_to_packed(njt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a nested jagged tensor back to packed tensor + sequence lengths.

    Args:
        njt: Nested tensor with jagged layout.

    Returns:
        Tuple of (packed_tensor, seq_lengths) where:
        - packed_tensor has shape (total_tokens, ...)
        - seq_lengths has shape (num_sequences,)
    """
    if not is_njt(njt):
        raise ValueError("Input must be a nested tensor with jagged layout")

    # Get sequence lengths from NJT offsets
    seq_lengths = get_njt_seq_lengths(njt)

    # Convert to padded and then pack (NJT provides efficient way to get values)
    # Access underlying values storage
    values = njt.values()  # Shape: (total_tokens, ...)

    return values, seq_lengths


def get_njt_seq_lengths(njt: torch.Tensor) -> torch.Tensor:
    """Extract sequence lengths from a nested jagged tensor.

    Args:
        njt: Nested tensor with jagged layout.

    Returns:
        1D tensor of sequence lengths, shape (num_sequences,).
    """
    if not is_njt(njt):
        raise ValueError("Input must be a nested tensor with jagged layout")

    # NJT stores offsets; compute lengths from consecutive differences
    offsets = njt.offsets()  # Shape: (num_sequences + 1,)
    seq_lengths = offsets[1:] - offsets[:-1]

    return seq_lengths


def get_njt_offsets(njt: torch.Tensor) -> torch.Tensor:
    """Get the offsets array from a nested jagged tensor.

    Args:
        njt: Nested tensor with jagged layout.

    Returns:
        1D tensor of offsets, shape (num_sequences + 1,).
        offsets[i] is the start index of sequence i in the values buffer.
    """
    if not is_njt(njt):
        raise ValueError("Input must be a nested tensor with jagged layout")

    return njt.offsets()


def njt_unflatten_heads(
    njt: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Unflatten the last dimension of an NJT into (num_heads, head_dim).

    Transforms NJT from shape (B, S*, hidden_dim) to (B, S*, num_heads, head_dim).

    Args:
        njt: Nested tensor with shape (B, S*, hidden_dim).
        num_heads: Number of attention heads.
        head_dim: Dimension per head.

    Returns:
        Nested tensor with shape (B, S*, num_heads, head_dim).
    """
    if not is_njt(njt):
        raise ValueError("Input must be a nested tensor with jagged layout")

    # NJT supports unflatten operation
    return njt.unflatten(-1, (num_heads, head_dim))


def njt_transpose_for_attention(
    njt: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Transform NJT for attention: unflatten heads and transpose.

    Transforms from (B, S*, hidden_dim) to (B, num_heads, S*, head_dim).

    Args:
        njt: Nested tensor with shape (B, S*, hidden_dim).
        num_heads: Number of attention heads.
        head_dim: Dimension per head.

    Returns:
        Nested tensor with shape (B, num_heads, S*, head_dim).
    """
    # First unflatten: (B, S*, hidden_dim) -> (B, S*, num_heads, head_dim)
    unflat = njt_unflatten_heads(njt, num_heads, head_dim)
    # Transpose: (B, S*, num_heads, head_dim) -> (B, num_heads, S*, head_dim)
    return unflat.transpose(1, 2)


def njt_transpose_from_attention(njt: torch.Tensor) -> torch.Tensor:
    """Transform NJT from attention format back to sequence format.

    Transforms from (B, num_heads, S*, head_dim) to (B, S*, num_heads, head_dim).

    Args:
        njt: Nested tensor with shape (B, num_heads, S*, head_dim).

    Returns:
        Nested tensor with shape (B, S*, num_heads, head_dim).
    """
    return njt.transpose(1, 2)


def njt_flatten_heads(njt: torch.Tensor) -> torch.Tensor:
    """Flatten the last two dimensions of an NJT (heads and head_dim).

    Transforms from (B, S*, num_heads, head_dim) to (B, S*, hidden_dim).

    Args:
        njt: Nested tensor with shape (B, S*, num_heads, head_dim).

    Returns:
        Nested tensor with shape (B, S*, hidden_dim).
    """
    if not is_njt(njt):
        raise ValueError("Input must be a nested tensor with jagged layout")

    return njt.flatten(-2, -1)


def create_position_ids_for_njt(njt: torch.Tensor) -> torch.Tensor:
    """Create position IDs for a nested jagged tensor.

    For each sequence in the batch, creates position IDs [0, 1, 2, ..., seq_len-1].
    Returns a packed 1D tensor of all position IDs.

    Args:
        njt: Nested tensor with jagged layout.

    Returns:
        1D tensor of position IDs, shape (total_tokens,).
    """
    seq_lengths = get_njt_seq_lengths(njt)
    device = njt.device

    # Create position IDs for each sequence
    position_ids_list = []
    for length in seq_lengths.tolist():
        position_ids_list.append(torch.arange(length, device=device))

    return torch.cat(position_ids_list, dim=0)


def batch_size_from_njt(njt: torch.Tensor) -> int:
    """Get the batch size (number of sequences) from an NJT.

    Args:
        njt: Nested tensor with jagged layout.

    Returns:
        Number of sequences in the batch.
    """
    if not is_njt(njt):
        raise ValueError("Input must be a nested tensor with jagged layout")

    # Offsets has length (batch_size + 1)
    return len(njt.offsets()) - 1


def total_tokens_from_njt(njt: torch.Tensor) -> int:
    """Get the total number of tokens across all sequences in an NJT.

    Args:
        njt: Nested tensor with jagged layout.

    Returns:
        Total number of tokens.
    """
    if not is_njt(njt):
        raise ValueError("Input must be a nested tensor with jagged layout")

    return njt.values().shape[0]


def apply_rope_to_njt(
    njt_q: torch.Tensor,
    njt_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding (RoPE) to NJT query and key tensors.

    This function applies RoPE to NJTs by operating on their underlying values.
    NJTs in attention format have shape (B, num_heads, S*, head_dim).

    For NJTs, position_ids should be a packed 1D tensor of shape (total_tokens,)
    created using create_position_ids_for_njt().

    Args:
        njt_q: NJT query tensor with shape (B, num_heads, S*, head_dim).
        njt_k: NJT key tensor with shape (B, num_heads, S*, head_dim).
        cos: Cosine embeddings, shape (max_seq_len, head_dim) or (max_seq_len, head_dim//2).
        sin: Sine embeddings, shape (max_seq_len, head_dim) or (max_seq_len, head_dim//2).
        position_ids: Optional position IDs, shape (total_tokens,).
                      If None, creates sequential position IDs for each sequence.

    Returns:
        Tuple of (njt_q_rope, njt_k_rope) with RoPE applied.
    """
    if not is_njt(njt_q) or not is_njt(njt_k):
        raise ValueError("Inputs must be nested tensors with jagged layout")

    # Squeeze cos/sin if needed
    cos = cos.squeeze()
    sin = sin.squeeze()

    # Get underlying values - these are packed contiguously
    # For NJT with shape (B, H, S*, D), values have shape (total_tokens * H, D)
    # But we need to be careful about the layout

    # First, transpose from (B, H, S*, D) to (B, S*, H, D) to get packed (total, H, D)
    njt_q_seq = njt_q.transpose(1, 2)  # (B, S*, H, D)
    njt_k_seq = njt_k.transpose(1, 2)  # (B, S*, H, D)

    # Get the values which are packed: shape (total_tokens, H, D)
    q_values = njt_q_seq.values()  # (total_tokens, num_heads, head_dim)
    k_values = njt_k_seq.values()  # (total_tokens, num_heads, head_dim)

    total_tokens, num_heads, head_dim = q_values.shape

    # Create position IDs if not provided
    if position_ids is None:
        position_ids = create_position_ids_for_njt(njt_q_seq)

    # Get cos/sin for the positions
    # Ensure position_ids are within bounds
    max_pos = cos.shape[0]
    position_ids = position_ids.clamp(0, max_pos - 1)

    # Index cos and sin by position IDs
    # cos shape: (total_tokens, head_dim) or (total_tokens, head_dim//2)
    cos_pos = cos[position_ids]  # (total_tokens, ...)
    sin_pos = sin[position_ids]  # (total_tokens, ...)

    # Handle half head_dim case (common for efficient RoPE)
    if cos_pos.shape[-1] == head_dim // 2:
        cos_pos = torch.cat([cos_pos, cos_pos], dim=-1)
        sin_pos = torch.cat([sin_pos, sin_pos], dim=-1)

    # Add head dimension: (total_tokens, head_dim) -> (total_tokens, 1, head_dim)
    cos_pos = cos_pos.unsqueeze(1)
    sin_pos = sin_pos.unsqueeze(1)

    # Apply RoPE: Q * cos + rotate_half(Q) * sin
    half = head_dim // 2
    q_rotated = torch.cat((-q_values[..., half:], q_values[..., :half]), dim=-1)
    k_rotated = torch.cat((-k_values[..., half:], k_values[..., :half]), dim=-1)

    q_rope = q_values * cos_pos + q_rotated * sin_pos
    k_rope = k_values * cos_pos + k_rotated * sin_pos

    # Reconstruct NJTs from the RoPE'd values
    # We need to rebuild NJTs with the same structure
    seq_lengths = get_njt_seq_lengths(njt_q_seq)

    # Split q_rope and k_rope back into sequences
    seq_lengths_list = seq_lengths.tolist()
    q_sequences = torch.split(q_rope, seq_lengths_list, dim=0)
    k_sequences = torch.split(k_rope, seq_lengths_list, dim=0)

    # Create new NJTs
    njt_q_rope_seq = torch.nested.nested_tensor(list(q_sequences), layout=torch.jagged)
    njt_k_rope_seq = torch.nested.nested_tensor(list(k_sequences), layout=torch.jagged)

    # Transpose back to attention format: (B, S*, H, D) -> (B, H, S*, D)
    njt_q_rope = njt_q_rope_seq.transpose(1, 2)
    njt_k_rope = njt_k_rope_seq.transpose(1, 2)

    return njt_q_rope, njt_k_rope


def apply_rope_to_njt_inplace(
    njt_q: torch.Tensor,
    njt_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to NJT Q and K tensors in-place (modifies underlying storage).

    This is more memory efficient but modifies the input tensors.

    Args:
        njt_q: NJT query tensor with shape (B, num_heads, S*, head_dim). Modified in-place.
        njt_k: NJT key tensor with shape (B, num_heads, S*, head_dim). Modified in-place.
        cos: Cosine embeddings.
        sin: Sine embeddings.
        position_ids: Optional position IDs.

    Returns:
        The same input tensors (now with RoPE applied).
    """
    # For now, use the non-inplace version as NJT inplace operations are tricky
    # This can be optimized later if needed
    return apply_rope_to_njt(njt_q, njt_k, cos, sin, position_ids)


__all__ = [
    "is_njt",
    "create_njt_from_list",
    "create_njt_from_packed",
    "njt_to_packed",
    "get_njt_seq_lengths",
    "get_njt_offsets",
    "njt_unflatten_heads",
    "njt_transpose_for_attention",
    "njt_transpose_from_attention",
    "njt_flatten_heads",
    "create_position_ids_for_njt",
    "batch_size_from_njt",
    "total_tokens_from_njt",
    "apply_rope_to_njt",
    "apply_rope_to_njt_inplace",
]
