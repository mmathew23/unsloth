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

"""Utilities for enabling packed (padding-free) batches across Unsloth."""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Any, Iterable, Optional, Sequence, Tuple

import torch

try:
    from xformers.ops.fmha.attn_bias import (
        BlockDiagonalCausalMask as _XFormersBlockMask,
    )
except Exception:
    try:
        from xformers.attn_bias import BlockDiagonalCausalMask as _XFormersBlockMask
    except Exception:
        _XFormersBlockMask = None

_XFORMERS_MASK_CACHE_MAXSIZE = 32
_XFORMERS_MASK_CACHE: OrderedDict[Tuple[Tuple[int, ...], int], Any] = OrderedDict()


def _window_cache_key(sliding_window: Optional[int]) -> int:
    if sliding_window is None or sliding_window <= 0:
        return 0
    return int(sliding_window)


def _get_cached_block_mask(
    lengths: Tuple[int, ...],
    sliding_window: Optional[int],
):
    if _XFormersBlockMask is None:
        return None

    window_key = _window_cache_key(sliding_window)
    cache_key = (lengths, window_key)
    cached = _XFORMERS_MASK_CACHE.get(cache_key)
    if cached is not None:
        _XFORMERS_MASK_CACHE.move_to_end(cache_key)
        return cached

    mask = _XFormersBlockMask.from_seqlens(list(lengths))
    if window_key and mask is not None and hasattr(mask, "make_local_attention"):
        mask = mask.make_local_attention(window_size = window_key)

    _XFORMERS_MASK_CACHE[cache_key] = mask
    if len(_XFORMERS_MASK_CACHE) > _XFORMERS_MASK_CACHE_MAXSIZE:
        _XFORMERS_MASK_CACHE.popitem(last = False)
    return mask


class _TrlPackingWarningFilter(logging.Filter):
    to_filter = (
        "attention implementation is not",
        "kernels-community",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(substring in message for substring in self.to_filter)


_TRL_FILTER_INSTALLED = False


def _ensure_trl_warning_filter():
    global _TRL_FILTER_INSTALLED
    if _TRL_FILTER_INSTALLED:
        return
    logging.getLogger("trl.trainer.sft_trainer").addFilter(_TrlPackingWarningFilter())
    _TRL_FILTER_INSTALLED = True


def mark_allow_overlength(module):
    """Mark a module hierarchy so padding-free batches can exceed max_seq_length."""
    if module is None:
        return
    if hasattr(module, "max_seq_length"):
        setattr(module, "_unsloth_allow_packed_overlength", True)
    children = getattr(module, "children", None)
    if children is None:
        return
    for child in children():
        mark_allow_overlength(child)


def configure_sample_packing(config):
    """Mutate an ``SFTConfig`` so TRL prepares packed batches."""
    _ensure_trl_warning_filter()
    setattr(config, "packing", True)
    setattr(config, "padding_free", True)


def configure_padding_free(config):
    """Mutate an ``SFTConfig`` so TRL enables padding-free batching without packing."""
    _ensure_trl_warning_filter()
    setattr(config, "padding_free", True)


def enable_sample_packing(
    model,
    trainer,
    *,
    sequence_lengths_key: str = "seq_lengths",
) -> None:
    """Enable runtime support for packed batches on an existing trainer."""
    if model is None or trainer is None:
        raise ValueError("model and trainer must not be None")

    mark_allow_overlength(model)

    if hasattr(trainer, "args") and hasattr(trainer.args, "remove_unused_columns"):
        trainer.args.remove_unused_columns = False

    collator = getattr(trainer, "data_collator", None)
    if collator is None or not hasattr(collator, "torch_call"):
        return
    if getattr(collator, "_unsloth_packing_wrapped", False):
        return

    if hasattr(collator, "padding_free"):
        collator.padding_free = True
    if hasattr(collator, "return_position_ids"):
        collator.return_position_ids = True

    original_torch_call = collator.torch_call

    def torch_call_with_lengths(examples: Sequence[dict]):
        batch = original_torch_call(examples)
        if examples and isinstance(examples[0], dict):
            seq_lengths: list[int] = []
            for example in examples:
                lengths = example.get(sequence_lengths_key)
                if isinstance(lengths, Iterable):
                    seq_lengths.extend(int(length) for length in lengths)
            if seq_lengths:
                batch["packed_seq_lengths"] = torch.tensor(
                    seq_lengths, dtype = torch.int32
                )
                if "attention_mask" in batch:
                    batch.pop("attention_mask")
        return batch

    collator.torch_call = torch_call_with_lengths
    collator._unsloth_packing_wrapped = True


def enable_padding_free_metadata(model, trainer):
    """Inject seq-length metadata when padding-free batching is enabled without packing."""
    collator = getattr(trainer, "data_collator", None)
    if collator is None or not hasattr(collator, "torch_call"):
        return
    if getattr(collator, "_unsloth_padding_free_lengths_wrapped", False):
        return

    # Check if padding_free is enabled in trainer args or on the collator
    trainer_args = getattr(trainer, "args", None)
    trainer_padding_free = trainer_args is not None and getattr(trainer_args, "padding_free", False)
    collator_padding_free = getattr(collator, "padding_free", False)

    if not trainer_padding_free and not collator_padding_free:
        return

    mark_allow_overlength(model)
    if hasattr(collator, "return_position_ids"):
        collator.return_position_ids = True

    original_torch_call = collator.torch_call

    def torch_call_with_padding_free_metadata(examples: Sequence[dict]):
        seq_lengths: list[int] = []
        if examples and isinstance(examples[0], dict):
            for example in examples:
                lengths = example.get("seq_lengths")
                if lengths is None:
                    ids = example.get("input_ids")
                    if ids is None:
                        continue
                    lengths = [len(ids)]
                    example["seq_lengths"] = lengths
                seq_lengths.extend(lengths)

        batch = original_torch_call(examples)
        if seq_lengths:
            batch["packed_seq_lengths"] = torch.tensor(
                seq_lengths,
                dtype = torch.int32,
            )
        return batch

    collator.torch_call = torch_call_with_padding_free_metadata
    collator._unsloth_padding_free_lengths_wrapped = True


def get_packed_info_from_kwargs(
    kwargs: dict,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
    """Return packed sequence metadata expected by the attention kernels."""

    seq_lengths = kwargs.get("packed_seq_lengths")
    if seq_lengths is None:
        return None

    lengths = seq_lengths.to(device = device, dtype = torch.int32, non_blocking = True)
    cu_seqlens = torch.empty(lengths.numel() + 1, dtype = torch.int32, device = device)
    cu_seqlens[0] = 0
    torch.cumsum(lengths, dim = 0, dtype = torch.int32, out = cu_seqlens[1:])

    max_seqlen = int(lengths.max().item())
    return lengths, cu_seqlens, max_seqlen


def build_xformers_block_causal_mask(
    seq_info: Optional[Tuple[torch.Tensor, torch.Tensor, int]],
    *,
    sliding_window: Optional[int] = None,
    base_mask: Optional[Any] = None,
):
    if _XFormersBlockMask is None:
        return None
    if seq_info is not None:
        seq_lengths, _, _ = seq_info
        lengths_tensor = seq_lengths.to("cpu", torch.int32)
        if lengths_tensor.numel() == 0:
            return None
        lengths = tuple(int(x) for x in lengths_tensor.tolist())
        mask = _get_cached_block_mask(lengths, sliding_window)
    else:
        mask = base_mask

        if (
            sliding_window is not None
            and sliding_window > 0
            and mask is not None
            and hasattr(mask, "make_local_attention")
        ):
            mask = mask.make_local_attention(window_size = sliding_window)
    return mask


def build_sdpa_packed_attention_mask(
    seq_info: Tuple[torch.Tensor, torch.Tensor, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    seq_lengths, _, _ = seq_info
    total_tokens = int(seq_lengths.sum().item())
    mask = torch.full(
        (total_tokens, total_tokens),
        float("-inf"),
        dtype = dtype,
        device = device,
    )
    offset = 0
    for length in seq_lengths.tolist():
        length = int(length)
        if length <= 0:
            continue
        block = torch.zeros((length, length), dtype = dtype, device = device)
        upper = torch.triu(
            torch.ones((length, length), device = device), diagonal = 1
        ).bool()
        block = block.masked_fill(upper, float("-inf"))
        if (
            sliding_window is not None
            and sliding_window > 0
            and length > sliding_window
        ):
            idx = torch.arange(length, device = device)
            dist = idx.unsqueeze(1) - idx.unsqueeze(0)
            window_mask = dist >= sliding_window
            block = block.masked_fill(window_mask, float("-inf"))
        mask[offset : offset + length, offset : offset + length] = block
        offset += length
    return mask.unsqueeze(0).unsqueeze(0)


def _normalize_packed_lengths(
    seq_lengths: Any,
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if seq_lengths is None:
        return None
    if isinstance(seq_lengths, torch.Tensor):
        lengths = seq_lengths.to(device = device, dtype = torch.int64)
    else:
        lengths = torch.tensor(seq_lengths, device = device, dtype = torch.int64)
    if lengths.ndim != 1:
        lengths = lengths.reshape(-1)
    if lengths.numel() == 0:
        return None
    return lengths


def mask_packed_sequence_boundaries(
    shift_labels: torch.Tensor,
    seq_lengths: Any,
    *,
    ignore_index: int = -100,
) -> bool:
    """Mark final token of every packed sample so CE ignores boundary predictions."""
    lengths = _normalize_packed_lengths(seq_lengths, device = shift_labels.device)
    if lengths is None:
        return False

    flat = shift_labels.reshape(-1)
    total_tokens = flat.shape[0]
    boundary_positions = torch.cumsum(lengths, dim = 0) - 1
    valid = boundary_positions < total_tokens
    if not torch.all(valid):
        boundary_positions = boundary_positions[valid]
    if boundary_positions.numel() == 0:
        return False
    flat[boundary_positions] = ignore_index
    return True


# =============================================================================
# Nested Jagged Tensor (NJT) Collation Utilities
# =============================================================================
# NJTs provide efficient representation of variable-length sequences without
# explicit padding, improving memory efficiency and performance.
# See: https://docs.pytorch.org/docs/stable/nested.html

# Check for NJT support (requires PyTorch 2.5+)
HAS_NJT_SUPPORT = hasattr(torch, "jagged") or (
    hasattr(torch, "nested") and hasattr(torch.nested, "nested_tensor")
)

# NJT logging support
_NJT_PACKING_LOGGING_ENABLED = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"
_NJT_PACKING_LOGGED_ONCE = False


def _log_njt_packing(message: str, once: bool = False) -> None:
    """Log NJT packing usage when UNSLOTH_ENABLE_LOGGING=1."""
    global _NJT_PACKING_LOGGED_ONCE
    if not _NJT_PACKING_LOGGING_ENABLED:
        return
    if once and _NJT_PACKING_LOGGED_ONCE:
        return
    print(f"🦥 Unsloth NJT Packing: {message}")
    if once:
        _NJT_PACKING_LOGGED_ONCE = True


def collate_to_njt(
    sequences: Sequence[torch.Tensor],
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Collate a list of variable-length sequences into a Nested Jagged Tensor.

    Args:
        sequences: List of tensors, each with shape (seq_len_i, ...).
                   The first dimension (sequence length) can vary.
        device: Target device. If None, uses device of first tensor.
        dtype: Target dtype. If None, uses dtype of first tensor.

    Returns:
        Nested tensor with jagged layout, shape (num_sequences, *, ...).

    Raises:
        ValueError: If sequences is empty or NJT support is unavailable.
    """
    if not sequences:
        raise ValueError("Cannot create NJT from empty sequence list")

    if not HAS_NJT_SUPPORT:
        raise RuntimeError(
            "NJT support requires PyTorch 2.5+. "
            "Please upgrade PyTorch to use NJT collation."
        )

    if device is None:
        device = sequences[0].device
    if dtype is None:
        dtype = sequences[0].dtype

    # Ensure all tensors are on target device/dtype
    processed = []
    for seq in sequences:
        if seq.device != device or seq.dtype != dtype:
            seq = seq.to(device=device, dtype=dtype)
        processed.append(seq)

    return torch.nested.nested_tensor(processed, layout=torch.jagged)


def collate_batch_to_njt(
    batch: dict,
    *,
    keys_to_convert: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Convert specified keys in a batch dict from packed tensors to NJTs.

    This function takes a batch dict that contains packed tensors and
    `packed_seq_lengths`, and converts specified keys to NJT format.

    Args:
        batch: Batch dictionary containing packed tensors and `packed_seq_lengths`.
        keys_to_convert: Keys to convert to NJT format. Defaults to
                         ["input_ids", "labels", "attention_mask"] if present.
        device: Target device for NJT creation.

    Returns:
        Modified batch dict with specified keys as NJTs and `njt_seq_lengths` added.
        Original `packed_seq_lengths` is preserved for backward compatibility.
    """
    seq_lengths = batch.get("packed_seq_lengths")
    if seq_lengths is None:
        return batch

    if keys_to_convert is None:
        keys_to_convert = ["input_ids", "labels"]

    if device is None:
        # Try to get device from first available tensor
        for key in keys_to_convert:
            if key in batch and isinstance(batch[key], torch.Tensor):
                device = batch[key].device
                break

    seq_lengths_list = seq_lengths.tolist()

    for key in keys_to_convert:
        if key not in batch:
            continue

        tensor = batch[key]
        if not isinstance(tensor, torch.Tensor):
            continue

        # Flatten to 1D if needed (packed format is typically 1D or 2D with batch=1)
        flat_tensor = tensor.reshape(-1)
        total_expected = sum(seq_lengths_list)

        if flat_tensor.shape[0] != total_expected:
            # Skip if sizes don't match (might be a different kind of tensor)
            continue

        # Split packed tensor into list by sequence lengths
        sequences = torch.split(flat_tensor, seq_lengths_list, dim=0)

        # Create NJT
        batch[key] = torch.nested.nested_tensor(list(sequences), layout=torch.jagged)

    # Store sequence lengths for NJT path
    batch["njt_seq_lengths"] = seq_lengths

    # Log NJT collation
    _log_njt_packing(
        f"Converted batch to NJT format ({len(seq_lengths_list)} sequences, "
        f"{sum(seq_lengths_list)} total tokens)",
        once=True,
    )

    return batch


def enable_njt_collation(
    model,
    trainer,
    *,
    keys_to_convert: Optional[Sequence[str]] = None,
) -> None:
    """Enable NJT collation for packed batches on an existing trainer.

    This wraps the data collator to convert packed batches to NJT format,
    which can improve memory efficiency and performance.

    Args:
        model: The model (used to mark for overlength support).
        trainer: The trainer whose collator will be wrapped.
        keys_to_convert: Batch keys to convert to NJT format.

    Note:
        This should be called after enable_sample_packing() or when the
        collator already produces packed_seq_lengths.
    """
    if model is None or trainer is None:
        raise ValueError("model and trainer must not be None")

    if not HAS_NJT_SUPPORT:
        raise RuntimeError(
            "NJT support requires PyTorch 2.5+. "
            "Please upgrade PyTorch to use NJT collation."
        )

    mark_allow_overlength(model)

    collator = getattr(trainer, "data_collator", None)
    if collator is None or not hasattr(collator, "torch_call"):
        return

    if getattr(collator, "_unsloth_njt_wrapped", False):
        return

    # Get the current torch_call (may already be wrapped by enable_sample_packing)
    original_torch_call = collator.torch_call

    def torch_call_with_njt(examples: Sequence[dict]):
        batch = original_torch_call(examples)

        # Convert to NJT format if packed_seq_lengths is present
        if "packed_seq_lengths" in batch:
            batch = collate_batch_to_njt(
                batch,
                keys_to_convert=keys_to_convert,
            )
            # Mark batch as using NJT
            batch["use_njt"] = True

        return batch

    collator.torch_call = torch_call_with_njt
    collator._unsloth_njt_wrapped = True

    _log_njt_packing("Enabled NJT collation for trainer data collator", once=True)


def get_njt_from_packed_kwargs(
    kwargs: dict,
    tensor_key: str,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Create an NJT from a packed tensor in kwargs using seq_lengths.

    Args:
        kwargs: Kwargs dict containing the tensor and packed_seq_lengths.
        tensor_key: Key of the tensor to convert.
        device: Target device.

    Returns:
        NJT tensor if conversion successful, None otherwise.
    """
    seq_lengths = kwargs.get("packed_seq_lengths")
    tensor = kwargs.get(tensor_key)

    if seq_lengths is None or tensor is None:
        return None

    if not isinstance(tensor, torch.Tensor):
        return None

    # Already an NJT
    if tensor.is_nested:
        return tensor

    seq_lengths_list = seq_lengths.to(device).tolist()
    flat_tensor = tensor.reshape(-1).to(device)

    total_expected = sum(seq_lengths_list)
    if flat_tensor.shape[0] != total_expected:
        return None

    sequences = torch.split(flat_tensor, seq_lengths_list, dim=0)
    return torch.nested.nested_tensor(list(sequences), layout=torch.jagged)


# NJT (Nested Jagged Tensor) collator wrapping
# This enables NJT to work with any collator by converting padded batches to concatenated format

UNSLOTH_USE_NJT = os.environ.get("UNSLOTH_USE_NJT", "0") == "1"
_NJT_LOGGING_ENABLED = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"


def _log_njt(message: str) -> None:
    """Log NJT message when logging is enabled."""
    if _NJT_LOGGING_ENABLED:
        print(f"🦥 Unsloth NJT: {message}")


def _convert_padded_to_concatenated(batch: dict) -> dict:
    """
    Convert a padded batch to concatenated format for NJT.

    Padded format: input_ids shape (batch_size, max_seq_len)
    Concatenated format: input_ids shape (1, total_tokens) + packed_seq_lengths

    Returns the batch unchanged if already in concatenated format or conversion fails.
    """
    if "packed_seq_lengths" in batch:
        # Already in concatenated format
        _log_njt(f"Batch already has packed_seq_lengths: {batch['packed_seq_lengths']}")
        return batch

    input_ids = batch.get("input_ids")
    attention_mask = batch.get("attention_mask")

    if input_ids is None:
        _log_njt("No input_ids in batch, skipping conversion")
        return batch

    _log_njt(f"Input shape: {input_ids.shape}, keys: {list(batch.keys())}")

    # Check if already concatenated (bsz=1 with potential packed format)
    if input_ids.dim() == 2 and input_ids.shape[0] == 1:
        # Could be concatenated already, but we don't know seq_lengths
        # Without attention_mask or explicit seq_lengths, we can't convert
        if attention_mask is None:
            return batch
        # If attention_mask is all 1s for single row, treat as one sequence
        if attention_mask.shape[0] == 1 and attention_mask.sum() == attention_mask.numel():
            batch["packed_seq_lengths"] = torch.tensor(
                [input_ids.shape[1]], dtype=torch.int32
            )
            return batch
        return batch

    # Padded format: (batch_size, max_seq_len)
    if input_ids.dim() != 2 or input_ids.shape[0] <= 1:
        return batch

    batch_size, max_seq_len = input_ids.shape

    # Compute actual sequence lengths from attention_mask
    if attention_mask is not None:
        seq_lengths = attention_mask.sum(dim=1).tolist()
    else:
        # No attention_mask - assume all positions are valid (no padding)
        seq_lengths = [max_seq_len] * batch_size

    # Concatenate sequences (remove padding)
    concatenated_ids = []
    concatenated_labels = []
    has_labels = "labels" in batch

    for i, length in enumerate(seq_lengths):
        length = int(length)
        concatenated_ids.append(input_ids[i, :length])
        if has_labels:
            concatenated_labels.append(batch["labels"][i, :length])

    # Create concatenated tensors
    new_input_ids = torch.cat(concatenated_ids, dim=0).unsqueeze(0)  # (1, total_tokens)
    batch["input_ids"] = new_input_ids
    batch["packed_seq_lengths"] = torch.tensor(seq_lengths, dtype=torch.int32)

    if has_labels:
        new_labels = torch.cat(concatenated_labels, dim=0).unsqueeze(0)
        batch["labels"] = new_labels

    # Create new attention_mask (all 1s for concatenated)
    total_tokens = sum(seq_lengths)
    batch["attention_mask"] = torch.ones((1, total_tokens), dtype=torch.long)

    # Handle position_ids if present
    if "position_ids" in batch:
        # Regenerate position_ids for each sequence
        position_ids = []
        for length in seq_lengths:
            position_ids.append(torch.arange(length, dtype=torch.long))
        batch["position_ids"] = torch.cat(position_ids, dim=0).unsqueeze(0)

    _log_njt(
        f"Converted padded batch ({batch_size}x{max_seq_len}) to concatenated "
        f"(1x{total_tokens}), seq_lengths={seq_lengths}"
    )
    return batch


def enable_njt_collator_wrapper(trainer) -> bool:
    """
    Wrap the trainer's data collator to enable NJT support.

    This converts padded batches to concatenated format and adds packed_seq_lengths.
    Only activates when UNSLOTH_USE_NJT=1 environment variable is set.

    Returns True if wrapping was applied, False otherwise.
    """
    if not UNSLOTH_USE_NJT:
        return False

    collator = getattr(trainer, "data_collator", None)
    if collator is None:
        return False

    # Already wrapped
    if getattr(collator, "_unsloth_njt_collator_wrapped", False):
        return False

    # Determine which method to wrap
    if hasattr(collator, "torch_call"):
        original_call = collator.torch_call
        call_attr = "torch_call"
    elif callable(collator):
        original_call = collator.__call__
        call_attr = "__call__"
    else:
        return False

    def njt_collator_wrapper(examples):
        _log_njt(f"NJT collator wrapper called with {len(examples)} examples")
        batch = original_call(examples)
        _log_njt(f"Original collator returned batch with keys: {list(batch.keys())}")
        # Convert padded to concatenated format for NJT
        batch = _convert_padded_to_concatenated(batch)
        return batch

    setattr(collator, call_attr, njt_collator_wrapper)
    collator._unsloth_njt_collator_wrapped = True

    _log_njt("Wrapped data collator for NJT support (padded -> concatenated)")
    return True


# =============================================================================
# NJT Standalone Data Collator
# =============================================================================
# This collator creates NJT tensors directly from variable-length samples,
# without using packing or padding. This is the "Option 1" approach for NJT.


class UnslothNJTDataCollator:
    """
    Data collator that creates Nested Jagged Tensors (NJT) from variable-length samples.

    This collator is used when UNSLOTH_USE_NJT=1 and replaces packing/padding_free.
    It creates NJT tensors directly, enabling efficient variable-length training
    without padding or packing overhead.

    Args:
        tokenizer: The tokenizer (used for padding token ID if needed).
        max_seq_length: Maximum sequence length (samples longer than this are truncated).
        padding_side: Not used for NJT, but kept for compatibility.
    """

    def __init__(
        self,
        tokenizer=None,
        max_seq_length: int = 2048,
        padding_side: str = "right",
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.padding_side = padding_side
        self._logged_once = False

        if not HAS_NJT_SUPPORT:
            raise RuntimeError(
                "NJT support requires PyTorch 2.5+. "
                "Please upgrade PyTorch to use NJT collation."
            )

    def __call__(self, examples: Sequence[dict]) -> dict:
        """Collate examples into a batch with NJT tensors."""
        return self.torch_call(examples)

    def torch_call(self, examples: Sequence[dict]) -> dict:
        """
        Collate examples into a batch with NJT tensors.

        Args:
            examples: List of dicts, each with 'input_ids' (and optionally 'labels', etc.)

        Returns:
            Batch dict with NJT tensors for 'input_ids', 'labels', plus 'njt_seq_lengths'.
        """
        if not examples:
            raise ValueError("Cannot collate empty batch")

        batch_size = len(examples)

        # Extract sequences
        input_ids_list = []
        labels_list = []
        seq_lengths = []

        for ex in examples:
            input_ids = ex.get("input_ids")
            if input_ids is None:
                raise ValueError("Example missing 'input_ids'")

            # Convert to tensor if needed
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, dtype=torch.long)

            # Truncate if needed
            if len(input_ids) > self.max_seq_length:
                input_ids = input_ids[:self.max_seq_length]

            input_ids_list.append(input_ids)
            seq_lengths.append(len(input_ids))

            # Always create labels - use input_ids if no labels provided (standard LM training)
            labels = ex.get("labels")
            if labels is None:
                # Use input_ids as labels (standard LM training)
                labels = input_ids.clone()
            elif not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)

            if len(labels) > self.max_seq_length:
                labels = labels[:self.max_seq_length]

            labels_list.append(labels)

        # Create NJT tensors
        njt_input_ids = torch.nested.nested_tensor(input_ids_list, layout=torch.jagged)

        batch = {
            "input_ids": njt_input_ids,
            "njt_seq_lengths": torch.tensor(seq_lengths, dtype=torch.int32),
            "use_njt": True,  # Flag to indicate NJT mode
        }

        # Always include labels (created from input_ids if not provided)
        njt_labels = torch.nested.nested_tensor(labels_list, layout=torch.jagged)
        batch["labels"] = njt_labels

        # Create position_ids for each sequence (0, 1, 2, ...)
        position_ids_list = [
            torch.arange(length, dtype=torch.long) for length in seq_lengths
        ]
        njt_position_ids = torch.nested.nested_tensor(position_ids_list, layout=torch.jagged)
        batch["position_ids"] = njt_position_ids

        # Log once
        if not self._logged_once and _NJT_PACKING_LOGGING_ENABLED:
            total_tokens = sum(seq_lengths)
            _log_njt_packing(
                f"NJT collator created batch: {batch_size} seqs, "
                f"{total_tokens} total tokens, lengths={seq_lengths}"
            )
            self._logged_once = True

        return batch


def get_njt_collator(tokenizer, max_seq_length: int = 2048) -> UnslothNJTDataCollator:
    """
    Get an NJT data collator for use with SFTTrainer.

    This collator creates NJT tensors directly from variable-length samples,
    enabling efficient training without padding or packing.

    Args:
        tokenizer: The tokenizer.
        max_seq_length: Maximum sequence length.

    Returns:
        UnslothNJTDataCollator instance.
    """
    return UnslothNJTDataCollator(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )


__all__ = [
    "configure_sample_packing",
    "configure_padding_free",
    "enable_sample_packing",
    "enable_padding_free_metadata",
    "mark_allow_overlength",
    "get_packed_info_from_kwargs",
    "build_xformers_block_causal_mask",
    "build_sdpa_packed_attention_mask",
    "mask_packed_sequence_boundaries",
    # NJT utilities
    "HAS_NJT_SUPPORT",
    "collate_to_njt",
    "collate_batch_to_njt",
    "enable_njt_collation",
    "get_njt_from_packed_kwargs",
    "enable_njt_collator_wrapper",
    "UNSLOTH_USE_NJT",
    # NJT standalone
    "UnslothNJTDataCollator",
    "get_njt_collator",
]
