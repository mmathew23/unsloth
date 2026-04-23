"""Patch Qwen3VLTextModel decoder layers so their forward routes through
torch.utils.checkpoint.checkpoint (which unsloth replaces with
unsloth_checkpoint when the smart-GC dispatcher is installed).

As of transformers 4.57.6 (HEAD 2026-04), ``Qwen3VLTextModel.forward``
contains a plain Python ``for layer_idx, decoder_layer in enumerate(self.layers):``
loop that calls ``decoder_layer(...)`` directly — no ``gradient_checkpointing``
branch. Therefore the standard HF _gradient_checkpointing_func path (which
unsloth patches) is never invoked for the VL language-model decoder. Effect:

  - During VL SFT, decoder activations are NOT offloaded to CPU even when
    ``gradient_checkpointing=True`` and unsloth's hooks backend is installed.
  - During VL GRPO, peak_alloc is identical between native and unsloth
    hooks because no offload ever happens for the decoder.

The fix: wrap each ``Qwen3VLTextDecoderLayer.__call__`` with a thin shim
that forwards through ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)``
when the owning text model has ``gradient_checkpointing=True`` and is in
training mode. Non-training forwards bypass the shim.

Gated by env ``UNSLOTH_PATCH_QWEN3VL_GC`` (default on).
"""
from __future__ import annotations

import os
import weakref

import torch


_PATCH_APPLIED = False


def _should_patch() -> bool:
    val = os.environ.get("UNSLOTH_PATCH_QWEN3VL_GC", "1").strip().lower()
    return val not in ("", "0", "false", "no", "off")


def patch_qwen3vl_text_model_checkpointing(model=None) -> bool:
    """Install the decoder-layer checkpoint shim on Qwen3VLTextModel.

    If ``model`` is provided, also walk it now and patch any existing
    Qwen3VLTextModel instances (their decoder layers). This covers the case
    where the model was loaded before ``patch_qwen3vl_text_model_checkpointing``
    was first called.

    Returns True if the patch was applied (or was already applied), False if
    it was skipped (env flag off, or transformers doesn't have the class).
    """
    global _PATCH_APPLIED
    if not _should_patch():
        return False
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl as m
    except Exception:
        return False

    text_cls = getattr(m, "Qwen3VLTextModel", None)
    decoder_cls = getattr(m, "Qwen3VLTextDecoderLayer", None)
    if text_cls is None or decoder_cls is None:
        return False

    # Patch __init__ so any future Qwen3VLTextModel picks up shims.
    if not _PATCH_APPLIED:
        original_text_init = text_cls.__init__
        if not getattr(original_text_init, "_unsloth_gc_patched", False):
            def _patched_init(self, config, *args, **kwargs):
                original_text_init(self, config, *args, **kwargs)
                parent_ref = weakref.ref(self)
                for layer in self.layers:
                    _install_layer_shim(layer, parent_ref)
            _patched_init._unsloth_gc_patched = True
            text_cls.__init__ = _patched_init
        _PATCH_APPLIED = True

    # Patch existing instances in ``model`` if provided.
    if model is not None:
        for mod in model.modules():
            if isinstance(mod, text_cls):
                parent_ref = weakref.ref(mod)
                for layer in mod.layers:
                    _install_layer_shim(layer, parent_ref)

    return True


def _install_layer_shim(layer, parent_ref):
    """Replace ``layer.forward`` with a checkpoint-wrapping shim.

    ``parent_ref`` is a weakref to the owning ``Qwen3VLTextModel`` so we
    can consult its current ``gradient_checkpointing`` flag at runtime.
    """
    if getattr(layer.forward, "_unsloth_gc_patched", False):
        return
    original_forward = layer.forward

    def _patched_forward(*args, **kwargs):
        parent = parent_ref()
        if parent is None or not (
            getattr(parent, "gradient_checkpointing", False) and parent.training
        ):
            return original_forward(*args, **kwargs)
        # Route through the current ``torch.utils.checkpoint.checkpoint``
        # attribute — which is ``unsloth_checkpoint`` when the smart-GC
        # dispatcher is installed. Using the attribute (rather than the
        # directly-imported symbol) means upstream or downstream patches
        # compose correctly.
        from torch.utils import checkpoint as _ckpt_mod

        def _call(*a, **kw):
            return original_forward(*a, **kw)

        return _ckpt_mod.checkpoint(_call, *args, use_reentrant=False, **kwargs)

    _patched_forward._unsloth_gc_patched = True
    layer.forward = _patched_forward
