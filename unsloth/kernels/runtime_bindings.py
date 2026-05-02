# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import FunctionType
from types import ModuleType
from typing import Any, Callable, Mapping

from ._backend_registry import get_kernel_backend, get_kernel_backend_state, get_kernel_impl


KernelCallable = Callable[..., Any]


@dataclass(frozen = True)
class KernelRuntimeBindings:
    """Concrete kernel globals for one generated runtime module instance.

    Generated cache source should keep direct global calls like
    ``fast_rms_layernorm(...)``.  This object is used only while importing and
    patching that generated module: its values are copied into the generated
    module's globals before Dynamo traces anything.
    """

    globals: Mapping[str, KernelCallable]
    state: Mapping[str, Any]


def _get_impl(name: str, fallback: KernelCallable) -> KernelCallable:
    try:
        return get_kernel_impl(name)
    except Exception:
        return fallback


def _get_backend(name: str) -> str | None:
    try:
        return get_kernel_backend(name)
    except Exception:
        return None


def _clone_function_with_globals(
    fn: KernelCallable,
    updates: Mapping[str, Any],
) -> KernelCallable:
    cloned = FunctionType(
        fn.__code__,
        {**fn.__globals__, **updates},
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )
    cloned.__kwdefaults__ = getattr(fn, "__kwdefaults__", None)
    cloned.__annotations__ = dict(getattr(fn, "__annotations__", {}))
    cloned.__dict__.update(getattr(fn, "__dict__", {}))
    cloned.__doc__ = getattr(fn, "__doc__", None)
    cloned.__module__ = getattr(fn, "__module__", None)
    cloned.__qualname__ = getattr(fn, "__qualname__", fn.__name__)
    return cloned


def _compiled_inner(fn: KernelCallable) -> KernelCallable:
    return getattr(fn, "__wrapped__", fn)


def _make_runtime_weight_dequant(
    fp8_mod: ModuleType,
    weight_dequant_block: KernelCallable,
) -> KernelCallable:
    return _clone_function_with_globals(
        fp8_mod.weight_dequant,
        {"weight_dequant_block": weight_dequant_block},
    )


def _make_runtime_fp8_linear(
    fp8_mod: ModuleType,
    *,
    act_quant: KernelCallable,
    fp8_block_matmul: KernelCallable,
    weight_dequant: KernelCallable,
    block_fp8_backend: str | None,
) -> KernelCallable:
    if block_fp8_backend == "eager":
        return fp8_mod._fp8_linear_eager

    torch = fp8_mod.torch
    RuntimeFP8BlockQuantLinear = type(
        "RuntimeFP8BlockQuantLinear",
        (torch.autograd.Function,),
        {
            "__module__": fp8_mod.__name__,
            "forward": staticmethod(
                _clone_function_with_globals(
                    fp8_mod.FP8BlockQuantLinear.forward,
                    {
                        "act_quant": act_quant,
                        "fp8_block_matmul": fp8_block_matmul,
                    },
                )
            ),
            "backward": staticmethod(
                _clone_function_with_globals(
                    fp8_mod.FP8BlockQuantLinear.backward,
                    {"weight_dequant": weight_dequant},
                )
            ),
        },
    )

    if fp8_mod.fp8_block_quant_linear is fp8_mod.fp8_torch_block_quant_forward:
        block_forward = _clone_function_with_globals(
            _compiled_inner(fp8_mod.fp8_torch_block_quant_forward),
            {"FP8BlockQuantLinear": RuntimeFP8BlockQuantLinear},
        )
        fp8_block_quant_linear = fp8_mod.torch_compile(block_forward)
    else:
        fp8_block_quant_linear = fp8_mod.fp8_block_quant_linear

    fp8_linear_static = fp8_mod.torch_compile(
        _clone_function_with_globals(
            _compiled_inner(fp8_mod._fp8_linear_static),
            {"fp8_block_quant_linear": fp8_block_quant_linear},
        )
    )

    def fp8_linear_training_autograd(X, weight, weight_scale, bias = None):
        return RuntimeFP8BlockQuantLinear.apply(X, weight, weight_scale, bias)

    def fp8_linear_runtime(X, weight, weight_scale, bias = None):
        if torch.is_grad_enabled() and X.requires_grad:
            return fp8_linear_training_autograd(X, weight, weight_scale, bias)
        return fp8_linear_static(X, weight, weight_scale, bias)

    return fp8_linear_runtime


def resolve_kernel_runtime_bindings() -> KernelRuntimeBindings:
    """Resolve concrete callables for generated-cache module globals.

    This function is intentionally setup-time only.  It may use the backend
    registry and current backend context to choose callables, but the returned
    functions are then bound directly into a generated module before tracing.
    """

    cross_entropy_mod = importlib.import_module("unsloth.kernels.cross_entropy_loss")
    fp8_mod = importlib.import_module("unsloth.kernels.fp8")
    geglu_mod = importlib.import_module("unsloth.kernels.geglu")
    grouped_gemm_mod = importlib.import_module("unsloth.kernels.grouped_gemm")
    layernorm_mod = importlib.import_module("unsloth.kernels.layernorm")
    rms_layernorm_mod = importlib.import_module("unsloth.kernels.rms_layernorm")
    rope_embedding_mod = importlib.import_module("unsloth.kernels.rope_embedding")
    swiglu_mod = importlib.import_module("unsloth.kernels.swiglu")

    rms_impl = _get_impl("unsloth.rms_layernorm", rms_layernorm_mod._fast_rms_layernorm_eager)
    fast_rms_layernorm = rms_layernorm_mod._make_fast_rms_layernorm_default(rms_impl)

    layernorm_impl = _get_impl("unsloth.layernorm", layernorm_mod._fast_layernorm_eager)
    fast_layernorm = layernorm_mod._make_fast_layernorm_default(layernorm_impl)

    fast_rope_impl = _get_impl(
        "unsloth.rope_embedding_qk",
        rope_embedding_mod._fast_rope_embedding_eager,
    )
    fast_rope_embedding = rope_embedding_mod._make_fast_rope_embedding_default(fast_rope_impl)
    rope_embedding = _get_impl("unsloth.rope_embedding", rope_embedding_mod._rope_embedding_eager)

    fast_cross_entropy_loss = _get_impl(
        "unsloth.cross_entropy_loss",
        cross_entropy_mod._fast_cross_entropy_loss_eager,
    )

    swiglu_fg_kernel = _get_impl("unsloth.swiglu_fg", swiglu_mod._swiglu_fg_eager)
    swiglu_DWf_DW_dfg_kernel = _get_impl(
        "unsloth.swiglu_bwd",
        swiglu_mod._swiglu_DWf_DW_dfg_eager,
    )

    geglu_exact_forward_kernel = _get_impl(
        "unsloth.geglu_exact_forward",
        geglu_mod._geglu_exact_forward_eager,
    )
    geglu_exact_backward_kernel = _get_impl(
        "unsloth.geglu_exact_backward",
        geglu_mod._geglu_exact_backward_eager,
    )
    geglu_approx_forward_kernel = _get_impl(
        "unsloth.geglu_approx_forward",
        geglu_mod._geglu_approx_forward_eager,
    )
    geglu_approx_backward_kernel = _get_impl(
        "unsloth.geglu_approx_backward",
        geglu_mod._geglu_approx_backward_eager,
    )

    grouped_gemm = _get_impl("unsloth.grouped_gemm", grouped_gemm_mod._grouped_gemm_eager)

    act_quant = _get_impl("unsloth.act_quant", fp8_mod._act_quant_eager)
    fp8_block_matmul = _get_impl(
        "unsloth.w8a8_block_fp8_matmul",
        fp8_mod._w8a8_block_fp8_matmul_eager,
    )
    if (
        fp8_block_matmul is fp8_mod.w8a8_block_fp8_matmul_triton
        and fp8_mod.torchao_blockwise_gemm is not None
    ):
        fp8_block_matmul = fp8_mod.torchao_block_matmul
    weight_dequant_block = _get_impl(
        "unsloth.weight_dequant",
        fp8_mod._weight_dequant_block_eager,
    )
    weight_dequant = _make_runtime_weight_dequant(fp8_mod, weight_dequant_block)
    block_fp8_backend = _get_backend("unsloth.w8a8_block_fp8_matmul")
    fp8_linear = _make_runtime_fp8_linear(
        fp8_mod,
        act_quant = act_quant,
        fp8_block_matmul = fp8_block_matmul,
        weight_dequant = weight_dequant,
        block_fp8_backend = block_fp8_backend,
    )

    globals_map: dict[str, KernelCallable] = {
        "fast_rms_layernorm": fast_rms_layernorm,
        "fast_rms_layernorm_default": fast_rms_layernorm,
        "fast_layernorm": fast_layernorm,
        "fast_layernorm_default": fast_layernorm,
        "fast_rope_embedding": fast_rope_embedding,
        "fast_rope_embedding_default": fast_rope_embedding,
        "rope_embedding": rope_embedding,
        "fast_cross_entropy_loss": fast_cross_entropy_loss,
        "fast_cross_entropy_loss_default": fast_cross_entropy_loss,
        "swiglu_fg_kernel": swiglu_fg_kernel,
        "swiglu_fg_kernel_default": swiglu_fg_kernel,
        "swiglu_DWf_DW_dfg_kernel": swiglu_DWf_DW_dfg_kernel,
        "swiglu_DWf_DW_dfg_kernel_default": swiglu_DWf_DW_dfg_kernel,
        "geglu_exact_forward_kernel": geglu_exact_forward_kernel,
        "geglu_exact_forward_kernel_default": geglu_exact_forward_kernel,
        "geglu_exact_backward_kernel": geglu_exact_backward_kernel,
        "geglu_exact_backward_kernel_default": geglu_exact_backward_kernel,
        "geglu_approx_forward_kernel": geglu_approx_forward_kernel,
        "geglu_approx_forward_kernel_default": geglu_approx_forward_kernel,
        "geglu_approx_backward_kernel": geglu_approx_backward_kernel,
        "geglu_approx_backward_kernel_default": geglu_approx_backward_kernel,
        "grouped_gemm": grouped_gemm,
        "grouped_gemm_default": grouped_gemm,
        "act_quant": act_quant,
        "fp8_block_matmul": fp8_block_matmul,
        "weight_dequant_block": weight_dequant_block,
        "weight_dequant": weight_dequant,
        "fp8_linear": fp8_linear,
    }
    return KernelRuntimeBindings(
        globals = globals_map,
        state = {
            **get_kernel_backend_state(),
            "resolved_block_fp8_backend": block_fp8_backend,
        },
    )


def bind_kernel_runtime_globals(
    module: ModuleType,
    bindings: KernelRuntimeBindings | None = None,
    *,
    only_existing: bool = False,
) -> KernelRuntimeBindings:
    """Copy resolved kernel globals into a generated runtime module.

    ``only_existing`` is useful for tests, but production binding generally
    sets the complete known surface so generated source can reference any hot
    kernel global without depending on imports from process-global modules.
    """

    if bindings is None:
        bindings = resolve_kernel_runtime_bindings()
    for name, value in bindings.globals.items():
        if only_existing and not hasattr(module, name):
            continue
        setattr(module, name, value)
    setattr(module, "_unsloth_kernel_binding_state", dict(bindings.state))
    return bindings
