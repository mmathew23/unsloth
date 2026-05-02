import os
import types
import unittest

import torch

from unsloth.kernels import (
    clear_kernel_backend_overrides,
    kernel_backend_context,
    register_kernel_backend,
)
from unsloth.kernels.runtime_bindings import (
    bind_kernel_runtime_globals,
    resolve_kernel_runtime_bindings,
)
import unsloth.kernels.layernorm as layernorm_module


class KernelRuntimeBindingTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("UNSLOTH_KERNEL_BACKEND", None)
        os.environ.pop("UNSLOTH_KERNEL_BACKEND_OVERRIDES", None)
        os.environ.pop("UNSLOTH_KERNEL_BACKEND_STRICT", None)
        clear_kernel_backend_overrides()

    def test_bindings_can_patch_module_globals_without_mutating_kernel_module(self):
        kernel_name = "unsloth.layernorm"

        def _dummy_layernorm(X, W, b, eps):
            return X + W + b + 3

        register_kernel_backend(kernel_name, "runtime_dummy", _dummy_layernorm)
        os.environ["UNSLOTH_KERNEL_BACKEND_OVERRIDES"] = (
            "unsloth.layernorm=runtime_dummy"
        )

        original_fast_layernorm = layernorm_module.fast_layernorm
        runtime_module = types.ModuleType("runtime_module")
        bindings = resolve_kernel_runtime_bindings()
        bind_kernel_runtime_globals(runtime_module, bindings)

        layernorm = torch.nn.LayerNorm(4)
        X = torch.zeros(2, 4)
        out = runtime_module.fast_layernorm(layernorm, X)

        self.assertTrue(torch.equal(out, X + layernorm.weight + layernorm.bias + 3))
        self.assertIs(layernorm_module.fast_layernorm, original_fast_layernorm)
        self.assertEqual(
            runtime_module._unsloth_kernel_binding_state["env_overrides"][
                "unsloth.layernorm"
            ],
            "runtime_dummy",
        )

    def test_fp8_runtime_bindings_do_not_follow_later_backend_rebinds(self):
        suffix = "runtime_fp8_a"
        backend_a = f"{suffix}_a"
        backend_b = f"{suffix}_b"

        def _act_a(X, block_size):
            return X, torch.ones(1, dtype = X.dtype)

        def _act_b(X, block_size):
            return X + 100, torch.ones(1, dtype = X.dtype)

        def _matmul_a(
            qinput,
            weight,
            scale,
            weight_scale,
            block_size,
            output_dtype = torch.bfloat16,
        ):
            shape = (*qinput.shape[:-1], weight.shape[0])
            return torch.full(shape, 11, dtype = output_dtype, device = qinput.device)

        def _matmul_b(
            qinput,
            weight,
            scale,
            weight_scale,
            block_size,
            output_dtype = torch.bfloat16,
        ):
            shape = (*qinput.shape[:-1], weight.shape[0])
            return torch.full(shape, 22, dtype = output_dtype, device = qinput.device)

        def _dequant_a(x, s, block_size = 128, dtype = torch.bfloat16):
            return torch.full_like(x, 33, dtype = dtype)

        def _dequant_b(x, s, block_size = 128, dtype = torch.bfloat16):
            return torch.full_like(x, 44, dtype = dtype)

        register_kernel_backend("unsloth.act_quant", backend_a, _act_a)
        register_kernel_backend("unsloth.act_quant", backend_b, _act_b)
        register_kernel_backend("unsloth.w8a8_block_fp8_matmul", backend_a, _matmul_a)
        register_kernel_backend("unsloth.w8a8_block_fp8_matmul", backend_b, _matmul_b)
        register_kernel_backend("unsloth.weight_dequant", backend_a, _dequant_a)
        register_kernel_backend("unsloth.weight_dequant", backend_b, _dequant_b)

        runtime_module = types.ModuleType("runtime_fp8_module")
        with kernel_backend_context(global_backend = backend_a):
            bind_kernel_runtime_globals(runtime_module)

        X = torch.zeros(2, 3, dtype = torch.float32, requires_grad = True)
        weight = torch.zeros(4, 3, dtype = torch.float32)
        scalar_scale = torch.ones(1, dtype = torch.float32)
        block_scale = torch.ones(2, 2, dtype = torch.float32)

        with kernel_backend_context(global_backend = backend_b):
            dequant = runtime_module.weight_dequant(
                torch.zeros(129, 129),
                block_scale,
                dtype = torch.float32,
            )
            with torch.enable_grad():
                linear = runtime_module.fp8_linear(X, weight, scalar_scale)

        self.assertTrue(torch.equal(dequant, torch.full_like(dequant, 33)))
        self.assertTrue(torch.equal(linear, torch.full_like(linear, 11)))


if __name__ == "__main__":
    unittest.main()
