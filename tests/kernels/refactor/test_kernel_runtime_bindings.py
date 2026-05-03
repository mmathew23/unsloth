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
    _make_runtime_fp8_linear,
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

    def test_distinct_runtime_modules_keep_distinct_backend_bindings(self):
        kernel_name = "unsloth.layernorm"
        backend_a = "runtime_module_a"
        backend_b = "runtime_module_b"

        def _layernorm_a(X, W, b, eps):
            return X + W + b + 10

        def _layernorm_b(X, W, b, eps):
            return X + W + b + 20

        register_kernel_backend(kernel_name, backend_a, _layernorm_a)
        register_kernel_backend(kernel_name, backend_b, _layernorm_b)

        runtime_a = types.ModuleType("runtime_module_a")
        runtime_b = types.ModuleType("runtime_module_b")

        with kernel_backend_context(global_backend = backend_a):
            bind_kernel_runtime_globals(runtime_a)
        with kernel_backend_context(global_backend = backend_b):
            bind_kernel_runtime_globals(runtime_b)

        layernorm = torch.nn.LayerNorm(4)
        X = torch.zeros(2, 4)

        with kernel_backend_context(global_backend = backend_b):
            out_a = runtime_a.fast_layernorm(layernorm, X)
        with kernel_backend_context(global_backend = backend_a):
            out_b = runtime_b.fast_layernorm(layernorm, X)

        self.assertTrue(torch.equal(out_a, X + layernorm.weight + layernorm.bias + 10))
        self.assertTrue(torch.equal(out_b, X + layernorm.weight + layernorm.bias + 20))
        self.assertEqual(
            runtime_a._unsloth_kernel_binding_state["runtime_global_backend"],
            backend_a,
        )
        self.assertEqual(
            runtime_b._unsloth_kernel_binding_state["runtime_global_backend"],
            backend_b,
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

    def test_runtime_fp8_rowwise_binding_isolates_fbgemm_dequant_global(self):
        def _global_weight_dequant(weight, scale):
            return torch.full((weight.shape[1], weight.shape[0]), 99.0)

        def _runtime_weight_dequant(weight, scale):
            return torch.full((weight.shape[1], weight.shape[0]), 7.0)

        namespace = {
            "torch": torch,
            "weight_dequant": _global_weight_dequant,
        }
        exec(
            """
class FbgemmFp8Linear_matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, weight_scale, bias = None):
        return torch.matmul(x, weight_dequant(weight, weight_scale))

    @staticmethod
    def backward(ctx, grad_output):
        raise AssertionError("backward is not used by this structural test")

class FP8_fbgemm_block_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, weight_scale, bias = None):
        return X

    @staticmethod
    def backward(ctx, grad_output):
        raise AssertionError("backward is not used by this structural test")

def fbgemm_fp8_linear(X, weight, weight_scale, bias = None):
    return FbgemmFp8Linear_matmul.apply(X, weight, weight_scale, bias)

def fp8_fbgemm_block_linear(X, weight, weight_scale, bias = None):
    return FP8_fbgemm_block_linear.apply(X, weight, weight_scale, bias)

def fp8_torch_block_quant_forward(X, weight, weight_scale, bias = None):
    return X

def _fp8_linear_static(X, weight, weight_scale, bias = None):
    return fbgemm_fp8_linear(X, weight, weight_scale, bias)
""",
            namespace,
        )

        fake_fp8_mod = types.SimpleNamespace(
            __name__ = "fake_fp8_mod",
            torch = torch,
            torch_compile = lambda fn: fn,
            FP8BlockQuantLinear = namespace["FP8_fbgemm_block_linear"],
            FbgemmFp8Linear_matmul = namespace["FbgemmFp8Linear_matmul"],
            FP8_fbgemm_block_linear = namespace["FP8_fbgemm_block_linear"],
            fp8_block_quant_linear = namespace["fp8_torch_block_quant_forward"],
            fp8_torch_block_quant_forward = namespace["fp8_torch_block_quant_forward"],
            fp8_fbgemm_block_linear = namespace["fp8_fbgemm_block_linear"],
            fbgemm_fp8_linear = namespace["fbgemm_fp8_linear"],
            _fp8_linear_static = namespace["_fp8_linear_static"],
            _fp8_linear_eager = lambda X, weight, weight_scale, bias = None: X,
        )

        runtime_fp8_linear = _make_runtime_fp8_linear(
            fake_fp8_mod,
            act_quant = lambda X, block_size: (X, torch.ones(1)),
            fp8_block_matmul = lambda *args, **kwargs: args[0],
            weight_dequant = _runtime_weight_dequant,
            block_fp8_backend = "runtime",
        )

        X = torch.ones(1, 2)
        weight = torch.ones(3, 2)
        weight_scale = torch.ones(3, 1)
        with torch.no_grad():
            out = runtime_fp8_linear(X, weight, weight_scale)

        self.assertTrue(torch.equal(out, torch.full((1, 3), 14.0)))


if __name__ == "__main__":
    unittest.main()
