import os
import types
import unittest

import torch

from unsloth.kernels import clear_kernel_backend_overrides, register_kernel_backend
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


if __name__ == "__main__":
    unittest.main()

