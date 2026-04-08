import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for repo in (ROOT / "unsloth", ROOT / "unsloth_zoo"):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from unsloth.kernels import (
    clear_kernel_backend_overrides,
    get_kernel_backend,
    is_kernel_backend_available,
    kernel_backend_context,
    register_kernel_backend,
    set_kernel_backend,
    set_kernel_backend_for_op,
)
from unsloth.kernels.layernorm import fast_layernorm
from unsloth.kernels.swiglu import swiglu_fg_kernel


class KernelBackendTests(unittest.TestCase):
    def tearDown(self):
        clear_kernel_backend_overrides()
        os.environ.pop("UNSLOTH_KERNEL_BACKEND", None)
        os.environ.pop("UNSLOTH_KERNEL_BACKEND_OVERRIDES", None)

    def test_unknown_backend_falls_back_to_triton(self):
        os.environ["UNSLOTH_KERNEL_BACKEND"] = "missing_backend"
        backend = get_kernel_backend("unsloth.cross_entropy_loss")
        self.assertEqual(backend, "triton")

        available, _ = is_kernel_backend_available("cutile")
        self.assertTrue(available)

    def test_runtime_override_selects_custom_backend_for_low_level_op(self):
        register_kernel_backend(
            "unsloth.swiglu_fg",
            "dummy",
            lambda e, g: e + 2 * g,
        )
        e = torch.ones(2, 3)
        g = torch.full((2, 3), 4.0)

        with kernel_backend_context(global_backend = "dummy"):
            out = swiglu_fg_kernel(e, g)

        self.assertTrue(torch.equal(out, e + 2 * g))

    def test_runtime_override_selects_custom_backend_for_module_style_op(self):
        captured = {}

        def _dummy_layernorm(X, W, b, eps):
            captured["weight_shape"] = tuple(W.shape)
            captured["bias_shape"] = tuple(b.shape)
            captured["eps"] = eps
            return X + W + b

        register_kernel_backend("unsloth.layernorm", "dummy", _dummy_layernorm)

        layernorm = torch.nn.LayerNorm(4)
        X = torch.zeros(2, 4)
        out = fast_layernorm(layernorm, X, backend = "dummy")

        self.assertEqual(captured["weight_shape"], (4,))
        self.assertEqual(captured["bias_shape"], (4,))
        self.assertEqual(captured["eps"], layernorm.eps)
        self.assertTrue(torch.equal(out, X + layernorm.weight + layernorm.bias))

    def test_per_op_override_beats_global_backend(self):
        register_kernel_backend("unsloth.swiglu_fg", "dummy", lambda e, g: e - g)
        set_kernel_backend("cutile")
        set_kernel_backend_for_op("swiglu_fg", "dummy")

        e = torch.ones(1, 2)
        g = torch.full((1, 2), 3.0)
        out = swiglu_fg_kernel(e, g)

        self.assertTrue(torch.equal(out, e - g))

    def test_cutile_kernel_wrappers_work_without_triton_imports(self):
        script = textwrap.dedent(
            f"""
            import builtins
            import os
            import sys
            from pathlib import Path

            root = Path({str(ROOT)!r})
            for repo in (root / "unsloth", root / "unsloth_zoo"):
                sys.path.insert(0, str(repo))

            os.environ["UNSLOTH_IS_PRESENT"] = "1"
            os.environ["UNSLOTH_KERNEL_BACKEND"] = "cutile"

            real_import = builtins.__import__
            def blocked(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "triton" or name.startswith("triton."):
                    raise ModuleNotFoundError("triton blocked for test")
                return real_import(name, globals, locals, fromlist, level)

            builtins.__import__ = blocked

            import unsloth
            import torch
            from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss
            from unsloth.kernels.layernorm import fast_layernorm
            from unsloth.kernels.swiglu import swiglu_fg_kernel

            x = torch.randn(2, 3, 64, device="cuda", dtype=torch.bfloat16)
            ln = torch.nn.LayerNorm(64, device="cuda", dtype=torch.bfloat16)
            y = fast_layernorm(ln, x, backend="cutile")
            z = swiglu_fg_kernel(x, x, backend="cutile")

            logits = torch.randn(2, 7, 257, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            labels = torch.randint(0, 257, (2, 7), device="cuda")
            labels[0, 0] = -100
            loss = fast_cross_entropy_loss(logits, labels, backend="cutile")
            loss.backward()

            assert y.shape == x.shape
            assert z.shape == x.shape
            assert loss.item() == loss.item()
            """
        )
        env = os.environ.copy()
        completed = subprocess.run(
            [sys.executable, "-c", script],
            env = env,
            capture_output = True,
            text = True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg = completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
