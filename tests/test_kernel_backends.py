import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
for repo in (ROOT / "unsloth", ROOT / "unsloth-zoo"):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_SKIP_MODEL_IMPORTS", "1")

from unsloth.kernels import (
    clear_kernel_backend_overrides,
    describe_kernel_backends,
    ensure_backend_loaded,
    get_kernel_backend,
    get_kernel_impl,
    get_registered_kernel_backends,
    is_kernel_backend_available,
    kernel_backend_context,
    register_kernel_backend,
    set_kernel_backend,
    set_kernel_backend_for_op,
)
from unsloth.kernels.grouped_gemm import grouped_gemm
from unsloth.kernels.moe.grouped_gemm.reference.moe_ops import (
    get_routing_indices,
    permute,
    torch_grouped_gemm,
    unpermute,
)
from unsloth.kernels.layernorm import fast_layernorm
from unsloth.kernels.swiglu import swiglu_fg_kernel
import unsloth.kernels.fp8 as fp8_module
import unsloth.kernels._backend_registry as backend_registry


class KernelBackendTests(unittest.TestCase):
    def tearDown(self):
        clear_kernel_backend_overrides()
        os.environ.pop("UNSLOTH_KERNEL_BACKEND", None)
        os.environ.pop("UNSLOTH_KERNEL_BACKEND_OVERRIDES", None)
        os.environ.pop("UNSLOTH_KERNEL_BACKEND_STRICT", None)

    def test_unknown_backend_falls_back_to_triton_or_eager(self):
        os.environ["UNSLOTH_KERNEL_BACKEND"] = "missing_backend"
        backend = get_kernel_backend("unsloth.cross_entropy_loss")
        self.assertIn(backend, {"triton", "eager"})

        available, reason = is_kernel_backend_available("cutile")
        self.assertIsInstance(available, bool)
        if not available:
            self.assertTrue(reason)

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

    def test_explicit_backend_request_skips_intermediate_fallback_backend(self):
        kernel_name = "unsloth.synthetic_explicit_request"
        backend_registry._AVAILABILITY_CHECKS["missing"] = lambda: (False, "blocked")
        register_kernel_backend(kernel_name, "dummy_default", lambda x: x + 10)
        register_kernel_backend(kernel_name, "eager", lambda x: x + 1)

        backend = get_kernel_backend(
            kernel_name,
            backend = "missing",
            fallback_backend = "dummy_default",
        )

        self.assertEqual(backend, "eager")
        backend_registry._AVAILABILITY_CHECKS.pop("missing", None)

    def test_implicit_backend_request_can_use_configured_fallback_backend(self):
        kernel_name = "unsloth.synthetic_implicit_request"
        backend_registry._AVAILABILITY_CHECKS["missing"] = lambda: (False, "blocked")
        register_kernel_backend(kernel_name, "dummy_default", lambda x: x + 10)
        register_kernel_backend(kernel_name, "eager", lambda x: x + 1)
        set_kernel_backend("missing")

        backend = get_kernel_backend(
            kernel_name,
            fallback_backend = "dummy_default",
        )

        self.assertEqual(backend, "dummy_default")
        backend_registry._AVAILABILITY_CHECKS.pop("missing", None)

    def test_describe_kernel_backends_reports_capabilities(self):
        description = describe_kernel_backends()

        self.assertIn("backends", description)
        self.assertIn("eager", description["backends"])
        eager = description["backends"]["eager"]
        self.assertIn("registered_ops", eager)
        self.assertIn("unsloth.cross_entropy_loss", eager["registered_ops"])

        triton = description["backends"]["triton"]
        self.assertEqual(
            triton["package"],
            "unsloth.kernels.backends.triton",
        )
        self.assertIn("registered_ops", triton)

    def test_backend_loader_registers_once(self):
        available, reason = is_kernel_backend_available("triton")
        if not available:
            self.skipTest(reason or "Triton is unavailable.")

        ensure_backend_loaded("triton")
        first = describe_kernel_backends()["backends"]["triton"]["registered_op_count"]
        ensure_backend_loaded("triton")
        second = describe_kernel_backends()["backends"]["triton"]["registered_op_count"]

        self.assertEqual(first, second)

    def test_grouped_gemm_explicit_cutile_request_falls_to_eager(self):
        backend_registry._AVAILABILITY_CHECKS["cutile"] = lambda: (False, "blocked")

        backend = get_kernel_backend("unsloth.grouped_gemm", backend = "cutile")

        self.assertEqual(backend, "eager")
        backend_registry._AVAILABILITY_CHECKS.pop("cutile", None)

    def test_grouped_gemm_eager_matches_reference(self):
        torch.manual_seed(3407)
        X = torch.randn(3, 4)
        W_up = torch.randn(2, 6, 4)
        selected_experts = torch.tensor([[1, 0], [0, 1], [1, 1]])
        m_sizes, gather_indices = get_routing_indices(selected_experts, 2)
        m_sizes = m_sizes.to(dtype = torch.int32)

        actual_up = grouped_gemm(
            X,
            W_up,
            m_sizes,
            topk = 2,
            gather_indices = gather_indices,
            permute_x = True,
            backend = "eager",
            autotune = True,
        )
        expected_up = torch_grouped_gemm(
            permute(X, gather_indices, 2),
            W_up,
            m_sizes,
            transpose = True,
        )
        torch.testing.assert_close(actual_up, expected_up)

        W_down = torch.randn(2, 4, 6)
        actual_down = grouped_gemm(
            actual_up,
            W_down,
            m_sizes,
            topk = 2,
            gather_indices = gather_indices,
            permute_y = True,
            backend = "eager",
            autotune = True,
        )
        expected_down = unpermute(
            torch_grouped_gemm(actual_up, W_down, m_sizes, transpose = True),
            gather_indices,
        )
        torch.testing.assert_close(actual_down, expected_down)

    def test_grouped_gemm_backend_registration(self):
        self.assertIn("eager", get_registered_kernel_backends("unsloth.grouped_gemm"))

        available, _ = is_kernel_backend_available("triton")
        if available:
            ensure_backend_loaded("triton")
            self.assertIn(
                "triton",
                get_registered_kernel_backends("unsloth.grouped_gemm"),
            )

        available, _ = is_kernel_backend_available("cutile")
        if available:
            ensure_backend_loaded("cutile")
            self.assertIn(
                "cutile",
                get_registered_kernel_backends("unsloth.grouped_gemm"),
            )

    def test_grouped_gemm_non_triton_backends_ignore_triton_only_kwargs(self):
        captured = {}

        def _dummy_grouped_gemm(*args, **kwargs):
            captured["num_args"] = len(args)
            captured["kwargs"] = dict(kwargs)
            X, W = args[:2]
            return torch.zeros(X.shape[0] * 2, W.shape[1], dtype = X.dtype)

        register_kernel_backend("unsloth.grouped_gemm", "dummy", _dummy_grouped_gemm)

        X = torch.randn(3, 4)
        W = torch.randn(2, 6, 4)
        m_sizes = torch.tensor([3, 3], dtype = torch.int32)
        gather_indices = torch.arange(6, dtype = torch.int32)
        out = grouped_gemm(
            X,
            W,
            m_sizes,
            topk = 2,
            gather_indices = gather_indices,
            permute_x = True,
            kernel_config_fwd = "fwd_config",
            kernel_config_bwd_dX = "dx_config",
            kernel_config_bwd_dW = "dw_config",
            autotune = True,
            backend = "dummy",
        )

        self.assertEqual(tuple(out.shape), (6, 6))
        self.assertEqual(captured["num_args"], 12)
        self.assertEqual(captured["kwargs"], {})

    def test_requested_backend_runtime_failure_is_surfaced(self):
        kernel_name = "unsloth.synthetic_runtime_failure"

        register_kernel_backend(kernel_name, "dummy", lambda x: (_ for _ in ()).throw(RuntimeError("kernel exploded")))
        register_kernel_backend(kernel_name, "eager", lambda x: x + 1)

        implementation = get_kernel_impl(kernel_name, backend = "dummy")

        with self.assertRaisesRegex(RuntimeError, "kernel exploded"):
            implementation(torch.tensor(1))

    def test_cutile_request_falls_back_to_eager_when_triton_is_blocked(self):
        script = textwrap.dedent(
            f"""
            import builtins
            import os
            import sys
            from pathlib import Path

            root = Path({str(ROOT)!r})
            for repo in (root / "unsloth", root / "unsloth-zoo"):
                sys.path.insert(0, str(repo))

            os.environ["UNSLOTH_IS_PRESENT"] = "1"
            os.environ["UNSLOTH_SKIP_MODEL_IMPORTS"] = "1"
            os.environ["UNSLOTH_KERNEL_BACKEND"] = "cutile"

            real_import = builtins.__import__

            def blocked(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "triton" or name.startswith("triton."):
                    raise ModuleNotFoundError("triton blocked for test")
                return real_import(name, globals, locals, fromlist, level)

            builtins.__import__ = blocked

            import torch
            import unsloth.kernels._backend_registry as backend_registry
            from unsloth.kernels import get_kernel_backend
            from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss
            from unsloth.kernels.layernorm import fast_layernorm
            from unsloth.kernels.rope_embedding import fast_rope_embedding
            from unsloth.kernels.swiglu import swiglu_fg_kernel

            backend_registry._AVAILABILITY_CHECKS["cutile"] = lambda: (False, "cutile blocked for test")
            assert get_kernel_backend("unsloth.cross_entropy_loss", backend="cutile") == "eager"

            x = torch.randn(2, 3, 8)
            ln = torch.nn.LayerNorm(8)
            y = fast_layernorm(ln, x, backend="cutile")
            z = swiglu_fg_kernel(x, x, backend="cutile")

            logits = torch.randn(2, 7, 17, requires_grad=True)
            labels = torch.randint(0, 17, (2, 7))
            labels[0, 0] = -100
            loss = fast_cross_entropy_loss(logits, labels, backend="cutile")
            loss.backward()

            q = torch.randn(2, 4, 9, 32)
            k = torch.randn(2, 2, 9, 32)
            cos = torch.randn(9, 32)
            sin = torch.randn(9, 32)
            q_out, k_out = fast_rope_embedding(q, k, cos, sin, backend="cutile")

            assert y.shape == x.shape
            assert z.shape == x.shape
            assert q_out.shape == q.shape
            assert k_out.shape == k.shape
            assert torch.isfinite(loss)
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

    def test_fp8_block_path_preserves_resolved_backend(self):
        X = torch.randn(2, 4)
        weight = torch.randn(4, 4)
        weight_scale = torch.ones(1, 1)
        captured = {}

        def _fake_block(X, weight, weight_scale, *, backend = None):
            captured["backend"] = backend
            return torch.zeros(X.shape[:-1] + (weight.shape[0],), dtype = X.dtype)

        with patch.object(fp8_module, "get_kernel_backend", return_value = "triton"):
            with patch.object(fp8_module, "fp8_block_quant_linear", side_effect = _fake_block):
                fp8_module.fp8_linear(X, weight, weight_scale, backend = "cutile")

        self.assertEqual(captured["backend"], "triton")

    def test_fp8_rowwise_path_preserves_resolved_backend(self):
        X = torch.randn(2, 4)
        weight = torch.randn(4, 4)
        weight_scale = torch.ones(4, 1)
        captured = {}

        def _fake_rowwise(X, weight, weight_scale, bias = None, *, backend = None):
            captured["backend"] = backend
            return torch.zeros(X.shape[:-1] + (weight.shape[0],), dtype = X.dtype)

        with patch.object(fp8_module, "get_kernel_backend", return_value = "triton"):
            with patch.object(fp8_module, "fbgemm_fp8_linear", side_effect = _fake_rowwise):
                fp8_module.fp8_linear(X, weight, weight_scale, backend = "cutile")

        self.assertEqual(captured["backend"], "triton")

    def test_fp8_eager_fallback_uses_eager_linear(self):
        X = torch.randn(2, 4)
        weight = torch.randn(4, 4)
        weight_scale = torch.ones(1, 1)

        with patch.object(fp8_module, "get_kernel_backend", return_value = "eager"):
            with patch.object(fp8_module, "_fp8_linear_eager", return_value = torch.ones(2, 4)) as eager_impl:
                out = fp8_module.fp8_linear(X, weight, weight_scale, backend = "cutile")

        eager_impl.assert_called_once()
        self.assertTrue(torch.equal(out, torch.ones(2, 4)))


if __name__ == "__main__":
    unittest.main()
