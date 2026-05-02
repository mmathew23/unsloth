import importlib
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
for repo in (ROOT / "unsloth", ROOT / "unsloth-zoo"):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

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
    set_kernel_backends,
    set_kernel_backend_for_op,
)
grouped_gemm_module = importlib.import_module("unsloth.kernels.grouped_gemm")
from unsloth.kernels.moe.grouped_gemm.reference.moe_ops import (
    get_routing_indices,
    permute,
    torch_grouped_gemm,
    unpermute,
)
layernorm_module = importlib.import_module("unsloth.kernels.layernorm")
swiglu_module = importlib.import_module("unsloth.kernels.swiglu")
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

    def test_set_kernel_backends_preserves_global_when_only_overrides_are_set(self):
        set_kernel_backend("dummy_global")
        set_kernel_backends(overrides = {"unsloth.swiglu_fg": "dummy_op"})

        state = backend_registry.get_kernel_backend_state()
        self.assertEqual(state["runtime_global_backend"], "dummy_global")
        self.assertEqual(
            state["runtime_overrides"]["unsloth.swiglu_fg"],
            "dummy_op",
        )

    def test_runtime_override_selects_custom_backend_for_low_level_op(self):
        import importlib

        swiglu_mod = importlib.import_module("unsloth.kernels.swiglu")
        register_kernel_backend(
            "unsloth.swiglu_fg",
            "dummy",
            lambda e, g: e + 2 * g,
        )
        e = torch.ones(2, 3)
        g = torch.full((2, 3), 4.0)

        with kernel_backend_context(global_backend = "dummy"):
            out = swiglu_mod.swiglu_fg_kernel(e, g)

        self.assertTrue(torch.equal(out, e + 2 * g))

    def test_runtime_override_selects_custom_backend_for_module_style_op(self):
        import importlib

        layernorm_mod = importlib.import_module("unsloth.kernels.layernorm")
        captured = {}

        def _dummy_layernorm(X, W, b, eps):
            captured["weight_shape"] = tuple(W.shape)
            captured["bias_shape"] = tuple(b.shape)
            captured["eps"] = eps
            return X + W + b

        register_kernel_backend("unsloth.layernorm", "dummy", _dummy_layernorm)

        layernorm = torch.nn.LayerNorm(4)
        X = torch.zeros(2, 4)
        with kernel_backend_context(global_backend = "dummy"):
            out = layernorm_mod.fast_layernorm(layernorm, X)

        self.assertEqual(captured["weight_shape"], (4,))
        self.assertEqual(captured["bias_shape"], (4,))
        self.assertEqual(captured["eps"], layernorm.eps)
        self.assertTrue(torch.equal(out, X + layernorm.weight + layernorm.bias))

    def test_implicit_request_walks_registered_builtins_before_eager(self):
        """A user with a cutile-only install (or any install where the default
        backend is unavailable but another registered builtin IS available)
        must get the registered builtin, not fall straight to eager.
        """
        from unsloth.kernels import _backend_registry as br

        kernel_name = "unsloth.synthetic_implicit_walk"
        # No "triton" / "cutile" entries for this kernel so the test is
        # decoupled from real backend installs. Use synthetic names that
        # mirror the production order: a "primary" that is unavailable, a
        # "secondary" that is available.
        br._BUILTIN_LOADERS["synthetic_primary"] = lambda: None
        br._AVAILABILITY_CHECKS["synthetic_primary"] = lambda: (False, "blocked")
        br._BUILTIN_LOADERS["synthetic_secondary"] = lambda: None
        br._AVAILABILITY_CHECKS["synthetic_secondary"] = lambda: (True, None)
        register_kernel_backend(kernel_name, "synthetic_secondary", lambda x: x + 1)
        register_kernel_backend(kernel_name, "eager", lambda x: x - 1)
        try:
            with patch.object(br, "DEFAULT_KERNEL_BACKEND", "synthetic_primary"):
                backend = get_kernel_backend(kernel_name)
            self.assertEqual(
                backend, "synthetic_secondary",
                "Implicit request must walk registered builtins before eager; "
                "got eager instead, meaning cutile-only installs would silently "
                "lose performance.",
            )
        finally:
            br._BUILTIN_LOADERS.pop("synthetic_primary", None)
            br._BUILTIN_LOADERS.pop("synthetic_secondary", None)
            br._AVAILABILITY_CHECKS.pop("synthetic_primary", None)
            br._AVAILABILITY_CHECKS.pop("synthetic_secondary", None)
            br._REGISTRY.pop(kernel_name, None)

    def test_explicit_request_does_not_walk_other_registered_builtins(self):
        """An explicit ``backend=`` request must NOT silently fall through to
        any other registered builtin. The chain is [requested, eager] only —
        a user who asks for X never silently runs Y."""
        from unsloth.kernels import _backend_registry as br

        kernel_name = "unsloth.synthetic_explicit_walk"
        br._BUILTIN_LOADERS["synthetic_decoy"] = lambda: None
        br._AVAILABILITY_CHECKS["synthetic_decoy"] = lambda: (True, None)
        register_kernel_backend(kernel_name, "synthetic_decoy", lambda x: x + 100)
        register_kernel_backend(kernel_name, "eager", lambda x: x - 1)
        try:
            backend = get_kernel_backend(kernel_name, backend = "missing_backend")
            self.assertEqual(
                backend, "eager",
                "Explicit unknown backend leaked into a non-eager fallback; "
                "explicit requests must only fall to eager.",
            )
        finally:
            br._BUILTIN_LOADERS.pop("synthetic_decoy", None)
            br._AVAILABILITY_CHECKS.pop("synthetic_decoy", None)
            br._REGISTRY.pop(kernel_name, None)

    def test_check_triton_backend_actually_imports_not_just_find_spec(self):
        """Regression: partial triton installs (find_spec returns a valid spec
        but import raises ABI/native-deps error) used to pass the availability
        check. Now the check imports the module so describe_kernel_backends()
        doesn't lie about a broken triton being available."""
        from unsloth.kernels._backend_registry import _check_triton_backend

        # Patch importlib.import_module to simulate a broken triton install:
        # find_spec would succeed (it scans for the module on disk), but the
        # actual import raises (e.g., undefined symbol on .so load).
        original_import = importlib.import_module
        def _fake_import(name, *args, **kwargs):
            if name == "triton":
                raise ImportError("undefined symbol: simulated ABI mismatch")
            return original_import(name, *args, **kwargs)

        with patch.object(importlib, "import_module", side_effect = _fake_import):
            available, reason = _check_triton_backend()
        self.assertFalse(available)
        self.assertIsNotNone(reason)
        self.assertIn("Triton import failed", reason)
        self.assertIn("simulated ABI mismatch", reason)

    def test_failed_backend_loader_is_cached_not_retried(self):
        """A loader that raises should be cached so subsequent dispatches
        don't re-run it. Also verifies clear_backend_load_cache() forces a
        retry."""
        from unsloth.kernels import clear_backend_load_cache
        from unsloth.kernels import _backend_registry as br

        backend_name = "synthetic_failing_backend"
        call_count = {"n": 0}

        def loader():
            call_count["n"] += 1
            raise ImportError("synthetic broken loader")

        br.register_builtin_backend_loader(backend_name, loader)
        # Make sure no stale cache is around.
        clear_backend_load_cache(backend_name)

        with self.assertRaises(ImportError):
            br.ensure_backend_loaded(backend_name)
        self.assertEqual(call_count["n"], 1)

        # Subsequent calls must NOT re-run the loader; they raise the cached exc.
        for _ in range(5):
            with self.assertRaises(ImportError):
                br.ensure_backend_loaded(backend_name)
        self.assertEqual(call_count["n"], 1, "Failed loader was retried; cache broken.")

        # Clearing the cache must allow a retry.
        clear_backend_load_cache(backend_name)
        with self.assertRaises(ImportError):
            br.ensure_backend_loaded(backend_name)
        self.assertEqual(call_count["n"], 2)

        # Cleanup so other tests don't see this synthetic backend.
        br._BUILTIN_LOADERS.pop(backend_name, None)
        clear_backend_load_cache(backend_name)

    def test_concurrent_failed_load_propagates_to_waiters(self):
        """Threads that wait on a concurrent load whose loader fails must also
        see the failure, not silently treat the wake-up as success.
        """
        import threading
        from unsloth.kernels import clear_backend_load_cache
        from unsloth.kernels import _backend_registry as br

        backend_name = "synthetic_concurrent_failing_backend"
        gate = threading.Event()  # held until the test releases the loader thread

        def loader():
            gate.wait(timeout = 5.0)
            raise ImportError("synthetic concurrent failure")

        br.register_builtin_backend_loader(backend_name, loader)
        clear_backend_load_cache(backend_name)

        results = {"loader_thread": None, "waiter_thread": None}

        def call(key):
            try:
                br.ensure_backend_loaded(backend_name)
            except BaseException as e:
                results[key] = e

        loader_thread = threading.Thread(target = call, args = ("loader_thread",))
        waiter_thread = threading.Thread(target = call, args = ("waiter_thread",))
        loader_thread.start()
        # Give loader_thread a moment to take the load slot before waiter_thread arrives.
        import time
        time.sleep(0.1)
        waiter_thread.start()
        gate.set()
        loader_thread.join(timeout = 5.0)
        waiter_thread.join(timeout = 5.0)

        self.assertIsInstance(results["loader_thread"], ImportError)
        self.assertIsInstance(
            results["waiter_thread"], ImportError,
            "Waiter thread did not receive the loader's failure; it was treated as success.",
        )

        br._BUILTIN_LOADERS.pop(backend_name, None)
        clear_backend_load_cache(backend_name)

    def test_kernel_backend_context_preserves_outer_global_when_omitted(self):
        """Regression: kernel_backend_context(overrides=...) must not silently
        wipe an outer set_kernel_backend("cutile") from the surrounding scope.

        Before the sentinel fix, the default ``global_backend=None`` was
        forwarded to ``_RUNTIME_GLOBAL_BACKEND.set(None)`` and replaced the
        outer "cutile" with None inside the block. Now omitting the kwarg
        preserves the outer state; passing ``None`` explicitly still clears
        it (matching ``set_kernel_backend(None)``).
        """
        register_kernel_backend("unsloth.swiglu_fg", "dummy_outer", lambda e, g: e * 0)
        register_kernel_backend("unsloth.swiglu_fg", "dummy_inner", lambda e, g: e + g)

        # Outer setter establishes a global backend.
        set_kernel_backend("dummy_outer")

        # Context with overrides ONLY (no global_backend kwarg) must not
        # clobber the outer "dummy_outer" — kernels not in the override map
        # must keep resolving to "dummy_outer".
        with kernel_backend_context(overrides = {"some_other_kernel": "x"}):
            self.assertEqual(
                backend_registry._RUNTIME_GLOBAL_BACKEND.get(),
                "dummy_outer",
                "outer global backend was wiped by kernel_backend_context "
                "default; sentinel preserve-semantics is broken.",
            )

        # Explicit None must still clear the global backend within the scope.
        with kernel_backend_context(global_backend = None):
            self.assertIsNone(
                backend_registry._RUNTIME_GLOBAL_BACKEND.get(),
                "explicit global_backend=None must clear the outer state "
                "(matches set_kernel_backend(None)).",
            )

        # After exit either way, the outer state is restored.
        self.assertEqual(
            backend_registry._RUNTIME_GLOBAL_BACKEND.get(),
            "dummy_outer",
            "outer global backend not restored after kernel_backend_context exit.",
        )

    def test_per_op_override_beats_global_backend(self):
        register_kernel_backend("unsloth.swiglu_fg", "dummy", lambda e, g: e - g)
        set_kernel_backend("cutile")
        set_kernel_backend_for_op("swiglu_fg", "dummy")

        e = torch.ones(1, 2)
        g = torch.full((1, 2), 3.0)
        out = swiglu_module.swiglu_fg_kernel(e, g)

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

    def test_implicit_default_uses_configured_fallback_backend(self):
        # When no kwarg / env / runtime override is set, the resolution is
        # implicit: it starts from DEFAULT_KERNEL_BACKEND ("triton") and walks
        # through the configured fallback before landing on eager. Anything
        # the user expresses (kwarg, env var, set_kernel_backend, override)
        # is treated as explicit and skips the configured fallback.
        kernel_name = "unsloth.synthetic_implicit_request"
        register_kernel_backend(kernel_name, "dummy_default", lambda x: x + 10)
        register_kernel_backend(kernel_name, "eager", lambda x: x + 1)
        # No registration under DEFAULT_KERNEL_BACKEND ("triton"): chain is
        # [triton, dummy_default, eager] -> triton step is skipped (no impl)
        # -> resolves to dummy_default.

        backend = get_kernel_backend(
            kernel_name,
            fallback_backend = "dummy_default",
        )

        self.assertEqual(backend, "dummy_default")

    def test_set_kernel_backend_request_is_explicit(self):
        # set_kernel_backend / env var should match the explicit kwarg path:
        # if the requested backend is unavailable, the chain skips the
        # configured fallback and goes straight to eager. This prevents a
        # user who said "cutile" from silently running on triton.
        kernel_name = "unsloth.synthetic_runtime_explicit"
        backend_registry._AVAILABILITY_CHECKS["missing"] = lambda: (False, "blocked")
        register_kernel_backend(kernel_name, "dummy_default", lambda x: x + 10)
        register_kernel_backend(kernel_name, "eager", lambda x: x + 1)
        set_kernel_backend("missing")
        try:
            backend = get_kernel_backend(
                kernel_name,
                fallback_backend = "dummy_default",
            )
            self.assertEqual(backend, "eager")
        finally:
            set_kernel_backend(None)
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

    def test_describe_kernel_backends_reports_loader_failure(self):
        backend_name = "synthetic_describe_failure"
        original_loader = backend_registry._BUILTIN_LOADERS.get(backend_name)
        original_package = backend_registry._BACKEND_PACKAGES.get(backend_name)
        original_check = backend_registry._AVAILABILITY_CHECKS.get(backend_name)

        def failing_loader():
            raise RuntimeError("synthetic loader exploded")

        backend_registry._BUILTIN_LOADERS[backend_name] = failing_loader
        backend_registry._BACKEND_PACKAGES[backend_name] = "synthetic.package"
        backend_registry._AVAILABILITY_CHECKS[backend_name] = lambda: (True, None)
        backend_registry.clear_backend_load_cache(backend_name)
        try:
            description = describe_kernel_backends()
            data = description["backends"][backend_name]

            self.assertFalse(data["available"])
            self.assertIn("Backend load failed", data["reason"])
            self.assertIn("synthetic loader exploded", data["reason"])
            self.assertFalse(data["loaded"])
        finally:
            if original_loader is None:
                backend_registry._BUILTIN_LOADERS.pop(backend_name, None)
            else:
                backend_registry._BUILTIN_LOADERS[backend_name] = original_loader
            if original_package is None:
                backend_registry._BACKEND_PACKAGES.pop(backend_name, None)
            else:
                backend_registry._BACKEND_PACKAGES[backend_name] = original_package
            if original_check is None:
                backend_registry._AVAILABILITY_CHECKS.pop(backend_name, None)
            else:
                backend_registry._AVAILABILITY_CHECKS[backend_name] = original_check
            backend_registry.clear_backend_load_cache(backend_name)

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
        original_check = backend_registry._AVAILABILITY_CHECKS.get("cutile")
        backend_registry._AVAILABILITY_CHECKS["cutile"] = lambda: (False, "blocked")

        try:
            backend = get_kernel_backend("unsloth.grouped_gemm", backend = "cutile")

            self.assertEqual(backend, "eager")
        finally:
            if original_check is None:
                backend_registry._AVAILABILITY_CHECKS.pop("cutile", None)
            else:
                backend_registry._AVAILABILITY_CHECKS["cutile"] = original_check

    def test_grouped_gemm_eager_matches_reference(self):
        torch.manual_seed(3407)
        X = torch.randn(3, 4)
        W_up = torch.randn(2, 6, 4)
        selected_experts = torch.tensor([[1, 0], [0, 1], [1, 1]])
        m_sizes, gather_indices = get_routing_indices(selected_experts, 2)
        m_sizes = m_sizes.to(dtype = torch.int32)

        with kernel_backend_context(global_backend = "eager"):
            actual_up = grouped_gemm_module.grouped_gemm(
                X,
                W_up,
                m_sizes,
                topk = 2,
                gather_indices = gather_indices,
                permute_x = True,
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
        with kernel_backend_context(global_backend = "eager"):
            actual_down = grouped_gemm_module.grouped_gemm(
                actual_up,
                W_down,
                m_sizes,
                topk = 2,
                gather_indices = gather_indices,
                permute_y = True,
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

    def test_grouped_gemm_dispatch_passes_unified_kwargs(self):
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
        with kernel_backend_context(global_backend = "dummy"):
            out = grouped_gemm_module.grouped_gemm(
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
            )

        # Static dispatch preserves the Python call shape: X, W, and m_sizes
        # are positional while topk and backend-specific options remain kwargs.
        self.assertEqual(tuple(out.shape), (6, 6))
        self.assertEqual(captured["num_args"], 3)
        self.assertEqual(captured["kwargs"]["topk"], 2)
        for key in (
            "gather_indices",
            "permute_x",
            "kernel_config_fwd",
            "kernel_config_bwd_dX",
            "kernel_config_bwd_dW",
            "autotune",
        ):
            self.assertIn(key, captured["kwargs"])
        self.assertEqual(captured["kwargs"]["kernel_config_fwd"], "fwd_config")
        self.assertTrue(captured["kwargs"]["autotune"])

    def test_grouped_gemm_eager_absorbs_triton_only_kwargs(self):
        # Confirms the registered eager impl tolerates kernel_config_* / autotune
        # without raising (uses **_unused).
        X = torch.randn(6, 4)
        W = torch.randn(2, 4, 4)
        m_sizes = torch.tensor([3, 3], dtype = torch.int32)
        with kernel_backend_context(global_backend = "eager"):
            out = grouped_gemm_module.grouped_gemm(
                X,
                W,
                m_sizes,
                topk = 1,
                kernel_config_fwd = "ignored",
                kernel_config_bwd_dX = "ignored",
                kernel_config_bwd_dW = "ignored",
                autotune = True,
            )
        self.assertEqual(out.shape[-1], W.shape[-2])

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
            import importlib
            import os
            import sys
            from pathlib import Path

            root = Path({str(ROOT)!r})
            for repo in (root / "unsloth", root / "unsloth-zoo"):
                sys.path.insert(0, str(repo))

            os.environ["UNSLOTH_KERNEL_BACKEND"] = "cutile"

            real_import = builtins.__import__

            def blocked(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "triton" or name.startswith("triton."):
                    raise ModuleNotFoundError("triton blocked for test")
                return real_import(name, globals, locals, fromlist, level)

            builtins.__import__ = blocked

            import torch
            import unsloth.kernels._backend_registry as backend_registry
            from unsloth.kernels import get_kernel_backend, kernel_backend_context
            cross_entropy_mod = importlib.import_module("unsloth.kernels.cross_entropy_loss")
            layernorm_mod = importlib.import_module("unsloth.kernels.layernorm")
            rope_mod = importlib.import_module("unsloth.kernels.rope_embedding")
            swiglu_mod = importlib.import_module("unsloth.kernels.swiglu")

            backend_registry._AVAILABILITY_CHECKS["cutile"] = lambda: (False, "cutile blocked for test")
            assert get_kernel_backend("unsloth.cross_entropy_loss", backend="cutile") == "eager"

            x = torch.randn(2, 3, 8)
            ln = torch.nn.LayerNorm(8)
            with kernel_backend_context(global_backend="cutile"):
                y = layernorm_mod.fast_layernorm(ln, x)
                z = swiglu_mod.swiglu_fg_kernel(x, x)

                logits = torch.randn(2, 7, 17, requires_grad=True)
                labels = torch.randint(0, 17, (2, 7))
                labels[0, 0] = -100
                loss = cross_entropy_mod.fast_cross_entropy_loss(logits, labels)
                loss.backward()

                q = torch.randn(2, 4, 9, 32)
                k = torch.randn(2, 2, 9, 32)
                cos = torch.randn(9, 32)
                sin = torch.randn(9, 32)
                q_out, k_out = rope_mod.fast_rope_embedding(q, k, cos, sin)

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

    def test_fp8_rowwise_eager_fallback_warns_once(self):
        """When FBGEMM rowwise is missing AND the user asked for non-eager,
        the eager fallback must emit a one-shot warning so the perf surprise
        shows up in logs. Subsequent calls must NOT re-warn (dedupe)."""
        X = torch.randn(2, 4)
        weight = torch.randn(4, 4)
        weight_scale = torch.ones(4, 1)

        def _fake_eager(X, weight, weight_scale, bias = None):
            return torch.zeros(X.shape[:-1] + (weight.shape[0],), dtype = X.dtype)

        with patch.object(fp8_module, "_HAS_FBGEMM_ROWWISE", False):
            with patch.object(fp8_module, "_warned_fp8_rowwise_eager_fallback", False):
                with patch.object(fp8_module, "_fp8_linear_eager", side_effect = _fake_eager):
                    with patch.object(fp8_module.logger, "warning") as warn:
                        fp8_module.fp8_linear(X, weight, weight_scale)
                        fp8_module.fp8_linear(X, weight, weight_scale)
                        fp8_module.fp8_linear(X, weight, weight_scale)
                self.assertEqual(
                    warn.call_count, 1,
                    "Row-wise FP8 → eager fallback warning must fire exactly once "
                    "across multiple calls (one-shot dedupe).",
                )
                msg = warn.call_args[0][0]
                self.assertIn("fbgemm_gpu", msg)
                self.assertIn("eager", msg)

    def test_fp8_rowwise_falls_to_eager_when_fbgemm_rowwise_missing(self):
        """Regression: cutile-only install (no fbgemm_gpu) used to crash with
        AttributeError on torch.ops.fbgemm.f8f8bf16_rowwise once a non-eager
        backend resolved to the row-wise FP8 dispatch. With the
        _HAS_FBGEMM_ROWWISE guard, this now falls to eager silently."""
        X = torch.randn(2, 4)
        weight = torch.randn(4, 4)
        weight_scale = torch.ones(4, 1)
        eager_called = {"n": 0}

        def _fake_eager(X, weight, weight_scale, bias = None):
            eager_called["n"] += 1
            return torch.zeros(X.shape[:-1] + (weight.shape[0],), dtype = X.dtype)

        def _explode_rowwise(*args, **kwargs):
            raise AttributeError(
                "torch.ops.fbgemm.f8f8bf16_rowwise — simulating absent FBGEMM"
            )

        with patch.object(fp8_module, "_HAS_FBGEMM_ROWWISE", False):
            with patch.object(fp8_module, "get_kernel_backend", return_value = "cutile"):
                with patch.object(fp8_module, "_fp8_linear_eager", side_effect = _fake_eager):
                    with patch.object(fp8_module, "fbgemm_fp8_linear", side_effect = _explode_rowwise):
                        # Pre-fix: would call _explode_rowwise → AttributeError.
                        # Post-fix: guard routes to _fake_eager.
                        fp8_module.fp8_linear(X, weight, weight_scale)

        self.assertEqual(
            eager_called["n"], 1,
            "Row-wise FP8 must fall to eager when FBGEMM rowwise op is absent.",
        )

    def test_fp8_block_path_uses_fp8_block_quant_linear(self):
        # Block FP8 must route through fp8_block_quant_linear; backend-specific
        # act/matmul symbols are already rebound before this call.
        X = torch.randn(2, 4)
        weight = torch.randn(4, 4)
        weight_scale = torch.ones(1, 1)
        captured = {}

        def _fake_block(X, weight, weight_scale, bias = None):
            captured["called"] = True
            return torch.zeros(X.shape[:-1] + (weight.shape[0],), dtype = X.dtype)

        with patch.object(fp8_module, "fp8_block_quant_linear", side_effect = _fake_block) as block_impl:
            with patch.object(fp8_module, "fp8_torch_block_quant_forward") as torch_block_impl:
                fp8_module.fp8_linear(X, weight, weight_scale)

        block_impl.assert_called_once()
        torch_block_impl.assert_not_called()
        self.assertTrue(captured["called"])

    def test_kernel_backend_runtime_switch_rebinds_aliases(self):
        """After `set_kernel_backend(...)` mutates the global ContextVar,
        each dispatcher module's `_resolved_<name>` aliases must rebind to
        the new backend's impl. Verifies the registry hook fires correctly
        for fp8 (act_quant), swiglu, layernorm, and grouped_gemm — one
        alias per refactored module is enough to confirm the unified
        contract."""
        from unsloth.kernels._backend_registry import (
            set_kernel_backend,
            kernel_backend_context,
        )
        import importlib
        import unsloth.kernels.fp8 as fp8_mod
        import unsloth.kernels.swiglu as swiglu_mod
        import unsloth.kernels.layernorm as layernorm_mod

        gg_mod = importlib.import_module("unsloth.kernels.grouped_gemm")

        default_act = fp8_mod._resolved_act_quant
        default_swiglu = swiglu_mod._resolved_swiglu_fg
        default_layernorm = layernorm_mod._resolved_fast_layernorm
        default_gg = gg_mod._resolved_grouped_gemm

        try:
            set_kernel_backend("eager")
            # Each alias must now point at the eager impl (asserted concretely
            # for fp8 — its eager impl is reachable as `_act_quant_eager`).
            self.assertIs(fp8_mod._resolved_act_quant, fp8_mod._act_quant_eager)
            # The other modules: just confirm the alias rebound to a non-None
            # callable. The eager impl symbol naming differs across modules.
            self.assertIsNotNone(swiglu_mod._resolved_swiglu_fg)
            self.assertIsNotNone(layernorm_mod._resolved_fast_layernorm)
            self.assertIsNotNone(gg_mod._resolved_grouped_gemm)
            # And the baked fp8 backend constant should track too.
            self.assertEqual(fp8_mod._BAKED_BLOCK_FP8_BACKEND, "eager")
        finally:
            set_kernel_backend(None)

        # After clearing, aliases restore to module-load defaults.
        self.assertIs(fp8_mod._resolved_act_quant, default_act)
        self.assertIs(swiglu_mod._resolved_swiglu_fg, default_swiglu)
        self.assertIs(layernorm_mod._resolved_fast_layernorm, default_layernorm)
        self.assertIs(gg_mod._resolved_grouped_gemm, default_gg)

        # `kernel_backend_context` must also fire the hook on enter AND exit.
        with kernel_backend_context(global_backend = "eager"):
            self.assertIs(fp8_mod._resolved_act_quant, fp8_mod._act_quant_eager)
        # Exiting the context restores the outer alias.
        self.assertIs(fp8_mod._resolved_act_quant, default_act)

    def test_static_default_hot_symbols_are_exported_callables(self):
        import importlib

        ce_mod = importlib.import_module("unsloth.kernels.cross_entropy_loss")
        fast_lora_mod = importlib.import_module("unsloth.kernels.fast_lora")
        geglu_mod = importlib.import_module("unsloth.kernels.geglu")
        gg_mod = importlib.import_module("unsloth.kernels.grouped_gemm")
        rms_mod = importlib.import_module("unsloth.kernels.rms_layernorm")
        rope_mod = importlib.import_module("unsloth.kernels.rope_embedding")
        swiglu_mod = importlib.import_module("unsloth.kernels.swiglu")

        for default_symbol in (
            ce_mod.fast_cross_entropy_loss_default,
            gg_mod.grouped_gemm_default,
            rms_mod.fast_rms_layernorm_default,
            rope_mod.fast_rope_embedding_default,
            rope_mod.rope_embedding_default,
            swiglu_mod.swiglu_fg_kernel_default,
            swiglu_mod.swiglu_DWf_DW_dfg_kernel_default,
            geglu_mod.geglu_exact_forward_kernel_default,
            geglu_mod.geglu_exact_backward_kernel_default,
            geglu_mod.geglu_approx_forward_kernel_default,
            geglu_mod.geglu_approx_backward_kernel_default,
        ):
            self.assertTrue(callable(default_symbol))

        self.assertIs(ce_mod.fast_cross_entropy_loss, ce_mod.fast_cross_entropy_loss_default)
        self.assertTrue(callable(gg_mod.grouped_gemm))
        self.assertIs(rms_mod.fast_rms_layernorm, rms_mod.fast_rms_layernorm_default)
        self.assertIs(rope_mod.fast_rope_embedding, rope_mod.fast_rope_embedding_default)
        self.assertIs(rope_mod.rope_embedding, rope_mod.rope_embedding_default)
        self.assertIs(swiglu_mod.swiglu_fg_kernel, swiglu_mod.swiglu_fg_kernel_default)
        self.assertIs(
            swiglu_mod.swiglu_DWf_DW_dfg_kernel,
            swiglu_mod.swiglu_DWf_DW_dfg_kernel_default,
        )
        self.assertIs(geglu_mod.geglu_exact_forward_kernel, geglu_mod.geglu_exact_forward_kernel_default)
        self.assertIs(geglu_mod.geglu_exact_backward_kernel, geglu_mod.geglu_exact_backward_kernel_default)
        self.assertIs(geglu_mod.geglu_approx_forward_kernel, geglu_mod.geglu_approx_forward_kernel_default)
        self.assertIs(geglu_mod.geglu_approx_backward_kernel, geglu_mod.geglu_approx_backward_kernel_default)

        self.assertIs(fast_lora_mod.swiglu_fg_kernel, swiglu_mod.swiglu_fg_kernel_default)
        self.assertIs(
            fast_lora_mod.swiglu_DWf_DW_dfg_kernel,
            swiglu_mod.swiglu_DWf_DW_dfg_kernel_default,
        )
        self.assertIs(
            fast_lora_mod.geglu_exact_forward_kernel,
            geglu_mod.geglu_exact_forward_kernel_default,
        )
        self.assertIs(
            fast_lora_mod.geglu_approx_backward_kernel,
            geglu_mod.geglu_approx_backward_kernel_default,
        )

    def test_grouped_gemm_context_dispatch_does_not_rebind_exported_symbol(self):
        import importlib

        from unsloth.kernels._backend_registry import register_kernel_backend

        gg_mod = importlib.import_module("unsloth.kernels.grouped_gemm")

        def _dummy_a(*args, **kwargs):
            return "a"

        def _dummy_b(*args, **kwargs):
            return "b"

        register_kernel_backend("unsloth.grouped_gemm", "dummy_a", _dummy_a)
        register_kernel_backend("unsloth.grouped_gemm", "dummy_b", _dummy_b)

        exported = gg_mod.grouped_gemm
        with kernel_backend_context(overrides = {"grouped_gemm": "dummy_a"}):
            self.assertIs(gg_mod.grouped_gemm, exported)
            self.assertEqual(gg_mod.grouped_gemm(), "a")
            with kernel_backend_context(overrides = {"grouped_gemm": "dummy_b"}):
                self.assertIs(gg_mod.grouped_gemm, exported)
                self.assertEqual(gg_mod.grouped_gemm(), "b")
            self.assertIs(gg_mod.grouped_gemm, exported)
            self.assertEqual(gg_mod.grouped_gemm(), "a")

        self.assertIs(gg_mod.grouped_gemm, exported)

    def test_static_default_hot_symbols_fall_back_to_direct_eager_not_dispatcher(self):
        import importlib

        ce_mod = importlib.import_module("unsloth.kernels.cross_entropy_loss")
        fast_lora_mod = importlib.import_module("unsloth.kernels.fast_lora")
        geglu_mod = importlib.import_module("unsloth.kernels.geglu")
        gg_mod = importlib.import_module("unsloth.kernels.grouped_gemm")
        rms_mod = importlib.import_module("unsloth.kernels.rms_layernorm")
        rope_mod = importlib.import_module("unsloth.kernels.rope_embedding")
        swiglu_mod = importlib.import_module("unsloth.kernels.swiglu")

        def _closure_has(fn, target):
            while hasattr(fn, "__wrapped__"):
                fn = fn.__wrapped__
            return any(cell.cell_contents is target for cell in (fn.__closure__ or ()))

        try:
            with patch.object(backend_registry, "get_kernel_impl", side_effect = RuntimeError("blocked")):
                ce_mod._rebind_cross_entropy_aliases()
                geglu_mod._rebind_geglu_aliases()
                gg_mod._rebind_grouped_gemm_aliases()
                rms_mod._rebind_rms_layernorm_aliases()
                rope_mod._rebind_rope_embedding_aliases()
                swiglu_mod._rebind_swiglu_aliases()

            self.assertIs(ce_mod.fast_cross_entropy_loss_default, ce_mod._fast_cross_entropy_loss_eager)
            self.assertIs(gg_mod.grouped_gemm_default, gg_mod._grouped_gemm_eager)
            self.assertTrue(
                _closure_has(rms_mod.fast_rms_layernorm_default, rms_mod._fast_rms_layernorm_eager)
            )
            self.assertIs(rope_mod.rope_embedding_default, rope_mod._rope_embedding_eager)
            if rope_mod.DEVICE_COUNT <= 1:
                self.assertTrue(
                    _closure_has(
                        rope_mod.fast_rope_embedding_default,
                        rope_mod._fast_rope_embedding_eager,
                    )
                )
            self.assertIs(swiglu_mod.swiglu_fg_kernel_default, swiglu_mod._swiglu_fg_eager)
            self.assertIs(
                swiglu_mod.swiglu_DWf_DW_dfg_kernel_default,
                swiglu_mod._swiglu_DWf_DW_dfg_eager,
            )
            self.assertIs(
                geglu_mod.geglu_exact_forward_kernel_default,
                geglu_mod._geglu_exact_forward_eager,
            )
            self.assertIs(
                fast_lora_mod.swiglu_fg_kernel,
                swiglu_mod.swiglu_fg_kernel_default,
            )
        finally:
            ce_mod._rebind_cross_entropy_aliases()
            geglu_mod._rebind_geglu_aliases()
            gg_mod._rebind_grouped_gemm_aliases()
            rms_mod._rebind_rms_layernorm_aliases()
            rope_mod._rebind_rope_embedding_aliases()
            swiglu_mod._rebind_swiglu_aliases()

    def test_utils_fp8_lazy_binding_replaces_wrapper_symbols(self):
        import importlib

        fp8_real = importlib.import_module("unsloth.kernels.fp8")
        utils_mod = importlib.import_module("unsloth.kernels.utils")

        bound_weight_dequant, bound_fp8_linear = utils_mod._bind_fp8_symbols()

        self.assertIs(bound_weight_dequant, fp8_real.weight_dequant)
        self.assertIs(bound_fp8_linear, fp8_real.fp8_linear)
        self.assertIs(utils_mod.weight_dequant, fp8_real.weight_dequant)
        self.assertIs(utils_mod.fp8_linear, fp8_real.fp8_linear)

    def test_fp8_block_path_forwards_bias(self):
        # The dispatcher must forward a non-None bias kwarg into
        # fp8_block_quant_linear -- prior to the fix it was silently dropped on
        # the cutile / probed-not-OK FBGEMM block path.
        X = torch.randn(2, 4)
        weight = torch.randn(4, 4)
        weight_scale = torch.ones(1, 1)
        bias = torch.randn(4)
        captured = {}

        def _fake_block(X, weight, weight_scale, bias = None):
            captured["bias_id"] = id(bias) if bias is not None else None
            captured["bias_is_none"] = bias is None
            return torch.zeros(X.shape[:-1] + (weight.shape[0],), dtype = X.dtype)

        with patch.object(fp8_module, "fp8_block_quant_linear", side_effect = _fake_block):
            fp8_module.fp8_linear(X, weight, weight_scale, bias = bias)

        self.assertFalse(
            captured["bias_is_none"],
            "fp8_linear must forward bias kwarg into fp8_block_quant_linear.",
        )
        self.assertEqual(captured["bias_id"], id(bias))

    def test_fp8_block_quant_no_bias_unchanged(self):
        # Regression: bias=None default must yield the same output as before
        # the bias-plumbing change.
        torch.manual_seed(0)
        X = torch.randn(2, 4, dtype = torch.float32)
        weight = torch.randn(4, 4, dtype = torch.float32)
        weight_scale = torch.ones(1, 1, dtype = torch.float32)

        with kernel_backend_context(global_backend = "eager"):
            out_no_bias = fp8_module.fp8_linear(X, weight, weight_scale, bias = None)
            out_default = fp8_module.fp8_linear(X, weight, weight_scale)

        self.assertTrue(torch.equal(out_no_bias, out_default))

    def test_fp8_eager_block_dequant_does_not_transpose_square_scales(self):
        # Square matrices have symmetric expected/reversed scale shapes. The
        # eager fallback must prefer the actual expected shape, otherwise Qwen3
        # FP8 square projections dequantize with transposed block scales.
        block_size = 128
        x = torch.ones((256, 256), dtype = torch.float8_e4m3fn)
        scales = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype = torch.float32,
        )

        out = fp8_module._weight_dequant_block_eager(
            x,
            scales,
            block_size = block_size,
            dtype = torch.float32,
        )

        self.assertEqual(out[0, 0].item(), 1.0)
        self.assertEqual(out[0, 128].item(), 2.0)
        self.assertEqual(out[128, 0].item(), 3.0)
        self.assertEqual(out[128, 128].item(), 4.0)

    def _make_block_quant_linear(self, in_features, out_features, bias):
        # Synthetic block-quant linear: float weights paired with all-ones
        # block scales so the FP8 path is exercised but the dequant collapses
        # to identity. Returns (weight, weight_scale, bias_tensor).
        weight = torch.randn(out_features, in_features, dtype = torch.float32, device = "cuda" if torch.cuda.is_available() else "cpu") * 0.05
        block_size = [128, 128]
        m, n = weight.shape
        p = (m + block_size[0] - 1) // block_size[0]
        q = (n + block_size[1] - 1) // block_size[1]
        weight_scale = torch.ones(p, q, dtype = torch.float32, device = weight.device)
        bias_t = torch.randn(out_features, dtype = torch.float32, device = weight.device) if bias else None
        return weight, weight_scale, bias_t

    def test_fp8_block_quant_bias_forward_correctness(self):
        # Forward parity vs eager reference under the eager fp8_linear path
        # (the only path runnable on CPU without GPU). With weight_scale=1 the
        # FP8 dequant is a no-op so the output should equal X @ W.T + b.
        weight, weight_scale, bias = self._make_block_quant_linear(128, 64, bias = True)
        X = torch.randn(4, 16, 128, dtype = torch.float32, device = weight.device)

        with kernel_backend_context(global_backend = "eager"):
            out = fp8_module.fp8_linear(X, weight, weight_scale, bias = bias)

        ref = X @ weight.T + bias
        self.assertEqual(out.shape, ref.shape)
        self.assertTrue(torch.allclose(out, ref, atol = 1e-2, rtol = 1e-2))

    def test_fp8_block_quant_bias_backward_gradient(self):
        # FP8BlockQuantLinear.backward must return a non-None d_bias in slot 4.
        # Drive the autograd Function directly so the test runs CPU-or-GPU.
        if not torch.cuda.is_available():
            self.skipTest("FP8BlockQuantLinear requires a CUDA device for triton-backed matmul.")
        torch.manual_seed(0)
        # Block-quant matmul kernel expects FP8 weights paired with float scales.
        weight_f, weight_scale, bias = self._make_block_quant_linear(128, 64, bias = True)
        weight_fp8 = weight_f.to(torch.float8_e4m3fn)
        X = torch.randn(4, 16, 128, dtype = torch.bfloat16, device = weight_fp8.device, requires_grad = True)
        bias_p = bias.detach().clone().to(torch.bfloat16).requires_grad_(True)

        out = fp8_module.FP8BlockQuantLinear.apply(X, weight_fp8, weight_scale, bias_p)
        out.sum().backward()

        self.assertIsNotNone(bias_p.grad, "d_bias must not be None when bias is provided.")
        self.assertEqual(bias_p.grad.shape, bias_p.shape)
        ref_grad = torch.ones_like(out).sum(dim = tuple(range(out.ndim - 1)))
        self.assertTrue(
            torch.allclose(bias_p.grad.to(torch.float32), ref_grad.to(torch.float32), atol = 1e-2, rtol = 1e-2),
            "d_bias must equal the reduction of grad_output over leading dims.",
        )

    def test_fp8_fbgemm_rowwise_bias_backward_gradient(self):
        if not fp8_module._HAS_FBGEMM_ROWWISE:
            self.skipTest("torch.ops.fbgemm.f8f8bf16_rowwise unavailable.")
        if not torch.cuda.is_available():
            self.skipTest("FBGEMM rowwise requires CUDA.")
        torch.manual_seed(0)
        in_features, out_features = 128, 64
        # Row-wise: weight_scale shape is (out_features, 1), per-row.
        weight = torch.randn(out_features, in_features, dtype = torch.bfloat16, device = "cuda") * 0.05
        weight_fp8 = weight.to(torch.float8_e4m3fn)
        weight_scale = torch.ones(out_features, 1, dtype = torch.bfloat16, device = "cuda")
        bias = torch.randn(out_features, dtype = torch.bfloat16, device = "cuda", requires_grad = True)
        X = torch.randn(2, 8, in_features, dtype = torch.bfloat16, device = "cuda", requires_grad = True)

        out = fp8_module.FbgemmFp8Linear_matmul.apply(X, weight_fp8, weight_scale, bias)
        out.sum().backward()

        self.assertIsNotNone(bias.grad, "FbgemmFp8Linear_matmul.backward must populate d_bias.")
        self.assertEqual(bias.grad.shape, bias.shape)
        ref_grad = torch.ones_like(out).sum(dim = tuple(range(out.ndim - 1)))
        self.assertTrue(
            torch.allclose(bias.grad.to(torch.float32), ref_grad.to(torch.float32), atol = 1e-2, rtol = 1e-2),
        )

    def test_fp8_fbgemm_block_bias_backward_gradient(self):
        # FBGEMM block path is gated by the runtime probe; skip when probe says no.
        if os.environ.get("UNSLOTH_HAS_FBGEMM", "0") != "1":
            self.skipTest("FBGEMM blockwise FP8 not probed-OK on this device.")
        if not torch.cuda.is_available():
            self.skipTest("FBGEMM block requires CUDA.")
        if fp8_module.triton_quantize_fp8_block is None:
            self.skipTest("triton_quantize_fp8_block unavailable.")
        torch.manual_seed(0)
        in_features, out_features = 128, 128
        weight = torch.randn(out_features, in_features, dtype = torch.bfloat16, device = "cuda") * 0.05
        weight_fp8 = weight.to(torch.float8_e4m3fn)
        weight_scale = torch.ones(1, 1, dtype = torch.float32, device = "cuda")
        bias = torch.randn(out_features, dtype = torch.bfloat16, device = "cuda", requires_grad = True)
        X = torch.randn(2, 8, in_features, dtype = torch.bfloat16, device = "cuda", requires_grad = True)

        out = fp8_module.FP8_fbgemm_block_linear.apply(X, weight_fp8, weight_scale, bias)
        out.sum().backward()

        self.assertIsNotNone(bias.grad, "FP8_fbgemm_block_linear.backward must populate d_bias.")
        self.assertEqual(bias.grad.shape, bias.shape)
        ref_grad = torch.ones_like(out).sum(dim = tuple(range(out.ndim - 1)))
        self.assertTrue(
            torch.allclose(bias.grad.to(torch.float32), ref_grad.to(torch.float32), atol = 1e-2, rtol = 1e-2),
        )


if __name__ == "__main__":
    unittest.main()
