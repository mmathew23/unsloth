import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

from unsloth.kernels import describe_kernel_backends, ensure_backend_loaded, is_kernel_backend_available
from unsloth.kernels.backends._manifests import CUTILE_EXPECTED_OPS, TRITON_EXPECTED_OPS

CUTILE_DIR = ROOT / "unsloth" / "unsloth" / "kernels" / "backends" / "cutile"


class KernelBackendArchitectureTests(unittest.TestCase):
    def test_vendored_cutile_files_use_local_adapter(self):
        vendored_files = sorted(
            path
            for path in CUTILE_DIR.glob("*.py")
            if path.name not in {"__init__.py", "ct_ops.py"} and not path.name.startswith("_")
        )
        self.assertTrue(vendored_files)

        for path in vendored_files:
            source = path.read_text()
            self.assertIn(
                "from ._adapter import register_impl",
                source,
                msg = f"{path.name} should register through the CuTile adapter.",
            )
            self.assertNotIn(
                "_backend_registry import register_impl",
                source,
                msg = f"{path.name} should not import the central registry directly.",
            )

    def test_glm4_moe_uses_pluggable_grouped_gemm(self):
        source = (
            ROOT / "unsloth" / "unsloth" / "models" / "glm4_moe.py"
        ).read_text()
        self.assertIn(
            "from ..kernels.grouped_gemm import grouped_gemm",
            source,
        )
        self.assertNotIn("sys.path.insert", source)
        self.assertNotIn("from grouped_gemm.interface import grouped_gemm", source)

    def test_backend_packages_register_expected_ops(self):
        checked_backends = []
        for backend_name, expected_ops in (
            ("triton", TRITON_EXPECTED_OPS),
            ("cutile", CUTILE_EXPECTED_OPS),
        ):
            available, reason = is_kernel_backend_available(backend_name)
            if not available:
                continue
            with self.subTest(backend = backend_name):
                ensure_backend_loaded(backend_name)
                registered_ops = set(
                    describe_kernel_backends()["backends"][backend_name]["registered_ops"]
                )
                self.assertTrue(
                    set(expected_ops).issubset(registered_ops),
                    msg = f"{backend_name} missing ops: {sorted(set(expected_ops) - registered_ops)}",
                )
            checked_backends.append(backend_name)

        if not checked_backends:
            self.skipTest("No pluggable kernel backend packages are available.")


if __name__ == "__main__":
    unittest.main()
