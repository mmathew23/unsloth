import unittest
from pathlib import Path

from unsloth.kernels import describe_kernel_backends, ensure_backend_loaded, is_kernel_backend_available
from unsloth.kernels.backends.cutile import EXPECTED_OPS as CUTILE_EXPECTED_OPS
from unsloth.kernels.backends.triton import EXPECTED_OPS as TRITON_EXPECTED_OPS

ROOT = Path(__file__).resolve().parents[2]
CUTILE_DIR = ROOT / "unsloth" / "unsloth" / "kernels" / "backends" / "cutile"


class KernelBackendArchitectureTests(unittest.TestCase):
    def test_vendored_cutile_files_use_local_adapter(self):
        vendored_files = sorted(
            path
            for path in CUTILE_DIR.glob("*.py")
            if path.name not in {"__init__.py", "_adapter.py", "ct_ops.py"}
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
        for backend_name, expected_ops in (
            ("triton", TRITON_EXPECTED_OPS),
            ("cutile", CUTILE_EXPECTED_OPS),
        ):
            available, reason = is_kernel_backend_available(backend_name)
            if not available:
                self.skipTest(reason or f"{backend_name} is unavailable.")
            ensure_backend_loaded(backend_name)
            registered_ops = set(
                describe_kernel_backends()["backends"][backend_name]["registered_ops"]
            )
            self.assertTrue(
                set(expected_ops).issubset(registered_ops),
                msg = f"{backend_name} missing ops: {sorted(set(expected_ops) - registered_ops)}",
            )


if __name__ == "__main__":
    unittest.main()
