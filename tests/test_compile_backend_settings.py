import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for package_root in (ROOT / "unsloth", ROOT / "unsloth-zoo"):
    package_root_str = str(package_root)
    if package_root_str not in sys.path:
        sys.path.insert(0, package_root_str)

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")


class CompileBackendSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from unsloth.models import _utils

        cls._utils = _utils

    def test_accelerate_inductor_kwargs_match_default_compile_shape(self):
        previous_backend = self._utils.UNSLOTH_COMPILE_BACKEND
        try:
            self._utils.UNSLOTH_COMPILE_BACKEND = "inductor"
            compile_kwargs = self._utils.torch_compile_kwargs()
        finally:
            self._utils.UNSLOTH_COMPILE_BACKEND = previous_backend

        self.assertNotIn("backend", compile_kwargs)
        self.assertEqual(compile_kwargs["dynamic"], True)
        self.assertEqual(compile_kwargs["fullgraph"], False)
        self.assertIn("options", compile_kwargs)

    def test_accelerate_non_inductor_kwargs_keep_explicit_backend(self):
        previous_backend = self._utils.UNSLOTH_COMPILE_BACKEND
        try:
            self._utils.UNSLOTH_COMPILE_BACKEND = "aot_eager"
            compile_kwargs = self._utils.torch_compile_kwargs()
        finally:
            self._utils.UNSLOTH_COMPILE_BACKEND = previous_backend

        self.assertEqual(compile_kwargs["backend"], "aot_eager")
        self.assertEqual(compile_kwargs["dynamic"], True)
        self.assertEqual(compile_kwargs["fullgraph"], False)
        self.assertNotIn("options", compile_kwargs)


if __name__ == "__main__":
    unittest.main()
