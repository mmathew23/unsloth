import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UNSLOTH_PYPROJECT = ROOT / "unsloth" / "pyproject.toml"
UNSLOTH_ZOO_PYPROJECT = ROOT / "unsloth-zoo" / "pyproject.toml"


def _load_optional_dependencies(pyproject_path: Path) -> dict[str, list[str]]:
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    return data["project"]["optional-dependencies"]


def _load_dependencies(pyproject_path: Path) -> list[str]:
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    return data["project"]["dependencies"]


class CutilePackagingTests(unittest.TestCase):
    def test_unsloth_cutile_extra_is_available_without_triton(self):
        optional = _load_optional_dependencies(UNSLOTH_PYPROJECT)
        self.assertIn("cutile", optional)
        cutile_deps = optional["cutile"]
        self.assertIn("unsloth[huggingfacenotorch]", cutile_deps)
        self.assertIn("torchvision", cutile_deps)
        self.assertTrue(
            any(dep.startswith("unsloth_zoo[cutile]") for dep in cutile_deps)
        )
        self.assertFalse(any("unsloth[triton]" in dep for dep in cutile_deps))
        self.assertFalse(any(dep.startswith("triton") for dep in cutile_deps))

    def test_unsloth_zoo_base_dependencies_do_not_require_triton(self):
        dependencies = _load_dependencies(UNSLOTH_ZOO_PYPROJECT)
        self.assertFalse(any(dep.startswith("triton") for dep in dependencies))

    def test_unsloth_zoo_cutile_extra_carries_cutile_stack(self):
        optional = _load_optional_dependencies(UNSLOTH_ZOO_PYPROJECT)
        self.assertIn("cutile", optional)
        cutile_deps = optional["cutile"]
        self.assertTrue(any(dep.startswith("cuda-tile") for dep in cutile_deps))
        self.assertTrue(any(dep.startswith("cut_cross_entropy") for dep in cutile_deps))

    def test_unsloth_zoo_huggingface_extra_keeps_triton_path(self):
        optional = _load_optional_dependencies(UNSLOTH_ZOO_PYPROJECT)
        self.assertIn("huggingface", optional)
        self.assertTrue(
            any("unsloth_zoo[core,triton]" == dep for dep in optional["huggingface"])
        )


if __name__ == "__main__":
    unittest.main()
