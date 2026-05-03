import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CUTILE_BACKENDS = ROOT / "unsloth" / "kernels" / "backends" / "cutile"


def _assignment_name_sets(path: Path, target_name: str) -> list[set[str]]:
    tree = ast.parse(path.read_text(), filename = str(path))
    name_sets = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == target_name for target in node.targets):
            continue
        name_sets.append(
            {name.id for name in ast.walk(node.value) if isinstance(name, ast.Name)}
        )
    if not name_sets:
        raise AssertionError(f"Could not find assignment to {target_name!r} in {path}")
    return name_sets


def _single_assignment_name_set(path: Path, target_name: str) -> set[str]:
    name_sets = _assignment_name_sets(path, target_name)
    if len(name_sets) != 1:
        raise AssertionError(
            f"Expected one assignment to {target_name!r} in {path}, found {len(name_sets)}"
        )
    return name_sets[0]


class CutileDispatcherCacheKeyTests(unittest.TestCase):
    def test_norm_cache_keys_exclude_launch_grid_rows(self):
        layernorm_path = CUTILE_BACKENDS / "layernorm.py"
        rms_path = CUTILE_BACKENDS / "rms_layernorm.py"

        for path, target_name in (
            (layernorm_path, "fwd_cache_key"),
            (layernorm_path, "bwd_cache_key"),
            (rms_path, "fwd_cache_key"),
            (rms_path, "bwd_cache_key"),
        ):
            with self.subTest(path = path.name, target_name = target_name):
                names = _single_assignment_name_set(path, target_name)
                self.assertNotIn("n_rows", names)
                self.assertIn("n_cols", names)

    def test_rope_cache_keys_exclude_batch_derived_rows(self):
        rope_path = CUTILE_BACKENDS / "rope_embedding.py"

        for target_name in ("single_cache_key", "qk_cache_key"):
            with self.subTest(target_name = target_name):
                names = _single_assignment_name_set(rope_path, target_name)
                self.assertNotIn("n_rows", names)
                self.assertNotIn("seq_len", names)
                self.assertIn("head_dim", names)
                self.assertIn("TILE_HD", names)
                self.assertIn("cos_row_stride", names)
                self.assertIn("no_padding", names)

    def test_rope_seqlen_is_runtime_scalar_not_compile_constant(self):
        rope_path = CUTILE_BACKENDS / "rope_embedding.py"
        tree = ast.parse(rope_path.read_text(), filename = str(rope_path))
        checked = 0

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name not in {"_rope_embedding_ct", "_rope_embedding_QK_ct"}:
                continue
            for arg in node.args.args:
                if arg.arg != "seqlen":
                    continue
                checked += 1
                self.assertIsInstance(arg.annotation, ast.Name)
                self.assertEqual(arg.annotation.id, "int")

        self.assertEqual(checked, 2)

    def test_element_count_constants_stay_in_elementwise_cache_keys(self):
        swiglu_path = CUTILE_BACKENDS / "swiglu.py"
        geglu_path = CUTILE_BACKENDS / "geglu.py"

        for path in (swiglu_path, geglu_path):
            for names in _assignment_name_sets(path, "cache_key"):
                with self.subTest(path = path.name, names = sorted(names)):
                    self.assertIn("n_elements", names)
                    self.assertIn("LONG_INDEXING", names)

    def test_rms_forward_casts_output_to_input_dtype_before_scatter(self):
        rms_path = CUTILE_BACKENDS / "rms_layernorm.py"
        tree = ast.parse(rms_path.read_text(), filename = str(rms_path))

        casts_to_input_dtype = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "output" for target in node.targets):
                continue
            value = node.value
            if not isinstance(value, ast.Call):
                continue
            if not (
                isinstance(value.func, ast.Attribute)
                and value.func.attr == "astype"
                and isinstance(value.func.value, ast.Name)
                and value.func.value.id == "ct"
            ):
                continue
            if len(value.args) != 2:
                continue
            input_arg, dtype_arg = value.args
            if not (isinstance(input_arg, ast.Name) and input_arg.id == "output"):
                continue
            if not (
                isinstance(dtype_arg, ast.Attribute)
                and dtype_arg.attr == "dtype"
                and isinstance(dtype_arg.value, ast.Name)
                and dtype_arg.value.id == "X"
            ):
                continue
            casts_to_input_dtype += 1

        self.assertGreaterEqual(casts_to_input_dtype, 1)


if __name__ == "__main__":
    unittest.main()
