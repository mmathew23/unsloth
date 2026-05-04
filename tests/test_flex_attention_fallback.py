import types
import unittest
from unittest.mock import patch


class FlexAttentionFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from unsloth.models import _utils

        cls._utils = _utils

    def test_requested_flex_falls_back_to_sdpa_without_inductor(self):
        class Model:
            _supports_sdpa = True
            _supports_flex_attn = True

        config = types.SimpleNamespace(
            model_type = "gemma3",
            _attn_implementation = "flex_attention",
            attention_dropout = 0,
        )
        previous_backend = self._utils.UNSLOTH_COMPILE_BACKEND
        try:
            self._utils.UNSLOTH_COMPILE_BACKEND = "aot_eager"
            resolved = self._utils.resolve_attention_implementation(
                Model,
                config,
                requested_attn_implementation = "flex_attention",
                supports_sdpa = True,
            )
        finally:
            self._utils.UNSLOTH_COMPILE_BACKEND = previous_backend

        self.assertEqual(resolved, "sdpa")
        self.assertEqual(config._attn_implementation, "sdpa")

    def test_requested_flex_falls_back_to_eager_without_sdpa(self):
        class Model:
            _supports_sdpa = False
            _supports_flex_attn = True

        config = types.SimpleNamespace(
            model_type = "synthetic",
            _attn_implementation = "flex_attention",
            attention_dropout = 0,
        )
        previous_backend = self._utils.UNSLOTH_COMPILE_BACKEND
        try:
            self._utils.UNSLOTH_COMPILE_BACKEND = "aot_eager"
            resolved = self._utils.resolve_attention_implementation(
                Model,
                config,
                requested_attn_implementation = "flex_attention",
                supports_sdpa = False,
            )
        finally:
            self._utils.UNSLOTH_COMPILE_BACKEND = previous_backend

        self.assertEqual(resolved, "eager")
        self.assertEqual(config._attn_implementation, "eager")

    def test_requested_flex_is_preserved_when_available(self):
        class Model:
            _supports_sdpa = True
            _supports_flex_attn = True

        config = types.SimpleNamespace(
            model_type = "gemma3",
            _attn_implementation = "flex_attention",
            attention_dropout = 0,
        )
        previous_backend = self._utils.UNSLOTH_COMPILE_BACKEND
        try:
            self._utils.UNSLOTH_COMPILE_BACKEND = "inductor"
            with patch(
                "transformers.utils.import_utils.is_torch_flex_attn_available",
                return_value = True,
            ):
                resolved = self._utils.resolve_attention_implementation(
                    Model,
                    config,
                    requested_attn_implementation = "flex_attention",
                    supports_sdpa = True,
                )
        finally:
            self._utils.UNSLOTH_COMPILE_BACKEND = previous_backend

        self.assertEqual(resolved, "flex_attention")
        self.assertEqual(config._attn_implementation, "flex_attention")


if __name__ == "__main__":
    unittest.main()
