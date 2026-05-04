# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_import_fixes_module():
    repo_root = Path(__file__).resolve().parents[2]
    import_fixes_path = repo_root / "unsloth" / "import_fixes.py"
    spec = importlib.util.spec_from_file_location(
        "unsloth_import_fixes_fp8_local", import_fixes_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_triton_finegrained_fp8_accepts_static_activation_scheme(monkeypatch):
    import_fixes = _load_import_fixes_module()

    class FakeFineGrainedFP8HfQuantizer:
        pass

    quantizer_module = types.ModuleType(
        "transformers.quantizers.quantizer_finegrained_fp8"
    )
    quantizer_module.FineGrainedFP8HfQuantizer = FakeFineGrainedFP8HfQuantizer

    quantizer_utils = types.ModuleType("transformers.quantizers.quantizers_utils")
    quantizer_utils.should_convert_module = lambda name, modules_to_not_convert: True

    monkeypatch.setitem(
        sys.modules,
        "transformers.quantizers.quantizer_finegrained_fp8",
        quantizer_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.quantizers.quantizers_utils",
        quantizer_utils,
    )
    monkeypatch.setattr(import_fixes, "_is_triton_importable", lambda: False)

    import_fixes.patch_finegrained_fp8_without_triton()

    model = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False))
    model._keep_in_fp32_modules = []
    quantizer = FakeFineGrainedFP8HfQuantizer()
    quantizer.quantization_config = types.SimpleNamespace(
        dequantize = False,
        modules_to_not_convert = None,
        weight_block_size = None,
        activation_scheme = "static",
    )
    quantizer.pre_quantized = False
    quantizer.get_modules_to_not_convert = lambda model, modules, keep: []

    patched = quantizer._process_model_before_weight_loading(model)

    assert patched is model
    assert model[0].activation_scheme == "static"
    assert model[0].activation_scale is not None
