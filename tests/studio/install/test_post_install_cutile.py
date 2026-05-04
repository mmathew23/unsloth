from pathlib import Path


def _script_source() -> str:
    return (
        Path(__file__).resolve().parents[3]
        / "studio"
        / "post_install_cutile.sh"
    ).read_text()


def test_cutile_only_launcher_defaults_to_dynamo_disabled():
    source = _script_source()

    assert 'COMPILE_BACKEND="dynamo_disable"' in source
    assert 'export UNSLOTH_TORCH_COMPILE_BACKEND=%s\\n' in source
    assert "export TORCHDYNAMO_DISABLE=1" in source
    assert "UNSLOTH_TORCH_COMPILE_BACKEND=aot_eager" in source


def test_hybrid_launcher_keeps_inductor_and_unsets_dynamo_disable():
    source = _script_source()

    assert 'COMPILE_BACKEND="inductor"' in source
    assert "unset TORCHDYNAMO_DISABLE" in source
    assert "--mode cutile-only|hybrid" in source
