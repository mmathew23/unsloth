from pathlib import Path
import subprocess


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


def test_fallback_defaults_to_upstream_main():
    source = _script_source()

    assert 'BRANCH="${UNSLOTH_CUTILE_BRANCH:-main}"' in source
    assert (
        'UNSLOTH_REPO_URL="${UNSLOTH_CUTILE_UNSLOTH_REPO_URL:-https://github.com/unslothai/unsloth.git}"'
        in source
    )
    assert (
        'ZOO_REPO_URL="${UNSLOTH_CUTILE_ZOO_REPO_URL:-https://github.com/unslothai/unsloth-zoo.git}"'
        in source
    )
    assert "--branch REF" in source


def test_fallback_git_urls_are_configurable_for_review_branches(tmp_path):
    script = (
        Path(__file__).resolve().parents[3]
        / "studio"
        / "post_install_cutile.sh"
    )
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dry-run",
            "--unsloth-root",
            str(tmp_path / "missing-unsloth"),
            "--zoo-root",
            str(tmp_path / "missing-zoo"),
            "--branch",
            "feat/cutile",
            "--unsloth-repo-url",
            "https://github.com/mmathew23/unsloth.git",
            "--zoo-repo-url",
            "https://github.com/mmathew23/unsloth-zoo.git",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "git+https://github.com/mmathew23/unsloth.git@feat/cutile" in result.stdout
    assert "git+https://github.com/mmathew23/unsloth-zoo.git@feat/cutile" in result.stdout


def test_requested_cutile_spec_is_installed_after_zoo_dependencies():
    source = _script_source()

    zoo_install = source.index('Installing local unsloth-zoo[$ZOO_EXTRA]')
    cutile_spec_install = source.index('Installing requested cuda-tile spec')
    unsloth_install = source.index('Installing local unsloth[$UNSLOTH_EXTRA]')

    assert zoo_install < cutile_spec_install < unsloth_install
