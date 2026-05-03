#!/usr/bin/env bash
# Post-install CuTile setup for Unsloth Studio.
#
# This script patches an existing Studio virtual environment after the normal
# Studio installer has created it. Default mode is strict cutile-only:
#   - install CuTile-capable Unsloth / Unsloth Zoo packages
#   - remove Triton from the Studio venv
#   - write a launcher that sets the required backend environment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNSLOTH_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$UNSLOTH_REPO_ROOT/.." && pwd)"

MODE="cutile-only"
STUDIO_VENV="${UNSLOTH_STUDIO_VENV:-$HOME/.unsloth/studio/unsloth_studio}"
UNSLOTH_ROOT="$UNSLOTH_REPO_ROOT"
ZOO_ROOT="${UNSLOTH_ZOO_ROOT:-}"
BRANCH="feature/pluggable-kernel-backend-v2"
CUTILE_SPEC=""
LAUNCHER_PATH=""
STRICT="auto"
DRY_RUN=0
VERBOSE=0
ALLOW_TRITON=0

usage() {
    cat <<'EOF'
Usage:
  post_install_cutile.sh [options]

Default:
  Configure an existing Unsloth Studio venv for strict cutile-only execution.

Options:
  --mode cutile-only|hybrid
      cutile-only: install CuTile support, uninstall Triton, launch with
                   UNSLOTH_KERNEL_BACKEND=cutile and
                   UNSLOTH_TORCH_COMPILE_BACKEND=aot_eager.
      hybrid:      install CuTile + Triton support, keep Triton, launch with
                   UNSLOTH_KERNEL_BACKEND=cutile and
                   UNSLOTH_TORCH_COMPILE_BACKEND=inductor.

  --studio-venv PATH
      Studio venv path. Default: ~/.unsloth/studio/unsloth_studio

  --unsloth-root PATH
      Local Unsloth repo root. If it contains pyproject.toml, the script uses
      an editable install. Default: parent of this script's studio/ directory.

  --zoo-root PATH
      Local unsloth-zoo repo root. If omitted, the script tries sibling
      directories named unsloth-zoo and unsloth_zoo. If no local repo is found,
      it installs from GitHub using --branch.

  --branch REF
      Git ref used when local source trees are not available.
      Default: feature/pluggable-kernel-backend-v2

  --cutile-spec SPEC
      Optional cuda-tile package spec to install first. Example:
      "cuda-tile[tileiras] @ git+https://github.com/nvidia/cutile-python.git@my-branch"

  --launcher PATH
      Launcher path to write. Defaults to:
        ~/.local/bin/unsloth-studio-cutile
      or:
        ~/.local/bin/unsloth-studio-cutile-hybrid

  --strict / --no-strict
      Whether launcher sets UNSLOTH_KERNEL_BACKEND_STRICT=1.
      Default: strict for cutile-only, not strict for hybrid.

  --allow-triton
      Do not fail verification if Triton remains importable in cutile-only mode.
      Useful only for debugging partial environments.

  --dry-run
      Print commands without executing.

  --verbose
      Print pip output directly.

Examples:
  ./studio/post_install_cutile.sh
  ./studio/post_install_cutile.sh --mode hybrid
  ./studio/post_install_cutile.sh --cutile-spec \
    "cuda-tile[tileiras] @ git+https://github.com/nvidia/cutile-python.git@review-branch"
EOF
}

log() {
    printf '%s\n' "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

run_cmd() {
    if [ "$DRY_RUN" = "1" ]; then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi

    if [ "$VERBOSE" = "1" ]; then
        "$@"
        return
    fi

    local tmp
    tmp="$(mktemp)"
    if "$@" >"$tmp" 2>&1; then
        rm -f "$tmp"
    else
        local rc=$?
        cat "$tmp" >&2
        rm -f "$tmp"
        return "$rc"
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --mode)
            [ "$#" -ge 2 ] || die "--mode requires an argument"
            MODE="$2"
            shift 2
            ;;
        --studio-venv)
            [ "$#" -ge 2 ] || die "--studio-venv requires an argument"
            STUDIO_VENV="$2"
            shift 2
            ;;
        --unsloth-root)
            [ "$#" -ge 2 ] || die "--unsloth-root requires an argument"
            UNSLOTH_ROOT="$2"
            shift 2
            ;;
        --zoo-root)
            [ "$#" -ge 2 ] || die "--zoo-root requires an argument"
            ZOO_ROOT="$2"
            shift 2
            ;;
        --branch)
            [ "$#" -ge 2 ] || die "--branch requires an argument"
            BRANCH="$2"
            shift 2
            ;;
        --cutile-spec)
            [ "$#" -ge 2 ] || die "--cutile-spec requires an argument"
            CUTILE_SPEC="$2"
            shift 2
            ;;
        --launcher)
            [ "$#" -ge 2 ] || die "--launcher requires an argument"
            LAUNCHER_PATH="$2"
            shift 2
            ;;
        --strict)
            STRICT=1
            shift
            ;;
        --no-strict)
            STRICT=0
            shift
            ;;
        --allow-triton)
            ALLOW_TRITON=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --verbose|-v)
            VERBOSE=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

case "$MODE" in
    cutile-only|hybrid) ;;
    *) die "--mode must be cutile-only or hybrid" ;;
esac

if [ "$STRICT" = "auto" ]; then
    if [ "$MODE" = "cutile-only" ]; then
        STRICT=1
    else
        STRICT=0
    fi
fi

if [ -z "$LAUNCHER_PATH" ]; then
    if [ "$MODE" = "cutile-only" ]; then
        LAUNCHER_PATH="$HOME/.local/bin/unsloth-studio-cutile"
    else
        LAUNCHER_PATH="$HOME/.local/bin/unsloth-studio-cutile-hybrid"
    fi
fi

if [ -z "$ZOO_ROOT" ]; then
    if [ -f "$WORKSPACE_ROOT/unsloth-zoo/pyproject.toml" ]; then
        ZOO_ROOT="$WORKSPACE_ROOT/unsloth-zoo"
    elif [ -f "$WORKSPACE_ROOT/unsloth_zoo/pyproject.toml" ]; then
        ZOO_ROOT="$WORKSPACE_ROOT/unsloth_zoo"
    fi
fi

STUDIO_PY="$STUDIO_VENV/bin/python"
STUDIO_UNSLOTH="$STUDIO_VENV/bin/unsloth"

[ "$(uname -s)" = "Linux" ] || die "CuTile package support is currently Linux-only"
if [ "$DRY_RUN" = "0" ]; then
    [ -d "$STUDIO_VENV" ] || die "Studio venv not found at $STUDIO_VENV. Run ./install.sh --local or the Studio installer first."
    [ -x "$STUDIO_PY" ] || die "Python not found at $STUDIO_PY"
    [ -x "$STUDIO_UNSLOTH" ] || die "Unsloth CLI not found at $STUDIO_UNSLOTH"
fi

log "Configuring Unsloth Studio for mode: $MODE"
log "Studio venv: $STUDIO_VENV"
log "Unsloth root: $UNSLOTH_ROOT"
if [ -n "$ZOO_ROOT" ]; then
    log "Unsloth Zoo root: $ZOO_ROOT"
else
    log "Unsloth Zoo root: none found; using GitHub branch $BRANCH"
fi

if [ "$DRY_RUN" = "0" ] && ! "$STUDIO_PY" -m pip --version >/dev/null 2>&1; then
    log "Bootstrapping pip in Studio venv"
    run_cmd "$STUDIO_PY" -m ensurepip --upgrade
fi

PIP=( "$STUDIO_PY" -m pip )

if [ -n "$CUTILE_SPEC" ]; then
    log "Installing requested cuda-tile spec"
    run_cmd "${PIP[@]}" install --force-reinstall "$CUTILE_SPEC"
fi

if [ "$MODE" = "hybrid" ]; then
    ZOO_EXTRA="cutile,triton"
    UNSLOTH_EXTRA="cutile,triton"
else
    ZOO_EXTRA="cutile"
    UNSLOTH_EXTRA="cutile"
fi

if [ -n "$ZOO_ROOT" ] && [ -f "$ZOO_ROOT/pyproject.toml" ]; then
    log "Installing local unsloth-zoo[$ZOO_EXTRA]"
    run_cmd "${PIP[@]}" install --force-reinstall -e "$ZOO_ROOT[$ZOO_EXTRA]"
else
    log "Installing unsloth-zoo[$ZOO_EXTRA] from GitHub branch $BRANCH"
    run_cmd "${PIP[@]}" install --force-reinstall \
        "unsloth_zoo[$ZOO_EXTRA] @ git+https://github.com/unslothai/unsloth-zoo.git@$BRANCH"
fi

if [ -f "$UNSLOTH_ROOT/pyproject.toml" ]; then
    log "Installing local unsloth[$UNSLOTH_EXTRA] without dependency replacement"
    run_cmd "${PIP[@]}" install --force-reinstall --no-deps -e "$UNSLOTH_ROOT[$UNSLOTH_EXTRA]"
else
    log "Installing unsloth[$UNSLOTH_EXTRA] from GitHub branch $BRANCH"
    run_cmd "${PIP[@]}" install --force-reinstall --no-deps \
        "unsloth[$UNSLOTH_EXTRA] @ git+https://github.com/unslothai/unsloth.git@$BRANCH"
fi

if [ "$MODE" = "cutile-only" ]; then
    log "Removing Triton packages for strict cutile-only mode"
    run_cmd "${PIP[@]}" uninstall -y triton triton-windows || true
fi

log "Writing launcher: $LAUNCHER_PATH"
if [ "$DRY_RUN" = "0" ]; then
    mkdir -p "$(dirname "$LAUNCHER_PATH")"
    if [ "$MODE" = "cutile-only" ]; then
        COMPILE_BACKEND="aot_eager"
    else
        COMPILE_BACKEND="inductor"
    fi
    {
        printf '%s\n' '#!/usr/bin/env sh'
        printf '%s\n' '# Generated by Unsloth Studio post_install_cutile.sh'
        printf '%s\n' 'export UNSLOTH_KERNEL_BACKEND=cutile'
        printf 'export UNSLOTH_TORCH_COMPILE_BACKEND=%s\n' "$COMPILE_BACKEND"
        if [ "$STRICT" = "1" ]; then
            printf '%s\n' 'export UNSLOTH_KERNEL_BACKEND_STRICT=1'
        else
            printf '%s\n' 'unset UNSLOTH_KERNEL_BACKEND_STRICT'
        fi
        printf '%s\n' 'unset UNSLOTH_COMPILE_DISABLE'
        printf '%s\n' 'unset TORCHDYNAMO_DISABLE'
        printf 'exec %s studio "$@"\n' "$(printf '%s' "$STUDIO_UNSLOTH" | sed "s/'/'\\\\''/g; s/.*/'&'/")"
    } > "$LAUNCHER_PATH"
    chmod +x "$LAUNCHER_PATH"
fi

if [ "$DRY_RUN" = "1" ]; then
    log "Skipping verification in dry-run mode"
else
    log "Verifying Studio environment"
    VERIFY_MODE="$MODE" VERIFY_ALLOW_TRITON="$ALLOW_TRITON" "$STUDIO_PY" - <<'PY'
import importlib.util
import os
import sys

mode = os.environ["VERIFY_MODE"]
allow_triton = os.environ.get("VERIFY_ALLOW_TRITON") == "1"
has_cuda_tile = importlib.util.find_spec("cuda.tile") is not None
has_cuda_tile_tune = importlib.util.find_spec("cuda.tile.tune") is not None
has_triton = importlib.util.find_spec("triton") is not None

print("cuda.tile importable:", has_cuda_tile)
print("cuda.tile.tune importable:", has_cuda_tile_tune)
print("triton importable:", has_triton)

if not has_cuda_tile or not has_cuda_tile_tune:
    raise SystemExit("cuda.tile and cuda.tile.tune must be importable")
if mode == "cutile-only" and has_triton and not allow_triton:
    raise SystemExit("triton is still importable in cutile-only mode")
if mode == "hybrid" and not has_triton:
    raise SystemExit("triton must be importable in hybrid mode")
PY
fi

log "Done."
log "Launch with:"
log "  $LAUNCHER_PATH -H 0.0.0.0 -p 8888"
