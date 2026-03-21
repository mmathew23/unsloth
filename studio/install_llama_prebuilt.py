#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import platform
import shutil
import site
import socket
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


EXIT_SUCCESS = 0
EXIT_FALLBACK = 2
EXIT_ERROR = 1

DEFAULT_LLAMA_TAG = os.environ.get("UNSLOTH_LLAMA_TAG")
DEFAULT_PUBLISHED_REPO = os.environ.get("UNSLOTH_LLAMA_RELEASE_REPO", "mmathew23/llama.cpp-prebuilt")
DEFAULT_PUBLISHED_TAG = os.environ.get("UNSLOTH_LLAMA_RELEASE_TAG")
DEFAULT_PUBLISHED_MANIFEST_ASSET = os.environ.get(
    "UNSLOTH_LLAMA_RELEASE_MANIFEST_ASSET", "llama-prebuilt-manifest.json"
)
UPSTREAM_REPO = "ggml-org/llama.cpp"
UPSTREAM_RELEASES_API = f"https://api.github.com/repos/{UPSTREAM_REPO}/releases/latest"
TEST_MODEL_URL = "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf"
LINUX_SHARED_BUNDLE_HOST_LIBRARIES = (
    "libssl.so.3",
    "libcrypto.so.3",
    "libstdc++.so.6",
    "libgcc_s.so.1",
    "libgomp.so.1",
)


@dataclass
class HostInfo:
    system: str
    machine: str
    is_windows: bool
    is_linux: bool
    is_macos: bool
    is_x86_64: bool
    is_arm64: bool
    nvidia_smi: str | None
    driver_cuda_version: tuple[int, int] | None
    compute_caps: list[str]
    has_nvidia: bool


@dataclass
class AssetChoice:
    repo: str
    tag: str
    name: str
    url: str
    source_label: str
    runtime_name: str | None = None
    runtime_url: str | None = None
    is_ready_bundle: bool = False
    install_kind: str = ""
    bundle_profile: str | None = None
    runtime_line: str | None = None
    coverage_class: str | None = None
    supported_sms: list[str] | None = None
    min_sm: int | None = None
    max_sm: int | None = None
    selection_log: list[str] | None = None


@dataclass(frozen=True)
class PublishedLlamaArtifact:
    asset_name: str
    install_kind: str
    runtime_line: str | None
    coverage_class: str | None
    supported_sms: list[str]
    min_sm: int | None
    max_sm: int | None
    bundle_profile: str | None
    rank: int


@dataclass
class PublishedReleaseBundle:
    repo: str
    release_tag: str
    upstream_tag: str
    assets: dict[str, str]
    manifest_asset_name: str
    artifacts: list[PublishedLlamaArtifact]
    selection_log: list[str]


@dataclass
class LinuxCudaSelection:
    attempts: list[AssetChoice]
    selection_log: list[str]

    @property
    def primary(self) -> AssetChoice:
        if not self.attempts:
            raise RuntimeError("linux CUDA selection unexpectedly had no attempts")
        return self.attempts[0]


class PrebuiltFallback(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"[llama-prebuilt] {message}")


def log_lines(lines: Iterable[str]) -> None:
    for line in lines:
        log(line)


def auth_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "unsloth-studio-llama-prebuilt",
    }
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=auth_headers())
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=auth_headers())
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def github_release_assets(repo: str, tag: str) -> dict[str, str]:
    payload = fetch_json(f"https://api.github.com/repos/{repo}/releases/tags/{urllib.parse.quote(tag)}")
    return {asset["name"]: asset["browser_download_url"] for asset in payload.get("assets", [])}


def github_release(repo: str, tag: str) -> dict[str, Any]:
    payload = fetch_json(f"https://api.github.com/repos/{repo}/releases/tags/{urllib.parse.quote(tag)}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected release payload for {repo}@{tag}")
    return payload


def github_releases(repo: str, *, per_page: int = 30) -> list[dict[str, Any]]:
    payload = fetch_json(f"https://api.github.com/repos/{repo}/releases?per_page={per_page}")
    if not isinstance(payload, list):
        raise RuntimeError(f"unexpected releases payload for {repo}")
    return [item for item in payload if isinstance(item, dict)]


def latest_upstream_release_tag() -> str:
    payload = fetch_json(UPSTREAM_RELEASES_API)
    tag = payload.get("tag_name")
    if not isinstance(tag, str) or not tag:
        raise RuntimeError(f"latest release tag was missing from {UPSTREAM_RELEASES_API}")
    return tag


def normalize_compute_caps(compute_caps: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in compute_caps:
        value = str(raw).strip()
        if not value:
            continue
        try:
            normalized_value = str(int(value))
        except ValueError:
            continue
        if normalized_value in seen:
            continue
        seen.add(normalized_value)
        normalized.append(normalized_value)
    normalized.sort(key=int)
    return normalized


def linux_runtime_dirs_for_required_libraries(required_libraries: Iterable[str]) -> list[str]:
    required = [library for library in required_libraries if library]
    candidates: list[str | Path] = []

    env_dirs = os.environ.get("CUDA_RUNTIME_LIB_DIR", "")
    if env_dirs:
        candidates.extend(part for part in env_dirs.split(os.pathsep) if part)
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if ld_library_path:
        candidates.extend(part for part in ld_library_path.split(os.pathsep) if part)

    cuda_roots: list[Path] = []
    for name in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
        value = os.environ.get(name)
        if value:
            cuda_roots.append(Path(value))
    cuda_roots.extend(Path(path) for path in glob_paths("/usr/local/cuda", "/usr/local/cuda-*"))

    for root in cuda_roots:
        candidates.extend(
            [
                root / "lib",
                root / "lib64",
                root / "targets" / "x86_64-linux" / "lib",
            ]
        )

    candidates.extend(
        Path(path)
        for path in glob_paths(
            "/lib",
            "/lib64",
            "/usr/lib",
            "/usr/lib64",
            "/usr/local/lib",
            "/usr/local/lib64",
            "/lib/x86_64-linux-gnu",
            "/usr/lib/x86_64-linux-gnu",
        )
    )
    candidates.extend(Path(path) for path in glob_paths("/usr/local/lib/ollama/cuda_v*", "/usr/lib/wsl/lib"))
    candidates.extend(Path(path) for path in python_runtime_dirs())
    candidates.extend(Path(path) for path in ldconfig_runtime_dirs(required))

    resolved = dedupe_existing_dirs(candidates)
    if not required:
        return resolved

    matched: list[tuple[int, str]] = []
    remainder: list[str] = []
    for directory in resolved:
        base = Path(directory)
        provided = 0
        for library in required:
            if any(base.glob(f"{library}*")):
                provided += 1
        if provided:
            matched.append((provided, directory))
        else:
            remainder.append(directory)

    matched.sort(key=lambda item: item[0], reverse=True)
    if matched:
        return [directory for _, directory in matched]
    return remainder


def detected_linux_runtime_lines() -> tuple[list[str], dict[str, list[str]]]:
    line_requirements = {
        "cuda13": ["libcudart.so.13", "libcublas.so.13"],
        "cuda12": ["libcudart.so.12", "libcublas.so.12"],
    }
    detected: list[str] = []
    runtime_dirs: dict[str, list[str]] = {}
    for line, required in line_requirements.items():
        dirs = linux_runtime_dirs_for_required_libraries(required)
        matching_dirs = [
            directory
            for directory in dirs
            if all(any(Path(directory).glob(f"{library}*")) for library in required)
        ]
        if matching_dirs:
            detected.append(line)
            runtime_dirs[line] = matching_dirs
    return detected, runtime_dirs


def release_asset_map(release: dict[str, Any]) -> dict[str, str]:
    assets = release.get("assets")
    if not isinstance(assets, list):
        return {}
    return {
        asset["name"]: asset.get("browser_download_url", "")
        for asset in assets
        if isinstance(asset, dict)
        and isinstance(asset.get("name"), str)
        and isinstance(asset.get("browser_download_url"), str)
    }


def parse_published_artifact(raw: Any) -> PublishedLlamaArtifact | None:
    if not isinstance(raw, dict):
        return None
    asset_name = raw.get("asset_name")
    install_kind = raw.get("install_kind")
    if not isinstance(asset_name, str) or not asset_name:
        return None
    if not isinstance(install_kind, str) or not install_kind:
        return None
    supported_sms = normalize_compute_caps(raw.get("supported_sms", []))
    min_sm_raw = raw.get("min_sm")
    max_sm_raw = raw.get("max_sm")
    min_sm = int(min_sm_raw) if min_sm_raw is not None else None
    max_sm = int(max_sm_raw) if max_sm_raw is not None else None
    runtime_line = raw.get("runtime_line")
    coverage_class = raw.get("coverage_class")
    bundle_profile = raw.get("bundle_profile")
    rank_raw = raw.get("rank", 1000)
    try:
        rank = int(rank_raw)
    except (TypeError, ValueError):
        rank = 1000
    return PublishedLlamaArtifact(
        asset_name=asset_name,
        install_kind=install_kind,
        runtime_line=runtime_line if isinstance(runtime_line, str) and runtime_line else None,
        coverage_class=coverage_class if isinstance(coverage_class, str) and coverage_class else None,
        supported_sms=supported_sms,
        min_sm=min_sm,
        max_sm=max_sm,
        bundle_profile=bundle_profile if isinstance(bundle_profile, str) and bundle_profile else None,
        rank=rank,
    )


def parse_published_release_bundle(repo: str, release: dict[str, Any]) -> PublishedReleaseBundle | None:
    release_tag = release.get("tag_name")
    if not isinstance(release_tag, str) or not release_tag:
        return None

    assets = release_asset_map(release)
    manifest_url = assets.get(DEFAULT_PUBLISHED_MANIFEST_ASSET)
    if not manifest_url:
        return None

    # Mixed repos are filtered by an explicit release-side manifest rather than
    # by release tag or asset filename conventions.
    manifest_payload = fetch_json(manifest_url)
    if not isinstance(manifest_payload, dict):
        raise RuntimeError(f"published manifest {DEFAULT_PUBLISHED_MANIFEST_ASSET} was not a JSON object")
    component = manifest_payload.get("component")
    upstream_tag = manifest_payload.get("upstream_tag")
    if component != "llama.cpp":
        return None
    if not isinstance(upstream_tag, str) or not upstream_tag:
        raise RuntimeError(
            f"published manifest {DEFAULT_PUBLISHED_MANIFEST_ASSET} in {repo}@{release_tag} omitted upstream_tag"
        )

    artifacts_payload = manifest_payload.get("artifacts")
    if not isinstance(artifacts_payload, list):
        raise RuntimeError(
            f"published manifest {DEFAULT_PUBLISHED_MANIFEST_ASSET} in {repo}@{release_tag} omitted artifacts"
        )

    artifacts = [artifact for raw in artifacts_payload if (artifact := parse_published_artifact(raw))]
    selection_log = [
        f"published_release: repo={repo}",
        f"published_release: tag={release_tag}",
        f"published_release: manifest={DEFAULT_PUBLISHED_MANIFEST_ASSET}",
        f"published_release: upstream_tag={upstream_tag}",
    ]
    return PublishedReleaseBundle(
        repo=repo,
        release_tag=release_tag,
        upstream_tag=upstream_tag,
        assets=assets,
        manifest_asset_name=DEFAULT_PUBLISHED_MANIFEST_ASSET,
        artifacts=artifacts,
        selection_log=selection_log,
    )


def iter_published_release_bundles(repo: str, published_release_tag: str = "") -> Iterable[PublishedReleaseBundle]:
    releases = [github_release(repo, published_release_tag)] if published_release_tag else github_releases(repo)
    for release in releases:
        if not published_release_tag and (release.get("draft") or release.get("prerelease")):
            continue
        try:
            bundle = parse_published_release_bundle(repo, release)
        except Exception as exc:
            release_tag = release.get("tag_name", "unknown")
            log(f"published release metadata ignored for {repo}@{release_tag}: {exc}")
            continue
        if bundle is None:
            continue
        yield bundle


def linux_cuda_choice_from_release(
    host: HostInfo,
    release: PublishedReleaseBundle,
) -> LinuxCudaSelection | None:
    host_sms = normalize_compute_caps(host.compute_caps)
    runtime_lines, runtime_dirs = detected_linux_runtime_lines()
    selection_log = list(release.selection_log) + [
        f"linux_cuda_selection: release={release.release_tag}",
        f"linux_cuda_selection: detected_sms={','.join(host_sms) if host_sms else 'unknown'}",
        "linux_cuda_selection: runtime_lines="
        + (",".join(runtime_lines) if runtime_lines else "none"),
    ]
    for runtime_line in ("cuda13", "cuda12"):
        selection_log.append(
            "linux_cuda_selection: runtime_dirs "
            f"{runtime_line}="
            + (",".join(runtime_dirs.get(runtime_line, [])) if runtime_dirs.get(runtime_line) else "none")
        )
    published_artifacts = [artifact for artifact in release.artifacts if artifact.install_kind == "linux-cuda"]
    published_asset_names = sorted(artifact.asset_name for artifact in published_artifacts)
    selection_log.append(
        "linux_cuda_selection: published_assets="
        + (",".join(published_asset_names) if published_asset_names else "none")
    )

    if not host_sms:
        selection_log.append("linux_cuda_selection: compute capability detection unavailable; prefer portable by runtime line")
    if not runtime_lines:
        selection_log.append("linux_cuda_selection: no compatible Linux CUDA runtime line detected on host")
        return None

    host_floor = min(int(value) for value in host_sms) if host_sms else None
    host_ceiling = max(int(value) for value in host_sms) if host_sms else None
    if host_floor is not None and host_ceiling is not None:
        selection_log.append(f"linux_cuda_selection: host_floor={host_floor} host_ceiling={host_ceiling}")

    attempts: list[AssetChoice] = []
    seen_attempts: set[str] = set()

    def add_attempt(artifact: PublishedLlamaArtifact, asset_url: str, reason: str) -> None:
        asset_name = artifact.asset_name
        if asset_name in seen_attempts:
            return
        seen_attempts.add(asset_name)
        attempts.append(
            AssetChoice(
                repo=release.repo,
                tag=release.release_tag,
                name=asset_name,
                url=asset_url,
                source_label="published",
                is_ready_bundle=True,
                install_kind="linux-cuda",
                bundle_profile=artifact.bundle_profile,
                runtime_line=artifact.runtime_line,
                coverage_class=artifact.coverage_class,
                supported_sms=artifact.supported_sms,
                min_sm=artifact.min_sm,
                max_sm=artifact.max_sm,
                selection_log=list(selection_log)
                + [
                    "linux_cuda_selection: selected "
                    f"{asset_name} runtime_line={artifact.runtime_line} coverage_class={artifact.coverage_class} reason={reason}"
                ],
            )
        )

    for runtime_line in runtime_lines:
        coverage_candidates: list[tuple[PublishedLlamaArtifact, str]] = []
        portable_candidate: tuple[PublishedLlamaArtifact, str] | None = None
        for artifact in published_artifacts:
            if artifact.runtime_line != runtime_line:
                continue
            asset_name = artifact.asset_name
            asset_url = release.assets.get(asset_name)
            if not asset_url:
                selection_log.append(f"linux_cuda_selection: reject {asset_name} missing asset")
                continue
            if not host_sms and artifact.coverage_class != "portable":
                selection_log.append(
                    "linux_cuda_selection: reject "
                    f"{asset_name} runtime_line={runtime_line} coverage_class={artifact.coverage_class} "
                    "reason=unknown_compute_caps_prefer_portable"
                )
                continue

            if not artifact.supported_sms:
                selection_log.append(
                    "linux_cuda_selection: reject "
                    f"{asset_name} runtime_line={runtime_line} coverage_class={artifact.coverage_class} "
                    "reason=artifact_missing_supported_sms"
                )
                continue
            if artifact.min_sm is None or artifact.max_sm is None:
                selection_log.append(
                    "linux_cuda_selection: reject "
                    f"{asset_name} runtime_line={runtime_line} coverage_class={artifact.coverage_class} "
                    "reason=artifact_missing_sm_bounds"
                )
                continue

            supported_sms = {str(value) for value in artifact.supported_sms}
            missing_sms = [sm for sm in host_sms if sm not in supported_sms]
            reasons: list[str] = []
            if host_floor is not None and host_floor < artifact.min_sm:
                reasons.append(f"host_floor<{artifact.min_sm}")
            if host_ceiling is not None and host_ceiling > artifact.max_sm:
                reasons.append(f"host_ceiling>{artifact.max_sm}")
            if missing_sms:
                reasons.append(f"missing_sms={','.join(missing_sms)}")
            if reasons:
                selection_log.append(
                    "linux_cuda_selection: reject "
                    f"{asset_name} runtime_line={runtime_line} coverage_class={artifact.coverage_class} "
                    f"coverage={artifact.min_sm}-{artifact.max_sm} supported={','.join(artifact.supported_sms)} "
                    f"reasons={' '.join(reasons)}"
                )
                continue

            selection_log.append(
                "linux_cuda_selection: accept "
                f"{asset_name} runtime_line={runtime_line} coverage_class={artifact.coverage_class} "
                f"coverage={artifact.min_sm}-{artifact.max_sm} supported={','.join(artifact.supported_sms)}"
            )
            if artifact.coverage_class == "portable":
                portable_candidate = (artifact, asset_url)
            else:
                coverage_candidates.append((artifact, asset_url))

        if coverage_candidates:
            artifact, url = sorted(
                coverage_candidates,
                key=lambda item: (
                    (item[0].max_sm or 0) - (item[0].min_sm or 0),
                    item[0].rank,
                    item[0].max_sm or 0,
                ),
            )[0]
            add_attempt(artifact, url, "best coverage for runtime line")
        if portable_candidate:
            artifact, url = portable_candidate
            add_attempt(artifact, url, "portable fallback for runtime line")

    if not attempts:
        return None

    selection_log.append(
        "linux_cuda_selection: attempt_order=" + ",".join(choice.name for choice in attempts)
    )
    for attempt in attempts:
        attempt.selection_log = list(selection_log) + [
            "linux_cuda_selection: attempt "
            f"{attempt.name} runtime_line={attempt.runtime_line} coverage_class={attempt.coverage_class}"
        ]
    return LinuxCudaSelection(attempts=attempts, selection_log=selection_log)


def latest_published_linux_cuda_tag(host: HostInfo, published_repo: str) -> str | None:
    for release in iter_published_release_bundles(published_repo):
        if linux_cuda_choice_from_release(host, release):
            return release.upstream_tag
    return None


def resolve_requested_llama_tag(requested_tag: str | None, host: HostInfo, published_repo: str) -> str:
    if requested_tag and requested_tag != "latest":
        return requested_tag
    if host.is_linux and host.is_x86_64 and host.has_nvidia:
        try:
            published_tag = latest_published_linux_cuda_tag(host, published_repo)
        except Exception as exc:
            log(f"linux CUDA latest-tag lookup failed for {published_repo}: {exc}")
        else:
            if published_tag:
                return published_tag
            log("no compatible published Linux CUDA release found for latest; falling back to upstream latest release")
    return latest_upstream_release_tag()


def run_capture(command: list[str], *, timeout: int = 30, check: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
    return result


def detect_host() -> HostInfo:
    system = platform.system()
    machine = platform.machine().lower()
    is_windows = system == "Windows"
    is_linux = system == "Linux"
    is_macos = system == "Darwin"
    is_x86_64 = machine in {"x86_64", "amd64"}
    is_arm64 = machine in {"arm64", "aarch64"}

    nvidia_smi = shutil.which("nvidia-smi")
    driver_cuda_version = None
    compute_caps: list[str] = []
    has_nvidia = False
    if nvidia_smi:
        try:
            result = run_capture([nvidia_smi], timeout=20)
            merged = "\n".join(part for part in (result.stdout, result.stderr) if part)
            if "NVIDIA-SMI" in merged:
                has_nvidia = True
            for line in merged.splitlines():
                if "CUDA Version:" in line:
                    raw = line.split("CUDA Version:", 1)[1].strip().split()[0]
                    major, minor = raw.split(".", 1)
                    driver_cuda_version = (int(major), int(minor))
                    break
        except Exception:
            pass

        try:
            caps = run_capture(
                [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
                timeout=20,
            )
            for raw in caps.stdout.splitlines():
                raw = raw.strip()
                if "." not in raw:
                    continue
                major, minor = raw.split(".", 1)
                cap = f"{int(major)}{int(minor)}"
                if cap not in compute_caps:
                    compute_caps.append(cap)
        except Exception:
            pass

    return HostInfo(
        system=system,
        machine=machine,
        is_windows=is_windows,
        is_linux=is_linux,
        is_macos=is_macos,
        is_x86_64=is_x86_64,
        is_arm64=is_arm64,
        nvidia_smi=nvidia_smi,
        driver_cuda_version=driver_cuda_version,
        compute_caps=compute_caps,
        has_nvidia=has_nvidia,
    )


def pick_windows_cuda_runtime(host: HostInfo) -> str | None:
    if not host.driver_cuda_version:
        return None
    major, minor = host.driver_cuda_version
    if major > 13 or (major == 13 and minor >= 1):
        return "13.1"
    if major > 12 or (major == 12 and minor >= 4):
        return "12.4"
    return None


def resolve_linux_cuda_choice(host: HostInfo, llama_tag: str, published_repo: str, published_release_tag: str) -> LinuxCudaSelection:
    for release in iter_published_release_bundles(published_repo, published_release_tag):
        if release.upstream_tag != llama_tag:
            log(
                "published release skipped "
                f"{published_repo}@{release.release_tag}: upstream_tag={release.upstream_tag} expected={llama_tag}"
            )
            continue
        selection = linux_cuda_choice_from_release(host, release)
        if selection is not None:
            return selection
    raise PrebuiltFallback("no compatible published Linux CUDA bundle was found")


def resolve_asset_choice(host: HostInfo, llama_tag: str, published_repo: str, published_release_tag: str) -> AssetChoice:
    upstream_assets = github_release_assets(UPSTREAM_REPO, llama_tag)

    if host.is_linux and host.is_x86_64:
        if host.has_nvidia:
            return resolve_linux_cuda_choice(host, llama_tag, published_repo, published_release_tag).primary

        upstream_name = f"llama-{llama_tag}-bin-ubuntu-x64.tar.gz"
        if upstream_name not in upstream_assets:
            raise PrebuiltFallback("upstream Linux CPU asset was not found")
        return AssetChoice(
            repo=UPSTREAM_REPO,
            tag=llama_tag,
            name=upstream_name,
            url=upstream_assets[upstream_name],
            source_label="upstream",
            install_kind="linux-cpu",
        )

    if host.is_windows and host.is_x86_64:
        if host.has_nvidia:
            runtime = pick_windows_cuda_runtime(host)
            if runtime:
                upstream_name = f"llama-{llama_tag}-bin-win-cuda-{runtime}-x64.zip"
                if upstream_name in upstream_assets:
                    return AssetChoice(
                        repo=UPSTREAM_REPO,
                        tag=llama_tag,
                        name=upstream_name,
                        url=upstream_assets[upstream_name],
                        source_label="upstream",
                        install_kind="windows-cuda",
                    )
            raise PrebuiltFallback("no compatible Windows CUDA asset was found")

        upstream_name = f"llama-{llama_tag}-bin-win-cpu-x64.zip"
        if upstream_name not in upstream_assets:
            raise PrebuiltFallback("upstream Windows CPU asset was not found")
        return AssetChoice(
            repo=UPSTREAM_REPO,
            tag=llama_tag,
            name=upstream_name,
            url=upstream_assets[upstream_name],
            source_label="upstream",
            install_kind="windows-cpu",
        )

    if host.is_macos and host.is_arm64:
        upstream_name = f"llama-{llama_tag}-bin-macos-arm64.tar.gz"
        if upstream_name not in upstream_assets:
            raise PrebuiltFallback("upstream macOS arm64 asset was not found")
        return AssetChoice(
            repo=UPSTREAM_REPO,
            tag=llama_tag,
            name=upstream_name,
            url=upstream_assets[upstream_name],
            source_label="upstream",
            install_kind="macos-arm64",
        )

    if host.is_macos and host.is_x86_64:
        upstream_name = f"llama-{llama_tag}-bin-macos-x64.tar.gz"
        if upstream_name not in upstream_assets:
            raise PrebuiltFallback("upstream macOS x64 asset was not found")
        return AssetChoice(
            repo=UPSTREAM_REPO,
            tag=llama_tag,
            name=upstream_name,
            url=upstream_assets[upstream_name],
            source_label="upstream",
            install_kind="macos-x64",
        )

    raise PrebuiltFallback(f"no prebuilt policy exists for {host.system} {host.machine}")


def extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if archive_path.name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
        return
    if archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(destination)
        return
    raise PrebuiltFallback(f"unsupported archive format: {archive_path.name}")


def payload_root(extract_dir: Path) -> Path:
    entries = list(extract_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extract_dir


def copy_globs(source_dir: Path, destination: Path, patterns: list[str], *, required: bool = True) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    matched_any = False
    for path in source_dir.iterdir():
        for pattern in patterns:
            if fnmatch.fnmatch(path.name, pattern):
                shutil.copy2(path, destination / path.name)
                matched_any = True
                break
    if required and not matched_any:
        raise PrebuiltFallback(f"required files missing from {source_dir}: {patterns}")


def ensure_converter_scripts(install_dir: Path, llama_tag: str) -> None:
    raw_base = f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{llama_tag}"
    converter_targets = [
        ("convert_hf_to_gguf.py", f"{raw_base}/convert_hf_to_gguf.py"),
        ("convert-hf-to-gguf.py", f"{raw_base}/convert_hf_to_gguf.py"),
    ]
    for file_name, url in converter_targets:
        download_file(url, install_dir / file_name)


def normalize_install_layout(install_dir: Path, host: HostInfo) -> tuple[Path, Path]:
    build_bin = install_dir / "build" / "bin"
    if host.is_windows:
        exec_dir = build_bin / "Release"
        exec_dir.mkdir(parents=True, exist_ok=True)
        return exec_dir / "llama-server.exe", exec_dir / "llama-quantize.exe"

    install_dir.mkdir(parents=True, exist_ok=True)
    build_bin.mkdir(parents=True, exist_ok=True)
    return install_dir / "llama-server", install_dir / "llama-quantize"


def install_from_archives(choice: AssetChoice, host: HostInfo, install_dir: Path, work_dir: Path) -> tuple[Path, Path]:
    main_archive = work_dir / choice.name
    log(f"downloading {choice.name} from {choice.source_label} release")
    download_file(choice.url, main_archive)

    if install_dir.exists():
        shutil.rmtree(install_dir)
    install_dir.mkdir(parents=True, exist_ok=True)

    if choice.is_ready_bundle:
        extract_dir = work_dir / "extract-main"
        extract_archive(main_archive, extract_dir)
        source_dir = payload_root(extract_dir)
        for item in source_dir.iterdir():
            if item.is_dir():
                shutil.copytree(item, install_dir / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, install_dir / item.name)
        if host.is_windows:
            exec_dir = install_dir / "build" / "bin" / "Release"
            exec_dir.mkdir(parents=True, exist_ok=True)
            for candidate in list(install_dir.iterdir()):
                if candidate == install_dir / "build" or candidate.is_dir():
                    continue
                if fnmatch.fnmatch(candidate.name, "*.exe") or fnmatch.fnmatch(candidate.name, "*.dll"):
                    shutil.copy2(candidate, exec_dir / candidate.name)
    else:
        extract_dir = work_dir / "extract-main"
        extract_archive(main_archive, extract_dir)
        source_dir = payload_root(extract_dir)
        if choice.install_kind == "linux-cpu":
            copy_globs(
                source_dir,
                install_dir,
                [
                    "llama-server",
                    "llama-quantize",
                    "libllama.so*",
                    "libggml.so*",
                    "libggml-base.so*",
                    "libmtmd.so*",
                    "libggml-cpu-*.so*",
                    "LICENSE",
                    "BUILD_INFO.txt",
                ],
                required=False,
            )
            copy_globs(source_dir, install_dir, ["llama-server", "llama-quantize"], required=True)
        elif choice.install_kind in {"macos-arm64", "macos-x64"}:
            copy_globs(
                source_dir,
                install_dir,
                [
                    "llama-server",
                    "llama-quantize",
                    "lib*.dylib",
                    "LICENSE",
                    "BUILD_INFO.txt",
                ],
                required=False,
            )
            copy_globs(source_dir, install_dir, ["llama-server", "llama-quantize"], required=True)
        elif choice.install_kind == "windows-cpu":
            exec_dir = install_dir / "build" / "bin" / "Release"
            copy_globs(source_dir, exec_dir, ["*.exe", "*.dll", "LICENSE", "BUILD_INFO.txt"], required=False)
            copy_globs(source_dir, exec_dir, ["llama-server.exe", "llama-quantize.exe"], required=True)
        elif choice.install_kind == "windows-cuda":
            exec_dir = install_dir / "build" / "bin" / "Release"
            copy_globs(source_dir, exec_dir, ["*.exe", "*.dll", "LICENSE", "BUILD_INFO.txt"], required=False)
            copy_globs(source_dir, exec_dir, ["llama-server.exe", "llama-quantize.exe"], required=True)
        else:
            raise PrebuiltFallback(f"unsupported upstream install kind: {choice.install_kind}")

    server_path, quantize_path = normalize_install_layout(install_dir, host)

    if host.is_windows:
        exec_dir = install_dir / "build" / "bin" / "Release"
        server_src = next(exec_dir.glob("llama-server.exe"), None)
        quantize_src = next(exec_dir.glob("llama-quantize.exe"), None)
        if server_src is None or quantize_src is None:
            raise PrebuiltFallback("windows executables were not installed correctly")
        return server_src, quantize_src

    source_server = install_dir / "llama-server"
    source_quantize = install_dir / "llama-quantize"
    if not source_server.exists():
        candidate = next((path for path in install_dir.rglob("llama-server") if path.is_file()), None)
        if candidate is None:
            raise PrebuiltFallback("llama-server was not installed")
        shutil.copy2(candidate, source_server)
    if not source_quantize.exists():
        candidate = next((path for path in install_dir.rglob("llama-quantize") if path.is_file()), None)
        if candidate is None:
            raise PrebuiltFallback("llama-quantize was not installed")
        shutil.copy2(candidate, source_quantize)

    os.chmod(source_server, 0o755)
    os.chmod(source_quantize, 0o755)

    build_bin = install_dir / "build" / "bin"
    build_bin.mkdir(parents=True, exist_ok=True)
    for src, dest in ((source_server, build_bin / "llama-server"), (source_quantize, build_bin / "llama-quantize")):
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            dest.symlink_to(src.relative_to(dest.parent))
        except Exception:
            shutil.copy2(src, dest)

    return source_server, source_quantize


def ensure_repo_shape(install_dir: Path) -> None:
    for relative in ("src", "ggml", "common"):
        (install_dir / relative).mkdir(parents=True, exist_ok=True)


def download_validation_model(path: Path) -> None:
    log("downloading tiny GGUF validation model")
    download_file(TEST_MODEL_URL, path)


def free_local_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    _, port = sock.getsockname()
    sock.close()
    return int(port)


def dedupe_existing_dirs(paths: Iterable[str | Path]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_dir():
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def linux_missing_libraries(binary_path: Path) -> list[str]:
    try:
        result = run_capture(["ldd", str(binary_path)], timeout=20)
    except Exception:
        return []

    missing: list[str] = []
    for line in (result.stdout + result.stderr).splitlines():
        line = line.strip()
        if "=> not found" not in line:
            continue
        library = line.split("=>", 1)[0].strip()
        if library and library not in missing:
            missing.append(library)
    return missing


def python_runtime_dirs() -> list[str]:
    candidates: list[Path] = []
    search_roots = [Path(entry) for entry in sys.path if entry]
    try:
        search_roots.extend(Path(path) for path in site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            search_roots.append(Path(user_site))
    except Exception:
        pass

    for root in search_roots:
        if not root.is_dir():
            continue
        candidates.extend(root.glob("nvidia/*/lib"))
        candidates.extend(root.glob("nvidia/*/bin"))
        candidates.extend(root.glob("torch/lib"))
    return dedupe_existing_dirs(candidates)


def ldconfig_runtime_dirs(required_libraries: Iterable[str]) -> list[str]:
    try:
        result = run_capture(["ldconfig", "-p"], timeout=20)
    except Exception:
        return []

    required = set(required_libraries)
    candidates: list[str] = []
    for line in result.stdout.splitlines():
        if "=>" not in line:
            continue
        library, _, location = line.partition("=>")
        library = library.strip().split()[0]
        if required and library not in required:
            continue
        path = Path(location.strip()).parent
        candidates.append(str(path))
    return dedupe_existing_dirs(candidates)


def linux_runtime_dirs(binary_path: Path) -> list[str]:
    missing = linux_missing_libraries(binary_path)
    return linux_runtime_dirs_for_required_libraries(missing)


def linux_library_matches(required_libraries: Iterable[str]) -> tuple[list[str], dict[str, list[str]], list[str]]:
    required = [library for library in required_libraries if library]
    runtime_dirs = linux_runtime_dirs_for_required_libraries(required)
    matches: dict[str, list[str]] = {}
    for library in required:
        matched_dirs = [
            directory for directory in runtime_dirs if any(Path(directory).glob(f"{library}*"))
        ]
        if matched_dirs:
            matches[library] = matched_dirs
    missing = [library for library in required if library not in matches]
    return runtime_dirs, matches, missing


def linux_required_host_libraries(choice: AssetChoice) -> tuple[str, ...]:
    if choice.install_kind in {"linux-cpu", "linux-cuda"}:
        return LINUX_SHARED_BUNDLE_HOST_LIBRARIES
    return ()


def preflight_linux_host_libraries(choice: AssetChoice, host: HostInfo) -> None:
    if not host.is_linux:
        return

    required = linux_required_host_libraries(choice)
    if not required:
        return

    runtime_dirs, matches, missing = linux_library_matches(required)
    if not missing:
        log(
            "linux host-library preflight passed: "
            + ",".join(f"{library}@{matches[library][0]}" for library in required)
        )
        return

    details = ", ".join(
        f"{library}={'|'.join(matches[library]) if library in matches else 'missing'}" for library in required
    )
    searched_dirs = ",".join(runtime_dirs) if runtime_dirs else "none"
    raise PrebuiltFallback(
        "linux host-library preflight failed for shared llama.cpp bundle: "
        + ", ".join(missing)
        + f"\nresolved={details}"
        + f"\nsearched_dirs={searched_dirs}"
    )


def glob_paths(*patterns: str) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        if any(char in pattern for char in "*?[]"):
            matches.extend(str(path) for path in Path("/").glob(pattern.lstrip("/")))
        else:
            matches.append(pattern)
    return matches


def windows_runtime_dirs() -> list[str]:
    candidates: list[str | Path] = []

    env_dirs = os.environ.get("CUDA_RUNTIME_DLL_DIR", "")
    if env_dirs:
        candidates.extend(part for part in env_dirs.split(os.pathsep) if part)

    path_dirs = os.environ.get("PATH", "")
    if path_dirs:
        candidates.extend(part for part in path_dirs.split(os.pathsep) if part)

    cuda_roots: list[Path] = []
    for name in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        value = os.environ.get(name)
        if value:
            cuda_roots.append(Path(value))

    for root in cuda_roots:
        candidates.extend([root / "bin", root / "lib" / "x64"])

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    toolkit_base = Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if toolkit_base.is_dir():
        candidates.extend(toolkit_base.glob("v*/bin"))
        candidates.extend(toolkit_base.glob("v*/lib/x64"))

    candidates.extend(Path(path) for path in python_runtime_dirs())
    return dedupe_existing_dirs(candidates)


def binary_env(binary_path: Path, install_dir: Path, host: HostInfo) -> dict[str, str]:
    env = os.environ.copy()
    if host.is_windows:
        path_dirs = [str(binary_path.parent), *windows_runtime_dirs()]
        existing = [part for part in env.get("PATH", "").split(os.pathsep) if part]
        env["PATH"] = os.pathsep.join(dedupe_existing_dirs([*path_dirs, *existing]))
    elif host.is_linux:
        ld_dirs = [str(install_dir), *linux_runtime_dirs(binary_path)]
        existing = [part for part in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(dedupe_existing_dirs([*ld_dirs, *existing]))
    elif host.is_macos:
        env["DYLD_LIBRARY_PATH"] = str(install_dir) + os.pathsep + env.get("DYLD_LIBRARY_PATH", "")
    return env


def validate_quantize(quantize_path: Path, probe_path: Path, quantized_path: Path, install_dir: Path, host: HostInfo) -> None:
    command = [str(quantize_path), str(probe_path), str(quantized_path), "Q6_K", "2"]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=120,
        env=binary_env(quantize_path, install_dir, host),
    )
    if result.returncode != 0 or not quantized_path.exists() or quantized_path.stat().st_size == 0:
        raise PrebuiltFallback(
            "llama-quantize validation failed:\n"
            + result.stdout
            + ("\n" + result.stderr if result.stderr else "")
        )


def validate_server(server_path: Path, probe_path: Path, host: HostInfo, install_dir: Path) -> None:
    port = free_local_port()
    command = [
        str(server_path),
        "-m",
        str(probe_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        "32",
        "--parallel",
        "1",
        "--threads",
        "1",
        "--ubatch-size",
        "32",
        "--batch-size",
        "32",
    ]
    if host.has_nvidia or (host.is_macos and host.is_arm64):
        command.extend(["--n-gpu-layers", "1"])

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=binary_env(server_path, install_dir, host),
    )
    lines: list[str] = []
    try:
        start = time.time()
        while time.time() - start < 20:
            if process.poll() is not None:
                break
            line = process.stdout.readline() if process.stdout else ""
            if line:
                lines.append(line.rstrip())
                if "server is listening on" in line.lower() or "starting the main loop" in line.lower():
                    break
            else:
                time.sleep(0.2)

        if process.poll() is not None:
            raise PrebuiltFallback("llama-server exited during startup:\n" + "\n".join(lines[-60:]))

        payload = json.dumps({"prompt": "a", "n_predict": 1}).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/completion",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        deadline = time.time() + 20
        response_body = ""
        status_code = None
        last_error = None
        while time.time() < deadline:
            if process.poll() is not None:
                raise PrebuiltFallback("llama-server exited before handling /completion:\n" + "\n".join(lines[-60:]))
            try:
                with urllib.request.urlopen(request, timeout=5) as response:
                    status_code = response.status
                    response_body = response.read().decode("utf-8", "replace")
                    break
            except urllib.error.HTTPError as exc:
                response_body = exc.read().decode("utf-8", "replace")
                last_error = exc
                break
            except Exception as exc:
                last_error = exc
                time.sleep(0.5)

        if status_code != 200:
            raise PrebuiltFallback(
                "llama-server completion validation failed"
                + (f" ({last_error})" if last_error else "")
                + ":\n"
                + "\n".join(lines[-60:])
                + ("\n" + response_body if response_body else "")
            )
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def collect_system_report(host: HostInfo, choice: AssetChoice | None, install_dir: Path) -> str:
    lines = [
        f"platform={host.system} machine={host.machine}",
        f"driver_cuda_version={host.driver_cuda_version}",
        f"compute_caps={','.join(host.compute_caps) if host.compute_caps else 'unknown'}",
        f"chosen_asset={(choice.name if choice else 'none')}",
        f"asset_source={(choice.source_label if choice else 'none')}",
    ]
    if host.is_linux and host.has_nvidia:
        runtime_lines, runtime_dirs = detected_linux_runtime_lines()
        lines.append("linux_runtime_lines=" + (",".join(runtime_lines) if runtime_lines else "none"))
        for runtime_line in ("cuda13", "cuda12"):
            lines.append(
                f"linux_runtime_dirs_{runtime_line}="
                + (",".join(runtime_dirs.get(runtime_line, [])) if runtime_dirs.get(runtime_line) else "none")
            )
    if choice and choice.selection_log:
        lines.append("selection_log:")
        lines.extend(choice.selection_log)
    if host.nvidia_smi:
        try:
            smi = run_capture([host.nvidia_smi], timeout=20)
            excerpt = "\n".join((smi.stdout + smi.stderr).splitlines()[:20])
            lines.append("nvidia-smi:")
            lines.append(excerpt)
        except Exception as exc:
            lines.append(f"nvidia-smi error: {exc}")

    if host.is_linux:
        required_host_libs = linux_required_host_libraries(choice) if choice else ()
        if required_host_libs:
            runtime_dirs, matches, missing = linux_library_matches(required_host_libs)
            lines.append("linux_required_host_libs=" + ",".join(required_host_libs))
            lines.append("linux_resolved_host_lib_dirs=" + (",".join(runtime_dirs) if runtime_dirs else "none"))
            for library in required_host_libs:
                lines.append(
                    f"linux_host_lib_{library}="
                    + ("|".join(matches.get(library, [])) if matches.get(library) else "missing")
                )
            lines.append("linux_missing_host_libs=" + (",".join(missing) if missing else "none"))
        server_binary = install_dir / "llama-server"
        if server_binary.exists():
            lines.append(
                "linux_missing_libs="
                + (",".join(linux_missing_libraries(server_binary)) or "none")
            )
            lines.append(
                "linux_runtime_dirs="
                + (",".join(linux_runtime_dirs(server_binary)) or "none")
            )
            try:
                ldd = run_capture(["ldd", str(server_binary)], timeout=20)
                lines.append("ldd llama-server:")
                lines.append((ldd.stdout + ldd.stderr).strip())
            except Exception as exc:
                lines.append(f"ldd error: {exc}")
    elif host.is_windows:
        lines.append("windows_runtime_dirs=" + (",".join(windows_runtime_dirs()) or "none"))
    elif host.is_macos:
        server_binary = install_dir / "llama-server"
        if server_binary.exists():
            try:
                otool = run_capture(["otool", "-L", str(server_binary)], timeout=20)
                lines.append("otool -L llama-server:")
                lines.append((otool.stdout + otool.stderr).strip())
            except Exception as exc:
                lines.append(f"otool error: {exc}")

    return "\n".join(lines)


def install_prebuilt(install_dir: Path, llama_tag: str, published_repo: str, published_release_tag: str) -> None:
    host = detect_host()
    choice: AssetChoice | None = None
    linux_cuda_attempts: list[AssetChoice] = []
    try:
        requested_tag = llama_tag
        llama_tag = resolve_requested_llama_tag(llama_tag, host, published_repo)
        if host.is_linux and host.is_x86_64 and host.has_nvidia:
            linux_cuda_selection = resolve_linux_cuda_choice(host, llama_tag, published_repo, published_release_tag)
            linux_cuda_attempts = linux_cuda_selection.attempts
            choice = linux_cuda_attempts[0]
            log_lines(linux_cuda_selection.selection_log)
        else:
            choice = resolve_asset_choice(host, llama_tag, published_repo, published_release_tag)
            if choice.selection_log:
                log_lines(choice.selection_log)
        log(f"selected {choice.name} ({choice.source_label}) for {host.system} {host.machine}")
        preflight_linux_host_libraries(choice, host)
        with tempfile.TemporaryDirectory(prefix="unsloth-llama-prebuilt-") as tmp:
            work_dir = Path(tmp)
            probe_path = work_dir / "stories260K.gguf"
            quantized_path = work_dir / "stories260K-q4.gguf"
            download_validation_model(probe_path)

            tried_fallback = False
            attempts = linux_cuda_attempts if linux_cuda_attempts else ([choice] if choice else [])
            for index, attempt in enumerate(attempts):
                if index > 0:
                    tried_fallback = True
                    log(
                        "retrying Linux CUDA prebuilt "
                        f"{attempt.name} runtime_line={attempt.runtime_line} coverage_class={attempt.coverage_class}"
                    )
                choice = attempt
                if choice.selection_log:
                    log_lines(choice.selection_log)
                server_path, quantize_path = install_from_archives(choice, host, install_dir, work_dir)
                ensure_repo_shape(install_dir)
                ensure_converter_scripts(install_dir, llama_tag)
                try:
                    validate_quantize(quantize_path, probe_path, quantized_path, install_dir, host)
                    validate_server(server_path, probe_path, host, install_dir)
                    break
                except PrebuiltFallback:
                    if index == len(attempts) - 1:
                        raise
                    log("selected Linux CUDA bundle failed validation; trying next prebuilt fallback")
            else:
                raise PrebuiltFallback("no Linux CUDA bundle passed validation")

            metadata = {
                "requested_tag": requested_tag,
                "tag": llama_tag,
                "asset": choice.name,
                "source": choice.source_label,
                "bundle_profile": choice.bundle_profile,
                "runtime_line": choice.runtime_line,
                "coverage_class": choice.coverage_class,
                "prebuilt_fallback_used": tried_fallback,
                "installed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text(json.dumps(metadata, indent=2) + "\n")
    except PrebuiltFallback as exc:
        log("prebuilt install path failed; falling back to source build")
        report = collect_system_report(host, choice, install_dir)
        print(report)
        raise SystemExit(EXIT_FALLBACK) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install and validate a prebuilt llama.cpp bundle for Unsloth Studio."
    )
    parser.add_argument("--install-dir", help="Target ~/.unsloth/llama.cpp directory")
    parser.add_argument(
        "--llama-tag",
        default=DEFAULT_LLAMA_TAG,
        help="llama.cpp release tag. Defaults to the latest upstream release tag.",
    )
    parser.add_argument("--published-repo", default=DEFAULT_PUBLISHED_REPO, help="Published bundle repository")
    parser.add_argument(
        "--published-release-tag",
        default=DEFAULT_PUBLISHED_TAG,
        help="Published GitHub release tag to pin. By default, scan releases until a compatible llama.cpp bundle is found.",
    )
    parser.add_argument(
        "--resolve-llama-tag",
        nargs="?",
        const="latest",
        help="Resolve a llama.cpp tag such as 'latest' for the current host variant.",
    )
    parser.add_argument(
        "--resolve-install-tag",
        nargs="?",
        const="latest",
        help="Resolve a llama.cpp tag such as 'latest' for the current host variant and print the concrete install tag.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resolve_llama_tag is not None:
        host = detect_host()
        print(resolve_requested_llama_tag(args.resolve_llama_tag, host, args.published_repo))
        return EXIT_SUCCESS

    if args.resolve_install_tag is not None:
        host = detect_host()
        print(resolve_requested_llama_tag(args.resolve_install_tag, host, args.published_repo))
        return EXIT_SUCCESS

    if not args.install_dir:
        raise SystemExit("install_llama_prebuilt.py: --install-dir is required unless --resolve-llama-tag is used")
    install_prebuilt(
        install_dir=Path(args.install_dir).expanduser().resolve(),
        llama_tag=args.llama_tag,
        published_repo=args.published_repo,
        published_release_tag=args.published_release_tag or "",
    )
    return EXIT_SUCCESS


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        message = textwrap.shorten(str(exc), width=400, placeholder="...")
        log(f"fatal helper error: {message}")
        raise SystemExit(EXIT_ERROR)
