import contextlib
import importlib
import importlib.util
import os
import sys
import threading
import warnings
from types import ModuleType
from typing import Any, Callable

DEFAULT_KERNEL_BACKEND = "triton"
EAGER_KERNEL_BACKEND = "eager"
GLOBAL_BACKEND_ENV = "UNSLOTH_KERNEL_BACKEND"
OVERRIDES_ENV = "UNSLOTH_KERNEL_BACKEND_OVERRIDES"
STRICT_ENV = "UNSLOTH_KERNEL_BACKEND_STRICT"

KernelImplementation = Callable[..., Any]
AvailabilityCheck = Callable[[], tuple[bool, str | None]]
BackendLoader = Callable[[], None]

_REGISTRY: dict[str, dict[str, KernelImplementation]] = {}
_BUILTIN_LOADERS: dict[str, BackendLoader] = {}
_AVAILABILITY_CHECKS: dict[str, AvailabilityCheck] = {}
_LOADED_BACKENDS: set[str] = set()
_GLOBAL_BACKEND_OVERRIDE: str | None = None
_KERNEL_BACKEND_OVERRIDES: dict[str, str] = {}
_WARNED_FALLBACKS: set[tuple[str, str, str]] = set()
_LOCK = threading.RLock()


def _normalize_kernel_name(name: str) -> str:
    normalized = str(name).strip()
    if not normalized:
        raise ValueError("Kernel name must not be empty.")
    if "." not in normalized:
        normalized = f"unsloth.{normalized}"
    return normalized


def _normalize_backend_name(backend: str | None) -> str | None:
    if backend is None:
        return None
    normalized = str(backend).strip().lower()
    if not normalized:
        return None
    return normalized


def _parse_env_overrides(raw_value: str | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if not raw_value:
        return overrides

    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid {OVERRIDES_ENV} entry {item!r}. Expected 'kernel=backend'."
            )
        kernel_name, backend_name = item.split("=", 1)
        overrides[_normalize_kernel_name(kernel_name)] = _normalize_backend_name(
            backend_name
        )
    return overrides


def _strict_mode_enabled() -> bool:
    return os.environ.get(STRICT_ENV, "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def register_impl(name: str, backend: str):
    def decorator(func: KernelImplementation) -> KernelImplementation:
        register_kernel_backend(name, backend, func)
        return func

    return decorator


def register_kernel_backend(
    name: str,
    backend: str,
    implementation: KernelImplementation,
) -> None:
    kernel_name = _normalize_kernel_name(name)
    backend_name = _normalize_backend_name(backend)
    if backend_name is None:
        raise ValueError(f"Backend for {kernel_name!r} must not be empty.")
    with _LOCK:
        _REGISTRY.setdefault(kernel_name, {})[backend_name] = implementation


def register_builtin_backend_loader(
    backend: str,
    loader: BackendLoader,
    *,
    availability_check: AvailabilityCheck | None = None,
) -> None:
    backend_name = _normalize_backend_name(backend)
    if backend_name is None:
        raise ValueError("Backend name must not be empty.")
    with _LOCK:
        _BUILTIN_LOADERS[backend_name] = loader
        if availability_check is not None:
            _AVAILABILITY_CHECKS[backend_name] = availability_check


def ensure_backend_loaded(backend: str) -> None:
    backend_name = _normalize_backend_name(backend)
    if backend_name is None:
        return
    with _LOCK:
        if backend_name in _LOADED_BACKENDS:
            return
        loader = _BUILTIN_LOADERS.get(backend_name)
        if loader is None:
            _LOADED_BACKENDS.add(backend_name)
            return
    loader()
    with _LOCK:
        _LOADED_BACKENDS.add(backend_name)


def is_kernel_backend_available(backend: str) -> tuple[bool, str | None]:
    backend_name = _normalize_backend_name(backend)
    if backend_name is None:
        return False, "Backend name is empty."
    if backend_name == DEFAULT_KERNEL_BACKEND:
        if importlib.util.find_spec("triton") is None:
            return False, "Triton is not installed."
        return True, None

    availability_check = _AVAILABILITY_CHECKS.get(backend_name)
    if availability_check is None:
        return True, None
    try:
        return availability_check()
    except Exception as exc:
        return False, str(exc)


def _get_requested_backend(kernel_name: str, backend: str | None = None) -> str:
    explicit_backend = _normalize_backend_name(backend)
    if explicit_backend is not None:
        return explicit_backend

    runtime_override = _KERNEL_BACKEND_OVERRIDES.get(kernel_name)
    if runtime_override is not None:
        return runtime_override

    env_overrides = _parse_env_overrides(os.environ.get(OVERRIDES_ENV))
    env_override = env_overrides.get(kernel_name)
    if env_override is not None:
        return env_override

    if _GLOBAL_BACKEND_OVERRIDE is not None:
        return _GLOBAL_BACKEND_OVERRIDE

    env_global = _normalize_backend_name(os.environ.get(GLOBAL_BACKEND_ENV))
    if env_global is not None:
        return env_global

    return DEFAULT_KERNEL_BACKEND


def _warn_fallback_once(kernel_name: str, requested_backend: str, fallback_backend: str):
    key = (kernel_name, requested_backend, fallback_backend)
    with _LOCK:
        if key in _WARNED_FALLBACKS:
            return
        _WARNED_FALLBACKS.add(key)
    warnings.warn(
        (
            f"Kernel backend '{requested_backend}' is unavailable for '{kernel_name}'. "
            f"Falling back to '{fallback_backend}'."
        ),
        RuntimeWarning,
        stacklevel = 3,
    )


def _candidate_backends(
    preferred_backend: str,
    fallback_backend: str,
) -> list[str]:
    candidates: list[str] = []
    for backend_name in (
        preferred_backend,
        fallback_backend,
        EAGER_KERNEL_BACKEND,
    ):
        normalized = _normalize_backend_name(backend_name)
        if normalized is None or normalized in candidates:
            continue
        candidates.append(normalized)
    return candidates


def get_kernel_backend(
    name: str,
    *,
    backend: str | None = None,
    fallback_backend: str = DEFAULT_KERNEL_BACKEND,
) -> str:
    kernel_name = _normalize_kernel_name(name)
    preferred_backend = _get_requested_backend(kernel_name, backend = backend)
    fallback_backend_name = _normalize_backend_name(fallback_backend) or DEFAULT_KERNEL_BACKEND
    available_backends = sorted(_REGISTRY.get(kernel_name, {}))

    available, reason = is_kernel_backend_available(preferred_backend)
    if available:
        ensure_backend_loaded(preferred_backend)
        if preferred_backend in _REGISTRY.get(kernel_name, {}):
            return preferred_backend

    if _strict_mode_enabled():
        detail = f": {reason}" if reason else ""
        raise RuntimeError(
            f"Kernel backend '{preferred_backend}' is unavailable for '{kernel_name}'{detail}"
        )

    for candidate_backend in _candidate_backends(
        preferred_backend,
        fallback_backend_name,
    ):
        if candidate_backend == preferred_backend:
            continue
        candidate_available, _ = is_kernel_backend_available(candidate_backend)
        if not candidate_available:
            continue
        ensure_backend_loaded(candidate_backend)
        if candidate_backend in _REGISTRY.get(kernel_name, {}):
            _warn_fallback_once(kernel_name, preferred_backend, candidate_backend)
            return candidate_backend

    raise NotImplementedError(
        f"No registered implementation for '{kernel_name}'. Available backends: {available_backends}"
    )


def get_kernel_impl(
    name: str,
    *,
    backend: str | None = None,
    fallback_backend: str = DEFAULT_KERNEL_BACKEND,
) -> KernelImplementation:
    kernel_name = _normalize_kernel_name(name)
    backend_name = get_kernel_backend(
        kernel_name,
        backend = backend,
        fallback_backend = fallback_backend,
    )
    return _REGISTRY[kernel_name][backend_name]


def dispatch_kernel(
    name: str,
    *args,
    backend: str | None = None,
    fallback_backend: str = DEFAULT_KERNEL_BACKEND,
    **kwargs,
):
    implementation = get_kernel_impl(
        name,
        backend = backend,
        fallback_backend = fallback_backend,
    )
    return implementation(*args, **kwargs)


def set_kernel_backend(backend: str | None) -> None:
    global _GLOBAL_BACKEND_OVERRIDE
    _GLOBAL_BACKEND_OVERRIDE = _normalize_backend_name(backend)


def set_kernel_backend_for_op(name: str, backend: str | None) -> None:
    kernel_name = _normalize_kernel_name(name)
    normalized_backend = _normalize_backend_name(backend)
    if normalized_backend is None:
        _KERNEL_BACKEND_OVERRIDES.pop(kernel_name, None)
    else:
        _KERNEL_BACKEND_OVERRIDES[kernel_name] = normalized_backend


def set_kernel_backends(
    *,
    global_backend: str | None = None,
    overrides: dict[str, str | None] | None = None,
) -> None:
    set_kernel_backend(global_backend)
    if overrides is None:
        return
    for kernel_name, backend_name in overrides.items():
        set_kernel_backend_for_op(kernel_name, backend_name)


def clear_kernel_backend_overrides() -> None:
    global _GLOBAL_BACKEND_OVERRIDE
    _GLOBAL_BACKEND_OVERRIDE = None
    _KERNEL_BACKEND_OVERRIDES.clear()


@contextlib.contextmanager
def kernel_backend_context(
    *,
    global_backend: str | None = None,
    overrides: dict[str, str | None] | None = None,
):
    previous_global = _GLOBAL_BACKEND_OVERRIDE
    previous_overrides = dict(_KERNEL_BACKEND_OVERRIDES)
    set_kernel_backends(global_backend = global_backend, overrides = overrides)
    try:
        yield
    finally:
        clear_kernel_backend_overrides()
        set_kernel_backend(previous_global)
        _KERNEL_BACKEND_OVERRIDES.update(previous_overrides)


def get_registered_kernel_backends(name: str | None = None) -> dict[str, list[str]] | list[str]:
    if name is not None:
        kernel_name = _normalize_kernel_name(name)
        return sorted(_REGISTRY.get(kernel_name, {}))
    return {
        kernel_name: sorted(backends)
        for kernel_name, backends in sorted(_REGISTRY.items())
    }


def get_kernel_backend_state() -> dict[str, Any]:
    env_overrides = _parse_env_overrides(os.environ.get(OVERRIDES_ENV))
    return {
        "default_backend": DEFAULT_KERNEL_BACKEND,
        "eager_backend": EAGER_KERNEL_BACKEND,
        "env_global_backend": _normalize_backend_name(os.environ.get(GLOBAL_BACKEND_ENV)),
        "env_overrides": dict(sorted(env_overrides.items())),
        "runtime_global_backend": _GLOBAL_BACKEND_OVERRIDE,
        "runtime_overrides": dict(sorted(_KERNEL_BACKEND_OVERRIDES.items())),
        "strict": _strict_mode_enabled(),
    }


def describe_kernel_backends() -> dict[str, Any]:
    for backend_name in sorted(_BUILTIN_LOADERS):
        available, _ = is_kernel_backend_available(backend_name)
        if available:
            ensure_backend_loaded(backend_name)

    backend_names = {
        DEFAULT_KERNEL_BACKEND,
        EAGER_KERNEL_BACKEND,
        *(_BUILTIN_LOADERS.keys()),
        *(backend for backends in _REGISTRY.values() for backend in backends.keys()),
    }
    availability = {
        backend_name: {
            "available": is_kernel_backend_available(backend_name)[0],
            "reason": is_kernel_backend_available(backend_name)[1],
        }
        for backend_name in sorted(backend_names)
    }
    return {
        "state": get_kernel_backend_state(),
        "availability": availability,
        "registry": get_registered_kernel_backends(),
    }


def _load_cutile_backend() -> None:
    if importlib.util.find_spec("cuda.tile_experimental") is None:
        import cuda.tile as ct

        tile_experimental = ModuleType("cuda.tile_experimental")

        def autotune_launch(
            stream,
            *,
            grid_fn,
            kernel,
            args_fn,
            hints_fn = None,
            search_space = None,
        ):
            config = None
            if search_space is not None:
                configs = search_space() if callable(search_space) else search_space
                config = next(iter(configs), None)
            grid = grid_fn(config)
            args = args_fn(config)
            # Older/newer cuda.tile Python bindings differ on whether launch
            # accepts occupancy-like keyword hints. The compatibility shim uses
            # the first autotune candidate and launches without keyword hints.
            return ct.launch(stream, grid, kernel, args)

        def clear_autotune_cache():
            return None

        tile_experimental.autotune_launch = autotune_launch
        tile_experimental.clear_autotune_cache = clear_autotune_cache
        sys.modules["cuda.tile_experimental"] = tile_experimental

    importlib.import_module("unsloth.kernels.backends.cutile")


def _check_cutile_backend() -> tuple[bool, str | None]:
    try:
        importlib.import_module("cuda.tile")
    except Exception as exc:
        return False, f"Missing cuda.tile: {exc}"
    return True, None


register_builtin_backend_loader(
    "cutile",
    _load_cutile_backend,
    availability_check = _check_cutile_backend,
)
