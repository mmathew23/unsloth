import contextlib
import contextvars
import importlib
import importlib.util
import os
import threading
import warnings
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
_BACKEND_PACKAGES: dict[str, str] = {}
_AVAILABILITY_CHECKS: dict[str, AvailabilityCheck] = {}
# One entry per backend whose loader has been attempted.
# Value is None on success, the raised exception on failure (cached so we
# don't retry a broken loader on every dispatch).
_BACKEND_LOAD_RESULT: dict[str, BaseException | None] = {}
_LOADING_BACKENDS: dict[str, threading.Event] = {}
_THREAD_LOCAL = threading.local()
_RUNTIME_GLOBAL_BACKEND: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "unsloth_kernel_backend_global",
    default = None,
)
_RUNTIME_KERNEL_OVERRIDES: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "unsloth_kernel_backend_overrides",
    default = {},
)
_WARNED_FALLBACKS: set[tuple[str, str, str]] = set()
_LOCK = threading.RLock()

# Hot-path caches. Hit on the steady-state training step where the same
# (kernel, backend) pair is dispatched 100s of times per second. Reading
# either is GIL-atomic so we skip _LOCK entirely on the common cache-hit
# path, eliminating ~750 RLock acquires per Qwen3-8B step.
#
# Invariants:
#  * `_LOADED_BACKENDS` is a strict subset of backends whose loader has
#    completed successfully. Once added, an entry is never removed (loader
#    failures are kept out). `clear_backend_load_cache` resets it.
#  * `_RESOLVE_CACHE` is keyed by the full resolution input (kernel_name,
#    explicit_backend kwarg, fallback_backend, env_global, env_overrides_str)
#    so any env-driven change is naturally a cache miss. Setter functions
#    that mutate `_REGISTRY` / `_BUILTIN_LOADERS` clear it.
_LOADED_BACKENDS: set[str] = set()
_RESOLVE_CACHE: dict[tuple[str, str | None, str, str | None, str | None], str] = {}

# Hooks fired when the global resolved backend changes — i.e. when
# `set_kernel_backend` / `set_kernel_backend_for_op` / `set_kernel_backends` /
# `clear_kernel_backend_overrides` mutates runtime backend state, when
# `kernel_backend_context` enters/exits, or when a new backend implementation is
# registered (so dispatcher kernels can rebind their `_resolved_<name>` module
# aliases).
GlobalBackendChangeHook = Callable[[str | None], None]
_GLOBAL_BACKEND_CHANGE_HOOKS: list[GlobalBackendChangeHook] = []
_GLOBAL_BACKEND_CHANGE_HOOK_KEYS: dict[tuple[str, str], GlobalBackendChangeHook] = {}


def register_global_backend_change_hook(fn: GlobalBackendChangeHook) -> None:
    """Register `fn` to be invoked after the resolved global backend changes.

    Called with the currently-resolved global backend string (e.g. ``"triton"``)
    or ``None`` if the global has been cleared. Fired by:
      * ``set_kernel_backend`` / ``set_kernel_backend_for_op`` /
        ``set_kernel_backends`` / ``clear_kernel_backend_overrides``
      * ``kernel_backend_context`` enter and exit (so nested-context restoration is honored)
      * ``register_kernel_backend`` (so a backend that registers AFTER a kernel
        module's `_resolved_<name>` alias was bound triggers a re-bind)
      * ``ensure_backend_loaded`` on first successful load (same reason)

    Hooks must be idempotent and side-effect-free beyond updating their own
    module-level aliases. Exceptions raised inside a hook are swallowed and
    logged (a faulty backend should not break global state mutation).
    """
    if not callable(fn):
        raise TypeError("Backend change hook must be callable.")
    hook_key = (getattr(fn, "__module__", ""), getattr(fn, "__qualname__", repr(fn)))
    with _LOCK:
        previous = _GLOBAL_BACKEND_CHANGE_HOOK_KEYS.get(hook_key)
        if previous is fn:
            return
        if previous in _GLOBAL_BACKEND_CHANGE_HOOKS:
            _GLOBAL_BACKEND_CHANGE_HOOKS.remove(previous)
        _GLOBAL_BACKEND_CHANGE_HOOK_KEYS[hook_key] = fn
        _GLOBAL_BACKEND_CHANGE_HOOKS.append(fn)


def _fire_global_backend_change_hooks() -> None:
    """Invoke every registered global-change hook with the current resolved
    global backend value. Hook exceptions are swallowed — a buggy hook must
    not corrupt the registry's state. Called after `_RUNTIME_GLOBAL_BACKEND`
    or the loaded-backend set has been mutated."""
    with _LOCK:
        hooks = tuple(_GLOBAL_BACKEND_CHANGE_HOOKS)
    if not hooks:
        return
    current = _RUNTIME_GLOBAL_BACKEND.get()
    for hook in hooks:
        try:
            hook(current)
        except Exception as exc:
            warnings.warn(
                f"Backend change hook {hook!r} raised {exc!r}; ignoring.",
                RuntimeWarning,
                stacklevel = 3,
            )


def _is_backend_loading_in_current_thread(backend: str) -> bool:
    return backend in getattr(_THREAD_LOCAL, "loading_backends", set())


def _invalidate_resolve_cache() -> None:
    """Drop the resolution cache. Cheap; called from setters that mutate
    process-wide registry state. Per-thread/contextvar setters do not need
    to call this — `get_kernel_backend` checks the contextvars before
    consulting the cache."""
    _RESOLVE_CACHE.clear()


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
        normalized_backend = _normalize_backend_name(backend_name)
        if normalized_backend is None:
            # An empty value (e.g. "rope_embedding=") is treated as "no
            # override" for that kernel rather than silently storing None.
            continue
        overrides[_normalize_kernel_name(kernel_name)] = normalized_backend
    return overrides


# Cache for the parsed UNSLOTH_KERNEL_BACKEND_OVERRIDES env value. Re-parsing
# the env var on every dispatch is measurable in tight training loops; the
# cache is invalidated whenever the env string changes. CPython attribute
# writes are atomic, so a benign race during cache refresh just re-parses
# once on each thread -- correctness is preserved.
_ENV_OVERRIDE_CACHE: tuple[str | None, dict[str, str]] = (None, {})


def _get_env_overrides() -> dict[str, str]:
    global _ENV_OVERRIDE_CACHE
    raw = os.environ.get(OVERRIDES_ENV)
    cached_raw, cached_value = _ENV_OVERRIDE_CACHE
    if raw == cached_raw:
        return cached_value
    parsed = _parse_env_overrides(raw)
    _ENV_OVERRIDE_CACHE = (raw, parsed)
    return parsed


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
        backend_is_loading = backend_name in _LOADING_BACKENDS
        _invalidate_resolve_cache()
    # New backend impl may unlock dispatcher aliases that were `None` before
    # (e.g. cutile registers its `act_quant` after fp8.py was imported).
    #
    # Do not fire while this backend's loader is still in-flight: hooks call
    # `get_kernel_impl`, which calls `ensure_backend_loaded`, and that would
    # wait on the same loader event before the loader can finish. The final
    # hook fired by `ensure_backend_loaded` after `loader()` returns rebinds
    # aliases once the backend is fully registered.
    if not backend_is_loading:
        _fire_global_backend_change_hooks()


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
        _invalidate_resolve_cache()


def register_backend_package(
    backend: str,
    package: str,
    *,
    availability_check: AvailabilityCheck | None = None,
) -> None:
    backend_name = _normalize_backend_name(backend)
    if backend_name is None:
        raise ValueError("Backend name must not be empty.")
    package_name = str(package).strip()
    if not package_name:
        raise ValueError("Backend package must not be empty.")

    def _load_package() -> None:
        module = importlib.import_module(package_name)
        load_backend = getattr(module, "load_backend", None)
        if callable(load_backend):
            load_backend()

    with _LOCK:
        _BACKEND_PACKAGES[backend_name] = package_name
    register_builtin_backend_loader(
        backend_name,
        _load_package,
        availability_check = availability_check,
    )


def ensure_backend_loaded(backend: str) -> None:
    backend_name = _normalize_backend_name(backend)
    if backend_name is None:
        return

    # Backend packages can import dispatcher modules while their loader is
    # still executing. Those modules may immediately resolve aliases via
    # `get_kernel_impl` / `get_kernel_backend`, which comes back here for the
    # same backend. Waiting on our own loader event would deadlock. Treat the
    # recursive same-thread lookup as "not loaded yet"; callers will either
    # see an incomplete registry and fall back temporarily, or the final hook
    # below will rebind aliases once the loader finishes.
    thread_loading = getattr(_THREAD_LOCAL, "loading_backends", set())
    if backend_name in thread_loading:
        return

    # Hot-path: already loaded successfully. `set.__contains__` is GIL-atomic
    # for str keys; no lock needed. This skips ~99% of dispatch_kernel calls
    # past the first per backend.
    if backend_name in _LOADED_BACKENDS:
        return

    with _LOCK:
        if backend_name in _BACKEND_LOAD_RESULT:
            cached = _BACKEND_LOAD_RESULT[backend_name]
            if cached is not None:
                raise cached
            _LOADED_BACKENDS.add(backend_name)
            return
        loader = _BUILTIN_LOADERS.get(backend_name)
        if loader is None:
            _BACKEND_LOAD_RESULT[backend_name] = None
            _LOADED_BACKENDS.add(backend_name)
            return
        availability_check = _AVAILABILITY_CHECKS.get(backend_name)
        event = _LOADING_BACKENDS.get(backend_name)
        if event is None:
            event = threading.Event()
            _LOADING_BACKENDS[backend_name] = event
            should_load = True
        else:
            should_load = False

    if not should_load:
        event.wait()
        with _LOCK:
            cached = _BACKEND_LOAD_RESULT.get(backend_name)
        if cached is not None:
            raise cached
        return

    exc_to_cache: BaseException | None = None
    newly_loaded = False
    try:
        loading_set = set(thread_loading)
        loading_set.add(backend_name)
        _THREAD_LOCAL.loading_backends = loading_set
        # Surface unavailability with a clear error before triggering the
        # backend package's import-time setup (which can crash on missing
        # native deps such as cuda.tile).
        if availability_check is not None:
            try:
                available, reason = availability_check()
            except Exception as exc:
                available, reason = False, str(exc)
            if not available:
                raise ImportError(
                    f"Backend '{backend_name}' is not available: {reason}"
                )
        loader()
    except BaseException as exc:
        exc_to_cache = exc
        raise
    finally:
        loading_set = set(getattr(_THREAD_LOCAL, "loading_backends", set()))
        loading_set.discard(backend_name)
        _THREAD_LOCAL.loading_backends = loading_set
        with _LOCK:
            _BACKEND_LOAD_RESULT[backend_name] = exc_to_cache
            pending = _LOADING_BACKENDS.pop(backend_name, None)
            if exc_to_cache is None:
                # Promote to lock-free fast-path set so future dispatches
                # skip the entire `with _LOCK` block.
                if backend_name not in _LOADED_BACKENDS:
                    newly_loaded = True
                _LOADED_BACKENDS.add(backend_name)
        if pending is not None:
            pending.set()
        # Backend just registered its kernels via `load_backend` — give the
        # dispatcher modules a chance to rebind `_resolved_<name>` aliases.
        if newly_loaded:
            _fire_global_backend_change_hooks()


def clear_backend_load_cache(backend: str | None = None) -> None:
    """Clear cached backend load results so the next dispatch retries the loader.

    Pass a backend name to clear only that backend, or omit to clear all.
    Useful in tests, and for users who fix a broken install mid-process.
    Does not interrupt an in-flight load.
    """
    with _LOCK:
        if backend is None:
            _BACKEND_LOAD_RESULT.clear()
            _LOADED_BACKENDS.clear()
            _invalidate_resolve_cache()
        else:
            backend_name = _normalize_backend_name(backend)
            if backend_name is not None:
                _BACKEND_LOAD_RESULT.pop(backend_name, None)
                _LOADED_BACKENDS.discard(backend_name)
                _invalidate_resolve_cache()
    _fire_global_backend_change_hooks()


def is_kernel_backend_available(backend: str) -> tuple[bool, str | None]:
    backend_name = _normalize_backend_name(backend)
    if backend_name is None:
        return False, "Backend name is empty."

    availability_check = _AVAILABILITY_CHECKS.get(backend_name)
    if availability_check is None:
        return True, None
    try:
        return availability_check()
    except Exception as exc:
        return False, str(exc)


def _get_requested_backend(
    kernel_name: str, backend: str | None = None
) -> tuple[str, bool]:
    """Resolve the requested backend and whether the request was explicit.

    "Explicit" means the user expressed intent through any of:
      - the ``backend=`` kwarg
      - ``set_kernel_backend_for_op`` / ``set_kernel_backend`` /
        ``kernel_backend_context``
      - ``UNSLOTH_KERNEL_BACKEND_OVERRIDES`` / ``UNSLOTH_KERNEL_BACKEND``

    Only the hardcoded ``DEFAULT_KERNEL_BACKEND`` is treated as implicit.
    Explicit requests skip the configured fallback (e.g. triton) and go
    straight to eager when the requested backend is unavailable, so a user
    who asks for cutile never silently runs Triton.
    """
    explicit_backend = _normalize_backend_name(backend)
    if explicit_backend is not None:
        return explicit_backend, True

    runtime_override = _RUNTIME_KERNEL_OVERRIDES.get().get(kernel_name)
    if runtime_override is not None:
        return runtime_override, True

    env_override = _get_env_overrides().get(kernel_name)
    if env_override is not None:
        return env_override, True

    runtime_global = _RUNTIME_GLOBAL_BACKEND.get()
    if runtime_global is not None:
        return runtime_global, True

    env_global = _normalize_backend_name(os.environ.get(GLOBAL_BACKEND_ENV))
    if env_global is not None:
        return env_global, True

    return DEFAULT_KERNEL_BACKEND, False


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
    *,
    explicit_request: bool,
) -> list[str]:
    """Return the ordered list of backends to try for a dispatch.

    For an *explicit* request (``backend=`` kwarg, env vars, runtime setters)
    the chain is intentionally short: ``[requested, eager]``. A user who asked
    for cutile must never silently end up running triton.

    For an *implicit* request (no user signal — only the hardcoded
    ``DEFAULT_KERNEL_BACKEND``) the chain walks every other registered
    builtin backend in registration order before falling to eager. This keeps
    ``pip install unsloth[cutile]`` working out of the box: triton is
    unavailable, the chain falls through to cutile (next registered builtin)
    rather than silently dropping to eager. Triton-installed users see no
    change because triton is the first candidate and succeeds.
    """
    ordered_backends = [preferred_backend]
    if not explicit_request:
        ordered_backends.append(fallback_backend)
        # Snapshot keys under the lock to avoid racing a concurrent
        # register_builtin_backend_loader call. Reads of _BUILTIN_LOADERS
        # elsewhere also rely on stable ordering established at registration.
        with _LOCK:
            registered_builtins = tuple(_BUILTIN_LOADERS)
        ordered_backends.extend(registered_builtins)
    ordered_backends.append(EAGER_KERNEL_BACKEND)

    candidates: list[str] = []
    for backend_name in ordered_backends:
        normalized = _normalize_backend_name(backend_name)
        if normalized is None or normalized in candidates:
            continue
        candidates.append(normalized)
    return candidates


def _backend_status_for_kernel(
    kernel_name: str,
    backend_name: str,
) -> tuple[bool, str | None]:
    available, reason = is_kernel_backend_available(backend_name)
    if not available:
        return False, reason

    try:
        ensure_backend_loaded(backend_name)
    except Exception as exc:
        return False, f"Failed to load backend '{backend_name}': {exc}"

    if backend_name not in _REGISTRY.get(kernel_name, {}):
        return (
            False,
            f"Kernel '{kernel_name}' has no registered '{backend_name}' implementation.",
        )
    return True, None


def get_kernel_backend(
    name: str,
    *,
    backend: str | None = None,
    fallback_backend: str = DEFAULT_KERNEL_BACKEND,
) -> str:
    kernel_name = _normalize_kernel_name(name)
    explicit_backend = _normalize_backend_name(backend)

    # Hot-path cache. Skipped when per-thread overrides are in play because
    # those mutate via ContextVar without notifying the cache. Env vars are
    # baked into the cache key so a mid-run env change naturally misses.
    runtime_global = _RUNTIME_GLOBAL_BACKEND.get()
    runtime_overrides = _RUNTIME_KERNEL_OVERRIDES.get()
    if not runtime_global and not runtime_overrides and not _strict_mode_enabled():
        cache_key = (
            kernel_name,
            explicit_backend,
            fallback_backend,
            os.environ.get(GLOBAL_BACKEND_ENV),
            os.environ.get(OVERRIDES_ENV),
        )
        cached = _RESOLVE_CACHE.get(cache_key)
        if cached is not None:
            return cached
    else:
        cache_key = None

    preferred_backend, is_explicit = _get_requested_backend(
        kernel_name, backend = backend,
    )
    fallback_backend_name = _normalize_backend_name(fallback_backend) or DEFAULT_KERNEL_BACKEND
    available_backends = sorted(_REGISTRY.get(kernel_name, {}))
    available, reason = _backend_status_for_kernel(kernel_name, preferred_backend)
    if available:
        if cache_key is not None:
            _RESOLVE_CACHE[cache_key] = preferred_backend
        return preferred_backend

    if _is_backend_loading_in_current_thread(preferred_backend):
        raise NotImplementedError(
            f"Kernel backend '{preferred_backend}' is still loading for "
            f"'{kernel_name}'. Deferring resolution until backend load completes."
        )

    if _strict_mode_enabled():
        detail = f": {reason}" if reason else ""
        raise RuntimeError(
            f"Kernel backend '{preferred_backend}' is unavailable for '{kernel_name}'{detail}"
        )

    for candidate_backend in _candidate_backends(
        preferred_backend,
        fallback_backend_name,
        explicit_request = is_explicit,
    ):
        if candidate_backend == preferred_backend:
            continue
        candidate_available, _ = _backend_status_for_kernel(kernel_name, candidate_backend)
        if not candidate_available:
            continue
        _warn_fallback_once(kernel_name, preferred_backend, candidate_backend)
        if cache_key is not None:
            _RESOLVE_CACHE[cache_key] = candidate_backend
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
    """Set the runtime global kernel backend. Setter only — returns ``None``.

    This is NOT a context manager. ``with set_kernel_backend("cutile"):`` will
    raise ``TypeError`` because ``None`` is not a context manager. For a
    scoped override that auto-restores on exit, use
    :func:`kernel_backend_context` instead::

        with kernel_backend_context(global_backend="cutile"):
            ...

    Scope: applies to the current contextvar context only. New threads,
    forked subprocesses (e.g. ``torch.utils.data.DataLoader`` workers,
    Ray actors), and ``multiprocessing`` children do NOT inherit it. For
    process-wide / cross-process configuration, set the
    ``UNSLOTH_KERNEL_BACKEND`` environment variable before spawning — the
    env var is read by every dispatch and inherited by subprocesses.
    """
    _RUNTIME_GLOBAL_BACKEND.set(_normalize_backend_name(backend))
    _fire_global_backend_change_hooks()


def set_kernel_backend_for_op(name: str, backend: str | None) -> None:
    """Set the runtime backend for a single kernel. Setter only — returns ``None``.

    This is NOT a context manager. For a scoped per-op override use
    :func:`kernel_backend_context` with ``overrides={...}`` instead::

        with kernel_backend_context(overrides={"rope_embedding": "cutile"}):
            ...

    Scope: current contextvar context only (see :func:`set_kernel_backend`
    for thread / subprocess caveats). For process-wide per-op overrides
    use the ``UNSLOTH_KERNEL_BACKEND_OVERRIDES`` env var (format:
    ``rope_embedding=cutile,layernorm=triton``).
    """
    kernel_name = _normalize_kernel_name(name)
    normalized_backend = _normalize_backend_name(backend)
    overrides = dict(_RUNTIME_KERNEL_OVERRIDES.get())
    if normalized_backend is None:
        overrides.pop(kernel_name, None)
    else:
        overrides[kernel_name] = normalized_backend
    _RUNTIME_KERNEL_OVERRIDES.set(overrides)
    _fire_global_backend_change_hooks()


def set_kernel_backends(
    *,
    global_backend: str | None = None,
    overrides: dict[str, str | None] | None = None,
) -> None:
    """Set the global backend and/or per-kernel overrides. Setter only — returns ``None``.

    This is NOT a context manager. For scoped, auto-restored configuration use
    :func:`kernel_backend_context`, which takes the same ``global_backend`` and
    ``overrides`` keyword arguments::

        with kernel_backend_context(global_backend="cutile",
                                    overrides={"rope_embedding": "triton"}):
            ...

    Scope: current contextvar context only (see :func:`set_kernel_backend`
    for thread / subprocess caveats). For process-wide configuration use
    the ``UNSLOTH_KERNEL_BACKEND`` and ``UNSLOTH_KERNEL_BACKEND_OVERRIDES``
    environment variables.
    """
    set_kernel_backend(global_backend)
    if overrides is None:
        return
    for kernel_name, backend_name in overrides.items():
        set_kernel_backend_for_op(kernel_name, backend_name)


def clear_kernel_backend_overrides() -> None:
    """Clear the runtime global backend and all per-kernel overrides.

    Setter only — returns ``None``. Resets the runtime contextvars to their
    default empty state. Environment-variable backends
    (``UNSLOTH_KERNEL_BACKEND``, ``UNSLOTH_KERNEL_BACKEND_OVERRIDES``) are not
    affected.
    """
    _RUNTIME_GLOBAL_BACKEND.set(None)
    _RUNTIME_KERNEL_OVERRIDES.set({})
    _fire_global_backend_change_hooks()


_UNSET: Any = object()


@contextlib.contextmanager
def kernel_backend_context(
    *,
    global_backend: str | None = _UNSET,
    overrides: dict[str, str | None] | None = None,
):
    """Scoped kernel-backend override; restores the previous state on exit.

    Use this instead of :func:`set_kernel_backend` /
    :func:`set_kernel_backend_for_op` / :func:`set_kernel_backends` when you
    want the override to last only for a block of code. Example::

        with kernel_backend_context(global_backend="cutile",
                                    overrides={"rope_embedding": "triton"}):
            run_step()
        # Outside the block the previous backend state is restored.

    Semantics for ``global_backend``:

    * Omitted (default sentinel) → **preserve** the outer global backend.
      An outer ``set_kernel_backend("cutile")`` keeps applying inside the
      scope.
    * ``None`` → **explicitly clear** the outer global backend within the
      scope (matches :func:`set_kernel_backend(None)`).
    * Any string → set the global backend to that value within the scope.

    Semantics for ``overrides``:

    * Omitted / ``None`` → preserve the outer per-kernel overrides.
    * Dict → merge with the outer overrides for the duration of the block.
      Passing ``None`` for a given kernel name clears that single override
      inside the scope.
    """
    if global_backend is _UNSET:
        global_token = None
    else:
        global_token = _RUNTIME_GLOBAL_BACKEND.set(_normalize_backend_name(global_backend))
    merged = dict(_RUNTIME_KERNEL_OVERRIDES.get())
    if overrides:
        for kernel_name, backend_name in overrides.items():
            normalized_kernel = _normalize_kernel_name(kernel_name)
            normalized_backend = _normalize_backend_name(backend_name)
            if normalized_backend is None:
                merged.pop(normalized_kernel, None)
            else:
                merged[normalized_kernel] = normalized_backend
    overrides_token = _RUNTIME_KERNEL_OVERRIDES.set(merged)
    # Fire after both contextvars are mutated so hooks observe the entered
    # state. Per-op overrides can change individual `_resolved_<name>` aliases
    # even when the global default is preserved.
    hooks_need_fire = global_token is not None or overrides is not None
    if hooks_need_fire:
        _fire_global_backend_change_hooks()
    try:
        yield
    finally:
        _RUNTIME_KERNEL_OVERRIDES.reset(overrides_token)
        if global_token is not None:
            _RUNTIME_GLOBAL_BACKEND.reset(global_token)
        if hooks_need_fire:
            # Fire on exit too so the outer context's resolved backend and
            # per-op aliases are restored.
            _fire_global_backend_change_hooks()


def get_registered_kernel_backends(name: str | None = None) -> dict[str, list[str]] | list[str]:
    if name is not None:
        kernel_name = _normalize_kernel_name(name)
        return sorted(_REGISTRY.get(kernel_name, {}))
    return {
        kernel_name: sorted(backends)
        for kernel_name, backends in sorted(_REGISTRY.items())
    }


def get_kernel_backend_state() -> dict[str, Any]:
    env_overrides = _get_env_overrides()
    return {
        "default_backend": DEFAULT_KERNEL_BACKEND,
        "eager_backend": EAGER_KERNEL_BACKEND,
        "env_global_backend": _normalize_backend_name(os.environ.get(GLOBAL_BACKEND_ENV)),
        "env_overrides": dict(sorted(env_overrides.items())),
        "runtime_global_backend": _RUNTIME_GLOBAL_BACKEND.get(),
        "runtime_overrides": dict(sorted(_RUNTIME_KERNEL_OVERRIDES.get().items())),
        "strict": _strict_mode_enabled(),
    }


def describe_kernel_backends() -> dict[str, Any]:
    load_errors: dict[str, str] = {}
    for backend_name in sorted(_BUILTIN_LOADERS):
        available, _ = is_kernel_backend_available(backend_name)
        if available:
            try:
                ensure_backend_loaded(backend_name)
            except Exception as exc:
                load_errors[backend_name] = str(exc)

    backend_names = {
        DEFAULT_KERNEL_BACKEND,
        EAGER_KERNEL_BACKEND,
        *(_BUILTIN_LOADERS.keys()),
        *(backend for backends in _REGISTRY.values() for backend in backends.keys()),
    }
    backends = {}
    for backend_name in sorted(backend_names):
        available, reason = is_kernel_backend_available(backend_name)
        if backend_name in load_errors:
            available = False
            reason = f"Backend load failed: {load_errors[backend_name]}"
        registered_ops = [
            kernel_name
            for kernel_name, implementations in sorted(_REGISTRY.items())
            if backend_name in implementations
        ]
        backends[backend_name] = {
            "available": available,
            "reason": reason,
            "loaded": _BACKEND_LOAD_RESULT.get(backend_name) is None
                       and backend_name in _BACKEND_LOAD_RESULT,
            "package": _BACKEND_PACKAGES.get(backend_name),
            "registered_op_count": len(registered_ops),
            "registered_ops": registered_ops,
        }
    availability = {
        backend_name: {
            "available": data["available"],
            "reason": data["reason"],
        }
        for backend_name, data in backends.items()
    }
    return {
        "state": get_kernel_backend_state(),
        "availability": availability,
        "backends": backends,
        "registry": get_registered_kernel_backends(),
    }

def _check_cutile_backend() -> tuple[bool, str | None]:
    try:
        importlib.import_module("cuda.tile")
    except Exception as exc:
        return False, f"Missing cuda.tile: {exc}"
    return True, None


def _check_triton_backend() -> tuple[bool, str | None]:
    # find_spec is not enough: a partial install (ABI mismatch, missing .so)
    # leaves the spec in place but raises on import. Mirror _check_cutile_backend
    # and actually attempt the import so describe_kernel_backends() doesn't
    # report a broken triton as available. After first call, sys.modules
    # caches the result so subsequent dispatches are a dict hit.
    try:
        importlib.import_module("triton")
    except Exception as exc:
        return False, f"Triton import failed: {exc}"
    return True, None


register_backend_package(
    "triton",
    "unsloth.kernels.backends.triton",
    availability_check = _check_triton_backend,
)

register_backend_package(
    "cutile",
    "unsloth.kernels.backends.cutile",
    availability_check = _check_cutile_backend,
)
