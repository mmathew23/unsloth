import math
from types import SimpleNamespace


class _KernelStub:
    def __init__(self, fn):
        self.fn = fn
        self.__name__ = getattr(fn, "__name__", "kernel_stub")
        self.__doc__ = getattr(fn, "__doc__", None)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __getitem__(self, _):
        raise RuntimeError(
            "Triton is not available in this environment, so Triton kernels cannot be launched."
        )


def _identity_decorator(fn):
    return fn


def _decorator_factory(*args, **kwargs):
    def decorator(fn):
        if isinstance(fn, _KernelStub):
            return fn
        return _KernelStub(fn)

    return decorator


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


try:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]

    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

    class _TLMath:
        tanh = staticmethod(math.tanh)

    class _TLLibdevice:
        tanh = staticmethod(math.tanh)

    tl = SimpleNamespace(
        constexpr = object(),
        cast = staticmethod(lambda x, dtype = None: x),
        math = _TLMath(),
        extra = SimpleNamespace(intel = SimpleNamespace(libdevice = _TLLibdevice())),
    )

    triton = SimpleNamespace(
        __version__ = "0.0.0",
        jit = _decorator_factory,
        heuristics = _decorator_factory,
        autotune = _decorator_factory,
        cdiv = staticmethod(lambda x, y: (x + y - 1) // y),
        next_power_of_2 = staticmethod(_next_power_of_2),
        runtime = SimpleNamespace(
            driver = SimpleNamespace(
                active = SimpleNamespace(
                    get_current_target = staticmethod(
                        lambda: SimpleNamespace(arch = "unsupported")
                    )
                )
            )
        ),
    )


__all__ = ["HAS_TRITON", "triton", "tl"]
