"""Internal adapter surface for Triton kernel registration."""

from ..._backend_registry import register_kernel_backend

BACKEND_NAME = "triton"


def register_impl(name: str, *, backend: str = BACKEND_NAME):
    if backend != BACKEND_NAME:
        raise ValueError(
            f"Triton adapter can only register backend={BACKEND_NAME!r}, got {backend!r}."
        )

    def decorator(func):
        register_kernel_backend(name, BACKEND_NAME, func)
        return func

    return decorator
