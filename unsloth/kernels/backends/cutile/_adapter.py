# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Internal adapter surface for CuTile kernel registration."""

# UNSLOTH_TILEGYM_DIFF_REVIEW: TileGym imports register_impl from
# tilegym.backend. Local Unsloth kernels route registration into
# unsloth.kernels._backend_registry instead.

from ..._backend_registry import register_kernel_backend

BACKEND_NAME = "cutile"


def register_impl(name: str, *, backend: str = BACKEND_NAME):
    if backend != BACKEND_NAME:
        raise ValueError(
            f"CuTile adapter can only register backend={BACKEND_NAME!r}, got {backend!r}."
        )

    def decorator(func):
        register_kernel_backend(name, BACKEND_NAME, func)
        return func

    return decorator
