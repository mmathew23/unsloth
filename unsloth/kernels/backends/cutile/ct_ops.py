# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CuTile helper ops and shared utilities for unsloth kernels.

Includes:
  - Host-side utilities: next_power_of_2, cdiv, calculate_settings, autotune_configs, select_launch_config
  - Device-side helpers (for @ct.kernel): erf_ct
"""

import math
import os
from types import SimpleNamespace

import cuda.tile as ct

# ---- Host-side utilities ----

MAX_FUSED_SIZE = 65536  # 2**16


def next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def cdiv(a, b):
    """Ceiling division."""
    return math.ceil(a / b)


def calculate_settings(n):
    """Calculate BLOCK_SIZE (next power of 2) for a given dimension n.

    Raises RuntimeError if n exceeds MAX_FUSED_SIZE.
    """
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch kernel since n = {n} exceeds the maximum blocksize = {MAX_FUSED_SIZE}.")
    return BLOCK_SIZE


def autotune_configs():
    """Yield standard occupancy configs for CuTile launches."""
    for occ in [1, 2, 4, 8]:
        yield SimpleNamespace(occupancy=occ)


# UNSLOTH_TILEGYM_DIFF_REVIEW: TileGym origin/main calls
# cuda.tile.tune.exhaustive_search at each tuned launch site. That is useful
# for profiling fixed shapes, but Unsloth padding-free training creates many
# runtime shapes and the repeated searches dominated wall time. Production
# defaults to the first config; set UNSLOTH_CUTILE_EXHAUSTIVE_TUNE=1 to review
# or profile the exact TileGym exhaustive-tuning behavior.
def select_launch_config(configs, stream, grid_fn, kernel, args_fn, hints_fn):
    """Select a CuTile launch config without benchmarking by default.

    TileGym's published kernels use cuda.tile.tune.exhaustive_search, but
    Unsloth's padding-free training sees many runtime shapes. Exhaustively
    benchmarking every new shape dominates runtime, so the production default
    keeps the old single-config behavior. Set UNSLOTH_CUTILE_EXHAUSTIVE_TUNE=1
    for intentional profiling/tuning runs.
    """
    configs = list(configs)
    if os.environ.get("UNSLOTH_CUTILE_EXHAUSTIVE_TUNE", "0") == "1":
        from cuda.tile.tune import exhaustive_search

        return exhaustive_search(configs, stream, grid_fn, kernel, args_fn, hints_fn)
    return SimpleNamespace(best=SimpleNamespace(config=configs[0]))


# ---- Device-side helpers (for use inside @ct.kernel) ----


def erf_ct(x):
    """Element-wise erf(x) via Abramowitz & Stegun polynomial approximation.

    Maximum error: |ε(x)| ≤ 1.5 × 10⁻⁷
    Reference: Abramowitz & Stegun, formula 7.1.26
    Source: src/tilegym/suites/unsloth/cutile/geglu.py

    Args:
        x: CuTile float32 tensor of any shape.

    Returns:
        Approximation of erf(x) with same shape as x.
    """
    abs_x = ct.maximum(x, ct.negative(x))
    t_denom = 1.0 + 0.3275911 * abs_x
    # 1/t via rsqrt: rsqrt(t_denom^2) = 1/|t_denom|; t_denom > 0 always
    t = ct.rsqrt(t_denom * t_denom)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    poly = 0.254829592 * t + (-0.284496736) * t2 + 1.421413741 * t3 + (-1.453152027) * t4 + 1.061405429 * t5
    erf_abs = 1.0 - poly * ct.exp(ct.negative(abs_x * abs_x))
    # erf is odd function
    return ct.where(x < 0.0, ct.negative(erf_abs), erf_abs)
