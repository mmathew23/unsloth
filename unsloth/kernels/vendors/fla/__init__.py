# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
FLA vendor integration namespace.

Upstream module copies are mirrored under:
`unsloth.kernels.vendors.fla.modules`
"""

from .rotary import (
    fla_rotary_embedding,
    fla_rotary_embedding_qk,
    fla_rotary_embedding_qk_positions,
)

__all__ = [
    "fla_rotary_embedding",
    "fla_rotary_embedding_qk",
    "fla_rotary_embedding_qk_positions",
]
