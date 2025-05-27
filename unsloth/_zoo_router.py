# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is a router for the unsloth_zoo package.
Choose behaviour with UNSLOTH_ZOO_MODE:
    external   -- use the pypi package first, fall back to vendored
    mixed      -- pypi package first except dotted names in _OVERRIDES
    local      -- local unsloth_zoo first, fall back to pypi package
    off        -- ignore pypi package completely, local unsloth_zoo only

This will eventually be removed, and unsloth_zoo will live inside the 
unsloth package.
"""

import importlib.abc
import importlib.machinery
import os
import sys
import pathlib
from typing import List


_MODE: str = os.getenv("UNSLOTH_ZOO_MODE", "external").lower()
_OVERRIDES: List[str] = []

_VALID_MODES = {"external", "mixed", "local", "off"}
if _MODE not in _VALID_MODES:
    raise ValueError(
        f"Invalid UNSLOTH_ZOO_MODE='{_MODE}'. "
        f"Must be one of: {', '.join(_VALID_MODES)}"
    )


# Absolute path to the vendored copy:  …/unsloth/unsloth_zoo
_LOCAL_ROOT: pathlib.Path = pathlib.Path(__file__).parent / "unsloth_zoo"
_LOCAL_PARENT = _LOCAL_ROOT.parent.resolve()      # …/unsloth

local_root_exists = _LOCAL_ROOT.is_dir()

# Verify vendored directory exists
if not local_root_exists:
    print(f"Unsloth: Local unsloth_zoo not found at {_LOCAL_ROOT}. "
          f"Using pypi package instead.")
    _MODE = "external"
    os.environ["UNSLOTH_ZOO_MODE"] = "external"


def _want_local(fullname: str) -> bool:
    """Return True if this import should resolve to the vendored copy."""
    if _MODE in ("off", "local"):
        return True
    if _MODE == "external":
        return False

    rel = fullname.split(".", 1)[1] if fullname != "unsloth_zoo" else ""
    for ov in _OVERRIDES:
        if rel == ov or rel.startswith(f"{ov}."):
            return True
    return False


class _ZooRouter(importlib.abc.MetaPathFinder):
    """Route between local unsloth_zoo and pypi package."""

    def _external_spec(self, fullname: str, pkg_path):
        """
        Search for the external copy, honouring the package search path
        if we're resolving a sub-module.
        """
        base_paths = list(pkg_path) if pkg_path is not None else sys.path
        search_paths = [
            p for p in base_paths
            if pathlib.Path(p).resolve() not in {_LOCAL_ROOT.resolve(), _LOCAL_PARENT}
        ]
        return importlib.machinery.PathFinder.find_spec(fullname, search_paths)

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("unsloth_zoo"):
            return None

        want_local = _want_local(fullname)

        if want_local:
            spec = importlib.machinery.PathFinder.find_spec(
                fullname, [str(_LOCAL_ROOT)]
            )
            if spec:
                return spec

            if _MODE == "off":
                raise ModuleNotFoundError(
                    f"{fullname!r} not found in vendored unsloth_zoo "
                    "and UNSLOTH_ZOO_MODE='off' forbids external fallback"
                )

        return self._external_spec(fullname, path)


# Install the finder before everything else
# But check if already installed to avoid duplicates
if not any(isinstance(finder, _ZooRouter) for finder in sys.meta_path):
    sys.meta_path.insert(0, _ZooRouter())
