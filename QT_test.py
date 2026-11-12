"""Compatibility shim for legacy imports.

Tests and external scripts may still import QT_test after the main app file
was renamed to M(H)ero.py. This module loads M(H)ero.py dynamically and
re-exports its public symbols.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).with_name("M(H)ero.py")
_SPEC = importlib.util.spec_from_file_location("mhero_main", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load module from {_MODULE_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

for _name in dir(_MODULE):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_MODULE, _name)

__all__ = [name for name in globals() if not name.startswith("_")]


if __name__ == "__main__":
    _MODULE.main()
