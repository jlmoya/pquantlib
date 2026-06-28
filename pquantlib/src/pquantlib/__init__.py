"""PQuantLib — Python port of QuantLib v1.42.1.

See the project README for migration discipline, ground-truth principle,
and the per-phase design/plan/completion docs under ``docs/migration/``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__: str = _version("pquantlib")
except PackageNotFoundError:  # raw source tree, not installed
    __version__ = "0.0.0+unknown"

__all__: list[str] = ["__version__"]
