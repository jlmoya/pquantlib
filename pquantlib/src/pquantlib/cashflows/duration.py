"""Duration enum.

# C++ parity: ql/cashflows/duration.hpp (v1.42.1).

C++ uses ``struct Duration { enum Type { Simple, Macaulay, Modified }; };``.
The nested-enum-inside-struct idiom is Python-translated as a plain
``IntEnum`` named ``Duration`` — callers write ``Duration.Simple`` /
``Duration.Macaulay`` / ``Duration.Modified``.
"""

from __future__ import annotations

from enum import IntEnum


class Duration(IntEnum):
    """Mirrors ``QuantLib::Duration::Type`` (v1.42.1 ql/cashflows/duration.hpp:36)."""

    Simple = 0
    Macaulay = 1
    Modified = 2
