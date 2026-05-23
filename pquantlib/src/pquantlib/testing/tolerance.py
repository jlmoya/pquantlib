"""Tolerance-tier assertions for cross-validation tests.

# C++ parity: none — this is harness, not a port.

Tiers (from CLAUDE.md operational rules):

- ``exact(actual, expected)`` — bit-identical via ``struct.pack('!d', x)``.
- ``tight(actual, expected)`` — ``math.isclose(abs_tol=1e-14, rel_tol=1e-12)``.
- ``loose(actual, expected)`` — ``math.isclose(abs_tol=1e-8, rel_tol=1e-8)``.

Per-test exceptions go through ``custom(..., reason="...")`` with a
mandatory written rationale.
"""

from __future__ import annotations

import math
import struct
from typing import Final

_TIGHT_ABS: Final[float] = 1e-14
_TIGHT_REL: Final[float] = 1e-12
_LOOSE_ABS: Final[float] = 1e-8
_LOOSE_REL: Final[float] = 1e-8


def _format_reason(reason: str | None) -> str:
    return f" (reason: {reason})" if reason else ""


def exact(actual: float, expected: float, *, reason: str | None = None) -> None:
    """Assert ``actual`` and ``expected`` are bit-identical IEEE-754 doubles."""
    if struct.pack("!d", actual) != struct.pack("!d", expected):
        raise AssertionError(
            f"EXACT tier mismatch: actual={actual!r} expected={expected!r}" + _format_reason(reason)
        )


def tight(actual: float, expected: float, *, reason: str | None = None) -> None:
    """Assert within TIGHT tier (abs_tol=1e-14, rel_tol=1e-12)."""
    if not math.isclose(actual, expected, abs_tol=_TIGHT_ABS, rel_tol=_TIGHT_REL):
        raise AssertionError(
            f"TIGHT tier mismatch: actual={actual!r} expected={expected!r}" + _format_reason(reason)
        )


def loose(actual: float, expected: float, *, reason: str | None = None) -> None:
    """Assert within LOOSE tier (abs_tol=1e-8, rel_tol=1e-8)."""
    if not math.isclose(actual, expected, abs_tol=_LOOSE_ABS, rel_tol=_LOOSE_REL):
        raise AssertionError(
            f"LOOSE tier mismatch: actual={actual!r} expected={expected!r}" + _format_reason(reason)
        )


def custom(
    actual: float,
    expected: float,
    *,
    abs_tol: float,
    rel_tol: float,
    reason: str,
) -> None:
    """Assert within a custom tolerance with a mandatory written ``reason``."""
    if not math.isclose(actual, expected, abs_tol=abs_tol, rel_tol=rel_tol):
        raise AssertionError(
            f"custom tier mismatch (abs_tol={abs_tol}, rel_tol={rel_tol}, "
            f"reason: {reason}): actual={actual!r} expected={expected!r}"
        )
