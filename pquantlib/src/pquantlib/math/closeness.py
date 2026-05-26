"""Floating-point closeness predicates.

# C++ parity: ql/math/comparison.hpp ``close(x, y[, n])`` + ``close_enough(x, y[, n])`` (v1.42.1).

Both follow the Knuth-style closeness check, but differ in the connective:
- ``close`` uses logical AND (|x-y| <= tol*|x| AND |x-y| <= tol*|y|).
- ``close_enough`` uses logical OR (|x-y| <= tol*|x| OR |x-y| <= tol*|y|).

The tolerance is ``n * QL_EPSILON`` where n defaults to 42. When either x
or y is exactly zero, the bound is ``tolerance * tolerance`` (a stricter
absolute check, since the relative form collapses to 0).

Equal inputs (including the +inf/-inf case) short-circuit to True.
"""

from __future__ import annotations

import math

from pquantlib.math.constants import QL_EPSILON

_DEFAULT_N: int = 42


def close(x: float, y: float, n: int = _DEFAULT_N) -> bool:
    """Knuth-style relative closeness with logical AND."""
    if x == y:
        return True
    diff = math.fabs(x - y)
    tolerance = n * QL_EPSILON
    if x == 0.0 or y == 0.0:
        return diff < tolerance * tolerance
    return diff <= tolerance * math.fabs(x) and diff <= tolerance * math.fabs(y)


def close_enough(x: float, y: float, n: int = _DEFAULT_N) -> bool:
    """Knuth-style relative closeness with logical OR (more permissive)."""
    if x == y:
        return True
    diff = math.fabs(x - y)
    tolerance = n * QL_EPSILON
    if x == 0.0 or y == 0.0:
        return diff < tolerance * tolerance
    return diff <= tolerance * math.fabs(x) or diff <= tolerance * math.fabs(y)
