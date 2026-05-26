"""Max copula (upper Frechet bound).

# C++ parity: ql/math/copulas/maxcopula.hpp + maxcopula.cpp (v1.42.1).

Per the C++ source: ``MaxCopula::operator()(x, y)`` returns ``min(x, y)`` —
the upper Frechet-Hoeffding bound (the "maximum" of the dependence ordering,
not the maximum of the arguments).
"""

from __future__ import annotations

from dataclasses import dataclass

from pquantlib import qassert


@dataclass(frozen=True, slots=True)
class MaxCopula:
    """Upper Frechet bound: ``C(x, y) = min(x, y)``."""

    def __call__(self, x: float, y: float) -> float:
        qassert.require(0.0 <= x <= 1.0, f"1st argument ({x}) must be in [0,1]")
        qassert.require(0.0 <= y <= 1.0, f"2nd argument ({y}) must be in [0,1]")
        return min(x, y)
