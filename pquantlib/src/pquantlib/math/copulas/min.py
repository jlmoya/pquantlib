"""Min copula (lower Frechet bound).

# C++ parity: ql/math/copulas/mincopula.hpp + mincopula.cpp (v1.42.1).

Per the C++ source: ``MinCopula::operator()(x, y)`` returns ``max(x+y-1, 0)`` —
the lower Frechet-Hoeffding bound (the "minimum" of the dependence ordering).
"""

from __future__ import annotations

from dataclasses import dataclass

from pquantlib import qassert


@dataclass(frozen=True, slots=True)
class MinCopula:
    """Lower Frechet bound: ``C(x, y) = max(x + y - 1, 0)``."""

    def __call__(self, x: float, y: float) -> float:
        qassert.require(0.0 <= x <= 1.0, f"1st argument ({x}) must be in [0,1]")
        qassert.require(0.0 <= y <= 1.0, f"2nd argument ({y}) must be in [0,1]")
        return max(x + y - 1.0, 0.0)
