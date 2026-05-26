"""Independent copula.

# C++ parity: ql/math/copulas/independentcopula.hpp + independentcopula.cpp (v1.42.1).
"""

from __future__ import annotations

from dataclasses import dataclass

from pquantlib import qassert


@dataclass(frozen=True, slots=True)
class IndependentCopula:
    """Independent bivariate copula: ``C(x, y) = x * y``."""

    def __call__(self, x: float, y: float) -> float:
        qassert.require(0.0 <= x <= 1.0, f"1st argument ({x}) must be in [0,1]")
        qassert.require(0.0 <= y <= 1.0, f"2nd argument ({y}) must be in [0,1]")
        return x * y
