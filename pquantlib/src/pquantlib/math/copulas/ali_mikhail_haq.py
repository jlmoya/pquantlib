"""Ali-Mikhail-Haq copula.

# C++ parity: ql/math/copulas/alimikhailhaqcopula.hpp + alimikhailhaqcopula.cpp (v1.42.1).
"""

from __future__ import annotations

from dataclasses import dataclass

from pquantlib import qassert


@dataclass(frozen=True, slots=True)
class AliMikhailHaqCopula:
    """Ali-Mikhail-Haq bivariate copula.

    ``C(x, y) = (x * y) / (1 - theta * (1 - x) * (1 - y))`` for ``theta in [-1, 1]``.
    """

    theta: float

    def __post_init__(self) -> None:
        qassert.require(-1.0 <= self.theta <= 1.0, f"theta ({self.theta}) must be in [-1,1]")

    def __call__(self, x: float, y: float) -> float:
        qassert.require(0.0 <= x <= 1.0, f"1st argument ({x}) must be in [0,1]")
        qassert.require(0.0 <= y <= 1.0, f"2nd argument ({y}) must be in [0,1]")
        return (x * y) / (1.0 - self.theta * (1.0 - x) * (1.0 - y))
