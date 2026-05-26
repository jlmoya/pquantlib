"""Marshall-Olkin copula.

# C++ parity: ql/math/copulas/marshallolkincopula.hpp + marshallolkincopula.cpp (v1.42.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pquantlib import qassert


@dataclass(frozen=True, slots=True)
class MarshallOlkinCopula:
    """Marshall-Olkin bivariate copula.

    ``C(x, y) = min(y * x^(1-alpha1), x * y^(1-alpha2))``
    for ``alpha1, alpha2 >= 0``.

    The C++ constructor stores ``a1_ = 1.0 - alpha1`` and ``a2_ = 1.0 - alpha2``
    as private members and uses them directly in ``operator()``. The Python
    dataclass keeps the user-facing ``alpha1``/``alpha2`` arguments as fields
    and computes ``1-alpha`` lazily in ``__call__`` (cheap, no per-call alloc).
    """

    alpha1: float
    alpha2: float
    _a1: float = field(init=False, repr=False, compare=False)
    _a2: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        qassert.require(self.alpha1 >= 0.0, f"1st parameter ({self.alpha1}) must be non-negative")
        qassert.require(self.alpha2 >= 0.0, f"2nd parameter ({self.alpha2}) must be non-negative")
        # Bypass frozen to cache derived fields, matching the C++ ctor's
        # ``a1_(1.0-a1), a2_(1.0-a2)`` initializer list semantics.
        object.__setattr__(self, "_a1", 1.0 - self.alpha1)
        object.__setattr__(self, "_a2", 1.0 - self.alpha2)

    def __call__(self, x: float, y: float) -> float:
        qassert.require(0.0 <= x <= 1.0, f"1st argument ({x}) must be in [0,1]")
        qassert.require(0.0 <= y <= 1.0, f"2nd argument ({y}) must be in [0,1]")
        return min(y * (x**self._a1), x * (y**self._a2))
