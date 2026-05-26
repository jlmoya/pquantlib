"""Factorial numbers.

# C++ parity: ql/math/factorial.hpp + ql/math/factorial.cpp (v1.42.1).

C++ tabulates factorials 0..27 (28 entries) and falls back to
``exp(GammaFunction.logValue(n+1))`` for larger n. The Python port uses
``math.lgamma`` (stdlib, C99-equivalent) for the fallback rather than
porting GammaFunction yet (which lives in distributions/, deferred to a
later cluster). Documented divergence: cross-validation against the C++
probe still pins agreement at TIGHT tolerance for n up to 170 (typical
QuantLib usage range).
"""

from __future__ import annotations

import math
from typing import Final

from pquantlib import qassert

# Mirrors the C++ ``firstFactorials`` array (indices 0..27 inclusive).
_FIRST_FACTORIALS: Final[tuple[float, ...]] = (
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5040.0,
    40320.0,
    362880.0,
    3628800.0,
    39916800.0,
    479001600.0,
    6227020800.0,
    87178291200.0,
    1307674368000.0,
    20922789888000.0,
    355687428096000.0,
    6402373705728000.0,
    121645100408832000.0,
    2432902008176640000.0,
    51090942171709440000.0,
    1124000727777607680000.0,
    25852016738884976640000.0,
    620448401733239439360000.0,
    15511210043330985984000000.0,
    403291461126605635584000000.0,
    10888869450418352160768000000.0,
)
_TABULATED: Final[int] = len(_FIRST_FACTORIALS) - 1  # 27


class Factorial:
    """Static-only factorial calculator (matches C++ ``class Factorial``)."""

    @staticmethod
    def get(n: int) -> float:
        # C++ uses Natural (unsigned). Python int can be negative, and the
        # tabulated-array path would silently fold negative n via Python
        # negative indexing (e.g. n=-1 returns the n=27 value). Guard here.
        qassert.require(n >= 0, f"Factorial.get requires n >= 0, got {n}")
        if n <= _TABULATED:
            return _FIRST_FACTORIALS[n]
        return math.exp(math.lgamma(n + 1))

    @staticmethod
    def ln(n: int) -> float:
        qassert.require(n >= 0, f"Factorial.ln requires n >= 0, got {n}")
        if n <= _TABULATED:
            return math.log(_FIRST_FACTORIALS[n])
        return math.lgamma(n + 1)
