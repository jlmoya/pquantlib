"""Gamma function (Lanczos approximation).

# C++ parity: ql/math/distributions/gammadistribution.{hpp,cpp} (v1.42.1)
#             ``class GammaFunction``.

The C++ implementation uses a fixed-order Lanczos approximation (the
"Numerical Recipes" coefficients, accurate to ~15 decimal digits for
``x > 1``). For ``x < 1`` the recurrence ``Gamma(x) = Gamma(x+1) / x``
shifts the argument into the convergent range; for ``x <= -20`` the
reflection formula ``Gamma(-x) = -pi / (Gamma(x) * x * sin(pi*x))`` is
used.

The Python port mirrors this byte-for-byte so cross-validation against
the C++ probe yields TIGHT-tolerance agreement for ``x >= 0.5``. We
keep the same hand-tuned constants ``c1..c6`` rather than substituting
``math.lgamma`` (the stdlib version uses a different — and slightly
more accurate — approximation; the L1 Factorial cluster carved this
delta out as a known divergence).
"""

from __future__ import annotations

import math
from typing import Final

from pquantlib import qassert

# Lanczos coefficients from Numerical Recipes 2nd ed. (same as C++).
# C++ parity: gammadistribution.cpp:62-67.
_C1: Final[float] = 76.18009172947146
_C2: Final[float] = -86.50532032941677
_C3: Final[float] = 24.01409824083091
_C4: Final[float] = -1.231739572450155
_C5: Final[float] = 0.1208650973866179e-2
_C6: Final[float] = -0.5395239384953e-5

# sqrt(2 * pi) — appears as the constant inside ``log(2.5066... * ser / x)``.
_SQRT_2PI: Final[float] = 2.5066282746310005


class GammaFunction:
    """Gamma function via the Lanczos approximation.

    # C++ parity: ``class GammaFunction`` — stateless; both methods are
    # member functions in C++ but logically static. The Python port
    # keeps a class for parity with C++ usage; either ``GammaFunction()
    # .value(x)`` or instance-level calls work.
    """

    @staticmethod
    def log_value(x: float) -> float:
        """Natural log of Gamma(x). Requires ``x > 0``.

        # C++ parity: ``GammaFunction::logValue`` — gammadistribution.cpp:69-82.
        """
        qassert.require(x > 0.0, "positive argument required")
        temp = x + 5.5
        temp -= (x + 0.5) * math.log(temp)
        ser = 1.000000000190015
        ser += _C1 / (x + 1.0)
        ser += _C2 / (x + 2.0)
        ser += _C3 / (x + 3.0)
        ser += _C4 / (x + 4.0)
        ser += _C5 / (x + 5.0)
        ser += _C6 / (x + 6.0)
        return -temp + math.log(_SQRT_2PI * ser / x)

    # C++ parity: ``logValue`` is the canonical camelCase name in C++.
    # Python convention is snake_case (``log_value``); expose camelCase
    # alias for symmetry with C++ probe field names and call-site
    # symmetry where pquantlib mirrors the C++ method-name letter for
    # letter (e.g. `BlackFormula.blackFormula`). Keep both.
    def logValue(self, x: float) -> float:  # noqa: N802 — C++ parity
        return self.log_value(x)

    @staticmethod
    def value(x: float) -> float:
        """Gamma(x) for any real ``x`` except non-positive integers.

        # C++ parity: ``GammaFunction::value`` — gammadistribution.cpp:84-98.
        """
        if x >= 1.0:
            return math.exp(GammaFunction.log_value(x))
        # x < 1.0
        if x > -20.0:
            # Gamma(x) = Gamma(x+1) / x. Recurse — at most ~21 levels.
            return GammaFunction.value(x + 1.0) / x
        # Reflection formula: Gamma(-x) = -pi / (Gamma(x) sin(pi x) x).
        # NB: This is the only branch that cannot be evaluated when
        # ``x`` is a negative integer (sin(pi*x) == 0 -> 1/0).
        return -math.pi / (GammaFunction.value(-x) * x * math.sin(math.pi * x))
