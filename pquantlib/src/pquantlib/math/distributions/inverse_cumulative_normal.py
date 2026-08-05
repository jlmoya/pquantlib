"""Inverse cumulative normal distribution — Acklam algorithm.

# C++ parity: ql/math/distributions/normaldistribution.hpp +
#             ql/math/distributions/normaldistribution.cpp (v1.42.1).

Implements Peter J. Acklam's rational approximation
(https://home.online.no/~pjacklam/notes/invnorm/) used in the C++
``InverseCumulativeNormal`` class. The standard region [x_low_, x_high_]
uses a single rational polynomial; tails use a separate ``tail_value``
rational form on ``sqrt(-2 * log(...))``.

Constants are copied verbatim from ``normaldistribution.cpp``.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Final

from scipy.special import (  # pyright: ignore[reportMissingTypeStubs]
    ndtri as _ndtri,  # pyright: ignore[reportUnknownVariableType]
)

from pquantlib import qassert

# Rational-approximation coefficients (a/b for the central region;
# c/d for the tails).
_A1: Final[float] = -3.969683028665376e01
_A2: Final[float] = 2.209460984245205e02
_A3: Final[float] = -2.759285104469687e02
_A4: Final[float] = 1.383577518672690e02
_A5: Final[float] = -3.066479806614716e01
_A6: Final[float] = 2.506628277459239e00

_B1: Final[float] = -5.447609879822406e01
_B2: Final[float] = 1.615858368580409e02
_B3: Final[float] = -1.556989798598866e02
_B4: Final[float] = 6.680131188771972e01
_B5: Final[float] = -1.328068155288572e01

_C1: Final[float] = -7.784894002430293e-03
_C2: Final[float] = -3.223964580411365e-01
_C3: Final[float] = -2.400758277161838e00
_C4: Final[float] = -2.549732539343734e00
_C5: Final[float] = 4.374664141464968e00
_C6: Final[float] = 2.938163982698783e00

_D1: Final[float] = 7.784695709041462e-03
_D2: Final[float] = 3.224671290700398e-01
_D3: Final[float] = 2.445134137142996e00
_D4: Final[float] = 3.754408661907416e00

_X_LOW: Final[float] = 0.02425
_X_HIGH: Final[float] = 1.0 - _X_LOW

_QL_EPSILON: Final[float] = sys.float_info.epsilon
_QL_MAX_REAL: Final[float] = sys.float_info.max
_QL_MIN_REAL: Final[float] = -sys.float_info.max


def _close_enough(a: float, b: float) -> bool:
    # Mirrors ql/math/comparison.hpp ``close_enough`` (n=42 ULPs default).
    # We use a generous 1e-12 here; only ``tail_value`` calls it on the
    # boundary cases (x ~ 1.0), where exact equality covers the common path.
    return math.isclose(a, b, abs_tol=1e-14, rel_tol=1e-12)


def _tail_value(x: float) -> float:
    if x <= 0.0 or x >= 1.0:
        if _close_enough(x, 1.0):
            return _QL_MAX_REAL
        if math.fabs(x) < _QL_EPSILON:
            return _QL_MIN_REAL
        qassert.fail(f"InverseCumulativeNormal({x}) undefined: must be 0 < x < 1")

    if x < _X_LOW:
        # Lower-tail rational approximation.
        z = math.sqrt(-2.0 * math.log(x))
        return (((((_C1 * z + _C2) * z + _C3) * z + _C4) * z + _C5) * z + _C6) / (
            (((_D1 * z + _D2) * z + _D3) * z + _D4) * z + 1.0
        )
    # Upper-tail rational approximation: negate the lower-tail numerator.
    z = math.sqrt(-2.0 * math.log(1.0 - x))
    return -(
        (((((_C1 * z + _C2) * z + _C3) * z + _C4) * z + _C5) * z + _C6)
        / ((((_D1 * z + _D2) * z + _D3) * z + _D4) * z + 1.0)
    )


def _standard_value(x: float) -> float:
    if x < _X_LOW or x > _X_HIGH:
        return _tail_value(x)
    # Central region: rational polynomial on (x - 0.5).
    z = x - 0.5
    r = z * z
    return (
        (((((_A1 * r + _A2) * r + _A3) * r + _A4) * r + _A5) * r + _A6)
        * z
        / (((((_B1 * r + _B2) * r + _B3) * r + _B4) * r + _B5) * r + 1.0)
    )


@dataclass(frozen=True, slots=True)
class InverseCumulativeNormal:
    """Inverse normal CDF via Acklam's approximation.

    ``InverseCumulativeNormal(average, sigma)(p)`` returns ``average + sigma * standard_value(p)``.
    """

    average: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        qassert.require(self.sigma > 0.0, f"sigma must be greater than 0.0 ({self.sigma} not allowed)")

    def __call__(self, x: float) -> float:
        return self.average + self.sigma * _standard_value(x)

    @staticmethod
    def standard_value(x: float) -> float:
        """Quantile for ``average=0, sigma=1`` — matches C++ ``static standard_value(x)``."""
        return _standard_value(x)


# Backward-compat alias used in C++ headers.
InvCumulativeNormalDistribution = InverseCumulativeNormal


# --- Moro (1995) ---------------------------------------------------------
# C++ parity: normaldistribution.cpp:143-159 (constants) and :161-190
# (evaluation). Beasley-Springer in the central region, a degree-8 polynomial
# in log(-log(tail)) outside it. Strictly less accurate than Acklam's above —
# the C++ docstring says so — and kept because published models are specified
# against it.
_MORO_A0: Final[float] = 2.50662823884
_MORO_A1: Final[float] = -18.61500062529
_MORO_A2: Final[float] = 41.39119773534
_MORO_A3: Final[float] = -25.44106049637

_MORO_B0: Final[float] = -8.47351093090
_MORO_B1: Final[float] = 23.08336743743
_MORO_B2: Final[float] = -21.06224101826
_MORO_B3: Final[float] = 3.13082909833

_MORO_C0: Final[float] = 0.3374754822726147
_MORO_C1: Final[float] = 0.9761690190917186
_MORO_C2: Final[float] = 0.1607979714918209
_MORO_C3: Final[float] = 0.0276438810333863
_MORO_C4: Final[float] = 0.0038405729373609
_MORO_C5: Final[float] = 0.0003951896511919
_MORO_C6: Final[float] = 0.0000321767881768
_MORO_C7: Final[float] = 0.0000002888167364
_MORO_C8: Final[float] = 0.0000003960315187


@dataclass(frozen=True, slots=True)
class MoroInverseCumulativeNormal:
    """Inverse normal CDF via Beasley-Springer with Moro's tail.

    # C++ parity: ``class MoroInverseCumulativeNormal`` —
    # normaldistribution.hpp:167-190, normaldistribution.cpp:161-190.
    """

    average: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        qassert.require(self.sigma > 0.0, f"sigma must be greater than 0.0 ({self.sigma} not allowed)")

    def __call__(self, x: float) -> float:
        # C++ parity: normaldistribution.cpp:162-189.
        qassert.require(x > 0.0 and x < 1.0, f"MoroInverseCumulativeNormal({x}) undefined: must be 0<x<1")

        temp = x - 0.5

        if math.fabs(temp) < 0.42:
            # Beasley and Springer, 1977.
            result = temp * temp
            result = (
                temp
                * (((_MORO_A3 * result + _MORO_A2) * result + _MORO_A1) * result + _MORO_A0)
                / ((((_MORO_B3 * result + _MORO_B2) * result + _MORO_B1) * result + _MORO_B0) * result + 1.0)
            )
        else:
            # Improved approximation for the tail (Moro 1995).
            result = x if x < 0.5 else 1.0 - x
            result = math.log(-math.log(result))
            result = _MORO_C0 + result * (
                _MORO_C1
                + result
                * (
                    _MORO_C2
                    + result
                    * (
                        _MORO_C3
                        + result
                        * (
                            _MORO_C4
                            + result
                            * (_MORO_C5 + result * (_MORO_C6 + result * (_MORO_C7 + result * _MORO_C8)))
                        )
                    )
                )
            )
            if x < 0.5:
                result = -result

        return self.average + result * self.sigma


# --- Maddock / Boost ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MaddockInverseCumulativeNormal:
    """Inverse normal CDF at full double precision.

    # C++ parity: ``class MaddockInverseCumulativeNormal`` —
    # normaldistribution.hpp:214-222, normaldistribution.cpp:192-200. C++
    # forwards to ``boost::math::quantile(normal_distribution<Real>(mu, sigma), x)``.

    This is the one delegation in the module, and it is a delegation of a
    *mathematical function*, not of an algorithm: both Boost's ``erfc_inv``
    rational approximation and SciPy's Cephes ``ndtri`` are documented to
    return the correctly-rounded normal quantile to within an ulp or two, and
    the probe confirms they agree to TIGHT across ``1e-15 .. 1 - 1e-12``.
    That is categorically different from swapping one interpolation scheme for
    another with the same name: there is only one normal quantile, and both
    implementations compute it.
    """

    average: float = 0.0
    sigma: float = 1.0

    def __call__(self, x: float) -> float:
        return float(_ndtri(x)) * self.sigma + self.average
