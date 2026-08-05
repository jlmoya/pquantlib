"""Exp-sinh (double-exponential) quadrature on a half-infinite interval.

# C++ parity: ql/math/integrals/expsinhintegral.hpp (v1.43).

# C++ parity divergence: the C++ class is a thin wrapper over
# ``boost::math::quadrature::exp_sinh``. There is no boost in Python, and
# ``scipy.integrate.quad`` is a different algorithm with different endpoint
# and tail behaviour, so the DE rule itself is implemented here and
# cross-validated against boost through the C++ probe.

The rule
--------

For ``[a, +inf)`` the substitution is ``x = a + exp((pi/2) sinh t)``, with

    dx/dt = (pi/2) cosh(t) exp((pi/2) sinh t)

so ``int_a^inf f(x) dx = int_{-inf}^{inf} f(x(t)) w(t) dt`` with
``w(t) = (pi/2) cosh(t) exp((pi/2) sinh t)``. The trapezoid rule in ``t``
converges doubly exponentially; as ``t -> -inf`` the abscissa approaches
``a`` from above (never reaching it, so an integrable singularity at ``a`` is
sampled rather than hit), and as ``t -> +inf`` it runs off to infinity while
the weight-times-integrand decays for any ``f`` that decays faster than
``1/x``.

``(-inf, b]`` is handled by the reflection ``x = b - exp((pi/2) sinh t)``,
which is ``int_{-inf}^b f = int_0^inf f(b - y) dy``.

Level ``k`` uses spacing ``h = 2^-k`` and reuses every point of level ``k-1``.
Termination matches the documented boost contract: stop when the change
between two successive levels is at most ``rel_tolerance`` times the L1 norm
of the level's sum.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final

from pquantlib import qassert
from pquantlib.math.constants import M_PI_2, QL_EPSILON, QL_MAX_REAL
from pquantlib.math.integrals.integrator import Integrator, RealFunction
from pquantlib.math.integrals.tanh_sinh_integral import DEFAULT_REL_TOLERANCE

# C++ ``Null<Size>()`` — ql/utilities/null.hpp yields
# ``std::numeric_limits<int>::max()``.
_NULL_SIZE: Final[int] = 2147483647

# exp(u) overflows above ~709.78 and underflows to zero below ~-745.13.
_MAX_U: Final[float] = 700.0
_MIN_U: Final[float] = -745.0


def _de_level_sum(
    f: Callable[[float], float],
    bound: float,
    sign: float,
    h: float,
    *,
    only_odd: bool,
) -> tuple[float, float]:
    """Accumulate ``(sum w f, sum w |f|)`` over the exp-sinh nodes at spacing ``h``.

    The abscissa is ``bound + sign * exp((pi/2) sinh t)``: ``sign = +1`` for
    ``[bound, +inf)``, ``sign = -1`` for ``(-inf, bound]``. ``only_odd``
    restricts the walk to the odd multiples of ``h`` a refinement adds.
    """
    total = 0.0
    l1 = 0.0

    if not only_odd:
        # t = 0: u = 0, x = bound + sign, weight (pi/2).
        y = f(bound + sign)
        total += M_PI_2 * y
        l1 += M_PI_2 * abs(y)

    step = 2 if only_odd else 1

    for direction in (1.0, -1.0):
        j = 1
        while True:
            t = direction * j * h
            sinh_t = math.sinh(t)
            u = M_PI_2 * sinh_t
            if u > _MAX_U or u < _MIN_U:
                break
            ex = math.exp(u)
            w = M_PI_2 * math.cosh(t) * ex
            if w == 0.0 or math.isinf(w):
                break
            y = f(bound + sign * ex)
            term = w * y
            total += term
            l1 += w * abs(y)
            if abs(t) > 1.0 and abs(term) <= QL_EPSILON * abs(total):
                break
            j += step

    return total, l1


class ExpSinhIntegral(Integrator):
    """Double-exponential quadrature over a half-infinite interval.

    # C++ parity: expsinhintegral.hpp:39-83.

    :param rel_tolerance: stop once the level-to-level change is within this
        fraction of the L1 norm of the sum.
    :param max_refinements: cap on halvings of the ``t`` spacing.
    """

    __slots__ = ("_max_refinements", "_rel_tolerance")

    def __init__(
        self,
        rel_tolerance: float = DEFAULT_REL_TOLERANCE,
        max_refinements: int = 9,
    ) -> None:
        # C++ parity: Integrator(QL_MAX_REAL, Null<Size>()).
        super().__init__(QL_MAX_REAL, _NULL_SIZE)
        self._rel_tolerance: float = rel_tolerance
        self._max_refinements: int = max_refinements

    def rel_tolerance(self) -> float:
        """The relative tolerance the stopping rule uses."""
        return self._rel_tolerance

    def integrate(self, f: RealFunction) -> float:
        """``int_0^inf f(x) dx``.

        # C++ parity: expsinhintegral.hpp:53-63 — the extra non-virtual
        # overload for the half-infinite interval ``[0, inf)``. It resets the
        # evaluation counter itself, because ``Integrator::operator()`` (which
        # normally does that) is bypassed.
        """
        self._set_number_of_evaluations(0)
        return self._de(f, 0.0, 1.0)

    def _integrate(self, f: RealFunction, a: float, b: float) -> float:
        # C++ parity: expsinhintegral.hpp:66-78. boost requires exactly one
        # infinite bound; QuantLib passes a and b straight through.
        a_inf = math.isinf(a)
        b_inf = math.isinf(b)
        qassert.require(
            a_inf != b_inf,
            "exactly one of the integration bounds must be infinite "
            "for the exp-sinh quadrature",
        )
        if b_inf:
            return self._de(f, a, 1.0)
        return self._de(f, b, -1.0)

    def _de(self, f: RealFunction, bound: float, sign: float) -> float:
        """Run the exp-sinh level refinement anchored at ``bound``."""

        def counted(x: float) -> float:
            self._increase_number_of_evaluations(1)
            return f(x)

        h = 1.0
        s, l1 = _de_level_sum(counted, bound, sign, h, only_odd=False)
        prev = h * s

        error = QL_MAX_REAL
        for _ in range(self._max_refinements):
            h *= 0.5
            ds, dl1 = _de_level_sum(counted, bound, sign, h, only_odd=True)
            s += ds
            l1 += dl1
            current = h * s
            error = abs(current - prev)
            prev = current
            if error <= self._rel_tolerance * abs(h * l1):
                break

        self._set_absolute_error(error)
        return prev


__all__ = ["ExpSinhIntegral"]
