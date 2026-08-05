"""Tanh-sinh (double-exponential) quadrature on a finite interval.

# C++ parity: ql/math/integrals/tanhsinhintegral.hpp (v1.43).

# C++ parity divergence: the C++ class is a thin wrapper over
# ``boost::math::quadrature::tanh_sinh``. There is no boost in Python, and
# ``scipy.integrate.quad`` is a *different* algorithm (adaptive Clenshaw-Curtis
# / QUADPACK), not a drop-in for a double-exponential rule — it has different
# endpoint-singularity behaviour, a different error estimate and a different
# evaluation pattern. So the DE rule itself is implemented here and
# cross-validated against boost through the C++ probe.

The rule
--------

With ``x = c + r * tanh((pi/2) sinh t)`` where ``c = (a+b)/2`` and
``r = (b-a)/2``,

    int_a^b f(x) dx = r int_{-inf}^{inf} f(x(t)) w(t) dt,
    w(t) = (pi/2) cosh(t) / cosh^2((pi/2) sinh t)

and the trapezoid rule in ``t`` converges doubly exponentially for functions
holomorphic on the open interval, *including* when they blow up at the
endpoints. Level ``k`` uses spacing ``h = 2^-k`` and reuses every point of
level ``k-1``, so refining only costs the odd multiples of ``h``.

Two implementation details are what make this usable rather than a textbook
toy, and both mirror what boost does:

* the abscissa is built from its **complement** ``1 - tanh(u) = 2/(e^{2u}+1)``
  and applied as ``a + r*c`` / ``b - r*c``. Forming ``c + r*tanh(u)`` directly
  rounds to exactly ``a`` or ``b`` once ``u`` passes about 19, which hands an
  endpoint singularity a ``+inf`` and poisons the sum;
* the point loop stops once the term stops moving the partial sum, so the far
  tail (where the weight is below 1e-40) is never evaluated.

Termination matches the documented boost contract: stop when the change
between two successive levels is at most ``rel_tolerance`` times the L1 norm
of the level's sum.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from typing import Final

from pquantlib.math.constants import M_PI_2, QL_EPSILON, QL_MAX_REAL
from pquantlib.math.integrals.integrator import Integrator, RealFunction

# C++ ``Null<Size>()`` — ql/utilities/null.hpp yields
# ``std::numeric_limits<int>::max()``.
_NULL_SIZE: Final[int] = 2147483647

# ``std::sqrt(std::numeric_limits<Real>::epsilon())`` — the C++ default
# relative tolerance for both DE integrators.
DEFAULT_REL_TOLERANCE: Final[float] = math.sqrt(sys.float_info.epsilon)

# (pi/2) sinh(t) beyond this overflows cosh(u)**2 (u > ~354) and underflows
# the abscissa complement 2/(e^{2u}+1); the DE weight there is below 1e-300.
_MAX_U: Final[float] = 350.0

# Below this the DE weight cannot move a double-precision partial sum for any
# integrand whose values stay under ~1e35 near the endpoints.
_MIN_WEIGHT: Final[float] = 1e-40


def _de_level_sum(
    f: Callable[[float], float],
    a: float,
    b: float,
    half: float,
    center: float,
    h: float,
    *,
    only_odd: bool,
) -> tuple[float, float]:
    """Accumulate ``(sum w f, sum w |f|)`` over the tanh-sinh nodes at spacing ``h``.

    ``only_odd`` restricts the walk to odd multiples of ``h`` — the points a
    refinement adds on top of the previous level.
    """
    total = 0.0
    l1 = 0.0

    if not only_odd:
        # t = 0: x = center, weight (pi/2) cosh(0) / cosh^2(0) = pi/2.
        y = f(center)
        total += M_PI_2 * y
        l1 += M_PI_2 * abs(y)

    j = 1
    step = 2 if only_odd else 1
    left_alive = True
    right_alive = True
    while left_alive or right_alive:
        t = j * h
        u = M_PI_2 * math.sinh(t)
        if u > _MAX_U:
            break
        cosh_u = math.cosh(u)
        w = M_PI_2 * math.cosh(t) / (cosh_u * cosh_u)
        if w < _MIN_WEIGHT:
            break
        # 1 - tanh(u), formed without cancellation.
        complement = 2.0 / (math.exp(2.0 * u) + 1.0)
        offset = half * complement
        # The two sides die at different t: ``a + offset`` stays distinct from
        # ``a`` until the offset underflows in absolute terms, while
        # ``b - offset`` collapses onto ``b`` as soon as the offset drops below
        # half an ulp of ``b``. With a = 0, b = 1 that is t ~ 3.5 on the right
        # and t ~ 6 on the left — so they must be retired independently, or an
        # integrand singular at ``a`` loses the whole tail that carries its
        # mass.
        term = 0.0
        if left_alive:
            xl = a + offset
            if xl <= a:
                left_alive = False
            else:
                yl = f(xl)
                term += w * yl
                l1 += w * abs(yl)
        if right_alive:
            xr = b - offset
            if xr >= b:
                right_alive = False
            else:
                yr = f(xr)
                term += w * yr
                l1 += w * abs(yr)
        total += term
        if t > 1.0 and abs(term) <= QL_EPSILON * abs(total):
            break
        j += step

    return total, l1


class TanhSinhIntegral(Integrator):
    """Double-exponential quadrature over a finite ``[a, b]``.

    # C++ parity: tanhsinhintegral.hpp:38-70.

    :param rel_tolerance: stop once the level-to-level change is within this
        fraction of the L1 norm of the sum.
    :param max_refinements: cap on halvings of the ``t`` spacing.
    :param min_complement: smallest abscissa complement boost would tabulate;
        accepted for signature parity, unused here because the point loop
        stops on the weight instead.
    """

    __slots__ = ("_max_refinements", "_min_complement", "_rel_tolerance")

    def __init__(
        self,
        rel_tolerance: float = DEFAULT_REL_TOLERANCE,
        max_refinements: int = 15,
        min_complement: float = sys.float_info.min * 4,
    ) -> None:
        # C++ parity: Integrator(QL_MAX_REAL, Null<Size>()).
        super().__init__(QL_MAX_REAL, _NULL_SIZE)
        self._rel_tolerance: float = rel_tolerance
        self._max_refinements: int = max_refinements
        self._min_complement: float = min_complement

    def rel_tolerance(self) -> float:
        """The relative tolerance the stopping rule uses."""
        return self._rel_tolerance

    def _integrate(self, f: RealFunction, a: float, b: float) -> float:
        # C++ parity: tanhsinhintegral.hpp:57-70 — boost does the work there;
        # the DE rule itself is spelled out here.
        half = (b - a) / 2.0
        center = (a + b) / 2.0

        def counted(x: float) -> float:
            self._increase_number_of_evaluations(1)
            return f(x)

        h = 1.0
        s, l1 = _de_level_sum(counted, a, b, half, center, h, only_odd=False)
        prev = half * h * s

        error = QL_MAX_REAL
        for _ in range(self._max_refinements):
            h *= 0.5
            ds, dl1 = _de_level_sum(counted, a, b, half, center, h, only_odd=True)
            s += ds
            l1 += dl1
            current = half * h * s
            error = abs(current - prev)
            prev = current
            if error <= self._rel_tolerance * abs(half * h * l1):
                break

        self._set_absolute_error(error)
        return prev


__all__ = ["DEFAULT_REL_TOLERANCE", "TanhSinhIntegral"]
