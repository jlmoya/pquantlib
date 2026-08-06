"""AdaptiveRungeKutta — Cash-Karp RK with adaptive step size.

# C++ parity: ql/math/ode/adaptiverungekutta.hpp (v1.43).

Runge-Kutta with adaptive step size as described in *Numerical
Recipes in C*, chapter 16.2: an embedded 5th/4th-order Cash-Karp
pair, with the step accepted or shrunk according to

.. code-block:: text

    yScale[i] = |y[i]| + |dydx[i] * h| + TINY
    errmax    = max_i |yerr[i] / yScale[i]| / eps

The three constructor arguments are, in order, the **prescribed
error** ``eps``, the **initial step size** ``h1`` and the **smallest
allowed step** ``hmin``. None of them is a relative tolerance in the
``scipy.integrate`` sense, and the error control is *not* the
``rtol``/``atol`` combination used by ``solve_ivp``: substituting an
RK45 integrator with ``rtol=h1``/``atol=eps`` reproduces QuantLib only
to roughly 1e-5 relative, which is why this class is ported verbatim
rather than delegated to scipy.

Only the real-valued (``T = Real``) instantiation is ported; QuantLib
templates the class over ``Real``/``std::complex<Real>`` but nothing
in the library instantiates the complex variant.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Final

from pquantlib import qassert

# C++ ADAPTIVERK_* constants (adaptiverungekutta.hpp private section).
_MAXSTP: Final[int] = 10000
_TINY: Final[float] = 1.0e-30
_SAFETY: Final[float] = 0.9
_PGROW: Final[float] = -0.2
_PSHRINK: Final[float] = -0.25
_ERRCON: Final[float] = 1.89e-4

# Cash-Karp tableau, spelled exactly as the C++ member initialiser list so the
# floating-point values are bit-identical.
_A2: Final[float] = 0.2
_A3: Final[float] = 0.3
_A4: Final[float] = 0.6
_A5: Final[float] = 1.0
_A6: Final[float] = 0.875
_B21: Final[float] = 0.2
_B31: Final[float] = 3.0 / 40.0
_B32: Final[float] = 9.0 / 40.0
_B41: Final[float] = 0.3
_B42: Final[float] = -0.9
_B43: Final[float] = 1.2
_B51: Final[float] = -11.0 / 54.0
_B52: Final[float] = 2.5
_B53: Final[float] = -70.0 / 27.0
_B54: Final[float] = 35.0 / 27.0
_B61: Final[float] = 1631.0 / 55296.0
_B62: Final[float] = 175.0 / 512.0
_B63: Final[float] = 575.0 / 13824.0
_B64: Final[float] = 44275.0 / 110592.0
_B65: Final[float] = 253.0 / 4096.0
_C1: Final[float] = 37.0 / 378.0
_C3: Final[float] = 250.0 / 621.0
_C4: Final[float] = 125.0 / 594.0
_C6: Final[float] = 512.0 / 1771.0
_DC1: Final[float] = _C1 - 2825.0 / 27648.0
_DC3: Final[float] = _C3 - 18575.0 / 48384.0
_DC4: Final[float] = _C4 - 13525.0 / 55296.0
_DC5: Final[float] = -277.0 / 14336.0
_DC6: Final[float] = _C6 - 0.25

#: ``F: R x R^n -> R^n`` giving ``f'(x) = F(x, f(x))``.
type OdeFct = Callable[[float, list[float]], Sequence[float]]
#: Scalar specialisation ``F: R x R -> R``.
type OdeFct1d = Callable[[float, float], float]


class AdaptiveRungeKutta:
    """Cash-Karp Runge-Kutta with Numerical-Recipes adaptive step control.

    # C++ parity: ``template <class T = Real> class AdaptiveRungeKutta``.

    Args:
        eps: prescribed error for the solution.
        h1: start step size.
        hmin: smallest step size allowed; a proposed step at or below
            it raises rather than silently stalling.
    """

    def __init__(self, eps: float = 1.0e-6, h1: float = 1.0e-4, hmin: float = 0.0) -> None:
        self._eps: float = eps
        self._h1: float = h1
        self._hmin: float = hmin

    def __call__(
        self,
        ode: OdeFct,
        y1: Sequence[float],
        x1: float,
        x2: float,
    ) -> list[float]:
        """Integrate ``f'(x) = F(x, f(x))`` from ``x1`` to ``x2`` with ``f(x1) = y1``.

        # C++ parity: ``AdaptiveRungeKutta<T>::operator()(OdeFct, vector, Real, Real)``.
        """
        n = len(y1)
        y = list(y1)
        y_scale = [0.0] * n
        x = x1
        h = self._h1 * (1 if x1 <= x2 else -1)

        for _nstp in range(1, _MAXSTP + 1):
            dydx = list(ode(x, y))
            for i in range(n):
                y_scale[i] = abs(y[i]) + abs(dydx[i] * h) + _TINY
            if (x + h - x2) * (x + h - x1) > 0.0:
                h = x2 - x
            x, _hdid, hnext = self._rkqs(y, dydx, x, h, self._eps, y_scale, ode)

            if (x - x2) * (x2 - x1) >= 0.0:
                return y

            qassert.require(
                math.fabs(hnext) > self._hmin,
                f"Step size ({hnext}) too small ({self._hmin} min) in AdaptiveRungeKutta",
            )
            h = hnext
        qassert.fail(f"Too many steps ({_MAXSTP}) in AdaptiveRungeKutta")

    def solve_1d(self, ode: OdeFct1d, y1: float, x1: float, x2: float) -> float:
        """Scalar convenience overload.

        # C++ parity: ``AdaptiveRungeKutta<T>::operator()(OdeFct1d, T, Real, Real)``
        # — which wraps the scalar function in ``detail::OdeFctWrapper`` and
        # forwards to the vector overload, so the arithmetic is identical.
        """

        def wrapped(x: float, y: list[float]) -> list[float]:
            return [ode(x, y[0])]

        return self(wrapped, [y1], x1, x2)[0]

    # --- internals ------------------------------------------------------

    def _rkqs(
        self,
        y: list[float],
        dydx: Sequence[float],
        x: float,
        htry: float,
        eps: float,
        y_scale: Sequence[float],
        derivs: OdeFct,
    ) -> tuple[float, float, float]:
        """Quality-controlled step; mutates ``y`` and returns ``(x, hdid, hnext)``.

        # C++ parity: ``AdaptiveRungeKutta<T>::rkqs`` — which returns via the
        # ``x``, ``hdid``, ``hnext`` reference parameters.
        """
        n = len(y)
        h = htry

        while True:
            ytemp, yerr = self._rkck(y, dydx, x, h, derivs)
            errmax = 0.0
            for i in range(n):
                errmax = max(errmax, abs(yerr[i] / y_scale[i]))
            errmax /= eps
            if errmax > 1.0:
                htemp1 = _SAFETY * h * (errmax**_PSHRINK)
                htemp2 = h / 10
                # C++ spells these as explicit ternaries rather than
                # std::min/std::max (a VC++14 inlining workaround); the
                # selection is the same.
                max_positive = htemp1 if htemp1 > htemp2 else htemp2
                max_negative = htemp1 if htemp1 < htemp2 else htemp2
                h = max_positive if h >= 0.0 else max_negative
                xnew = x + h
                qassert.require(
                    xnew != x,
                    f"Stepsize underflow ({h} at x = {x}) in AdaptiveRungeKutta::rkqs",
                )
                continue
            hnext = _SAFETY * h * (errmax**_PGROW) if errmax > _ERRCON else 5.0 * h
            hdid = h
            x += hdid
            for i in range(n):
                y[i] = ytemp[i]
            return x, hdid, hnext

    def _rkck(
        self,
        y: Sequence[float],
        dydx: Sequence[float],
        x: float,
        h: float,
        derivs: OdeFct,
    ) -> tuple[list[float], list[float]]:
        """One Cash-Karp step; returns ``(yout, yerr)``.

        # C++ parity: ``AdaptiveRungeKutta<T>::rkck``.
        """
        n = len(y)
        ytemp = [0.0] * n

        # first step
        for i in range(n):
            ytemp[i] = y[i] + _B21 * h * dydx[i]

        # second step
        ak2 = list(derivs(x + _A2 * h, ytemp))
        for i in range(n):
            ytemp[i] = y[i] + h * (_B31 * dydx[i] + _B32 * ak2[i])

        # third step
        ak3 = list(derivs(x + _A3 * h, ytemp))
        for i in range(n):
            ytemp[i] = y[i] + h * (_B41 * dydx[i] + _B42 * ak2[i] + _B43 * ak3[i])

        # fourth step
        ak4 = list(derivs(x + _A4 * h, ytemp))
        for i in range(n):
            ytemp[i] = y[i] + h * (_B51 * dydx[i] + _B52 * ak2[i] + _B53 * ak3[i] + _B54 * ak4[i])

        # fifth step
        ak5 = list(derivs(x + _A5 * h, ytemp))
        for i in range(n):
            ytemp[i] = y[i] + h * (
                _B61 * dydx[i] + _B62 * ak2[i] + _B63 * ak3[i] + _B64 * ak4[i] + _B65 * ak5[i]
            )

        # sixth step
        ak6 = list(derivs(x + _A6 * h, ytemp))
        yout = [0.0] * n
        yerr = [0.0] * n
        for i in range(n):
            yout[i] = y[i] + h * (_C1 * dydx[i] + _C3 * ak3[i] + _C4 * ak4[i] + _C6 * ak6[i])
            yerr[i] = h * (
                _DC1 * dydx[i] + _DC3 * ak3[i] + _DC4 * ak4[i] + _DC5 * ak5[i] + _DC6 * ak6[i]
            )
        return yout, yerr


__all__ = ["AdaptiveRungeKutta", "OdeFct", "OdeFct1d"]
