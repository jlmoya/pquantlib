"""Adaptive Runge-Kutta (Cash-Karp 4/5) ODE integrator.

# C++ parity: ql/math/ode/adaptiverungekutta.hpp (v1.43) — header-only.

Not ``scipy.integrate.solve_ivp(method="RK45")``. Both are embedded
Runge-Kutta pairs, but the step controller is where the answer comes from and
the two differ in every detail that matters: QuantLib's error norm is
``max_i |yerr_i / yScale_i| / eps`` with ``yScale_i = |y_i| + |h * dydx_i| +
1e-30`` recomputed at the *outer* step (not per stage), its shrink factor is
clamped to ``h/10`` by a hand-written min/max pair kept for a VC++14 inlining
bug, its growth is capped at 5x with a 1.89e-4 error threshold, and it fails
loudly after 10000 steps or below ``hmin`` rather than returning a status.

The generic ``T`` of the C++ template is instantiated for ``Real`` and for
``std::complex<Real>``; the Python port is written against ``complex``, which
subsumes both, and the 1-D entry point returns whatever the caller's ODE
returns.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Final

from pquantlib import qassert

# C++ parity: adaptiverungekutta.hpp:93-95.
_MAXSTP: Final[int] = 10000
_TINY: Final[float] = 1.0e-30
_SAFETY: Final[float] = 0.9
_PGROW: Final[float] = -0.2
_PSHRINK: Final[float] = -0.25
_ERRCON: Final[float] = 1.89e-4

# Cash-Karp coefficients. C++ parity: adaptiverungekutta.hpp:52-59, 90-92.
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

# The C++ typedefs, as Python type aliases.
type OdeFct = Callable[[float, Sequence[float]], list[float]]
type OdeFct1d = Callable[[float, float], float]


class OdeFctWrapper:
    """Adapt a 1-D ODE to the vector signature.

    # C++ parity: ``detail::OdeFctWrapper`` — adaptiverungekutta.hpp:131-143.
    Exists in C++ only so the scalar overload can delegate to the vector one;
    kept as a real class because the delegation is observable — the scalar
    entry point runs the *vector* step controller over a length-1 state.
    """

    __slots__ = ("_ode1d",)

    def __init__(self, ode1d: OdeFct1d) -> None:
        self._ode1d: OdeFct1d = ode1d

    def __call__(self, x: float, y: Sequence[float]) -> list[float]:
        return [self._ode1d(x, y[0])]


class AdaptiveRungeKutta:
    """Cash-Karp 4/5 with adaptive step size.

    # C++ parity: ``template <class T> class AdaptiveRungeKutta`` —
    # adaptiverungekutta.hpp:38-96 and the out-of-line members at :99-247.

    Parameters
    ----------
    eps: prescribed error for the solution.
    h1: start step size.
    hmin: smallest step size allowed.
    """

    __slots__ = ("_eps", "_h1", "_hmin")

    def __init__(self, eps: float = 1.0e-6, h1: float = 1.0e-4, hmin: float = 0.0) -> None:
        self._eps: float = eps
        self._h1: float = h1
        self._hmin: float = hmin

    def solve(self, ode: OdeFct, y1: Sequence[float], x1: float, x2: float) -> list[float]:
        """Integrate ``f'(x) = F(x, f(x))`` from ``x1`` to ``x2``.

        # C++ parity: the vector ``operator()`` — adaptiverungekutta.hpp:100-129.
        """
        n = len(y1)
        y = [float(v) for v in y1]
        y_scale = [0.0] * n
        x = x1
        h = self._h1 * (1 if x1 <= x2 else -1)

        for _ in range(1, _MAXSTP + 1):
            dydx = ode(x, y)
            for i in range(n):
                y_scale[i] = abs(y[i]) + abs(dydx[i] * h) + _TINY
            if (x + h - x2) * (x + h - x1) > 0.0:
                h = x2 - x
            x, _hdid, hnext = self._rkqs(y, dydx, x, h, self._eps, y_scale, ode)

            if (x - x2) * (x2 - x1) >= 0.0:
                return y

            if math.fabs(hnext) <= self._hmin:
                qassert.fail(f"Step size ({hnext}) too small ({self._hmin} min) in AdaptiveRungeKutta")
            h = hnext
        return qassert.fail(f"Too many steps ({_MAXSTP}) in AdaptiveRungeKutta")

    def solve_1d(self, ode: OdeFct1d, y1: float, x1: float, x2: float) -> float:
        """Scalar entry point.

        # C++ parity: the scalar ``operator()`` — adaptiverungekutta.hpp:147-154.
        """
        return self.solve(OdeFctWrapper(ode), [y1], x1, x2)[0]

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
        """One quality-controlled step. Mutates ``y``; returns (x, hdid, hnext).

        # C++ parity: ``rkqs`` — adaptiverungekutta.hpp:156-217. C++ passes
        # ``x``, ``hdid`` and ``hnext`` by reference; Python returns them.
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
                htemp1 = _SAFETY * h * math.pow(errmax, _PSHRINK)
                htemp2 = h / 10
                # C++ keeps these as explicit ternaries rather than std::min /
                # std::max because of a VC++14 inlining bug; the shape is
                # preserved so the tie-breaking on equal values matches.
                max_positive = htemp1 if htemp1 > htemp2 else htemp2
                max_negative = htemp1 if htemp1 < htemp2 else htemp2
                h = max_positive if h >= 0.0 else max_negative
                xnew = x + h
                if xnew == x:
                    qassert.fail(f"Stepsize underflow ({h} at x = {x}) in AdaptiveRungeKutta::rkqs")
                continue
            hnext = _SAFETY * h * math.pow(errmax, _PGROW) if errmax > _ERRCON else 5.0 * h
            hdid = h
            x += hdid
            for i in range(n):
                y[i] = ytemp[i]
            return x, hdid, hnext

    @staticmethod
    def _rkck(
        y: Sequence[float],
        dydx: Sequence[float],
        x: float,
        h: float,
        derivs: OdeFct,
    ) -> tuple[list[float], list[float]]:
        """One Cash-Karp step; returns (yout, yerr).

        # C++ parity: ``rkck`` — adaptiverungekutta.hpp:219-247.
        """
        n = len(y)

        # first step
        ytemp = [y[i] + _B21 * h * dydx[i] for i in range(n)]

        # second step
        ak2 = derivs(x + _A2 * h, ytemp)
        ytemp = [y[i] + h * (_B31 * dydx[i] + _B32 * ak2[i]) for i in range(n)]

        # third step
        ak3 = derivs(x + _A3 * h, ytemp)
        ytemp = [y[i] + h * (_B41 * dydx[i] + _B42 * ak2[i] + _B43 * ak3[i]) for i in range(n)]

        # fourth step
        ak4 = derivs(x + _A4 * h, ytemp)
        ytemp = [
            y[i] + h * (_B51 * dydx[i] + _B52 * ak2[i] + _B53 * ak3[i] + _B54 * ak4[i]) for i in range(n)
        ]

        # fifth step
        ak5 = derivs(x + _A5 * h, ytemp)
        ytemp = [
            y[i] + h * (_B61 * dydx[i] + _B62 * ak2[i] + _B63 * ak3[i] + _B64 * ak4[i] + _B65 * ak5[i])
            for i in range(n)
        ]

        # sixth step
        ak6 = derivs(x + _A6 * h, ytemp)
        yout = [y[i] + h * (_C1 * dydx[i] + _C3 * ak3[i] + _C4 * ak4[i] + _C6 * ak6[i]) for i in range(n)]
        yerr = [
            h * (_DC1 * dydx[i] + _DC3 * ak3[i] + _DC4 * ak4[i] + _DC5 * ak5[i] + _DC6 * ak6[i])
            for i in range(n)
        ]
        return yout, yerr


__all__ = ["AdaptiveRungeKutta", "OdeFct", "OdeFct1d", "OdeFctWrapper"]
