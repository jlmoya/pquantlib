"""Concentrating1dMesher — 1-D grid concentrated around critical points.

# C++ parity: ql/methods/finitedifferences/meshers/concentrating1dmesher.{hpp,cpp}
# (v1.43).

Two constructors, which are two different algorithms:

* **single critical point** — the classical ``sinh`` transform of Tavella-Randall.
  ``locations[i] = cPoint + density*sinh(c1(1-l) + c2 l)`` with
  ``c1 = asinh((start-cPoint)/density)``, ``c2 = asinh((end-cPoint)/density)``.
  With ``require_c_point`` the uniform ``l = i*dx`` is first bent through a
  three-knot linear map so that one grid node lands exactly on ``cPoint``.

* **several critical points** — solve
  ``y'(x) = a / sqrt(sum_i 1/(beta_i + (y-p_i)^2))``
  with adaptive Runge-Kutta, calibrating ``a`` by Brent so ``y(1) = end``, then
  bend the uniform parameterisation through a linear map so that every point
  flagged ``required`` lands on a node.

Python spells the C++ ``std::pair<Real,Real>`` critical point as an optional
``(c_point, density)`` tuple, and the ``std::tuple<Real,Real,bool>`` vector as
a sequence of ``(point, density, required)`` tuples. The two constructors
become ``__init__`` (single) and the ``from_critical_points`` classmethod
(multiple), because Python cannot overload.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.math.closeness import close, close_enough
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.math.ode.adaptive_runge_kutta import AdaptiveRungeKutta
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.methods.finitedifferences.meshers.fdm_1d_mesher import Fdm1dMesher


class _OdeIntegrationFct:
    """# C++ parity: the anonymous-namespace ``OdeIntegrationFct``."""

    __slots__ = ("_betas", "_points", "_rk")

    def __init__(self, points: Sequence[float], betas: Sequence[float], tol: float) -> None:
        self._points: Sequence[float] = points
        self._betas: Sequence[float] = betas
        self._rk: AdaptiveRungeKutta = AdaptiveRungeKutta(tol)

    def _jac(self, a: float, y: float) -> float:
        s = 0.0
        for p, b in zip(self._points, self._betas, strict=True):
            s += 1.0 / (b + (y - p) ** 2)
        return a / math.sqrt(s)

    def solve(self, a: float, y0: float, x0: float, x1: float) -> float:
        return self._rk.solve_1d(lambda _x, y: self._jac(a, y), y0, x0, x1)


@final
class Concentrating1dMesher(Fdm1dMesher):
    """1-D mesher concentrating nodes around one or more critical points.

    # C++ parity: ``class Concentrating1dMesher : public Fdm1dMesher``.
    """

    def __init__(
        self,
        start: float,
        end: float,
        size: int,
        c_points: tuple[float | None, float | None] = (None, None),
        require_c_point: bool = False,
        critical_points: Sequence[tuple[float, float, bool]] | None = None,
        tol: float = 1e-8,
    ) -> None:
        """# C++ parity: the single-critical-point constructor.

        ``c_points`` is ``(cPoint, density)``; ``None`` stands for C++
        ``Null<Real>()``. Note C++ rescales the density by ``(end - start)``.

        C++ overloads this constructor: the fourth argument is either a
        ``std::pair<Real, Real>`` (one critical point) or a
        ``std::vector<std::tuple<Real, Real, bool>>`` (several, solved through
        the ODE). Python cannot overload on argument type, so the multi-point
        form is reachable two ways — ``critical_points=`` here, or the
        :meth:`from_critical_points` classmethod. Both run the same code.

        Supplying both forms at once is rejected rather than silently resolved:
        in C++ the two are separate overloads, so no caller can express "both",
        and a silent precedence rule would let a caller believe a single
        critical point was honoured when it had been discarded.
        """
        if critical_points is not None:
            qassert.require(
                c_points == (None, None) and not require_c_point,
                "give either c_points or critical_points, not both",
            )
            self._init_from_critical_points(start, end, size, critical_points, tol)
            return

        super().__init__(size)
        qassert.require(end > start, "end must be larger than start")

        c_point = c_points[0]
        density = None if c_points[1] is None else c_points[1] * (end - start)

        qassert.require(
            c_point is None or (start <= c_point <= end),
            "cPoint must be between start and end",
        )
        qassert.require(density is None or density > 0.0, "density > 0 required")
        qassert.require(
            c_point is None or density is not None,
            "density must be given if cPoint is given",
        )
        qassert.require(
            not require_c_point or c_point is not None,
            "cPoint is required in grid but not given",
        )

        dx = 1.0 / (size - 1)

        if c_point is not None:
            assert density is not None
            transform: LinearInterpolation | None = None
            c1 = math.asinh((start - c_point) / density)
            c2 = math.asinh((end - c_point) / density)
            if require_c_point:
                u: list[float] = [0.0]
                z: list[float] = [0.0]
                if not close(c_point, start) and not close(c_point, end):
                    z0 = -c1 / (c2 - c1)
                    # C++: max(min(lround(z0*(size-1)), size-2), 1) / (size-1).
                    u0 = max(min(_lround(z0 * (size - 1)), size - 2), 1) / float(size - 1)
                    u.append(u0)
                    z.append(z0)
                u.append(1.0)
                z.append(1.0)
                transform = LinearInterpolation(
                    np.array(u, dtype=np.float64), np.array(z, dtype=np.float64)
                )

            for i in range(1, size - 1):
                li = transform(i * dx) if transform is not None else i * dx
                self._locations[i] = c_point + density * math.sinh(c1 * (1.0 - li) + c2 * li)
        else:
            for i in range(1, size - 1):
                self._locations[i] = start + i * dx * (end - start)

        self._finalise(start, end, size)

    @classmethod
    def from_critical_points(
        cls,
        start: float,
        end: float,
        size: int,
        c_points: Sequence[tuple[float, float, bool]],
        tol: float = 1e-8,
    ) -> Concentrating1dMesher:
        """# C++ parity: the multi-critical-point (ODE) constructor.

        ``c_points[i]`` is ``(point, density, required)``. Equivalent to
        passing ``critical_points=`` to the constructor.
        """
        self = cls.__new__(cls)
        self._init_from_critical_points(start, end, size, c_points, tol)
        return self

    def _init_from_critical_points(
        self,
        start: float,
        end: float,
        size: int,
        c_points: Sequence[tuple[float, float, bool]],
        tol: float,
    ) -> None:
        """Shared body of the multi-critical-point construction."""
        Fdm1dMesher.__init__(self, size)
        qassert.require(end > start, "end must be larger than start")

        points = [p for p, _, _ in c_points]
        betas = [(d * (end - start)) ** 2 for _, d, _ in c_points]

        # Initial guess for the scaling factor a so that y(1) = end.
        a_init = 0.0
        for p, b in zip(points, betas, strict=True):
            c1 = math.asinh((start - p) / b)
            c2 = math.asinh((end - p) / b)
            a_init += (c2 - c1) / len(points)

        fct = _OdeIntegrationFct(points, betas, tol)
        a = Brent().solve(
            lambda x: fct.solve(x, start, 0.0, 1.0) - end, tol, a_init, 0.1 * a_init
        )

        # Solve the ODE for all grid points.
        x = np.zeros(size, dtype=np.float64)
        y = np.zeros(size, dtype=np.float64)
        y[0] = start
        dx = 1.0 / (size - 1)
        for i in range(1, size):
            x[i] = i * dx
            y[i] = fct.solve(a, float(y[i - 1]), float(x[i - 1]), float(x[i]))

        # Eliminate numerical noise and ensure y(1) = end.
        dy = float(y[-1]) - end
        for i in range(1, size):
            y[i] -= i * dx * dy

        ode_solution = LinearInterpolation(x, y)

        # Ensure required points are part of the grid.
        w: list[tuple[float, float]] = [(0.0, 0.0)]
        for i, (p, _, required) in enumerate(c_points):
            if required and start < points[i] < end:
                j = int(np.searchsorted(y, p, side="left"))
                e = Brent().solve(
                    lambda xx, _p=p: ode_solution(xx, allow_extrapolation=True) - _p,
                    QL_EPSILON,
                    float(x[j]),
                    0.5 / size,
                )
                w.append((min(float(x[size - 2]), float(x[j])), e))
        w.append((1.0, 1.0))
        w.sort()
        # C++: std::unique with equal_on_first (close_enough on .first, n=1000).
        uniq: list[tuple[float, float]] = []
        for item in w:
            if uniq and close_enough(uniq[-1][0], item[0], 1000):
                continue
            uniq.append(item)

        transform = LinearInterpolation(
            np.array([p[0] for p in uniq], dtype=np.float64),
            np.array([p[1] for p in uniq], dtype=np.float64),
        )

        for i in range(size):
            self._locations[i] = ode_solution(transform(i * dx))

        self._finalise(start, end, size, clamp_ends=False)

    # --- shared tail ----------------------------------------------------

    def _finalise(self, start: float, end: float, size: int, clamp_ends: bool = True) -> None:
        if clamp_ends:
            self._locations[0] = start
            self._locations[-1] = end
        for i in range(size - 1):
            self._dplus[i] = self._dminus[i + 1] = (
                self._locations[i + 1] - self._locations[i]
            )
        self._dplus[-1] = math.nan
        self._dminus[0] = math.nan


def _lround(v: float) -> int:
    """``std::lround`` — round half away from zero, not Python's banker's rounding."""
    return math.floor(v + 0.5) if v >= 0.0 else -math.floor(-v + 0.5)


__all__ = ["Concentrating1dMesher"]
