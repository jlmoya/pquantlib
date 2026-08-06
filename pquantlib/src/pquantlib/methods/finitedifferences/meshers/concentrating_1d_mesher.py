"""Concentrating1dMesher — 1-D grid clustered around critical points.

# C++ parity: ql/methods/finitedifferences/meshers/concentrating1dmesher.{hpp,cpp}
# (v1.43).

Two constructions, matching the two C++ overloads:

**Single critical point** (``c_points=(c, density)``) — nodes are laid
out as ``c + density*(end-start)*sinh(c1*(1-l) + c2*l)`` for ``l``
uniform on ``[0, 1]``, which clusters them around ``c``. With
``require_c_point=True`` a piecewise-linear reparameterisation of
``l`` forces ``c`` to land exactly on a node.

**Several critical points** (``critical_points=[(c, density, required),
...]``) — the node positions solve
``y'(x) = a / sqrt(sum_i 1/(beta_i + (y - c_i)^2))`` with ``y(0) =
start``, the scale ``a`` being Brent-solved so that ``y(1) = end``.
The ODE is integrated with :class:`AdaptiveRungeKutta`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.closeness import close, close_enough
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.math.ode.adaptive_runge_kutta import AdaptiveRungeKutta
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.methods.finitedifferences.meshers.fdm_1d_mesher import Fdm1dMesher


def _lround(x: float) -> int:
    """Round half away from zero, as C++ ``std::lround``.

    Python's built-in ``round`` is round-half-to-even, which differs at
    exact ``.5`` — reachable here because ``z0*(size-1)`` lands on a half
    integer whenever the critical point sits midway between two nodes.
    """
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


class _OdeIntegrationFct:
    """ODE right-hand side + integrator for the multi-point construction.

    # C++ parity: the anonymous-namespace ``OdeIntegrationFct`` in
    # concentrating1dmesher.cpp.
    """

    def __init__(self, points: Sequence[float], betas: Sequence[float], tol: float) -> None:
        # C++: AdaptiveRungeKutta<> rk_(tol) — eps = tol, h1 and hmin default.
        self._rk: AdaptiveRungeKutta = AdaptiveRungeKutta(tol)
        self._points: Sequence[float] = points
        self._betas: Sequence[float] = betas

    def solve(self, a: float, y0: float, x0: float, x1: float) -> float:
        return self._rk.solve_1d(lambda x, y: self._jac(a, x, y), y0, x0, x1)

    def _jac(self, a: float, _x: float, y: float) -> float:
        s = 0.0
        for point, beta in zip(self._points, self._betas, strict=True):
            s += 1.0 / (beta + (y - point) * (y - point))
        return a / math.sqrt(s)


@final
class Concentrating1dMesher(Fdm1dMesher):
    """1-D mesher concentrating nodes around one or more critical points.

    # C++ parity: ``class Concentrating1dMesher : public Fdm1dMesher``.

    Exactly one of ``c_points`` / ``critical_points`` may be supplied;
    they select the two C++ constructor overloads.

    Args:
        start: leftmost location (always a node).
        end: rightmost location (always a node); must exceed ``start``.
        size: number of nodes.
        c_points: ``(critical point, density)``; either element may be
            ``None`` for C++'s ``Null<Real>``. ``None`` for the whole
            pair means a uniform grid. The density is scaled by
            ``end - start`` internally, exactly as C++ does.
        require_c_point: force the critical point onto a node.
        critical_points: ``[(point, density, required), ...]`` for the
            multi-point overload.
        tol: Brent / Runge-Kutta tolerance for the multi-point overload.
    """

    def __init__(
        self,
        start: float,
        end: float,
        size: int,
        c_points: tuple[float | None, float | None] | None = None,
        require_c_point: bool = False,
        *,
        critical_points: Sequence[tuple[float, float, bool]] | None = None,
        tol: float = 1e-8,
    ) -> None:
        super().__init__(size)
        qassert.require(end > start, "end must be larger than start")
        qassert.require(
            c_points is None or critical_points is None,
            "pass either c_points (single critical point) or critical_points "
            "(multiple), not both",
        )
        if critical_points is not None:
            self._build_multi(start, end, size, critical_points, tol)
        else:
            self._build_single(start, end, size, c_points, require_c_point)
        self._fill_spacings(size)

    # --- single critical point ------------------------------------------

    def _build_single(
        self,
        start: float,
        end: float,
        size: int,
        c_points: tuple[float | None, float | None] | None,
        require_c_point: bool,
    ) -> None:
        """# C++ parity: the ``std::pair<Real, Real> cPoints`` constructor."""
        c_point = None if c_points is None else c_points[0]
        raw_density = None if c_points is None else c_points[1]
        density = None if raw_density is None else raw_density * (end - start)

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
            c1 = math.asinh((start - c_point) / density)
            c2 = math.asinh((end - c_point) / density)
            transform: LinearInterpolation | None = None
            if require_c_point:
                u = [0.0]
                z = [0.0]
                if not close(c_point, start) and not close(c_point, end):
                    z0 = -c1 / (c2 - c1)
                    u0 = max(min(_lround(z0 * (size - 1)), size - 2), 1) / (size - 1)
                    u.append(u0)
                    z.append(z0)
                u.append(1.0)
                z.append(1.0)
                transform = LinearInterpolation(
                    np.asarray(u, dtype=np.float64), np.asarray(z, dtype=np.float64)
                )
            for i in range(1, size - 1):
                li = transform(i * dx) if transform is not None else i * dx
                self._locations[i] = c_point + density * math.sinh(c1 * (1.0 - li) + c2 * li)
        else:
            for i in range(1, size - 1):
                self._locations[i] = start + i * dx * (end - start)

        self._locations[0] = start
        self._locations[-1] = end

    # --- several critical points ----------------------------------------

    def _build_multi(
        self,
        start: float,
        end: float,
        size: int,
        c_points: Sequence[tuple[float, float, bool]],
        tol: float,
    ) -> None:
        """# C++ parity: the ``vector<tuple<Real, Real, bool>> cPoints`` constructor."""
        points = [float(p[0]) for p in c_points]
        # NOTE: C++ squares density*(end-start) here but then feeds the SQUARED
        # value to asinh's denominator below (rather than the un-squared one),
        # so betas plays both roles. Reproduced verbatim.
        betas = [(float(p[1]) * (end - start)) ** 2 for p in c_points]

        # scaling factor a such that y(1) = end
        a_init = 0.0
        for point, beta in zip(points, betas, strict=True):
            c1 = math.asinh((start - point) / beta)
            c2 = math.asinh((end - point) / beta)
            a_init += (c2 - c1) / len(points)

        fct = _OdeIntegrationFct(points, betas, tol)
        a = Brent().solve(lambda x: fct.solve(x, start, 0.0, 1.0) - end, tol, a_init, 0.1 * a_init)

        # solve the ODE for all grid points
        xs = [0.0] * size
        ys = [0.0] * size
        ys[0] = start
        dx = 1.0 / (size - 1)
        for i in range(1, size):
            xs[i] = i * dx
            ys[i] = fct.solve(a, ys[i - 1], xs[i - 1], xs[i])

        # eliminate numerical noise and ensure y(1) = end
        dy = ys[-1] - end
        for i in range(1, size):
            ys[i] -= i * dx * dy

        xs_arr: Array = np.asarray(xs, dtype=np.float64)
        ys_arr: Array = np.asarray(ys, dtype=np.float64)
        ode_solution = LinearInterpolation(xs_arr, ys_arr)

        # ensure required points are part of the grid
        w: list[tuple[float, float]] = [(0.0, 0.0)]
        for i, point in enumerate(points):
            if c_points[i][2] and start < point < end:
                j = int(np.searchsorted(ys_arr, point, side="left"))
                e = Brent().solve(
                    lambda x, _p=point: ode_solution(x, allow_extrapolation=True) - _p,
                    QL_EPSILON,
                    xs[j],
                    0.5 / size,
                )
                w.append((min(xs[size - 2], xs[j]), e))
        w.append((1.0, 1.0))
        w.sort()
        # C++ std::unique with equal_on_first — drops CONSECUTIVE entries whose
        # first components are close_enough within 1000 epsilons.
        deduped: list[tuple[float, float]] = [w[0]]
        for entry in w[1:]:
            if not close_enough(deduped[-1][0], entry[0], 1000):
                deduped.append(entry)

        u = np.asarray([p[0] for p in deduped], dtype=np.float64)
        z = np.asarray([p[1] for p in deduped], dtype=np.float64)
        transform = LinearInterpolation(u, z)

        for i in range(size):
            self._locations[i] = ode_solution(transform(i * dx))

    # --- shared tail ------------------------------------------------------

    def _fill_spacings(self, size: int) -> None:
        for i in range(size - 1):
            d = float(self._locations[i + 1]) - float(self._locations[i])
            self._dplus[i] = d
            self._dminus[i + 1] = d
        # C++ stores Null<Real> at the two boundary sentinels; the Python
        # Fdm1dMesher convention is NaN (see its docstring).
        self._dplus[-1] = math.nan
        self._dminus[0] = math.nan


__all__ = ["Concentrating1dMesher"]
