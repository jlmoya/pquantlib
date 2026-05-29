"""ChebyshevInterpolation — Lagrange interpolation at Chebyshev nodes.

# C++ parity: ql/math/interpolations/chebyshevinterpolation.{hpp,cpp}
#             (v1.42.1).

C++ ``ChebyshevInterpolation`` wires a ``LagrangeInterpolationImpl``
through the Chebyshev nodes of either the *first kind* (Gauss points)
or *second kind* (Gauss-Lobatto endpoints-included points). On the
canonical domain ``[-1, 1]``:

- **FirstKind** :math:`t_i = -\\cos((i + 0.5)\\,\\pi / n)` for
  :math:`i = 0..n-1`. Interior nodes only.
- **SecondKind** :math:`t_i = -\\cos(i\\,\\pi / (n-1))` for
  :math:`i = 0..n-1`. Includes endpoints :math:`t_0 = -1`,
  :math:`t_{n-1} = +1`.

The barycentric weights for these node families are closed-form:

- FirstKind: :math:`w_i = (-1)^i \\sin((i + 0.5)\\,\\pi / n)`.
- SecondKind: :math:`w_0 = w_{n-1} = 0.5 \\cdot (-1)^i`; interior
  :math:`w_i = (-1)^i`.

This port:

1. Generates the canonical Chebyshev nodes on ``[-1, 1]``.
2. Linearly remaps them to ``[x_min, x_max]``.
3. Evaluates via the second-form barycentric Lagrange formula
   (Berrut & Trefethen, 2004 — the same form C++'s
   ``LagrangeInterpolationImpl`` uses).

If a callable ``f`` is supplied at construction time, the y-values
are evaluated as ``f(x_i)`` at the (remapped) nodes.

Cross-validated against the C++ probe in
``migration-harness/cpp/probes/cluster_l10c/probe.cpp`` —
``ChebyshevInterpolation`` matches TIGHT on the standard
``f(x) = sin(x)`` test case.

# C++ parity: the C++ class has no ``x_min``/``x_max`` parameters;
# its nodes live on ``[-1, 1]`` only. We add the canonical linear
# remap because the typical use-case (probability density or option
# payoff interpolation) doesn't live on ``[-1, 1]``. Documented
# divergence: the no-remap call ``ChebyshevInterpolation(n, f)`` is
# faithful to C++; the optional ``x_min``, ``x_max`` arguments are a
# Python-side convenience.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from enum import IntEnum

import numpy as np

from pquantlib import qassert
from pquantlib.math.interpolations.interpolation import Interpolation


class PointsType(IntEnum):
    """Chebyshev node families.

    # C++ parity: ``ChebyshevInterpolation::PointsType``
    #             (chebyshevinterpolation.hpp:36).
    """

    FirstKind = 0
    SecondKind = 1


def chebyshev_nodes_canonical(n: int, points_type: PointsType) -> np.ndarray:
    """Generate the n Chebyshev nodes on ``[-1, 1]``.

    # C++ parity: ``ChebyshevInterpolation::nodes(n, pointsType)``
    #             (chebyshevinterpolation.cpp:58-74).
    """
    qassert.require(n >= 2, f"ChebyshevInterpolation needs n>=2: {n} not allowed")
    if points_type == PointsType.FirstKind:
        idx = np.arange(n, dtype=np.float64)
        return -np.cos((idx + 0.5) * math.pi / n)
    # SecondKind (default) — includes endpoints +/-1.
    idx = np.arange(n, dtype=np.float64)
    return -np.cos(idx * math.pi / (n - 1))


def _barycentric_weights(n: int, points_type: PointsType) -> np.ndarray:
    """Closed-form barycentric weights for the Chebyshev nodes.

    Berrut & Trefethen, *Barycentric Lagrange Interpolation*,
    SIAM Review 46(3):501-517, 2004 — eq. (5.1) and (5.2).
    """
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    if points_type == PointsType.FirstKind:
        # w_i = (-1)^i * sin((i + 0.5) * pi / n)
        idx = np.arange(n, dtype=np.float64)
        return signs * np.sin((idx + 0.5) * math.pi / n)
    # SecondKind
    w = signs.copy()
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def _barycentric_value(
    xs: np.ndarray,
    ys: np.ndarray,
    weights: np.ndarray,
    x: float,
) -> float:
    """Second-form barycentric Lagrange value at ``x``.

    .. math::

       p(x) = \\frac{\\sum_i \\frac{w_i\\,y_i}{x - x_i}}{\\sum_i \\frac{w_i}{x - x_i}}

    If ``x`` coincides with a node, return that node's y-value
    directly (the formula has a removable singularity).
    """
    # Detect coincidence with any node.
    for i in range(xs.shape[0]):
        if math.isclose(x, float(xs[i]), abs_tol=1e-15, rel_tol=1e-13):
            return float(ys[i])
    numer = 0.0
    denom = 0.0
    for i in range(xs.shape[0]):
        diff = x - float(xs[i])
        t = float(weights[i]) / diff
        numer += t * float(ys[i])
        denom += t
    return numer / denom


class ChebyshevInterpolation(Interpolation):
    """Chebyshev-node Lagrange interpolation.

    # C++ parity: ``ChebyshevInterpolation``
    #             (chebyshevinterpolation.hpp:34-58).

    Two construction modes:

    - From an explicit y-array: ``ChebyshevInterpolation(values=..., n=None)``
      — the size of ``values`` determines n, and nodes are computed
      to match.
    - From a callable ``f``: ``ChebyshevInterpolation(n=10, f=lambda x: ..)``
      — evaluates ``f`` at each Chebyshev node (after the optional
      ``[x_min, x_max]`` remap) to seed the y-values.

    Args:
        n: number of nodes (>=2). Required unless ``values`` is given.
        f: callable to evaluate at each node. If ``None`` the user
           may post-construct call :meth:`update_y` with explicit
           y-values.
        points_type: ``FirstKind`` (Gauss points, interior only) or
           ``SecondKind`` (Gauss-Lobatto, endpoints included).
           Default ``SecondKind`` matches C++.
        x_min, x_max: linear remap from ``[-1, 1]``. Default identity
           (``[-1, 1]``). Documented divergence — C++ does not provide
           this; the linear remap is a Python convenience.
        values: pre-computed y-values. Mutually exclusive with ``f``.
    """

    def __init__(
        self,
        n: int | None = None,
        f: Callable[[float], float] | None = None,
        points_type: PointsType = PointsType.SecondKind,
        x_min: float = -1.0,
        x_max: float = 1.0,
        values: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        qassert.require(
            x_max > x_min,
            f"ChebyshevInterpolation requires x_max > x_min: "
            f"got x_min={x_min}, x_max={x_max}",
        )
        if values is not None:
            qassert.require(
                f is None,
                "ChebyshevInterpolation: pass either 'values' or 'f', not both",
            )
            vals_arr = np.ascontiguousarray(values, dtype=np.float64)
            qassert.require(
                vals_arr.ndim == 1 and vals_arr.shape[0] >= 2,
                "ChebyshevInterpolation 'values' must be a 1-D array of "
                f"at least 2 elements; got shape {vals_arr.shape}",
            )
            n_used = vals_arr.shape[0]
        else:
            qassert.require(
                n is not None and n >= 2,
                f"ChebyshevInterpolation 'n' must be >= 2: {n} not allowed",
            )
            assert n is not None
            n_used = n
        # Compute Chebyshev nodes on the canonical domain, then remap.
        t = chebyshev_nodes_canonical(n_used, points_type)
        # Affine map [-1, 1] -> [x_min, x_max]:
        #   x = (x_max + x_min)/2 + t * (x_max - x_min)/2
        half_span = 0.5 * (x_max - x_min)
        mid = 0.5 * (x_max + x_min)
        x_nodes = mid + t * half_span
        if values is not None:
            assert values is not None
            y_nodes = np.ascontiguousarray(values, dtype=np.float64)
        elif f is not None:
            y_nodes = np.array(
                [float(f(float(x))) for x in x_nodes], dtype=np.float64
            )
        else:
            # Allow deferred seeding via ``update_y`` (matches C++ pattern
            # ``ChebyshevInterpolation(n, f)`` only — but Python lets us
            # post-seed for testability).
            y_nodes = np.zeros(n_used, dtype=np.float64)
        super().__init__(x_nodes, y_nodes, required_points=2)
        self._points_type: PointsType = points_type
        self._x_min: float = x_min
        self._x_max: float = x_max
        self._weights: np.ndarray = _barycentric_weights(n_used, points_type)

    # --- public API specific to Chebyshev ---------------------------------

    def update_y(self, y_values: Sequence[float] | np.ndarray) -> None:
        """Replace the y-values at the (already-fixed) Chebyshev nodes.

        # C++ parity: ``ChebyshevInterpolation::updateY``
        #             (chebyshevinterpolation.cpp:76-81).
        """
        y_arr = np.ascontiguousarray(y_values, dtype=np.float64)
        qassert.require(
            y_arr.shape[0] == self._ys.shape[0],
            f"ChebyshevInterpolation.update_y: length mismatch "
            f"(expected {self._ys.shape[0]}, got {y_arr.shape[0]})",
        )
        self._ys = y_arr

    def nodes(self) -> np.ndarray:
        """Return the Chebyshev nodes (after the linear remap).

        # C++ parity: ``ChebyshevInterpolation::nodes()`` (chebyshev
        # interpolation.cpp:54-56).
        """
        return self._xs.copy()

    # --- Interpolation API ------------------------------------------------

    def _value(self, x: float) -> float:
        return _barycentric_value(self._xs, self._ys, self._weights, x)
