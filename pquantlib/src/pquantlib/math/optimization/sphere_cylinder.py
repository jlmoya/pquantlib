"""Sphere-cylinder closest-point optimizer.

# C++ parity: ql/math/optimization/spherecylinder.{hpp,cpp} (v1.42.1).

Given a sphere centred at the origin (radius ``r``), a vertical cylinder centred
at ``(alpha, 0)`` (radius ``s``) and a point ``Z`` in R^3, find the point on the
sphere-cylinder intersection closest to ``Z`` (the intersection may be empty).
This is the inner solver of the max-homogeneity caplet calibration's
single-rate closest-point finder.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from pquantlib import qassert


def _brent_minimize(
    low: float,
    mid: float,
    high: float,
    tolerance: float,
    max_it: int,
    objective_function: Callable[[float], float],
) -> float:
    # C++ parity: template BrentMinimize<F> in spherecylinder.cpp (anonymous ns)
    # — golden-section search (NOT Brent despite the name).
    w = 0.5 * (3.0 - math.sqrt(5.0))
    x = w * low + (1 - w) * high
    if low < mid < high:
        x = mid
    mid_value = objective_function(x)

    iterations = 0
    while high - low > tolerance and iterations < max_it:
        if x - low > high - x:  # left interval is bigger
            tentative_new_mid = w * low + (1 - w) * x
            tentative_new_mid_value = objective_function(tentative_new_mid)
            if tentative_new_mid_value < mid_value:  # go left
                high = x
                x = tentative_new_mid
                mid_value = tentative_new_mid_value
            else:  # go right
                low = tentative_new_mid
        else:
            tentative_new_mid = w * x + (1 - w) * high
            tentative_new_mid_value = objective_function(tentative_new_mid)
            if tentative_new_mid_value < mid_value:  # go right
                low = x
                x = tentative_new_mid
                mid_value = tentative_new_mid_value
            else:  # go left
                high = tentative_new_mid
        iterations += 1
    return x


class SphereCylinderOptimizer:
    """Closest point of a sphere-cylinder intersection to a given point.

    # C++ parity: SphereCylinderOptimizer.
    """

    def __init__(
        self,
        r: float,
        s: float,
        alpha: float,
        z1: float,
        z2: float,
        z3: float,
        zweight: float = 1.0,
    ) -> None:
        # C++ parity: SphereCylinderOptimizer ctor.
        self._r = r
        self._s = s
        self._alpha = alpha
        self._z1 = z1
        self._z2 = z2
        self._z3 = z3
        self._zweight = zweight

        qassert.require(r > 0, "sphere must have positive radius")
        s = max(s, 0.0)
        qassert.require(alpha > 0, "cylinder centre must have positive coordinate")

        self._non_empty = math.fabs(alpha - s) <= r

        cylinder_inside = r * r - (s + alpha) * (s + alpha)
        if cylinder_inside > 0.0:
            self._top_value = alpha + s
            self._bottom_value = alpha - s
        else:
            self._bottom_value = alpha - s
            tmp = r * r - (s * s + alpha * alpha)
            if tmp <= 0:  # max to left of maximum
                top_value2 = math.sqrt(s * s - tmp * tmp / (4 * alpha * alpha))
                self._top_value = alpha - math.sqrt(s * s - top_value2 * top_value2)
            else:
                self._top_value = alpha + tmp / (2.0 * alpha)

    def is_intersection_non_empty(self) -> bool:
        # C++ parity: SphereCylinderOptimizer::isIntersectionNonEmpty.
        return self._non_empty

    def _objective_function(self, x1: float) -> float:
        # C++ parity: SphereCylinderOptimizer::objectiveFunction.
        x2sq = self._s * self._s - (x1 - self._alpha) * (x1 - self._alpha)
        # a negative number is rounding error
        x2 = math.sqrt(x2sq) if x2sq >= 0.0 else 0.0
        x3 = math.sqrt(self._r * self._r - x1 * x1 - x2 * x2)

        err = 0.0
        err += (x1 - self._z1) * (x1 - self._z1)
        err += (x2 - self._z2) * (x2 - self._z2)
        err += (x3 - self._z3) * (x3 - self._z3) * self._zweight
        return err

    def find_closest(
        self, max_iterations: int, tolerance: float
    ) -> tuple[float, float, float]:
        """Closest point via golden-section minimisation. Returns ``(y1,y2,y3)``.

        # C++ parity: SphereCylinderOptimizer::findClosest (writes y1/y2/y3).
        """
        x1, _x2, _x3 = self.find_by_projection()
        y1 = _brent_minimize(
            self._bottom_value,
            x1,
            self._top_value,
            tolerance,
            max_iterations,
            self._objective_function,
        )
        y2 = math.sqrt(self._s * self._s - (y1 - self._alpha) * (y1 - self._alpha))
        y3 = math.sqrt(self._r * self._r - y1 * y1 - y2 * y2)
        return y1, y2, y3

    def find_by_projection(self) -> tuple[float, float, float]:
        """Closest point by direct projection. Returns ``(y1,y2,y3)``.

        # C++ parity: SphereCylinderOptimizer::findByProjection (returns the
        success flag, writes y1/y2/y3; Python returns just the triple — the
        flag is unused by callers).
        """
        z1moved = self._z1 - self._alpha
        distance = math.sqrt(z1moved * z1moved + self._z2 * self._z2)
        scale = self._s / distance
        y1moved = z1moved * scale
        y1 = self._alpha + y1moved
        y2 = scale * self._z2
        residual = self._r * self._r - y1 * y1 - y2 * y2
        if residual >= 0.0:
            return y1, y2, math.sqrt(residual)
        if not self.is_intersection_non_empty():
            return y1, y2, 0.0
        # intersection non-empty but projection outside sphere: rightmost point
        y1 = self._top_value
        y2 = math.sqrt(self._r * self._r - y1 * y1)
        return y1, y2, 0.0


def sphere_cylinder_optimizer_closest(
    r: float,
    s: float,
    alpha: float,
    z1: float,
    z2: float,
    z3: float,
    max_iterations: int,
    tolerance: float,
    final_weight: float = 1.0,
) -> list[float]:
    """Free-function wrapper returning the closest point as ``[y1, y2, y3]``.

    # C++ parity: sphereCylinderOptimizerClosest.
    """
    optimizer = SphereCylinderOptimizer(r, s, alpha, z1, z2, z3, final_weight)
    qassert.require(
        optimizer.is_intersection_non_empty(),
        "intersection empty so no solution",
    )
    if max_iterations == 0:
        y1, y2, y3 = optimizer.find_by_projection()
    else:
        y1, y2, y3 = optimizer.find_closest(max_iterations, tolerance)
    return [y1, y2, y3]
