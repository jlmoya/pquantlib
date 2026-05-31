"""LagrangeInterpolation — barycentric Lagrange interpolation.

# C++ parity: ql/math/interpolations/lagrangeinterpolation.hpp (v1.42.1).

Berrut-Trefethen barycentric Lagrange interpolation (SIAM Review 46(3),
2004, https://people.maths.ox.ac.uk/trefethen/barycentric.pdf):

    p(x) = sum_i [lambda_i / (x - x_i)] y_i  /  sum_i [lambda_i / (x - x_i)]

with barycentric weights

    lambda_i = 1 / prod_{j != i} cM1 * (x_i - x_j),   cM1 = 4 / (x_max - x_min)

The ``cM1`` rescaling keeps the product magnitudes near unity (numerical
stability) and cancels in the value ratio. A query exactly on a node
returns that node's y-value (handled by the C++ ``close_enough`` /
``lower_bound`` guard, replicated here).

This class is the interpolator the CLV models (NormalCLVModel /
SquareRootCLVModel) use for their collocation mapping. Its distinctive
feature is :meth:`value_with` — evaluate against a *fresh* y-vector
reusing the cached weights (the C++ ``value(const Array&, Real)``
overload backed by ``UpdatedYInterpolation::updatedValue``).

``primitive`` / ``second_derivative`` are intentionally unimplemented
(the C++ class ``QL_FAIL``s on both).
"""

from __future__ import annotations

from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.closeness import close_enough
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.interpolations.interpolation import Interpolation


@final
class LagrangeInterpolation(Interpolation):
    """Barycentric Lagrange interpolation through ``(x, y)``.

    # C++ parity: ``class LagrangeInterpolation : public Interpolation``
    # (Impl at lagrangeinterpolation.hpp:42-134).
    """

    __slots__ = ("_lambda", "_n")

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(x_seq, y_seq, required_points=1)
        self._n: int = self._xs.shape[0]
        self._lambda: Array = np.zeros(self._n, dtype=np.float64)
        self.update()

    def update(self) -> None:
        # C++ parity: LagrangeInterpolationImpl::update (lines 57-70).
        xs = self._xs
        n = self._n
        if n == 1:
            # Single node: the (i != j) product loop never runs, so the
            # cM1 rescaling is never applied. C++ computes cM1 = 4/0 = +inf
            # (harmless, unused); Python would raise, so skip it.
            self._lambda[0] = 1.0
            return
        c_m1 = 4.0 / (float(xs[-1]) - float(xs[0]))
        for i in range(n):
            lam = 1.0
            x_i = float(xs[i])
            for j in range(n):
                if i != j:
                    lam *= c_m1 * (x_i - float(xs[j]))
            self._lambda[i] = 1.0 / lam

    def _value(self, x: float) -> float:
        # C++ parity: LagrangeInterpolationImpl::value -> _value(yBegin_, x).
        return self._eval(self._ys, x)

    def value_with(self, y: Array, x: float) -> float:
        """Interpolate at ``x`` using a fresh y-vector (cached weights).

        # C++ parity: ``LagrangeInterpolation::value(const Array&, Real)``
        # backed by ``UpdatedYInterpolation::updatedValue`` (lines 108-110,
        # 152-155). The CLV mapping functions call this each time the
        # collocation y-points change with the maturity.
        """
        ys = np.ascontiguousarray(y, dtype=np.float64)
        qassert.require(
            ys.shape[0] == self._n,
            f"y vector length {ys.shape[0]} != node count {self._n}",
        )
        return self._eval(ys, x)

    def _eval(self, ys: Array, x: float) -> float:
        # C++ parity: LagrangeInterpolationImpl::_value (lines 113-130).
        xs = self._xs
        n = self._n
        eps = 10.0 * QL_EPSILON * abs(x)
        # lower_bound(xBegin, xEnd, x - eps): first node >= x - eps; if it is
        # within eps of x, snap to that node's y.
        idx = int(np.searchsorted(xs, x - eps, side="left"))
        if idx < n and float(xs[idx]) - x < eps:
            return float(ys[idx])
        # Guard exact node hits where the C++ relies on IEEE inf/inf -> y_i
        # (the lower_bound+eps test misses x == node when eps == 0, e.g. at
        # x == 0). Python raises on /0, so snap explicitly via close_enough.
        numer = 0.0
        denom = 0.0
        for i in range(n):
            dx = x - float(xs[i])
            if close_enough(x, float(xs[i])):
                return float(ys[i])
            alpha = float(self._lambda[i]) / dx
            numer += alpha * float(ys[i])
            denom += alpha
        return numer / denom

    def _derivative(self, x: float) -> float:
        # C++ parity: LagrangeInterpolationImpl::derivative (lines 74-97).
        xs = self._xs
        ys = self._ys
        n = self._n
        numer = 0.0
        denom = 0.0
        numer_d = 0.0
        denom_d = 0.0
        for i in range(n):
            x_i = float(xs[i])
            if close_enough(x, x_i):
                p = 0.0
                for j in range(n):
                    if i != j:
                        p += (
                            float(self._lambda[j])
                            / (x - float(xs[j]))
                            * (float(ys[j]) - float(ys[i]))
                        )
                return p / float(self._lambda[i])
            alpha = float(self._lambda[i]) / (x - x_i)
            alpha_d = -alpha / (x - x_i)
            numer += alpha * float(ys[i])
            denom += alpha
            numer_d += alpha_d * float(ys[i])
            denom_d += alpha_d
        return (numer_d * denom - numer * denom_d) / (denom * denom)


__all__ = ["LagrangeInterpolation"]
