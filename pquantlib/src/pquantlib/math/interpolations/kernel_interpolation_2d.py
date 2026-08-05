"""2-D kernel interpolation over a rectangular grid.

# C++ parity: ql/math/interpolations/kernelinterpolation2d.hpp (v1.43).

The 2-D analogue of
:class:`~pquantlib.math.interpolations.kernel_interpolation.KernelInterpolation`
(Hakala/Wystup, *Foreign Exchange Risk*, p. 256)::

    f(X) = sum_n alpha_n K(||X - X_n||) / gamma(X)
    gamma(X) = sum_n K(||X - X_n||)

where ``X = (x, y)``, ``X_n`` runs over all ``xSize * ySize`` grid nodes and
``||.||`` is the Euclidean norm (C++ ``Norm2``) in the raw ``(x, y)``
coordinates — there is no per-axis scaling, so the kernel bandwidth has to
suit *both* axes at once. The weights come from the dense solve
``M alpha = z`` with ``M[r][c] = K(||X_r - X_c||) / gamma(X_r)``.

**Node ordering.** C++ flattens the grid with ``y`` as the outer loop and
``x`` as the inner one, so node ``n`` is ``(x[n % xSize], y[n // xSize])``.
That is the order of both ``alphaVec_`` and the rows/columns of ``M_``.

**z layout — the one QuantLib inconsistency in this file.** C++
``KernelInterpolation2DImpl`` requires ``zData.rows() == xSize`` and
``zData.columns() == ySize`` and reads ``zData_[i][j]`` with ``i`` the x
index: it is the *only* 2-D interpolation in QuantLib indexed ``[x][y]``.
``BilinearInterpolation``, ``BicubicSpline`` and
``BackwardflatLinearInterpolation`` all use ``[y][x]``. PQuantLib
normalises on ``[y][x]`` for every 2-D interpolation, so **this class takes
``z`` shaped ``(len(ys), len(xs))``** like its siblings, and the C++
``zData_[i][j]`` becomes ``z[j, i]`` internally. A C++ matrix has to be
transposed on the way in; the cross-validation reference stores the C++
orientation under ``z`` and the test transposes it explicitly.

**Inversion guard.** C++ checks ``|M alpha - z| < invPrec_`` after the solve,
with ``invPrec_`` defaulting to ``1.0e-10`` — three orders tighter than the
1-D class's ``1e-7``. ``setInverseResultPrecision`` can change it, but since
the constructor already ran ``calculate()`` the new value only bites on the
next :meth:`update`. Both behaviours are reproduced.

The linear-solve note on
:class:`~pquantlib.math.interpolations.kernel_interpolation.KernelInterpolation`
applies here verbatim.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation_2d import Interpolation2D
from pquantlib.math.matrix import Matrix

_DEFAULT_INV_PREC: float = 1.0e-10


def _ieee_divide(numerator: float, denominator: float) -> float:
    """``numerator / denominator`` with C++/IEEE-754 semantics at zero.

    C++ evaluates ``res / gammaFunc(X)`` as a plain double division, so a
    compactly-supported kernel evaluated where *every* node is out of support
    gives ``0.0 / 0.0 == nan`` (and a non-zero numerator would give a signed
    infinity). Python raises ``ZeroDivisionError`` instead, so the IEEE result
    is spelled out. This is reachable: the C++ test-suite's Epanechnikov
    kernel on a grid spaced by 10 has no support at all between the x pillars.
    """
    if denominator == 0.0:
        if numerator == 0.0:
            return math.nan
        return math.copysign(math.inf, numerator)
    return numerator / denominator


class KernelInterpolation2D(Interpolation2D):
    """Kernel interpolation over a rectangular ``(x, y)`` grid.

    # C++ parity: ``class KernelInterpolation2D : public Interpolation2D``
    # (kernelinterpolation2d.hpp:195-212), Impl at lines 55-179.

    Args:
        xs: x abscissae, sorted ascending.
        ys: y abscissae, sorted ascending.
        z: grid values shaped ``(len(ys), len(xs))`` — ``z[j, i]`` is the
            value at ``(xs[i], ys[j])``. Note this is the **transpose** of
            the C++ argument; see the module docstring.
        kernel: any callable taking one float and returning a float.
    """

    __slots__ = ("_alpha", "_inv_prec", "_kernel", "_m", "_x_size", "_xy_size", "_y_size")

    def __init__(
        self,
        xs: Array,
        ys: Array,
        z: Matrix,
        kernel: Callable[[float], float],
    ) -> None:
        super().__init__(xs, ys, z, required_points=2)
        self._x_size: int = self._xs.shape[0]
        self._y_size: int = self._ys.shape[0]
        self._xy_size: int = self._x_size * self._y_size
        self._kernel: Callable[[float], float] = kernel
        self._inv_prec: float = _DEFAULT_INV_PREC
        self._m: Matrix = np.zeros((self._xy_size, self._xy_size), dtype=np.float64)
        self._alpha: Array = np.zeros(self._xy_size, dtype=np.float64)
        # C++ parity: the ctor calls this->update() (kernelinterpolation2d.hpp:210).
        self.update()

    # ----- construction ---------------------------------------------------

    def set_inverse_result_precision(self, inv_prec: float) -> None:
        """Set the post-solve residual bound used by the *next* :meth:`update`.

        # C++ parity: ``setInverseResultPrecision``
        # (kernelinterpolation2d.hpp:102-104). C++ does not re-run the solve
        # either, so a value set after construction only applies from the next
        # ``update()`` onwards.
        """
        self._inv_prec = inv_prec

    def update(self) -> None:
        """Rebuild ``M`` and re-solve for the kernel weights.

        # C++ parity: ``calculate`` -> ``updateAlphaVec``
        # (kernelinterpolation2d.hpp:76, 129-171).
        """
        n = self._xy_size
        xs = self._xs
        ys = self._ys
        m = np.zeros((n, n), dtype=np.float64)
        z_vec = np.zeros(n, dtype=np.float64)

        # Loop order is C++'s: j (y) outer, i (x) inner.
        row = 0
        for j in range(self._y_size):
            for i in range(self._x_size):
                # C++ ``zData_[i][j]`` with zData indexed [x][y]; our z is [y][x].
                z_vec[row] = float(self._z[j, i])
                xk0 = float(xs[i])
                xk1 = float(ys[j])
                # C++ computes ``1/gammaFunc(Xk)`` as a raw double division;
                # see the 1-D class for why that matters.
                tmp = _ieee_divide(1.0, self._gamma(xk0, xk1))
                col = 0
                for j_m in range(self._y_size):
                    for i_m in range(self._x_size):
                        m[row, col] = (
                            self._kernel_abs(xk0, xk1, float(xs[i_m]), float(ys[j_m])) * tmp
                        )
                        col += 1
                row += 1

        self._m = m
        self._alpha = np.ascontiguousarray(np.linalg.solve(m, z_vec), dtype=np.float64)

        # C++ parity: kernelinterpolation2d.hpp:164-170 — strict ``<``.
        diff = np.abs(m @ self._alpha - z_vec)
        for value in diff:
            qassert.require(
                float(value) < self._inv_prec,
                "inversion failed in 2d kernel interpolation",
            )

    # ----- evaluation -----------------------------------------------------

    def _value(self, x: float, y: float) -> float:
        # C++ parity: kernelinterpolation2d.hpp:78-96.
        res = 0.0
        cnt = 0
        for j in range(self._y_size):
            for i in range(self._x_size):
                res += float(self._alpha[cnt]) * self._kernel_abs(
                    x, y, float(self._xs[i]), float(self._ys[j])
                )
                cnt += 1
        return _ieee_divide(res, self._gamma(x, y))

    # ----- inspectors (not in C++: the Impl members are private there) ----

    @property
    def alpha(self) -> Array:
        """The solved kernel weights, as a copy, in C++ node order."""
        return self._alpha.copy()

    @property
    def kernel_matrix(self) -> Matrix:
        """The gamma-normalised kernel matrix ``M``, as a copy."""
        return self._m.copy()

    # ----- helpers --------------------------------------------------------

    def _kernel_abs(self, x0: float, x1: float, y0: float, y1: float) -> float:
        """``K(||X - Y||)`` for the 2-vectors ``X = (x0, x1)``, ``Y = (y0, y1)``.

        # C++ parity: kernelinterpolation2d.hpp:109-111 — ``kernel_(Norm2(X-Y))``,
        # and ``Norm2`` is ``sqrt(DotProduct(v, v))``, so the sum of squares is
        # formed first (not ``math.hypot``, which rescales).
        """
        dx = x0 - y0
        dy = x1 - y1
        return self._kernel(math.sqrt(dx * dx + dy * dy))

    def _gamma(self, x: float, y: float) -> float:
        # C++ parity: kernelinterpolation2d.hpp:113-127.
        res = 0.0
        for j in range(self._y_size):
            for i in range(self._x_size):
                res += self._kernel_abs(x, y, float(self._xs[i]), float(self._ys[j]))
        return res


__all__ = ["KernelInterpolation2D"]
