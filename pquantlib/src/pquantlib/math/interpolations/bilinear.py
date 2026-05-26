"""Bilinear interpolation over a 2-D grid.

# C++ parity: ql/math/interpolations/bilinearinterpolation.hpp (v1.42.1)
# + ql/math/interpolations/interpolation2d.hpp ``Interpolation2D``.

Standard textbook bilinear interpolation on a 2-D regular (but not
necessarily uniform) grid. Given sorted ``xs`` (length ``n``) and ``ys``
(length ``m``) and a matrix ``z`` shaped ``(m, n)``, the value at
``(x, y)`` is:

    z(x, y) = (1-t)(1-u)*z[j,i] + t(1-u)*z[j,i+1]
            + (1-t)*u*z[j+1,i] + t*u*z[j+1,i+1]

where ``i = locate_x(x)``, ``j = locate_y(y)``,
``t = (x - xs[i]) / (xs[i+1] - xs[i])``,
``u = (y - ys[j]) / (ys[j+1] - ys[j])``.

**Indexing convention** (matches C++ ``zData_[j][i]``): the matrix is
indexed as ``z[y_index, x_index]`` — rows are y, columns are x. This
matches the test fixture in ``phase1-l1-E-design.md`` where
``z[i, j] = ys[i] + xs[j]``.

The C++ ``Interpolation2D::checkRange`` enforces in-range unless
``allow_extrapolation`` is set; we replicate that semantics on
``__call__``.
"""

from __future__ import annotations

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.closeness import close
from pquantlib.math.matrix import Matrix


class BilinearInterpolation:
    """Bilinear interpolation on a 2-D grid indexed as ``z[y, x]``."""

    def __init__(self, xs: Array, ys: Array, z: Matrix) -> None:
        # Defensive copy + dtype normalization, mirroring the 1-D base.
        xs_arr = np.ascontiguousarray(xs, dtype=np.float64)
        ys_arr = np.ascontiguousarray(ys, dtype=np.float64)
        z_arr = np.ascontiguousarray(z, dtype=np.float64)
        qassert.require(
            xs_arr.ndim == 1 and ys_arr.ndim == 1,
            "BilinearInterpolation requires 1-D xs and ys sequences",
        )
        qassert.require(z_arr.ndim == 2, "BilinearInterpolation requires a 2-D z matrix")
        qassert.require(
            xs_arr.shape[0] >= 2,
            f"not enough x points to interpolate: at least 2 required, {xs_arr.shape[0]} provided",
        )
        qassert.require(
            ys_arr.shape[0] >= 2,
            f"not enough y points to interpolate: at least 2 required, {ys_arr.shape[0]} provided",
        )
        # C++ ``zData_[j][i]`` indexes rows by y and columns by x; the
        # matrix shape is (len(ys), len(xs)).
        qassert.require(
            z_arr.shape == (ys_arr.shape[0], xs_arr.shape[0]),
            f"z matrix shape {z_arr.shape} does not match (len(ys), len(xs)) = "
            f"({ys_arr.shape[0]}, {xs_arr.shape[0]})",
        )
        self._xs: Array = xs_arr
        self._ys: Array = ys_arr
        self._z: Matrix = z_arr
        self._allow_extrapolation: bool = False

    # ----- public API -----------------------------------------------------

    def __call__(self, x: float, y: float, *, allow_extrapolation: bool = False) -> float:
        self._check_range(x, y, allow_extrapolation)
        i = self._locate_x(x)
        j = self._locate_y(y)
        z = self._z
        z1 = float(z[j, i])
        z2 = float(z[j, i + 1])
        z3 = float(z[j + 1, i])
        z4 = float(z[j + 1, i + 1])
        t = (x - float(self._xs[i])) / float(self._xs[i + 1] - self._xs[i])
        u = (y - float(self._ys[j])) / float(self._ys[j + 1] - self._ys[j])
        return (1.0 - t) * (1.0 - u) * z1 + t * (1.0 - u) * z2 + (1.0 - t) * u * z3 + t * u * z4

    @property
    def x_min(self) -> float:
        return float(self._xs[0])

    @property
    def x_max(self) -> float:
        return float(self._xs[-1])

    @property
    def y_min(self) -> float:
        return float(self._ys[0])

    @property
    def y_max(self) -> float:
        return float(self._ys[-1])

    def is_in_range(self, x: float, y: float) -> bool:
        x_ok = (self.x_min <= x <= self.x_max) or close(x, self.x_min) or close(x, self.x_max)
        if not x_ok:
            return False
        return (self.y_min <= y <= self.y_max) or close(y, self.y_min) or close(y, self.y_max)

    @property
    def allows_extrapolation(self) -> bool:
        return self._allow_extrapolation

    def enable_extrapolation(self, b: bool = True) -> None:
        self._allow_extrapolation = b

    # ----- helpers --------------------------------------------------------

    def _locate_x(self, x: float) -> int:
        return _locate_1d(self._xs, x)

    def _locate_y(self, y: float) -> int:
        return _locate_1d(self._ys, y)

    def _check_range(self, x: float, y: float, allow_extrapolation: bool) -> None:
        if allow_extrapolation or self._allow_extrapolation:
            return
        qassert.require(
            self.is_in_range(x, y),
            f"interpolation range is [{self.x_min}, {self.x_max}] x "
            f"[{self.y_min}, {self.y_max}]: extrapolation at ({x}, {y}) not allowed",
        )


def _locate_1d(values: Array, x: float) -> int:
    """C++ ``locateX`` / ``locateY`` parity (Interpolation2D::templateImpl)."""
    n = values.shape[0]
    if x < float(values[0]):
        return 0
    if x > float(values[-1]):
        return n - 2
    return int(np.searchsorted(values[:-1], x, side="right")) - 1
