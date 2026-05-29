"""Abstract base for 2-D interpolations.

# C++ parity: ql/math/interpolations/interpolation2d.hpp (v1.42.1) —
# ``class Interpolation2D``.

C++ ``Interpolation2D`` is a PIMPL handle exposing ``value(x, y)``,
``xMin``/``xMax``, ``yMin``/``yMax``, ``locateX``/``locateY``,
``isInRange``, and ``calculate``. The Impl is a templated
``templateImpl<I1, I2, M>`` carrying iterator pairs into externally-owned
x/y sequences and a Matrix view of z.

The Python port collapses this to a concrete abstract class:

- We **copy** ``xs``, ``ys``, and ``z`` into owned numpy arrays at
  construction. Documented divergence (matches the same choice in the
  1-D ``Interpolation`` base).
- ``Extrapolator`` mixin collapses into the ``allow_extrapolation``
  kwarg on ``__call__``.
- ``Impl`` PIMPL hierarchy disappears — each concrete subclass inherits
  directly.

**Indexing convention** (matches C++ ``zData_[j][i]``): the matrix is
indexed as ``z[y_index, x_index]`` — rows are y, columns are x. The
existing ``BilinearInterpolation`` (concrete-only) uses the same
layout, so a ``BilinearInterpolation`` and ``BicubicSpline`` are
drop-in API-compatible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.closeness import close
from pquantlib.math.matrix import Matrix


class Interpolation2D(ABC):
    """Abstract base for 2-D interpolations indexed as ``z[y, x]``.

    Subclasses must implement ``_value(x, y)``.
    """

    def __init__(self, xs: Array, ys: Array, z: Matrix, *, required_points: int = 2) -> None:
        xs_arr = np.ascontiguousarray(xs, dtype=np.float64)
        ys_arr = np.ascontiguousarray(ys, dtype=np.float64)
        z_arr = np.ascontiguousarray(z, dtype=np.float64)
        qassert.require(
            xs_arr.ndim == 1 and ys_arr.ndim == 1,
            "Interpolation2D requires 1-D xs and ys sequences",
        )
        qassert.require(z_arr.ndim == 2, "Interpolation2D requires a 2-D z matrix")
        qassert.require(
            xs_arr.shape[0] >= required_points,
            f"not enough x points to interpolate: at least {required_points} required, "
            f"{xs_arr.shape[0]} provided",
        )
        qassert.require(
            ys_arr.shape[0] >= required_points,
            f"not enough y points to interpolate: at least {required_points} required, "
            f"{ys_arr.shape[0]} provided",
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
        return self._value(x, y)

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

    def update(self) -> None:  # noqa: B027 — intentional default no-op; subclasses override
        """Recompute cached state if the underlying x/y/z data changed.

        Default no-op — subclasses that cache spline coefficients override.
        """

    # ----- hooks for subclasses ------------------------------------------

    @abstractmethod
    def _value(self, x: float, y: float) -> float: ...

    # ----- helpers --------------------------------------------------------

    def _check_range(self, x: float, y: float, allow_extrapolation: bool) -> None:
        if allow_extrapolation or self._allow_extrapolation:
            return
        qassert.require(
            self.is_in_range(x, y),
            f"interpolation range is [{self.x_min}, {self.x_max}] x "
            f"[{self.y_min}, {self.y_max}]: extrapolation at ({x}, {y}) not allowed",
        )
