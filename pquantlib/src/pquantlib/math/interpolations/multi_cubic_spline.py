"""MultiCubicSpline — n-D cubic interpolation on a rectilinear grid.

# C++ parity: ql/math/interpolations/multicubicspline.hpp (v1.43).

C++ ``MultiCubicSpline<N>`` is 571 lines of template recursion, but the
algorithm underneath is small: a **tensor product of 1-D natural cubic
splines**. ``detail::base_cubic_spline`` is the Numerical-Recipes tridiagonal
solve with ``y2[0] = y2[dim] = 0`` — natural boundary conditions — and
``detail::n_cubic_splint`` recursively collapses one axis at a time, splining
the collapsed values along the next axis up. It is exactly
:class:`~pquantlib.math.interpolations.bicubic_spline.BicubicSpline`
generalised to n dimensions.

**This module used to delegate to**
``scipy.interpolate.RegularGridInterpolator(method='cubic')``, whose per-axis
segments are *local* cubics fitted to numerical derivative estimates rather
than a global natural spline. The old module docstring called that a
"documented divergence" of "~1e-4 magnitude"; measured against the v1.43
probe on a non-uniform 5x4 grid the actual disagreement is **8 %** in 2-D and
**9 %** in 3-D. The recursion is now transcribed and agrees with C++ to
4e-16.

Because the natural-spline operator on each axis is linear and the operators
act on different axes, they commute — the collapse order does not change the
answer, only the last-bit rounding. This port collapses the last axis first,
which lets the innermost layer of splines be built once at construction (the
only layer that does not depend on the query point).

**Point count.** C++ requires **4** points on every axis
(``set_shared_increments`` needs ``size() - 1 > 2``). This port keeps the
2-point minimum it already had, since the 1-D natural spline is well defined
there; a 2- or 3-point axis simply has no C++ counterpart to cross-validate
against.

**Boundary caveat.** The C++ header carries a standing ``\bug`` note: "cannot
interpolate at the grid points on the boundary surface of the N-dimensional
region", and the C++ test-suite only ever checks strictly interior nodes. The
reference probe follows that restriction. This port has no such limitation —
it reproduces the grid values on the boundary too — so the boundary is
deliberately *not* cross-validated.

**Indexing convention.** ``values[i_0, i_1, ..., i_{n-1}]`` where ``i_k``
indexes ``grid[k]`` — grid order, matching C++ ``y[i][j][k]``. For ``n == 2``
that means ``values[x_index, y_index]``, which is the **transpose** of the
``z[y, x]`` convention used by
:class:`~pquantlib.math.interpolations.bicubic_spline.BicubicSpline` and the
rest of the 2-D family. C++ has the same split, for the same reason: this
class is indexed by grid axis, the 2-D family by (row, column).

This class is *not* a subclass of :class:`Interpolation` /
:class:`Interpolation2D` because it generalizes to n>2; the abstract
bases only specialize the 1-D and 2-D cases. The public API exposes
``__call__(point)`` for evaluation and ``x_min``/``x_max`` properties
on each axis via :meth:`axis_range`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.interpolations.cubic_interpolation import CubicNaturalSpline
from pquantlib.math.matrix import Matrix


class MultiCubicSpline:
    """n-D cubic interpolation on a rectilinear grid.

    # C++ parity: ``MultiCubicSpline<N>``
    #             (multicubicspline.hpp:50+).

    Args:
        grid: list of 1-D axis sequences. Length determines the
            dimensionality. Each axis must be strictly ascending and
            have at least 2 points (4 for the cubic to be exact).
        values: n-D array of values shaped ``(len(grid[0]),
            len(grid[1]), ..., len(grid[n-1]))``.

    For ``len(grid) == 1`` this degenerates to a plain
    :class:`~pquantlib.math.interpolations.cubic_interpolation.CubicNaturalSpline`,
    which is what the tensor product collapses to.
    """

    def __init__(
        self,
        grid: Sequence[Sequence[float] | np.ndarray | Array],
        values: np.ndarray | Matrix,
    ) -> None:
        if len(grid) == 0:
            qassert.require(False, "MultiCubicSpline requires at least one axis")
        axes = [np.ascontiguousarray(g, dtype=np.float64) for g in grid]
        for k, axis in enumerate(axes):
            qassert.require(
                axis.ndim == 1,
                f"MultiCubicSpline axis {k} must be 1-D; got shape {axis.shape}",
            )
            qassert.require(
                axis.shape[0] >= 2,
                f"MultiCubicSpline axis {k} needs >=2 points; got {axis.shape[0]}",
            )
            qassert.require(
                bool(np.all(np.diff(axis) > 0.0)),
                f"MultiCubicSpline axis {k} must be strictly ascending",
            )
        vals = np.ascontiguousarray(values, dtype=np.float64)
        expected_shape = tuple(a.shape[0] for a in axes)
        qassert.require(
            vals.shape == expected_shape,
            f"MultiCubicSpline values shape {vals.shape} does not match "
            f"grid shape {expected_shape}",
        )
        self._axes: list[np.ndarray] = axes
        self._values: np.ndarray = vals
        self._n_dim: int = len(axes)
        # The innermost layer of the recursion — one natural cubic spline per
        # line along the LAST axis — does not depend on the query point, so it
        # is built once here. C++ likewise precomputes all second derivatives
        # (``y2_``) in the constructor.
        last = axes[-1]
        self._inner: list[CubicNaturalSpline] = [
            CubicNaturalSpline(last, line) for line in vals.reshape(-1, last.shape[0])
        ]
        self._inner_shape: tuple[int, ...] = vals.shape[:-1]

    # --- inspectors -------------------------------------------------------

    @property
    def n_dim(self) -> int:
        return self._n_dim

    def axis(self, k: int) -> np.ndarray:
        qassert.require(
            0 <= k < self._n_dim,
            f"axis index {k} out of range [0, {self._n_dim})",
        )
        return self._axes[k].copy()

    def axis_range(self, k: int) -> tuple[float, float]:
        a = self.axis(k)
        return float(a[0]), float(a[-1])

    # --- evaluation -------------------------------------------------------

    def __call__(self, point: float | Sequence[float] | np.ndarray) -> float:
        """Evaluate the spline at a single point.

        For n=1, ``point`` is a scalar.
        For n>=2, ``point`` is a length-n sequence.

        # C++ parity: ``MultiCubicSpline<i>::operator()`` ->
        # ``detail::n_cubic_splint`` / ``detail::base_cubic_spline``: collapse
        # one axis at a time with a natural cubic spline, feeding the results
        # up to the next axis.
        """
        if self._n_dim == 1 and isinstance(point, (int, float)):
            coords = [float(point)]
        else:
            point_arr = np.ascontiguousarray(point, dtype=np.float64).ravel()
            qassert.require(
                point_arr.shape == (self._n_dim,),
                f"MultiCubicSpline {self._n_dim}-D call expects an n-vector "
                f"(n={self._n_dim}); got shape {point_arr.shape}",
            )
            coords = [float(c) for c in point_arr]

        # Innermost layer: the cached last-axis splines, evaluated at the last
        # coordinate. Extrapolation is allowed at every internal layer, as in
        # C++ where the recursion calls the raw splint with no range check.
        current = np.array(
            [s(coords[-1], allow_extrapolation=True) for s in self._inner],
            dtype=np.float64,
        ).reshape(self._inner_shape)

        # Then the remaining axes, last to first. Each step splines along the
        # trailing axis of ``current`` and drops it.
        for k in range(self._n_dim - 2, -1, -1):
            axis = self._axes[k]
            flat = current.reshape(-1, axis.shape[0])
            current = np.array(
                [
                    CubicNaturalSpline(axis, line)(coords[k], allow_extrapolation=True)
                    for line in flat
                ],
                dtype=np.float64,
            ).reshape(current.shape[:-1])

        return float(current.item())
