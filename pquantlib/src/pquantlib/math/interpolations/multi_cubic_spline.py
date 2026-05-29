"""MultiCubicSpline — n-D cubic interpolation on a rectilinear grid.

# C++ parity: ql/math/interpolations/multicubicspline.hpp (v1.42.1).

The C++ ``MultiCubicSpline`` is 571 LOC of templated machinery that
recursively composes 1-D natural cubic splines over each axis of a
``MultiArray<n>``. PQuantLib delegates to
``scipy.interpolate.RegularGridInterpolator(method='cubic')`` for the
2-D / 3-D cases (and supports n-D in general); for the 1-D degenerate
case it falls back to ``scipy.interpolate.CubicSpline(bc_type='natural')``
(matching the 1-D ``CubicNaturalSpline`` already ported in Phase 9 L9-A).

**Documented divergence — boundary condition.** scipy's
``RegularGridInterpolator(method='cubic')`` uses a *Hermite cubic*
piecewise interpolation: the per-axis 1-D segments are
``Bernstein/de-Boor`` cubics matched to numerical derivative estimates,
not the natural-spline tridiagonal solve used by QuantLib's
``MultiCubicSpline``. At pillar nodes both implementations roundtrip
exactly (both interpolate, neither approximates). At off-pillar
interior points on a coarse grid the two BCs measurably disagree
(observed ~1e-4 magnitude on the L10-C probe's 4x4 grid). For the
typical use-case (vol surfaces with 10+ points per axis) the
difference is below the tier-LOOSE threshold.

**Indexing convention.** Matches the existing
:class:`~pquantlib.math.interpolations.bicubic_spline.BicubicSpline`:
the 2-D ``z`` matrix is indexed as ``z[y_index, x_index]`` — rows
are y, columns are x. For n>=3 the indexing is in *grid order*:
``values[i_0, i_1, ..., i_{n-1}]`` where ``i_k`` indexes
``grid[k]``. This matches scipy's
``RegularGridInterpolator(points=grid, values=values)`` contract.

This class is *not* a subclass of :class:`Interpolation` /
:class:`Interpolation2D` because it generalizes to n>2; the abstract
bases only specialize the 1-D and 2-D cases. The public API exposes
``__call__(point)`` for evaluation and ``x_min``/``x_max`` properties
on each axis via :meth:`axis_range`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.interpolate import (  # type: ignore[import-untyped]
    CubicSpline,
    RegularGridInterpolator,
)

from pquantlib import qassert
from pquantlib.math.array import Array
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

    For ``len(grid) == 1`` we fall back to
    ``scipy.interpolate.CubicSpline(bc_type='natural')`` to match the
    1-D ``CubicNaturalSpline`` already ported in L9-A.
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
        self._spline: Any
        self._n_dim: int = len(axes)
        if self._n_dim == 1:
            # Match L9-A CubicNaturalSpline (natural BC, scipy.CubicSpline).
            self._spline = CubicSpline(
                axes[0], vals, bc_type="natural", extrapolate=True
            )
        else:
            # 2-D and higher — delegate to scipy's cubic RGI.
            # ``method='cubic'`` is the Hermite cubic; available since
            # scipy 1.9.
            # ``fill_value=None`` is a sentinel asking scipy to extrapolate
            # (rather than return ``nan`` or a fixed fill); the C++
            # MultiCubicSpline supports the same via
            # ``Extrapolator::enableExtrapolation()``. The argument type
            # in scipy's stubs is ``float`` despite accepting ``None``;
            # we cast through ``Any`` and pyright-ignore the assignment.
            self._spline = RegularGridInterpolator(
                tuple(axes),
                vals,
                method="cubic",
                bounds_error=False,
                fill_value=None,  # type: ignore[arg-type]
            )

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
        """
        if self._n_dim == 1:
            if isinstance(point, (int, float)):
                x = float(point)
            else:
                point_arr = np.ascontiguousarray(point, dtype=np.float64)
                qassert.require(
                    point_arr.shape == (1,),
                    f"MultiCubicSpline 1-D call expects scalar; got shape {point_arr.shape}",
                )
                x = float(point_arr[0])
            return float(self._spline(x))
        # n>=2 — RGI expects a (1, n) array of points.
        point_arr = np.atleast_2d(np.ascontiguousarray(point, dtype=np.float64))
        qassert.require(
            point_arr.shape == (1, self._n_dim),
            f"MultiCubicSpline {self._n_dim}-D call expects an n-vector "
            f"(n={self._n_dim}); got shape {point_arr.shape}",
        )
        return float(self._spline(point_arr)[0])
