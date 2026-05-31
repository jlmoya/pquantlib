"""LaplaceInterpolation — Laplace-equation in-filling of missing grid values.

# C++ parity: ql/experimental/math/laplaceinterpolation.{hpp,cpp} @ v1.42.1 (099987f0).

Reconstruction of missing values on an arbitrary-dimensional (``n >= 1``),
possibly non-equidistant grid by solving the discrete Laplace equation
``laplacian(u) = 0`` with the known values held fixed (Numerical Recipes, 3rd
ed., ch. 3.8). For ``n = 1`` this reduces to linear interpolation with flat
extrapolation.

# C++ parity divergence (delegation): the C++ implementation threads the full
# QuantLib FD stack — ``Predefined1dMesher`` + ``FdmMesherComposite`` +
# ``SecondDerivativeOp`` (assembled to a ``SparseMatrix`` via ``toMatrix``) +
# ``BiCGstab``. The Python port assembles the identical linear system directly
# with :mod:`scipy.sparse` (the second-derivative stencil weights
# ``2/zeta`` are inlined exactly as in ``SecondDerivativeOp``; the corner rows
# use the Numerical-Recipes 3.8.6 weighting) and solves it with
# :func:`scipy.sparse.linalg.bicgstab` — the same Bi-CG-Stab algorithm C++
# uses. The numerical result is identical. See ``docs/carve-outs.md`` Cat. 3.

Missing values are encoded as ``math.nan`` (the pquantlib representation of the
C++ ``Null<Real>`` sentinel for numeric grids, as used throughout the FD
meshers), both in the value function and in the matrix-convenience overload.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from scipy.sparse.linalg import bicgstab  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from pquantlib import qassert


def _is_null(v: float) -> bool:
    # C++ parity: ``Null<Real>()`` is represented as NaN for numeric grids.
    return math.isnan(v)


class LaplaceInterpolation:
    """Laplace-equation reconstruction of missing values on an n-D grid.

    :param y: value function ``y(coordinates) -> float``; missing entries
        return :data:`~pquantlib.types.NULL_REAL`.
    :param x: per-dimension coordinate vectors. Dimensions with a single
        coordinate are projected out (treated as degenerate).
    :param rel_tol: relative tolerance for the Bi-CG-Stab solve.
    :param max_iter_multiplier: iteration cap is ``max_iter_multiplier * N``.
    """

    __slots__ = (
        "_coordinate_included",
        "_dim",
        "_included_dims",
        "_interpolated",
        "_num_included",
        "_strides",
        "_x",
        "_y",
    )

    def __init__(
        self,
        y: Callable[[Sequence[int]], float],
        x: Sequence[Sequence[float]],
        rel_tol: float = 1e-6,
        max_iter_multiplier: int = 10,
    ) -> None:
        self._y = y
        self._x = [list(xi) for xi in x]

        self._coordinate_included = [len(xi) > 1 for xi in self._x]
        self._included_dims = [i for i, inc in enumerate(self._coordinate_included) if inc]
        self._dim = [len(self._x[i]) for i in self._included_dims]
        self._num_included = len(self._dim)
        self._interpolated: npt.NDArray[np.float64] = np.zeros(0)
        # row-major strides over the projected (included) grid.
        self._strides = self._compute_strides(self._dim)

        if self._num_included == 0:
            return

        self._solve(rel_tol, max_iter_multiplier)

    # ---- coordinate plumbing ----

    @staticmethod
    def _compute_strides(dim: Sequence[int]) -> list[int]:
        strides = [1] * len(dim)
        for i in range(len(dim) - 2, -1, -1):
            strides[i] = strides[i + 1] * dim[i + 1]
        return strides

    def _index(self, proj_coord: Sequence[int]) -> int:
        return int(sum(c * s for c, s in zip(proj_coord, self._strides, strict=True)))

    def _projected(self, coordinates: Sequence[int]) -> list[int]:
        return [coordinates[i] for i in self._included_dims]

    def _full(self, projected: Sequence[int]) -> list[int]:
        full = [0] * len(self._coordinate_included)
        for k, i in enumerate(self._included_dims):
            full[i] = projected[k]
        return full

    def _value_at(self, proj_coord: Sequence[int]) -> float:
        if self._num_included == len(self._x):
            return self._y(list(proj_coord))
        return self._y(self._full(proj_coord))

    # ---- system assembly + solve ----

    def _solve(self, rel_tol: float, max_iter_multiplier: int) -> None:
        dim = self._dim
        n_total = 1
        for d in dim:
            n_total *= d
        # per-included-dimension grid coordinates and spacings
        grids = [self._x[i] for i in self._included_dims]

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        rhs = np.zeros(n_total)
        guess = np.zeros(n_total)
        guess_tmp = 0.0

        for coord in itertools.product(*(range(d) for d in dim)):
            count = self._index(coord)
            val = self._value_at(coord)
            if _is_null(val):
                is_corner, corner_h, corner_neighbour = self._corner_info(coord, grids)
                if is_corner:
                    self._add_corner_row(
                        rows, cols, data, count, coord, corner_h, corner_neighbour
                    )
                else:
                    self._add_laplace_row(rows, cols, data, count, coord, grids)
                rhs[count] = 0.0
                guess[count] = guess_tmp
            else:
                rows.append(count)
                cols.append(count)
                data.append(1.0)
                rhs[count] = val
                guess[count] = guess_tmp = val

        mat = csr_matrix((data, (rows, cols)), shape=(n_total, n_total))
        sol, _info = bicgstab(  # pyright: ignore[reportUnknownVariableType]
            mat, rhs, x0=guess, rtol=rel_tol, maxiter=max_iter_multiplier * n_total
        )
        self._interpolated = np.asarray(sol, dtype=np.float64)

    def _corner_info(
        self, coord: Sequence[int], grids: Sequence[Sequence[float]]
    ) -> tuple[bool, list[float], list[int]]:
        """Whether ``coord`` is an all-boundary corner + its NR 3.8.6 data."""
        dim = self._dim
        corner_h = [0.0] * len(dim)
        corner_neighbour = [0] * len(dim)
        is_corner = True
        for d in range(len(dim)):
            g = grids[d]
            if coord[d] == 0:
                corner_h[d] = g[1] - g[0]  # dplus(0)
                corner_neighbour[d] = 1
            elif coord[d] == dim[d] - 1:
                corner_h[d] = g[dim[d] - 1] - g[dim[d] - 2]  # dminus(last)
                corner_neighbour[d] = dim[d] - 2
            else:
                is_corner = False
                break
        return is_corner, corner_h, corner_neighbour

    def _add_corner_row(
        self,
        rows: list[int],
        cols: list[int],
        data: list[float],
        count: int,
        coord: Sequence[int],
        corner_h: Sequence[float],
        corner_neighbour: Sequence[int],
    ) -> None:
        dim = self._dim
        sum_corner_h = sum(corner_h)
        for jdir in range(len(dim)):
            coord_j = list(coord)
            coord_j[jdir] = corner_neighbour[jdir]
            weight = sum(corner_h[i] for i in range(len(dim)) if i != jdir)
            weight = 1.0 if len(dim) == 1 else weight / sum_corner_h
            rows.append(count)
            cols.append(self._index(coord_j))
            data.append(-weight)
        rows.append(count)
        cols.append(count)
        data.append(1.0)

    def _add_laplace_row(
        self,
        rows: list[int],
        cols: list[int],
        data: list[float],
        count: int,
        coord: Sequence[int],
        grids: Sequence[Sequence[float]],
    ) -> None:
        # Sum of per-direction second-derivative stencils (only directions with
        # an interior coordinate contribute; boundary directions are zero).
        diag = 0.0
        for d in range(len(self._dim)):
            c = coord[d]
            if c == 0 or c == self._dim[d] - 1:
                continue
            g = grids[d]
            hm = g[c] - g[c - 1]
            hp = g[c + 1] - g[c]
            lower = 2.0 / (hm * (hm + hp))
            center = -2.0 / (hm * hp)
            upper = 2.0 / (hp * (hm + hp))

            lo = list(coord)
            lo[d] = c - 1
            up = list(coord)
            up[d] = c + 1
            rows.append(count)
            cols.append(self._index(lo))
            data.append(lower)
            rows.append(count)
            cols.append(self._index(up))
            data.append(upper)
            diag += center
        rows.append(count)
        cols.append(count)
        data.append(diag)

    # ---- query ----

    def __call__(self, coordinates: Sequence[int]) -> float:
        """Interpolated value at ``coordinates`` (full, un-projected)."""
        qassert.require(
            len(coordinates) == len(self._x),
            f"expected {len(self._x)} coordinates, got {len(coordinates)}",
        )
        if self._num_included == 0:
            val = self._y(list(coordinates))
            return 0.0 if _is_null(val) else val
        proj = (
            list(coordinates)
            if self._num_included == len(self._x)
            else self._projected(coordinates)
        )
        return float(self._interpolated[self._index(proj)])


def laplace_interpolation(
    matrix: npt.NDArray[np.float64],
    x: Sequence[float] | None = None,
    y: Sequence[float] | None = None,
    rel_tol: float = 1e-6,
    max_iter_multiplier: int = 10,
) -> None:
    """In-fill :data:`~pquantlib.types.NULL_REAL` entries of ``matrix`` in place.

    If the ``x`` (column) or ``y`` (row) grid is omitted an equidistant grid is
    assumed. Mirrors the C++ free function ``laplaceInterpolation(Matrix&, ...)``.
    """
    n_rows, n_cols = matrix.shape
    y_grid = list(y) if y else list(range(n_rows))
    x_grid = list(x) if x else list(range(n_cols))

    interp = LaplaceInterpolation(
        lambda coord: float(matrix[coord[0], coord[1]]),
        [y_grid, x_grid],
        rel_tol,
        max_iter_multiplier,
    )
    for i in range(n_rows):
        for jcol in range(n_cols):
            if _is_null(float(matrix[i, jcol])):
                matrix[i, jcol] = interp([i, jcol])
