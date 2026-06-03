"""TransformedGrid / LogGrid — non-uniform grid spacing helpers for the PDE layer.

# Retired-API compat layer — see package docstring.

C++ parity: ``ql/math/transformedgrid.hpp`` (v1.42.1) — ``TransformedGrid`` and
``LogGrid``. Java parity: ``org.jquantlib.math.TransformedGrid`` /
``org.jquantlib.math.LogGrid``.

A :class:`TransformedGrid` precomputes, for each interior node ``i``, the
backward/forward/total spacings of the (possibly transformed) grid:

- ``dxm[i] = g[i]   - g[i-1]``   (backward step)
- ``dxp[i] = g[i+1] - g[i]``     (forward step)
- ``dx[i]  = dxm[i] + dxp[i]``   (total)

where ``g`` is the transformed grid (``g = grid`` for the base class,
``g = log(grid)`` for :class:`LogGrid`). Boundary entries (``0`` and ``n-1``)
are left at zero, exactly as the C++/Java source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from pquantlib.math.array import Array


class TransformedGrid:
    """Non-uniform grid with precomputed node spacings.

    C++ parity: ``class TransformedGrid``. Java parity: ``TransformedGrid``.
    """

    def __init__(self, grid: Array, f: Callable[[Array], Array] | None = None) -> None:
        """Build from ``grid``, optionally transformed elementwise by ``f``.

        ``f`` is a vectorised transform applied to the whole grid (e.g.
        ``numpy.log`` for the log-grid). When omitted the grid is used as-is.
        """
        self._grid: Array = np.array(grid, dtype=np.float64)
        base: Array = np.array(grid, dtype=np.float64)
        self._transformed_grid: Array = base if f is None else np.asarray(f(base), dtype=np.float64)
        n = int(self._grid.shape[0])
        self._dxm: Array = np.zeros(n, dtype=np.float64)
        self._dxp: Array = np.zeros(n, dtype=np.float64)
        self._dx: Array = np.zeros(n, dtype=np.float64)
        tg = self._transformed_grid
        for i in range(1, n - 1):
            dxm = float(tg[i]) - float(tg[i - 1])
            dxp = float(tg[i + 1]) - float(tg[i])
            self._dxm[i] = dxm
            self._dxp[i] = dxp
            self._dx[i] = dxm + dxp

    def grid_array(self) -> Array:
        """The untransformed grid."""
        return self._grid

    def transformed_grid_array(self) -> Array:
        """The transformed grid."""
        return self._transformed_grid

    def grid(self, i: int) -> float:
        """Untransformed grid value at node ``i``."""
        return float(self._grid[i])

    def transformed_grid(self, i: int) -> float:
        """Transformed grid value at node ``i``."""
        return float(self._transformed_grid[i])

    def dxm(self, i: int) -> float:
        """Backward spacing at interior node ``i``."""
        return float(self._dxm[i])

    def dxp(self, i: int) -> float:
        """Forward spacing at interior node ``i``."""
        return float(self._dxp[i])

    def dx(self, i: int) -> float:
        """Total spacing at interior node ``i``."""
        return float(self._dx[i])

    def size(self) -> int:
        """Number of grid nodes."""
        return int(self._grid.shape[0])


class LogGrid(TransformedGrid):
    """Log-transformed grid (``g = log(grid)``).

    C++ parity: ``class LogGrid``. Java parity: ``LogGrid``.
    """

    def __init__(self, grid: Array) -> None:
        """Build a log-transformed grid from ``grid``."""
        super().__init__(grid, np.log)

    def log_grid_array(self) -> Array:
        """The log-transformed grid."""
        return self.transformed_grid_array()

    def log_grid(self, i: int) -> float:
        """Log-transformed grid value at node ``i``."""
        return self.transformed_grid(i)


__all__ = ["LogGrid", "TransformedGrid"]
