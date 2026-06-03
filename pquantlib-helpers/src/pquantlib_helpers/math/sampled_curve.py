"""SampledCurve — a (grid, values) pair sampled from a payoff/curve.

# Retired-API compat layer — superseded in C++ QuantLib v1.42.1 by the modern
# Fdm* curve machinery, but still shipped (``ql/math/sampledcurve.hpp``).

Java parity: ``org.jquantlib.math.SampledCurve``.
C++ parity: ``ql/math/sampledcurve.hpp`` (v1.42.1).

A :class:`SampledCurve` holds an x-grid and a matching values array. The legacy
FD vanilla engines use it to lay out a (log) underlying grid, sample the
intrinsic payoff on it, and extract the option value plus its first/second
central derivatives at the grid centre (``valueAtCenter`` / ``firstDerivative``
/ ``secondDerivativeAtCenter``).

Only the surface consumed by the FD engine layer is ported; ``regrid`` (cubic
re-interpolation onto a new grid) is intentionally omitted — the dividend
engines never call it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pquantlib.exceptions import LibraryException
from pquantlib_helpers.math.grid import bounded_log_grid

if TYPE_CHECKING:
    from collections.abc import Callable

    from pquantlib.math.array import Array


class SampledCurve:
    """A grid of x-values paired with sampled y-values.

    C++ parity: ``class SampledCurve``. Java parity: ``SampledCurve``.
    """

    def __init__(self, arg: int | Array | SampledCurve = 0) -> None:
        """Construct from a size, an existing grid array, or another curve.

        - ``SampledCurve(size)`` — zero grid + zero values of length ``size``.
        - ``SampledCurve(grid_array)`` — adopt ``grid_array`` (a numpy array),
          zero values of matching length.
        - ``SampledCurve(other)`` — DEEP copy of another curve.

        # Java parity: the *copy constructor* ``SampledCurve(SampledCurve)``
        # deep-copies (``grid.clone()`` / ``values.clone()``); the separate
        # ``clone()`` method is a SHALLOW ``Object.clone`` that shares the grid
        # and values array references — see :meth:`clone`.
        """
        if isinstance(arg, SampledCurve):
            self._grid: Array = np.array(arg._grid, dtype=np.float64)
            self._values: Array = np.array(arg._values, dtype=np.float64)
        elif isinstance(arg, int):
            self._grid = np.zeros(arg, dtype=np.float64)
            self._values = np.zeros(arg, dtype=np.float64)
        else:
            # numpy array grid
            self._grid = np.asarray(arg, dtype=np.float64)
            self._values = np.zeros(int(self._grid.shape[0]), dtype=np.float64)

    def clone(self) -> SampledCurve:
        """Return a SHALLOW copy sharing the grid + values array references.

        # Java parity: ``SampledCurve.clone()`` uses ``Object.clone()`` — a
        # field-by-field shallow copy, so the returned curve initially SHARES the
        # ``grid`` and ``values`` numpy arrays with ``self``. This aliasing is
        # load-bearing: in ``FDMultiPeriodEngine`` the dividend-scaling step
        # mutates ``intrinsicValues``'s grid array IN PLACE (``scale_grid``)
        # *before* ``initializeInitialCondition`` un-shares it via ``set_log_grid``,
        # which scales the cloned ``prices`` grid an extra time on the first
        # event. A deep copy here breaks the derivative-at-centre (delta/gamma)
        # cross-validation against the Java FD engine.
        """
        copy = SampledCurve.__new__(SampledCurve)
        copy._grid = self._grid
        copy._values = self._values
        return copy

    def size(self) -> int:
        """Number of grid nodes."""
        return int(self._grid.shape[0])

    def grid(self) -> Array:
        """The x-grid (live reference, as in C++/Java)."""
        return self._grid

    def values(self) -> Array:
        """The sampled values (live reference)."""
        return self._values

    def grid_value(self, i: int) -> float:
        """Grid x-value at node ``i``."""
        return float(self._grid[i])

    def value(self, i: int) -> float:
        """Sampled y-value at node ``i``."""
        return float(self._values[i])

    def empty(self) -> bool:
        """``True`` iff the grid has no nodes."""
        return int(self._grid.shape[0]) == 0

    def set_grid(self, g: Array) -> None:
        """Replace the x-grid. C++ parity: ``setGrid``."""
        self._grid = g

    def set_values(self, v: Array) -> None:
        """Replace the values array. C++ parity: ``setValues``."""
        self._values = v

    def set_log_grid(self, min_: float, max_: float) -> None:
        """Lay out a bounded log-grid of ``size`` nodes from ``min_`` to ``max_``.

        C++ parity: ``setLogGrid`` -> ``BoundedLogGrid(min, max, size() - 1)``.
        """
        self.set_grid(bounded_log_grid(min_, max_, self.size() - 1))

    def sample(self, func: Callable[[float], float]) -> None:
        """Sample ``func`` over the grid into the values array.

        C++ parity: ``sample`` (``values[i] = func(grid[i])``).
        """
        for i in range(int(self._grid.shape[0])):
            self._values[i] = func(float(self._grid[i]))

    def shift_grid(self, s: float) -> None:
        """Add ``s`` to every grid node IN PLACE.

        C++ parity: ``shiftGrid`` -> ``grid.addAssign(s)`` (mutates the array, so
        any curve sharing the grid via a shallow :meth:`clone` sees the shift).
        """
        self._grid += s

    def scale_grid(self, s: float) -> None:
        """Multiply every grid node by ``s`` IN PLACE.

        C++ parity: ``scaleGrid`` -> ``grid.mulAssign(s)`` (mutates the array, so
        any curve sharing the grid via a shallow :meth:`clone` sees the scale).
        """
        self._grid *= s

    def value_at_center(self) -> float:
        """Curve value at the grid centre.

        C++ parity: ``valueAtCenter`` — the mid node for odd sizes, the mean of
        the two central nodes for even sizes.
        """
        if self.empty():
            raise LibraryException("empty sampled curve")
        n = self.size()
        jmid = n // 2
        if n % 2 != 0:
            return float(self._values[jmid])
        return (float(self._values[jmid]) + float(self._values[jmid - 1])) / 2.0

    def first_derivative_at_center(self) -> float:
        """Central first derivative at the grid centre.

        C++ parity: ``firstDerivativeAtCenter``.
        """
        n = self.size()
        if n < 3:
            raise LibraryException("the size of the curve must be at least 3")
        jmid = n // 2
        v = self._values
        g = self._grid
        if n % 2 != 0:
            return (float(v[jmid + 1]) - float(v[jmid - 1])) / (
                float(g[jmid + 1]) - float(g[jmid - 1])
            )
        return (float(v[jmid]) - float(v[jmid - 1])) / (
            float(g[jmid]) - float(g[jmid - 1])
        )

    def second_derivative_at_center(self) -> float:
        """Central second derivative at the grid centre.

        C++ parity: ``secondDerivativeAtCenter``.
        """
        n = self.size()
        if n < 4:
            raise LibraryException("the size of the curve must be at least 4")
        jmid = n // 2
        v = self._values
        g = self._grid
        if n % 2 != 0:
            delta_plus = (float(v[jmid + 1]) - float(v[jmid])) / (
                float(g[jmid + 1]) - float(g[jmid])
            )
            delta_minus = (float(v[jmid]) - float(v[jmid - 1])) / (
                float(g[jmid]) - float(g[jmid - 1])
            )
            d_s = (float(g[jmid + 1]) - float(g[jmid - 1])) / 2.0
            return (delta_plus - delta_minus) / d_s
        delta_plus = (float(v[jmid + 1]) - float(v[jmid - 1])) / (
            float(g[jmid + 1]) - float(g[jmid - 1])
        )
        delta_minus = (float(v[jmid]) - float(v[jmid - 2])) / (
            float(g[jmid]) - float(g[jmid - 2])
        )
        return (delta_plus - delta_minus) / (float(g[jmid]) - float(g[jmid - 1]))


__all__ = ["SampledCurve"]
