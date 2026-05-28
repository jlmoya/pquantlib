"""Path — single-factor random walk.

# C++ parity: ql/methods/montecarlo/path.hpp (v1.42.1) — ``class Path``.

Thin wrapper around a ``TimeGrid`` plus a ``numpy.ndarray[float64]``
of values, one per grid point.  The first value is the initial asset
value (path includes its starting point).

Python divergences:

* C++ ``Array`` becomes ``numpy.ndarray[float64]`` (consistent with
  L1 carve-out — no separate Array / Matrix classes).
* The C++ ``operator[]`` / ``at`` / ``value`` / ``front`` / ``back``
  / ``time`` / ``length`` accessors all collapse into Pythonic
  equivalents: ``__getitem__``, ``__len__``, ``front()``, ``back()``,
  ``time(i)``, ``length()``, ``values`` property.
* Reverse-iterator support is provided through ``reversed(path.values)``.
* No mutable element access via ``path[i] = x`` is exposed in the
  public API — the C++ mutable overloads of ``operator[]`` are used
  internally by ``PathGenerator`` only; Python's ``PathGenerator``
  rebuilds the values ndarray each call so no in-place setter is
  needed at the Python layer.  ``values`` itself is mutable.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.time.time_grid import TimeGrid


class Path:
    """Single-factor random walk over a discrete time grid.

    # C++ parity: ql/methods/montecarlo/path.hpp ``class Path``.

    Stores a :class:`pquantlib.time.time_grid.TimeGrid` and a
    ``numpy.ndarray[float64]`` of length ``len(time_grid)``.
    """

    __slots__ = ("_time_grid", "_values")

    def __init__(
        self,
        time_grid: TimeGrid,
        values: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """Construct from a grid plus an optional values array.

        # C++ parity: ``Path::Path(TimeGrid, Array)`` (path.hpp:82-88).
        """
        self._time_grid: TimeGrid = time_grid
        if values is None:
            self._values = np.zeros(len(time_grid), dtype=np.float64)
        else:
            self._values = np.asarray(values, dtype=np.float64)
        qassert.require(
            self._values.size == len(self._time_grid),
            "different number of times and asset values",
        )

    # --- inspectors --------------------------------------------------------

    def empty(self) -> bool:
        """True iff the time grid is empty."""
        return self._time_grid.empty()

    def length(self) -> int:
        """Number of points in the path (= grid size).

        # C++ parity: ``Path::length`` returns ``timeGrid_.size()``.
        """
        return len(self._time_grid)

    def __len__(self) -> int:
        return self.length()

    def __getitem__(self, i: int) -> float:
        return float(self._values[i])

    def at(self, i: int) -> float:
        """Bounds-checked element access — mirrors C++ ``Path::at``."""
        qassert.require(0 <= i < self._values.size, f"path index {i} out of range")
        return float(self._values[i])

    def value(self, i: int) -> float:
        """Alias for ``__getitem__`` — mirrors C++ ``Path::value(Size)``."""
        return float(self._values[i])

    def time(self, i: int) -> float:
        """Time at the i-th grid point — mirrors C++ ``Path::time(Size)``."""
        return self._time_grid[i]

    def front(self) -> float:
        """Initial value — mirrors C++ ``Path::front``."""
        return float(self._values[0])

    def back(self) -> float:
        """Final value — mirrors C++ ``Path::back``."""
        return float(self._values[-1])

    @property
    def time_grid(self) -> TimeGrid:
        """The underlying time grid (read-only reference)."""
        return self._time_grid

    @property
    def values(self) -> npt.NDArray[np.float64]:
        """The underlying values array (mutable in-place — C++ parity).

        # C++ parity: the C++ class exposes mutable ``Array&`` accessors
        # so ``PathGenerator`` can write directly into the path. The
        # Python port surfaces the underlying ndarray with the same
        # mutation semantics; callers MUST NOT rebind the array (use
        # ``values[:] = ...`` for an in-place rewrite).
        """
        return self._values

    # --- iteration --------------------------------------------------------

    def __iter__(self):
        return iter(self._values)


__all__ = ["Path"]
