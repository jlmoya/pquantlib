"""BrownianBridge — Wiener-path construction by bisection.

# C++ parity: ql/methods/montecarlo/brownianbridge.{hpp,cpp} (v1.42.1).

Implements Peter Jaeckel's Brownian-bridge construction (see
"Monte Carlo Methods in Finance", 2002).  The bridge maps an input
sequence of independent Gaussian variates into a sequence of
*Wiener path increments*; combined with the standard deviation of
each step, it lets a path generator preserve the rough-path
structure when paired with a low-discrepancy sequence (Sobol).

Three constructors mirror the C++ overload set:

* ``BrownianBridge(steps=n)`` — uniform unit-time grid ``[1, 2, ..., n]``.
* ``BrownianBridge.from_times(times)`` — explicit times list (0 is
  not included — it's the implicit path start).
* ``BrownianBridge.from_time_grid(time_grid)`` — size is
  ``len(time_grid) - 1`` (the grid's t=0 anchor is dropped).

The Python port keeps the C++ accessor surface verbatim:
``size``, ``times``, ``bridge_index``, ``left_index``, ``right_index``,
``left_weight``, ``right_weight``, ``std_deviation``, ``transform``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.time.time_grid import TimeGrid


class BrownianBridge:
    """Brownian-bridge re-ordering of a Gaussian variate sequence.

    # C++ parity: ``BrownianBridge`` (brownianbridge.{hpp,cpp}).
    """

    __slots__ = (
        "_bridge_index",
        "_left_index",
        "_left_weight",
        "_right_index",
        "_right_weight",
        "_size",
        "_sqrtdt",
        "_std_dev",
        "_t",
    )

    def __init__(self, steps: int) -> None:
        """Unit-time grid ``[1, 2, ..., steps]``.

        # C++ parity: ``BrownianBridge(Size steps)`` (brownianbridge.cpp:34-41).
        """
        self._size: int = steps
        # Note: float() cast keeps Python int↔float symmetry with C++ Time=double.
        self._t: list[float] = [float(i + 1) for i in range(steps)]
        self._init_arrays()
        self._initialize()

    @classmethod
    def from_times(cls, times: Sequence[float]) -> BrownianBridge:
        """Explicit times list (must be > 0 and sorted; t=0 implicit).

        # C++ parity: ``BrownianBridge(const std::vector<Time>&)``.
        """
        bb = cls.__new__(cls)
        bb._size = len(times)
        bb._t = list(times)
        bb._init_arrays()
        bb._initialize()
        return bb

    @classmethod
    def from_time_grid(cls, time_grid: TimeGrid) -> BrownianBridge:
        """Drop the t=0 anchor and use the remaining grid times.

        # C++ parity: ``BrownianBridge(const TimeGrid&)``.
        """
        size = len(time_grid) - 1
        bb = cls.__new__(cls)
        bb._size = size
        bb._t = [time_grid[i + 1] for i in range(size)]
        bb._init_arrays()
        bb._initialize()
        return bb

    def _init_arrays(self) -> None:
        # Pre-allocate scratch buffers (matches C++ default-init in ctor body).
        n = self._size
        self._sqrtdt: list[float] = [0.0] * n
        self._bridge_index: list[int] = [0] * n
        self._left_index: list[int] = [0] * n
        self._right_index: list[int] = [0] * n
        self._left_weight: list[float] = [0.0] * n
        self._right_weight: list[float] = [0.0] * n
        self._std_dev: list[float] = [0.0] * n

    def _initialize(self) -> None:
        """Populate index + weight arrays via Jaeckel's bisection.

        # C++ parity: ``BrownianBridge::initialize`` (brownianbridge.cpp:60-110).
        """
        # 1) sqrtdt[0] = sqrt(t[0]); sqrtdt[i] = sqrt(t[i] - t[i-1])
        self._sqrtdt[0] = math.sqrt(self._t[0])
        for i in range(1, self._size):
            self._sqrtdt[i] = math.sqrt(self._t[i] - self._t[i - 1])

        # 2) map[i] tracks which path-point is populated by which variate.
        path_map: list[int] = [0] * self._size

        # 3) Last point is constructed from the first variate.
        path_map[self._size - 1] = 1
        self._bridge_index[0] = self._size - 1
        self._std_dev[0] = math.sqrt(self._t[self._size - 1])
        self._left_weight[0] = 0.0
        self._right_weight[0] = 0.0

        # 4) Bisection loop: find unpopulated points and assign each to
        # the next variate.
        j = 0
        for i in range(1, self._size):
            # Find next unpopulated entry
            while path_map[j] != 0:
                j += 1
            k = j
            # Find next populated entry to the right of j
            while path_map[k] == 0:
                k += 1
            # l-1 is the index of the point to be constructed next
            ell = j + ((k - 1 - j) >> 1)
            path_map[ell] = i
            self._bridge_index[i] = ell
            self._left_index[i] = j
            self._right_index[i] = k
            if j != 0:
                t_jm1 = self._t[j - 1]
                t_k = self._t[k]
                t_ell = self._t[ell]
                width = t_k - t_jm1
                self._left_weight[i] = (t_k - t_ell) / width
                self._right_weight[i] = (t_ell - t_jm1) / width
                self._std_dev[i] = math.sqrt((t_ell - t_jm1) * (t_k - t_ell) / width)
            else:
                t_k = self._t[k]
                t_ell = self._t[ell]
                self._left_weight[i] = (t_k - t_ell) / t_k
                self._right_weight[i] = t_ell / t_k
                self._std_dev[i] = math.sqrt(t_ell * (t_k - t_ell) / t_k)
            j = k + 1
            if j >= self._size:
                j = 0  # wrap around

    # --- inspectors --------------------------------------------------------

    def size(self) -> int:
        """Number of bridge points (= number of input variates)."""
        return self._size

    def times(self) -> tuple[float, ...]:
        """Bridge time points."""
        return tuple(self._t)

    def bridge_index(self) -> tuple[int, ...]:
        """For each variate i, the path-point index it constructs."""
        return tuple(self._bridge_index)

    def left_index(self) -> tuple[int, ...]:
        """For each variate i, the left anchor's path index."""
        return tuple(self._left_index)

    def right_index(self) -> tuple[int, ...]:
        """For each variate i, the right anchor's path index."""
        return tuple(self._right_index)

    def left_weight(self) -> tuple[float, ...]:
        """For each variate i, the left anchor's linear-blend weight."""
        return tuple(self._left_weight)

    def right_weight(self) -> tuple[float, ...]:
        """For each variate i, the right anchor's linear-blend weight."""
        return tuple(self._right_weight)

    def std_deviation(self) -> tuple[float, ...]:
        """For each variate i, the standard deviation of its contribution."""
        return tuple(self._std_dev)

    # --- transform --------------------------------------------------------

    def transform(self, variates: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Map an iid Gaussian sequence to bridge variations.

        # C++ parity: ``BrownianBridge::transform`` (brownianbridge.hpp:107-137).

        Given ``size`` independent unit-normal variates, returns a
        ``size``-length array of *Wiener-path increments* — each entry
        ``[i]`` is the ``W(t[i]) - W(t[i-1])`` normalized to unit
        time-step (i.e. divided by ``sqrtdt[i]``).  Multiply by
        ``sqrt(t[i] - t[i-1])`` to recover the raw Brownian increments.

        Note: C++ uses two iterators (``begin``, ``end``) plus an
        output iterator; the Python port takes a single ndarray of
        length ``size`` and returns a new ndarray of the same length.
        """
        variates = np.asarray(variates, dtype=np.float64)
        qassert.require(
            variates.size == self._size,
            f"incompatible sequence size: {variates.size} != {self._size}",
        )
        output = np.zeros(self._size, dtype=np.float64)
        # We use output to store the path...
        output[self._size - 1] = self._std_dev[0] * float(variates[0])
        for i in range(1, self._size):
            ell = self._bridge_index[i]
            j = self._left_index[i]
            k = self._right_index[i]
            if j != 0:
                output[ell] = (
                    self._left_weight[i] * output[j - 1]
                    + self._right_weight[i] * output[k]
                    + self._std_dev[i] * float(variates[i])
                )
            else:
                output[ell] = (
                    self._right_weight[i] * output[k]
                    + self._std_dev[i] * float(variates[i])
                )
        # ...then back-difference + normalize by sqrt(dt).
        # NOTE: C++ iterates ``for (Size i = size_-1; i >= 1; --i)`` with
        # post-decrement on the inner ``output[i] -= output[i-1]``. Python
        # range goes (size-1, 0, -1) — same semantics.
        for i in range(self._size - 1, 0, -1):
            output[i] -= output[i - 1]
            output[i] /= self._sqrtdt[i]
        output[0] /= self._sqrtdt[0]
        return output


__all__ = ["BrownianBridge"]
