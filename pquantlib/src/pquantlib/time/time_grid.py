"""TimeGrid — discrete time grid for FDM / Monte Carlo schemes.

# C++ parity: ql/timegrid.hpp + ql/timegrid.cpp (v1.42.1).

Three construction modes (matching C++):

1. ``regular(end, steps)`` — uniformly-spaced grid ``[0, end/steps, ..., end]``.
2. ``with_mandatory(times)`` — exactly the union of ``{0}`` and the sorted
   deduplicated mandatory times.
3. ``with_mandatory_and_steps(times, steps)`` — same as above but extra
   intermediate points are inserted between mandatory points with
   ``dt_max = end / steps`` (or ``min(diff)`` if ``steps == 0``).

The Python port uses classmethods for each construction mode rather than
relying on overloaded constructors (clearer call sites).
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable
from typing import Final

from pquantlib import qassert

# Tolerances for the C++ ``close_enough(a, b, n=42)`` analogue.
# Inlined here rather than borrowed from testing.tolerance to keep
# production code free of test-only module dependencies.
_CLOSE_ENOUGH_ABS: Final[float] = 1e-14
_CLOSE_ENOUGH_REL: Final[float] = 42 * 1e-12


def _close_enough(a: float, b: float) -> bool:
    """Mirrors C++ ``close_enough(a, b, 42)`` — relative-tolerance check."""
    return math.isclose(a, b, abs_tol=_CLOSE_ENOUGH_ABS, rel_tol=_CLOSE_ENOUGH_REL)


class TimeGrid:
    """Discrete monotonically-increasing time grid (non-negative)."""

    def __init__(self, times: list[float], mandatory_times: list[float]) -> None:
        """Direct internal constructor — prefer the factory classmethods."""
        self._times: tuple[float, ...] = tuple(times)
        self._mandatory_times: tuple[float, ...] = tuple(mandatory_times)
        # Pre-compute dt[i] = times[i+1] - times[i].
        self._dt: tuple[float, ...] = tuple(
            self._times[i + 1] - self._times[i] for i in range(len(self._times) - 1)
        )

    # --- factory classmethods ---------------------------------------------

    @classmethod
    def regular(cls, end: float, steps: int) -> TimeGrid:
        """Mirrors C++ ``TimeGrid(Time end, Size steps)``."""
        qassert.require(end > 0.0, "negative times not allowed")
        dt = end / steps
        times = [dt * i for i in range(steps + 1)]
        return cls(times, [end])

    @classmethod
    def with_mandatory(cls, times: Iterable[float]) -> TimeGrid:
        """Mirrors C++ mandatory-only iterator constructor."""
        mt = sorted(times)
        qassert.require(len(mt) > 0, "empty time sequence")
        qassert.require(mt[0] >= 0.0, "negative times not allowed")
        mt = cls._dedupe_close(mt)
        all_times: list[float] = []
        if mt[0] > 0.0:
            all_times.append(0.0)
        all_times.extend(mt)
        return cls(all_times, mt)

    @classmethod
    def with_mandatory_and_steps(cls, times: Iterable[float], steps: int) -> TimeGrid:
        """Mirrors C++ mandatory+steps iterator constructor."""
        mt = sorted(times)
        qassert.require(len(mt) > 0, "empty time sequence")
        qassert.require(mt[0] >= 0.0, "negative times not allowed")
        mt = cls._dedupe_close(mt)
        last = mt[-1]
        if steps == 0:
            diffs = [mt[i] - mt[i - 1] for i in range(1, len(mt))]
            if mt[0] == 0.0 and len(diffs) > 1:
                # C++ erases the first diff if it is 0 (start of mandatory at 0).
                pass  # no-op: we already started at index 1
            qassert.require(len(diffs) > 0, "at least two distinct points required in time grid")
            dt_max = min(diffs)
        else:
            dt_max = last / steps

        all_times: list[float] = [0.0]
        period_begin = 0.0
        for t in mt:
            if t != 0.0:
                n_steps = max(round((t - period_begin) / dt_max), 1)
                dt = (t - period_begin) / n_steps
                all_times.extend(period_begin + n * dt for n in range(1, n_steps + 1))
            period_begin = t
        return cls(all_times, mt)

    @staticmethod
    def _dedupe_close(sorted_times: list[float]) -> list[float]:
        """Drop consecutive entries that are ``close_enough``."""
        if not sorted_times:
            return sorted_times
        out = [sorted_times[0]]
        for t in sorted_times[1:]:
            if not _close_enough(out[-1], t):
                out.append(t)
        return out

    # --- inspectors --------------------------------------------------------

    def __len__(self) -> int:
        return len(self._times)

    def size(self) -> int:
        return len(self._times)

    def __getitem__(self, i: int) -> float:
        return self._times[i]

    def at(self, i: int) -> float:
        return self._times[i]

    def __iter__(self):
        return iter(self._times)

    def empty(self) -> bool:
        return len(self._times) == 0

    def front(self) -> float:
        return self._times[0]

    def back(self) -> float:
        return self._times[-1]

    @property
    def mandatory_times(self) -> tuple[float, ...]:
        return self._mandatory_times

    @property
    def times(self) -> tuple[float, ...]:
        return self._times

    def dt(self, i: int) -> float:
        return self._dt[i]

    # --- lookups -----------------------------------------------------------

    def closest_index(self, t: float) -> int:
        """Index of the grid point closest to ``t``."""
        idx = bisect.bisect_left(self._times, t)
        if idx == 0:
            return 0
        if idx == len(self._times):
            return len(self._times) - 1
        dt1 = self._times[idx] - t
        dt2 = t - self._times[idx - 1]
        return idx if dt1 < dt2 else idx - 1

    def closest_time(self, t: float) -> float:
        return self._times[self.closest_index(t)]

    def index(self, t: float) -> int:
        """Index ``i`` such that ``grid[i] ≈ t``; raises if ``t`` is not on the grid."""
        i = self.closest_index(t)
        if _close_enough(t, self._times[i]):
            return i
        if t < self._times[0]:
            qassert.fail(
                f"using inadequate time grid: all nodes are later than the required "
                f"time t = {t} (earliest node is t1 = {self._times[0]})"
            )
        if t > self._times[-1]:
            qassert.fail(
                f"using inadequate time grid: all nodes are earlier than the required "
                f"time t = {t} (latest node is t1 = {self._times[-1]})"
            )
        if t > self._times[i]:
            j, k = i, i + 1
        else:
            j, k = i - 1, i
        qassert.fail(
            f"using inadequate time grid: the nodes closest to the required time "
            f"t = {t} are t1 = {self._times[j]} and t2 = {self._times[k]}"
        )
