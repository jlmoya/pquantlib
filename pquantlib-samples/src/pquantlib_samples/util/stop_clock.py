"""StopClock — a simple start/stop elapsed-time helper.

Port of ``org.jquantlib.samples.util.StopClock`` (itself a JQuantLib helper).
The Java original exposes a millisecond / nanosecond ``Unit`` selector,
``startClock()`` / ``stopClock()`` / ``getElapsedTime()`` accessors, a
``reset()``, and a ``log()`` that writes the elapsed time to the logger.

The Python port keeps that surface (snake_case) and additionally supports use
as a context manager, which is the idiomatic Python timing pattern::

    with StopClock() as clock:
        do_work()
    print(clock)            # "Time taken: 12ms"

``time.monotonic_ns()`` is used as the clock source (monotonic, immune to wall
-clock adjustments) and converted to the requested unit on read.
"""

from __future__ import annotations

import time
from enum import Enum
from types import TracebackType


class Unit(Enum):
    """Resolution of the reported elapsed time."""

    MS = "ms"
    """milliseconds"""

    NS = "ns"
    """nanoseconds"""


class StopClock:
    """A minimal start/stop stopwatch reporting in ms or ns."""

    def __init__(self, unit: Unit = Unit.MS) -> None:
        self._unit: Unit = unit
        self._start_ns: int = 0
        self._stop_ns: int = 0

    def start_clock(self) -> None:
        self._start_ns = time.monotonic_ns()
        self._stop_ns = self._start_ns

    def stop_clock(self) -> None:
        self._stop_ns = time.monotonic_ns()

    def get_elapsed_time(self) -> int:
        """Elapsed time between the last start and stop, in ``self.unit``."""
        elapsed_ns = self._stop_ns - self._start_ns
        if self._unit is Unit.MS:
            return elapsed_ns // 1_000_000
        return elapsed_ns

    @property
    def unit(self) -> Unit:
        return self._unit

    def reset(self) -> None:
        self._start_ns = 0
        self._stop_ns = 0

    def __str__(self) -> str:
        return f"Time taken: {self.get_elapsed_time()}{self._unit.value}"

    def log(self) -> None:
        """Print the elapsed time (Java ``QL.info`` analogue — to stdout)."""
        print(str(self))

    # --- context-manager sugar (idiomatic Python timing) -----------------
    def __enter__(self) -> StopClock:
        self.start_clock()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop_clock()


__all__ = ["StopClock", "Unit"]
