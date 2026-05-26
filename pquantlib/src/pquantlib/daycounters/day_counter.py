"""DayCounter — abstract base for day-count conventions.

# C++ parity: ql/time/daycounter.hpp (v1.42.1).

The C++ class uses a pImpl Bridge (``DayCounter`` holds a
``shared_ptr<Impl>`` whose ``name``, ``dayCount``, ``yearFraction`` are
virtual). The Python port collapses this to direct ``abc.ABC`` inheritance,
matching the Calendar port.

Subclasses implement ``name()`` and ``year_fraction(d1, d2, ref_start,
ref_end)``. ``day_count`` has a default implementation of ``d2 - d1``;
override only when a different day-count convention is needed
(``ActualActual``, ``Business252``, ``OneDayCounter`` all override).

Equality is name-based, matching C++.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pquantlib.time.date import Date


class DayCounter(ABC):
    """Abstract base class for day-count conventions."""

    @abstractmethod
    def name(self) -> str: ...

    def day_count(self, d1: Date, d2: Date) -> int:
        """Mirrors the default C++ ``Impl::dayCount`` which returns ``d2 - d1``."""
        return d2 - d1

    @abstractmethod
    def year_fraction(
        self,
        d1: Date,
        d2: Date,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
    ) -> float: ...

    # --- equality + repr ---------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DayCounter):
            return NotImplemented
        return self.name() == other.name()

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.name())

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name()!r})"

    def __str__(self) -> str:
        return self.name()
