"""TimeSeries — sparse historical data indexed by Date.

# C++ parity: ql/timeseries.hpp (v1.42.1).

The C++ class is templated on the value type and on the underlying
``Container`` (default ``std::map<Date, T>``); the Python port is generic
over ``T`` via PEP 695 and uses a plain ``dict[Date, T]`` underneath
(Python 3.7+ preserves insertion order, but the public API exposes
date-sorted access).

Note: ``Series`` is not a C++ class in v1.42.1 — only ``TimeSeries`` exists.
The L1-A plan's mention of a separate Series class was a jquantlib-port
artifact; no Series module is created here. Documented divergence.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from pquantlib import qassert
from pquantlib.time.date import Date


class TimeSeries[T]:
    """Mutable date-keyed sparse series.

    Lookup of a missing date returns ``None`` (mirrors C++ ``Null<T>()``).
    """

    def __init__(self) -> None:
        self._values: dict[Date, T] = {}

    # --- factory constructors mirroring the C++ overloads -----------------

    @classmethod
    def from_pairs(cls, dates: Iterable[Date], values: Iterable[T]) -> TimeSeries[T]:
        """Equivalent of C++ ``TimeSeries(dBegin, dEnd, vBegin)``."""
        out: TimeSeries[T] = cls()
        for d, v in zip(dates, values, strict=False):
            out._values[d] = v
        return out

    @classmethod
    def from_first_date(cls, first_date: Date, values: Iterable[T]) -> TimeSeries[T]:
        """Equivalent of C++ ``TimeSeries(firstDate, begin, end)`` — consecutive dates."""
        out: TimeSeries[T] = cls()
        d = first_date
        for v in values:
            out._values[d] = v
            d = d + 1
        return out

    # --- core access -------------------------------------------------------

    def __setitem__(self, d: Date, value: T) -> None:
        self._values[d] = value

    def __getitem__(self, d: Date) -> T | None:
        return self._values.get(d)

    def __contains__(self, d: object) -> bool:
        return d in self._values

    def __len__(self) -> int:
        return len(self._values)

    def size(self) -> int:
        return len(self._values)

    def empty(self) -> bool:
        return not self._values

    def first_date(self) -> Date:
        qassert.require(bool(self._values), "empty time series")
        return min(self._values.keys())

    def last_date(self) -> Date:
        qassert.require(bool(self._values), "empty time series")
        return max(self._values.keys())

    def dates(self) -> tuple[Date, ...]:
        return tuple(sorted(self._values.keys()))

    def values(self) -> tuple[T, ...]:
        return tuple(self._values[d] for d in sorted(self._values.keys()))

    def items(self) -> tuple[tuple[Date, T], ...]:
        return tuple((d, self._values[d]) for d in sorted(self._values.keys()))

    def __iter__(self) -> Iterator[Date]:
        return iter(sorted(self._values.keys()))
