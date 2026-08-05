"""Index — purely virtual base for all rate / equity / inflation indexes.

# C++ parity: ql/index.hpp (v1.43)

Index IS both Observer and Observable (C++): rate-helper-style downstream
consumers register with the index; the index may itself register with
upstream sources (term structures, fixing-date settings) and re-notify
on update().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from pquantlib import qassert
from pquantlib.indexes.index_manager import IndexManager
from pquantlib.patterns.observer import Observable
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.time_series import TimeSeries


class Index(Observable, ABC):
    """Abstract index. Subclasses define name, fixing-date validity, and ``fixing()``."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def name(self) -> str:
        """Identifier; used for output and equality. Case-insensitive."""

    @abstractmethod
    def fixing_calendar(self) -> Calendar:
        """Calendar defining valid fixing dates."""

    @abstractmethod
    def is_valid_fixing_date(self, fixing_date: Date) -> bool:
        """Whether ``fixing_date`` is a valid fixing date for this index."""

    @abstractmethod
    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float:
        """Value of the fixing on ``fixing_date``."""

    def past_fixing(self, fixing_date: Date) -> float:
        qassert.require(
            self.is_valid_fixing_date(fixing_date),
            f"{fixing_date} is not a valid fixing date",
        )
        history = IndexManager().get_history(self.name())
        value = history[fixing_date]
        qassert.require(value is not None, f"no past fixing stored for {fixing_date}")
        assert value is not None
        return value

    def has_historical_fixing(self, fixing_date: Date) -> bool:
        return IndexManager().has_historical_fixing(self.name(), fixing_date)

    def time_series(self) -> TimeSeries[float]:
        return IndexManager().get_history(self.name())

    def allows_native_fixings(self) -> bool:
        return True

    def add_fixing(self, fixing_date: Date, fixing: float, force_overwrite: bool = False) -> None:
        """Mirror C++ ``Index::addFixing`` — a one-element ``add_fixings``.

        Routing through ``add_fixings`` is what applies the index's own
        ``is_valid_fixing_date`` check, so a fixing dated on a non-business day
        is rejected rather than silently stored.
        """
        self.add_fixings([fixing_date], [fixing], force_overwrite)

    def add_fixings(
        self,
        dates: Iterable[Date],
        values: Iterable[float],
        force_overwrite: bool = False,
    ) -> None:
        """Mirror C++ ``Index::addFixings(dBegin, dEnd, vBegin, forceOverwrite)``."""
        self._check_native_fixings_allowed()
        IndexManager().add_fixings(
            self.name(), dates, values, force_overwrite, self.is_valid_fixing_date,
        )

    def add_fixings_from_time_series(
        self, series: TimeSeries[float], force_overwrite: bool = False,
    ) -> None:
        """Mirror C++ ``Index::addFixings(const TimeSeries<Real>&, bool)``.

        Separate name rather than an overload: Python has no overload
        resolution, and a runtime type switch on the first argument would be
        the kind of implicit behaviour this port avoids.
        """
        self._check_native_fixings_allowed()
        self.add_fixings(series.dates(), series.values(), force_overwrite)

    def clear_fixings(self) -> None:
        """Mirror C++ ``Index::clearFixings`` — native fixings must be allowed."""
        self._check_native_fixings_allowed()
        IndexManager().clear_history(self.name())

    def _check_native_fixings_allowed(self) -> None:
        """Mirror C++ ``Index::checkNativeFixingsAllowed``."""
        qassert.require(
            self.allows_native_fixings(),
            f"native fixings not allowed for {self.name()}; "
            f"refer to underlying indices instead",
        )

    def update(self) -> None:
        self.notify_observers()
