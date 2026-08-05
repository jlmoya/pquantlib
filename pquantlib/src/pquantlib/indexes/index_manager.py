"""IndexManager — singleton repository of historical fixings.

# C++ parity: ql/indexes/indexmanager.hpp + ql/indexes/indexmanager.cpp (v1.43)

C++ uses Singleton<IndexManager> with a custom case-insensitive comparator.
Python uses pquantlib.patterns.Singleton + name.lower() normalization for
the same effect.

The C++ per-index ``Observable`` notifier subsystem is **not ported** —
it's marked deprecated in v1.42.1, and modern client code calls
``Index.update()`` directly. Re-add if a downstream consumer needs it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from pquantlib import qassert
from pquantlib.math.closeness import close
from pquantlib.patterns.singleton import Singleton
from pquantlib.time.date import Date
from pquantlib.time.time_series import TimeSeries


class IndexManager(Singleton):
    """Global repository for past index fixings (singleton; case-insensitive names)."""

    def __init__(self) -> None:
        super().__init__()
        self._data: dict[str, TimeSeries[float]] = {}

    @staticmethod
    def _norm(name: str) -> str:
        return name.lower()

    def has_history(self, name: str) -> bool:
        return self._norm(name) in self._data

    def get_history(self, name: str) -> TimeSeries[float]:
        key = self._norm(name)
        if key not in self._data:
            self._data[key] = TimeSeries[float]()
        return self._data[key]

    def set_history(self, name: str, history: TimeSeries[float]) -> None:
        self._data[self._norm(name)] = history

    def clear_history(self, name: str) -> None:
        self._data.pop(self._norm(name), None)

    def clear_histories(self) -> None:
        self._data.clear()

    def histories(self) -> list[str]:
        return list(self._data.keys())

    def has_historical_fixing(self, name: str, fixing_date: Date) -> bool:
        key = self._norm(name)
        return key in self._data and fixing_date in self._data[key]

    def add_fixing(
        self,
        name: str,
        fixing_date: Date,
        fixing: float,
        force_overwrite: bool = False,
    ) -> None:
        """Mirror C++ ``IndexManager::addFixing`` — a one-element ``add_fixings``."""
        self.add_fixings(name, [fixing_date], [fixing], force_overwrite)

    def add_fixings(
        self,
        name: str,
        dates: Iterable[Date],
        values: Iterable[float],
        force_overwrite: bool = False,
        is_valid_fixing_date: Callable[[Date], bool] | None = None,
    ) -> None:
        """Mirror C++ ``IndexManager::addFixings``.

        Every acceptable fixing is stored first and the rejections are reported
        afterwards, exactly as in C++: the two ``QL_REQUIRE``s sit *after* the
        loop, so a batch with one bad entry still commits the good ones before
        raising. Only the *last* offender of each kind is named — C++ keeps
        overwriting its single slot as it scans.

        A fixing that repeats a stored value is accepted silently, and the
        comparison is ``close()`` rather than ``==`` — a round-tripped double
        must not count as a conflict.
        """
        history = self.get_history(name)
        invalid: tuple[Date, float] | None = None
        duplicated: tuple[Date, float] | None = None
        for d, v in zip(dates, values, strict=True):
            if is_valid_fixing_date is not None and not is_valid_fixing_date(d):
                invalid = (d, v)
                continue
            current = history[d]
            if force_overwrite or current is None:
                history[d] = v
            elif not close(current, v):
                duplicated = (d, v)
        if invalid is not None:
            qassert.fail(
                f"At least one invalid fixing provided: "
                f"{invalid[0].weekday().name} {invalid[0]}, {invalid[1]}",
            )
        if duplicated is not None:
            qassert.fail(
                f"At least one duplicated fixing provided: "
                f"{duplicated[0]}, {duplicated[1]} while {history[duplicated[0]]} "
                f"value is already present",
            )
