"""IndexManager — singleton repository of historical fixings.

# C++ parity: ql/indexes/indexmanager.hpp + ql/indexes/indexmanager.cpp (v1.42.1)

C++ uses Singleton<IndexManager> with a custom case-insensitive comparator.
Python uses pquantlib.patterns.Singleton + name.lower() normalization for
the same effect.

The C++ per-index ``Observable`` notifier subsystem is **not ported** —
it's marked deprecated in v1.42.1, and modern client code calls
``Index.update()`` directly. Re-add if a downstream consumer needs it.
"""

from __future__ import annotations

from pquantlib import qassert
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
        history = self.get_history(name)
        existing = history[fixing_date]
        qassert.require(
            force_overwrite or existing is None or existing == fixing,
            f"duplicated fixing for {name} on {fixing_date}: "
            f"existing={existing}, new={fixing}",
        )
        history[fixing_date] = fixing
