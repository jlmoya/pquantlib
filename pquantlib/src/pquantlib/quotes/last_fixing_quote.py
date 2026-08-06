"""LastFixingQuote — quote adapter for the last available fixing of an index.

# C++ parity: ql/quotes/lastfixingquote.{hpp,cpp} (v1.43)

``referenceDate()`` is ``min(timeSeries().lastDate(), evaluationDate())``, so
the quote reads the newest fixing that is not in the future: move the
evaluation date back behind the history and the quote reads *back through* the
series rather than off its end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.quote import Quote

if TYPE_CHECKING:
    from pquantlib.indexes.index import Index
    from pquantlib.time.date import Date


class LastFixingQuote(Quote):
    """Quote adapter for the last fixing available of a given Index.

    # C++ parity: ``class LastFixingQuote : public Quote, public Observer``.
    """

    __slots__ = ("_index",)

    def __init__(self, index: Index) -> None:
        super().__init__()
        self._index: Index = index
        index.register_with(self)

    # --- Quote interface --------------------------------------------------

    def value(self) -> float:
        qassert.require(self.is_valid(), f"{self._index.name()} has no fixing")
        return self._index.fixing(self.reference_date())

    def is_valid(self) -> bool:
        return not self._index.time_series().empty()

    # --- Observer interface -----------------------------------------------

    def update(self) -> None:
        self.notify_observers()

    # --- LastFixingQuote interface ----------------------------------------

    def index(self) -> Index:
        return self._index

    def reference_date(self) -> Date:
        """Newest fixing date that is not after the evaluation date."""
        return min(
            self._index.time_series().last_date(),
            ObservableSettings().evaluation_date_or_today(),
        )
