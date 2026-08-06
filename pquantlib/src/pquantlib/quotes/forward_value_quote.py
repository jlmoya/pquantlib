"""ForwardValueQuote — quote adapter for an index fixing on a future date.

# C++ parity: ql/quotes/forwardvaluequote.{hpp,cpp} (v1.43)

A thin adapter: ``value()`` is ``index->fixing(fixingDate)``, so whether the
answer is a forecast or a stored historical fixing is entirely the index's
decision. ``isValid()`` is unconditionally ``true`` — including for a date the
index cannot fix at all, in which case ``value()`` raises. The C++ source
carries the comment ``// not sure this is the best approach...`` on exactly
that; the port keeps the behaviour rather than improving on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.quotes.quote import Quote

if TYPE_CHECKING:
    from pquantlib.indexes.index import Index
    from pquantlib.time.date import Date


class ForwardValueQuote(Quote):
    """Quote for the forward value of an index.

    # C++ parity: ``class ForwardValueQuote : public Quote, public Observer``.
    # PQuantLib's ``Observer`` is a structural Protocol, so implementing
    # ``update()`` is the whole of the Observer side.
    """

    __slots__ = ("_fixing_date", "_index")

    def __init__(self, index: Index, fixing_date: Date) -> None:
        super().__init__()
        self._index: Index = index
        self._fixing_date: Date = fixing_date
        index.register_with(self)

    # --- Quote interface --------------------------------------------------

    def value(self) -> float:
        return self._index.fixing(self._fixing_date)

    def is_valid(self) -> bool:
        # C++ comment, verbatim: "not sure this is the best approach..."
        return True

    # --- Observer interface -----------------------------------------------

    def update(self) -> None:
        self.notify_observers()
