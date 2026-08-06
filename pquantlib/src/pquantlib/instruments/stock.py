"""Stock — an instrument whose NPV is a quote.

# C++ parity: ql/instruments/stock.hpp + .cpp (v1.43).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.instruments.instrument import Instrument

if TYPE_CHECKING:
    from pquantlib.quotes.quote import Quote


class Stock(Instrument):
    """Simple stock: NPV is the value of the quote it is built on.

    C++ parity: stock.cpp:23-30. The constructor registers with the quote,
    so bumping the quote invalidates the cached NPV.
    """

    def __init__(self, quote: Quote | None) -> None:
        super().__init__()
        self._quote: Quote | None = quote
        # C++ ``registerWith(quote_)`` — an empty Handle registers with
        # nothing, hence the None guard.
        if quote is not None:
            quote.register_with(self)

    def is_expired(self) -> bool:
        return False

    def _perform_calculations(self) -> None:
        # C++ parity: stock.cpp:25-28 — an empty quote handle raises.
        qassert.require(self._quote is not None, "null quote set")
        assert self._quote is not None
        self._npv = self._quote.value()
        self._error_estimate = 0.0
        self._additional_results = {}


__all__ = ["Stock"]
