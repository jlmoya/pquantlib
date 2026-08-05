"""Cdi — Certificado de Depósito Interbancário, the Brazilian overnight rate.

# C++ parity: ql/indexes/ibor/cdi.{hpp,cpp} (v1.43).

Unlike the rest of the overnight family, CDI is quoted as an *annually
compounded* rate over 252 business days, so it overrides ``forecast_fixing``
with ``(D(start)/D(end))**(1/yf) - 1`` instead of the simple
``(D(start)/D(end) - 1)/yf`` that ``IborIndex`` uses.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.currencies.america import BRLCurrency
from pquantlib.daycounters.business_252 import Business252
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.brazil import Brazil
from pquantlib.time.date import Date


class Cdi(OvernightIndex):
    """CDI — BRL overnight rate on the Brazilian settlement calendar, Business/252."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(
            "CDI", 0, BRLCurrency(), Brazil(Brazil.Market.Settlement), Business252(Brazil()), h,
        )

    def forecast_fixing(self, fixing_date: Date) -> float:
        """Mirror C++ ``Cdi::forecastFixing`` — compounded, not simple."""
        start_date = self.value_date(fixing_date)
        end_date = self.maturity_date(start_date)
        yf = self._day_counter.year_fraction(start_date, end_date)
        qassert.require(yf > 0.0, f"year fraction ({yf}) must be positive")
        qassert.require(
            self._term_structure is not None,
            f"null term structure set to this instance of {self.name()}",
        )
        assert self._term_structure is not None
        discount_start = self._term_structure.discount(start_date)
        discount_end = self._term_structure.discount(end_date)
        return (discount_start / discount_end) ** (1.0 / yf) - 1.0
