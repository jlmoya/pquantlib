"""ProxyIbor — an IborIndex whose fixing is a proxy of another IborIndex.

# C++ parity: ql/experimental/coupons/proxyibor.hpp (v1.42.1, 099987f0).

The forecast fixing is

    forecastFixing(d) = gearing.value() * iborIndex.fixing(d) * spread.value()

(note: the C++ formula *multiplies* by spread rather than adding — verbatim
from proxyibor.hpp lines 55-58).

# C++ parity divergence (Handle<Quote>): per the pquantlib convention a
# ``Handle<Quote>`` collapses to a direct :class:`~pquantlib.quotes.quote.Quote`
# reference (Quote is itself Observable).
"""

from __future__ import annotations

from pquantlib.currencies.currency import Currency
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.quotes.quote import Quote
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class ProxyIbor(IborIndex):
    """IborIndex calculated as a (gearing, spread)-scaled proxy of another index."""

    def __init__(
        self,
        family_name: str,
        tenor: Period,
        settlement_days: int,
        currency: Currency,
        fixing_calendar: Calendar,
        convention: BusinessDayConvention,
        end_of_month: bool,
        day_counter: DayCounter,
        gearing: Quote,
        ibor_index: IborIndex,
        spread: Quote,
    ) -> None:
        super().__init__(
            family_name,
            tenor,
            settlement_days,
            currency,
            fixing_calendar,
            convention,
            end_of_month,
            day_counter,
            # C++ ProxyIbor has no own forwarding curve; fixings come from the
            # proxied iborIndex_.
            None,
        )
        self._gearing: Quote = gearing
        self._ibor_index: IborIndex = ibor_index
        self._spread: Quote = spread

    def forecast_fixing(self, fixing_date: Date) -> float:
        """C++ parity: proxyibor.hpp lines 55-58 — gearing * proxy * spread."""
        proxy = self._ibor_index.fixing(fixing_date)
        return self._gearing.value() * proxy * self._spread.value()
