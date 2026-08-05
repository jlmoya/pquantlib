"""EquityIndex — spot-and-curve driven equity index.

# C++ parity: ql/indexes/equityindex.{hpp,cpp} (v1.43)

Unlike the interest-rate indexes, this one derives from ``Index`` directly:
there is no tenor, no fixing days and no day counter. A fixing is either read
from history or forecast as a forward off the spot quote and two curves::

    forward = spot * D_dividend(t) / D_interest(t)

with the dividend leg dropped when no dividend curve is supplied. The spot
itself falls back to the last stored historical fixing when the quote is
absent, so an index carrying only history can still forecast.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.currencies.currency import Currency
from pquantlib.indexes.index import Index
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.patterns.observer import Observable
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class EquityIndex(Index):
    """Equity index driven by a spot quote and interest / dividend curves."""

    def __init__(
        self,
        name: str,
        fixing_calendar: Calendar,
        currency: Currency,
        interest: YieldTermStructureProtocol | None = None,
        dividend: YieldTermStructureProtocol | None = None,
        spot: Quote | None = None,
    ) -> None:
        super().__init__()
        self._name: str = name
        self._fixing_calendar: Calendar = fixing_calendar
        self._currency: Currency = currency
        self._interest: YieldTermStructureProtocol | None = interest
        self._dividend: YieldTermStructureProtocol | None = dividend
        self._spot: Quote | None = spot
        # C++ registers with each handle and with the evaluation date, so the
        # index re-notifies when any of them moves. PQuantLib passes concrete
        # objects rather than Handles, so only the ones that are Observables
        # can be registered with; Index.update() re-notifies downstream.
        for source in (interest, dividend, spot):
            if isinstance(source, Observable):
                source.register_with(self)
        ObservableSettings().register_with(self)

    # --- Index interface -------------------------------------------------------

    def name(self) -> str:
        return self._name

    def fixing_calendar(self) -> Calendar:
        return self._fixing_calendar

    def is_valid_fixing_date(self, fixing_date: Date) -> bool:
        """Mirror C++ ``EquityIndex::isValidFixingDate`` — any business day."""
        return self._fixing_calendar.is_business_day(fixing_date)

    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float:
        """Mirror C++ ``EquityIndex::fixing``.

        Forecast strictly in the future (or today when asked); otherwise read
        history, falling back to the spot quote on today's date.
        """
        qassert.require(
            self.is_valid_fixing_date(fixing_date),
            f"Fixing date {fixing_date} is not valid",
        )
        today = ObservableSettings().evaluation_date_or_today()
        if fixing_date > today or (fixing_date == today and forecast_todays_fixing):
            return self.forecast_fixing(fixing_date)
        # C++ pastFixing returns Null<Real> when absent; PQuantLib's raises,
        # so the presence check moves ahead of the read.
        if self.has_historical_fixing(fixing_date):
            return self.past_fixing(fixing_date)
        if fixing_date == today and self._spot is not None:
            return self._spot.value()
        qassert.fail(f"Missing {self.name()} fixing for {fixing_date}")

    # --- forecasting -----------------------------------------------------------

    def forecast_fixing(self, fixing_date: Date) -> float:
        """Mirror C++ ``EquityIndex::forecastFixing`` — spot carried to the date."""
        qassert.require(
            self._interest is not None,
            f"null interest rate term structure set to this instance of {self.name()}",
        )
        assert self._interest is not None
        spot = self._resolve_spot()
        if self._dividend is not None:
            return (
                spot
                * self._dividend.discount(fixing_date)
                / self._interest.discount(fixing_date)
            )
        return spot / self._interest.discount(fixing_date)

    def _resolve_spot(self) -> float:
        """Mirror the anonymous ``resolveSpot`` helper in equityindex.cpp."""
        if self._spot is not None:
            return self._spot.value()
        today = ObservableSettings().evaluation_date_or_today()
        last_fixing_date = self._fixing_calendar.adjust(
            today, BusinessDayConvention.Preceding,
        )
        qassert.require(
            self.has_historical_fixing(last_fixing_date),
            "Cannot forecast equity index, missing both spot and historical index",
        )
        return self.past_fixing(last_fixing_date)

    # --- inspectors ------------------------------------------------------------

    def currency(self) -> Currency:
        return self._currency

    def equity_interest_rate_curve(self) -> YieldTermStructureProtocol | None:
        return self._interest

    def equity_dividend_curve(self) -> YieldTermStructureProtocol | None:
        return self._dividend

    def spot(self) -> Quote | None:
        return self._spot

    # --- mutators --------------------------------------------------------------

    def clone(
        self,
        interest: YieldTermStructureProtocol | None,
        dividend: YieldTermStructureProtocol | None,
        spot: Quote | None,
    ) -> EquityIndex:
        """Mirror C++ ``EquityIndex::clone`` — same identity, new market data."""
        return EquityIndex(
            self._name, self._fixing_calendar, self._currency, interest, dividend, spot,
        )
