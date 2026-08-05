"""EquityIndex — base class for equity indexes.

# C++ parity: ql/indexes/equityindex.hpp + ql/indexes/equityindex.cpp (v1.43).

The index retrieves past fixings from the ``IndexManager`` history and
projects future fixings in one of two ways:

- with both a risk-free interest curve and a dividend curve::

      I(t, T) = I(t, t) * P_D(t, T) / P_R(t, T)

- with the interest curve alone, in which case that curve is an *equity
  forward* curve implied from, e.g., option prices::

      I(t, T) = I(t, t) / P_F(t, T)

``I(t, t)`` comes from the spot quote when one is supplied, otherwise from
the last historical fixing on or before today.

Python divergences from C++:

- ``Handle<YieldTermStructure>`` / ``Handle<Quote>`` are replaced by
  ``... | None``; an empty C++ handle is ``None`` here. This is the same
  convention the rest of this port's index hierarchy uses (see
  ``BMAIndex``, ``IborIndex``).
- C++ ``resolveSpot`` (equityindex.cpp:27-33) evaluates ``pastFixing`` even
  when the spot handle is non-empty, relying on ``Null<Real>()`` for a
  missing fixing. PQuantLib's ``Index.past_fixing`` raises instead of
  returning a null, so the lookup here is done lazily behind
  ``has_historical_fixing``. The observable behaviour is identical: the
  date passed to ``past_fixing`` is always a business day (it comes out of
  ``Calendar.adjust(..., Preceding)``), so the only other C++ throw path is
  unreachable in both.
- C++ additionally does ``registerWith(notifier())`` so that an
  ``addFixing`` on *any* index sharing the name notifies this instance's
  observers. PQuantLib's ``IndexManager`` has no per-name notifier, so that
  registration has no analogue here.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.currencies.currency import Currency
from pquantlib.indexes.index import Index
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class EquityIndex(Index):
    """Equity index with an optional interest curve, dividend curve and spot.

    # C++ parity: ``EquityIndex`` (equityindex.hpp:64-106).
    """

    def __init__(
        self,
        name: str,
        fixing_calendar: Calendar,
        currency: Currency,
        interest: YieldTermStructure | None = None,
        dividend: YieldTermStructure | None = None,
        spot: Quote | None = None,
    ) -> None:
        super().__init__()
        self._name: str = name
        self._fixing_calendar: Calendar = fixing_calendar
        self._currency: Currency = currency
        self._interest: YieldTermStructure | None = interest
        self._dividend: YieldTermStructure | None = dividend
        self._spot: Quote | None = spot

        # C++ parity: equityindex.cpp:48-52.
        if interest is not None:
            interest.register_with(self)
        if dividend is not None:
            dividend.register_with(self)
        if spot is not None:
            spot.register_with(self)
        ObservableSettings().register_with(self)

    # ---- Index interface -----------------------------------------------------

    def name(self) -> str:
        return self._name

    def fixing_calendar(self) -> Calendar:
        return self._fixing_calendar

    def is_valid_fixing_date(self, fixing_date: Date) -> bool:
        """# C++ parity: inline ``EquityIndex::isValidFixingDate`` (hpp:108-110)."""
        return self._fixing_calendar.is_business_day(fixing_date)

    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float:
        """Index level on ``fixing_date``.

        # C++ parity: ``EquityIndex::fixing`` (equityindex.cpp:52-71).

        Dispatch order: forecast for future dates (and for today when
        ``forecast_todays_fixing``), then the stored history, then the spot
        quote as a proxy for a missing fixing *for today only*, then fail.
        """
        qassert.require(
            self.is_valid_fixing_date(fixing_date),
            f"Fixing date {fixing_date} is not valid",
        )

        today = ObservableSettings().evaluation_date_or_today()

        if fixing_date > today or (fixing_date == today and forecast_todays_fixing):
            return self.forecast_fixing(fixing_date)

        if self.has_historical_fixing(fixing_date):
            # if historical fixing is present use it
            return self.past_fixing(fixing_date)

        if fixing_date == today and self._spot is not None:
            # Today's fixing is missing, but spot is provided, so use it as proxy
            return self._spot.value()

        qassert.fail(f"Missing {self.name()} fixing for {fixing_date}")

    # ---- inspectors ----------------------------------------------------------

    def currency(self) -> Currency:
        return self._currency

    def equity_interest_rate_curve(self) -> YieldTermStructure | None:
        """The rate curve used to forecast fixings (``None`` == empty handle)."""
        return self._interest

    def equity_dividend_curve(self) -> YieldTermStructure | None:
        """The dividend curve used to forecast fixings (``None`` == empty handle)."""
        return self._dividend

    def spot(self) -> Quote | None:
        """Index spot value (``None`` == empty handle)."""
        return self._spot

    # ---- fixing calculations -------------------------------------------------

    def forecast_fixing(self, fixing_date: Date) -> float:
        """Forward index level at ``fixing_date``.

        # C++ parity: ``EquityIndex::forecastFixing`` (equityindex.cpp:73-89).
        """
        qassert.require(
            self._interest is not None,
            f"null interest rate term structure set to this instance of {self.name()}",
        )
        assert self._interest is not None

        today = ObservableSettings().evaluation_date_or_today()
        last_fixing_date = self._fixing_calendar.adjust(today, BusinessDayConvention.Preceding)

        spot = self._resolve_spot(last_fixing_date)

        if self._dividend is not None:
            return spot * self._dividend.discount(fixing_date) / self._interest.discount(fixing_date)
        return spot / self._interest.discount(fixing_date)

    def _resolve_spot(self, last_fixing_date: Date) -> float:
        """Spot quote if present, else the last historical fixing.

        # C++ parity: anonymous ``resolveSpot`` (equityindex.cpp:27-33).
        """
        if self._spot is not None:
            return self._spot.value()
        qassert.require(
            self.has_historical_fixing(last_fixing_date),
            "Cannot forecast equity index, missing both spot and historical index",
        )
        return self.past_fixing(last_fixing_date)

    # ---- other methods -------------------------------------------------------

    def clone(
        self,
        interest: YieldTermStructure | None,
        dividend: YieldTermStructure | None,
        spot: Quote | None,
    ) -> EquityIndex:
        """A copy of this index linked to different curves / spot quote.

        # C++ parity: ``EquityIndex::clone`` (equityindex.cpp:91-96). The
        # clone keeps the name, so it shares this index's fixing history.
        """
        return EquityIndex(self.name(), self.fixing_calendar(), self.currency(), interest, dividend, spot)
