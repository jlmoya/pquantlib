"""OvernightIndexFuture — future on a compounded/averaged overnight investment.

# C++ parity: ql/instruments/overnightindexfuture.{hpp,cpp} (v1.43).

Compatible with the SOFR and SONIA futures listed on CME and ICE.  The price
is ``100 * (1 - (convexity_adjustment + rate))`` where ``rate`` is either the
simple average (``RateAveraging.Simple``) or the compounded rate
(``RateAveraging.Compound``) of the overnight index over
``[value_date, maturity_date)``.

The instrument has **no pricing engine** in C++ — it computes in
``performCalculations()`` off the index's own forwarding curve and its
historical fixings — so the Python port overrides
``_perform_calculations`` rather than taking an engine.

Python port notes:

- ``Handle<Quote> convexityAdjustment`` becomes an optional ``Quote``
  (this port has no ``Handle`` indirection); an absent quote means a zero
  adjustment, exactly as an empty handle does in C++.
- The forwarding curve is read from the index and must be a concrete
  :class:`YieldTermStructure` — ``averagedRate`` needs ``forward_rate``,
  which the structural ``YieldTermStructureProtocol`` deliberately omits.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.cashflows.rate_averaging import RateAveraging
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.instruments.instrument import Instrument
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.time_unit import TimeUnit


class OvernightIndexFuture(Instrument):
    """Future on a compounded overnight index investment.

    # C++ parity: ``OvernightIndexFuture`` (overnightindexfuture.hpp:38-64,
    # .cpp:28-140).
    """

    def __init__(
        self,
        overnight_index: OvernightIndex,
        value_date: Date,
        maturity_date: Date,
        convexity_adjustment: Quote | None = None,
        averaging_method: RateAveraging = RateAveraging.Compound,
    ) -> None:
        # # C++ parity: ctor (overnightindexfuture.cpp:28-41).
        super().__init__()
        # # C++ parity: ``QL_REQUIRE(overnightIndex_, "null overnight index")``.
        # The annotation already forbids ``None``, so pyright calls the check
        # redundant — but it is the runtime guard C++ has, and it still fires
        # for callers reaching this constructor from untyped code.
        qassert.require(
            overnight_index is not None,  # pyright: ignore[reportUnnecessaryComparison]
            "null overnight index",
        )
        self._overnight_index: OvernightIndex = overnight_index
        self._value_date: Date = value_date
        self._maturity_date: Date = maturity_date
        self._convexity_adjustment: Quote | None = convexity_adjustment
        self._averaging_method: RateAveraging = averaging_method

        overnight_index.register_with(self)
        if convexity_adjustment is not None:
            convexity_adjustment.register_with(self)
        ObservableSettings().register_with(self)

    # --- inspectors ----------------------------------------------------

    def overnight_index(self) -> OvernightIndex:
        return self._overnight_index

    def value_date(self) -> Date:
        return self._value_date

    def maturity_date(self) -> Date:
        return self._maturity_date

    def averaging_method(self) -> RateAveraging:
        return self._averaging_method

    def convexity_adjustment(self) -> float:
        """Zero when no quote was supplied.

        # C++ parity: ``OvernightIndexFuture::convexityAdjustment``
        (overnightindexfuture.cpp:132-134).
        """
        return 0.0 if self._convexity_adjustment is None else self._convexity_adjustment.value()

    def is_expired(self) -> bool:
        """Expired once the maturity date is reached.

        # C++ parity: ``OvernightIndexFuture::isExpired``
        (overnightindexfuture.cpp:128-130) uses
        ``detail::simple_event(maturityDate_).hasOccurred()``, i.e.
        ``maturityDate_ <= Settings::evaluationDate()`` under the default
        ``includeReferenceDateEvents() == false``.
        """
        today = ObservableSettings().evaluation_date_or_today()
        return self._maturity_date <= today

    # --- rate calculation ----------------------------------------------

    def _forward_curve(self) -> YieldTermStructure:
        ts = self._overnight_index.forecast_term_structure()
        qassert.require(
            isinstance(ts, YieldTermStructure),
            f"null term structure set to this instance of {self._overnight_index.name()}",
        )
        assert isinstance(ts, YieldTermStructure)
        return ts

    def averaged_rate(self) -> float:
        """Simple (arithmetic) average of the daily fixings over the period.

        # C++ parity: ``OvernightIndexFuture::averagedRate``
        (overnightindexfuture.cpp:43-73).
        """
        today = ObservableSettings().evaluation_date_or_today()
        calendar = self._overnight_index.fixing_calendar()
        day_counter = self._overnight_index.day_counter()
        forward_curve = self._forward_curve()
        avg = 0.0
        d1 = self._value_date
        # d1 could be a holiday.
        fixing_date = calendar.adjust(d1, BusinessDayConvention.Preceding)
        history = self._overnight_index.time_series()
        while d1 < self._maturity_date:
            d2 = calendar.advance(d1, 1, TimeUnit.Days)
            fwd: float | None
            if fixing_date < today:
                fwd = history[fixing_date]
                qassert.require(
                    fwd is not None,
                    f"missing rate on {fixing_date} for index {self._overnight_index.name()}",
                )
            elif fixing_date == today:
                fwd = history[fixing_date]
                if fwd is None:
                    fwd = forward_curve.forward_rate(
                        fixing_date, d2, Compounding.Simple, Frequency.Annual, False, day_counter
                    ).rate()
            else:
                fwd = forward_curve.forward_rate(
                    fixing_date, d2, Compounding.Simple, Frequency.Annual, False, day_counter
                ).rate()
            assert fwd is not None
            # The rate accrues from d1 even when the fixing date is earlier;
            # d2 might be beyond the maturity date if the latter is a holiday.
            avg += fwd * day_counter.year_fraction(d1, min(d2, self._maturity_date))
            fixing_date = d1 = d2

        return avg / day_counter.year_fraction(self._value_date, self._maturity_date)

    def compounded_rate(self) -> float:
        """Daily-compounded rate over the period, telescoped past the last fixing.

        # C++ parity: ``OvernightIndexFuture::compoundedRate``
        (overnightindexfuture.cpp:75-116).
        """
        today = ObservableSettings().evaluation_date_or_today()
        calendar = self._overnight_index.fixing_calendar()
        day_counter = self._overnight_index.day_counter()
        forward_curve = self._forward_curve()
        prod = 1.0
        forward_discount_start = self._value_date
        if today > self._value_date:
            # Can't value on a weekend inside the reference period because the
            # reset rate is unknown until the next business day; the user can
            # supply an estimate through the history if they really want to.
            today = calendar.adjust(today)
            forward_discount_start = today
            # For valuations inside the reference period, index quotes must
            # have been populated in the history.
            history = self._overnight_index.time_series()
            d1 = self._value_date
            # d1 could be a holiday.
            fixing_date = calendar.adjust(d1, BusinessDayConvention.Preceding)
            while d1 < today:
                r = history[fixing_date]
                qassert.require(
                    r is not None,
                    f"missing rate on {fixing_date} for index {self._overnight_index.name()}",
                )
                assert r is not None
                d2 = calendar.advance(d1, 1, TimeUnit.Days)
                # We can't reach the maturity date inside this loop, so d2
                # needs no capping (unlike averaged_rate above).
                prod *= 1 + r * day_counter.year_fraction(d1, d2)
                fixing_date = d1 = d2
            # Here d1 == today, and today's fixing may already be known.
            if today < self._maturity_date:
                r = history[today]
                if r is not None:
                    tomorrow = calendar.advance(today, 1, TimeUnit.Days)
                    prod *= 1 + r * day_counter.year_fraction(today, tomorrow)
                    forward_discount_start = tomorrow
        # The telescopic part runs from the end of the last known fixing to
        # the maturity date.
        forward_discount = forward_curve.discount(self._maturity_date) / forward_curve.discount(
            forward_discount_start
        )
        prod /= forward_discount

        return (prod - 1) / day_counter.year_fraction(self._value_date, self._maturity_date)

    def rate(self) -> float:
        """# C++ parity: ``OvernightIndexFuture::rate`` (overnightindexfuture.cpp:118-126)."""
        if self._averaging_method == RateAveraging.Simple:
            return self.averaged_rate()
        if self._averaging_method == RateAveraging.Compound:
            return self.compounded_rate()
        qassert.fail(f"unknown compounding convention ({int(self._averaging_method)})")

    # --- Instrument interface ------------------------------------------

    def _perform_calculations(self) -> None:
        """# C++ parity: ``OvernightIndexFuture::performCalculations``
        (overnightindexfuture.cpp:136-139)."""
        r = self.convexity_adjustment() + self.rate()
        self._npv = 100.0 * (1.0 - r)


__all__ = ["OvernightIndexFuture"]
