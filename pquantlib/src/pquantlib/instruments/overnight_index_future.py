"""OvernightIndexFuture — future on a compounded overnight-index investment.

# C++ parity: ql/instruments/overnightindexfuture.{hpp,cpp} (v1.43)

Compatible with the SOFR and SONIA futures listed on CME and ICE. The quoted
price is ``100 * (1 - (convexityAdjustment + rate))`` where ``rate`` is either
the simple average or the compounded average of the overnight fixings over
``[value_date, maturity_date)``, depending on ``averaging_method``.

The instrument prices itself: it overrides ``perform_calculations`` rather than
taking a pricing engine, exactly as C++ does.
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
    """Future on a compounded (or simply averaged) overnight index."""

    def __init__(
        self,
        overnight_index: OvernightIndex,
        value_date: Date,
        maturity_date: Date,
        convexity_adjustment: Quote | None = None,
        averaging_method: RateAveraging = RateAveraging.Compound,
    ) -> None:
        super().__init__()
        # C++ QL_REQUIREs a non-null index pointer; the Python signature is
        # already non-optional, so the check is the type annotation.
        self._overnight_index: OvernightIndex = overnight_index
        self._value_date: Date = value_date
        self._maturity_date: Date = maturity_date
        self._convexity_adjustment: Quote | None = convexity_adjustment
        self._averaging_method: RateAveraging = averaging_method
        overnight_index.register_with(self)
        if convexity_adjustment is not None:
            convexity_adjustment.register_with(self)
        ObservableSettings().register_with(self)

    # ---- inspectors ---------------------------------------------------------

    def overnight_index(self) -> OvernightIndex:
        return self._overnight_index

    def value_date(self) -> Date:
        return self._value_date

    def maturity_date(self) -> Date:
        return self._maturity_date

    def convexity_adjustment(self) -> float:
        # C++ parity: an empty handle means no adjustment.
        return 0.0 if self._convexity_adjustment is None else self._convexity_adjustment.value()

    # ---- rate ---------------------------------------------------------------

    def _forecast_curve(self) -> YieldTermStructure:
        ts = self._overnight_index.forecast_term_structure()
        qassert.require(
            ts is not None,
            f"null term structure set to this instance of {self._overnight_index.name()}",
        )
        assert isinstance(ts, YieldTermStructure)
        return ts

    def averaged_rate(self) -> float:
        """Simple average of the overnight fixings over the reference period.

        # C++ parity: ``OvernightIndexFuture::averagedRate`` (overnightindexfuture.cpp:42-70).
        """
        today = ObservableSettings().evaluation_date_or_today()
        calendar = self._overnight_index.fixing_calendar()
        day_counter = self._overnight_index.day_counter()
        forward_curve = self._forecast_curve()
        history = self._overnight_index.time_series()

        avg = 0.0
        d1 = self._value_date
        # d1 could be a holiday.
        fixing_date = calendar.adjust(d1, BusinessDayConvention.Preceding)
        while d1 < self._maturity_date:
            d2 = calendar.advance(d1, 1, TimeUnit.Days)
            if fixing_date < today:
                fwd = history[fixing_date]
                qassert.require(
                    fwd is not None,
                    f"missing rate on {fixing_date} for index "
                    f"{self._overnight_index.name()}",
                )
                assert fwd is not None
            else:
                fwd = history[fixing_date] if fixing_date == today else None
                if fwd is None:
                    fwd = forward_curve.forward_rate(
                        fixing_date, d2, Compounding.Simple, Frequency.Annual,
                        result_day_counter=day_counter,
                    ).rate()
            # The rate accrues from d1 even when the fixing date is earlier, and
            # d2 can overshoot maturity when maturity falls on a holiday.
            avg += fwd * day_counter.year_fraction(d1, min(d2, self._maturity_date))
            fixing_date = d1 = d2

        return avg / day_counter.year_fraction(self._value_date, self._maturity_date)

    def compounded_rate(self) -> float:
        """Compounded average of the overnight fixings over the reference period.

        # C++ parity: ``OvernightIndexFuture::compoundedRate`` (overnightindexfuture.cpp:72-127).
        """
        today = ObservableSettings().evaluation_date_or_today()
        calendar = self._overnight_index.fixing_calendar()
        day_counter = self._overnight_index.day_counter()
        forward_curve = self._forecast_curve()

        prod = 1.0
        forward_discount_start = self._value_date
        if today > self._value_date:
            # Can't value on a weekend inside the reference period, because the
            # reset rate isn't known until the start of the next business day;
            # a user who really wants to can supply an estimate.
            today = calendar.adjust(today)
            forward_discount_start = today
            history = self._overnight_index.time_series()
            d1 = self._value_date
            # d1 could be a holiday.
            fixing_date = calendar.adjust(d1, BusinessDayConvention.Preceding)
            while d1 < today:
                r = history[fixing_date]
                qassert.require(
                    r is not None,
                    f"missing rate on {fixing_date} for index "
                    f"{self._overnight_index.name()}",
                )
                assert r is not None
                d2 = calendar.advance(d1, 1, TimeUnit.Days)
                # The rate accrues from d1 even when the fixing date is earlier.
                # This loop cannot reach maturity, so d2 needs no cap.
                prod *= 1.0 + r * day_counter.year_fraction(d1, d2)
                fixing_date = d1 = d2
            # d1 == today here, and today's fixing may already be published.
            if today < self._maturity_date:
                r = history[today]
                if r is not None:
                    tomorrow = calendar.advance(today, 1, TimeUnit.Days)
                    prod *= 1.0 + r * day_counter.year_fraction(today, tomorrow)
                    forward_discount_start = tomorrow

        # The telescopic part runs from the end of the last known fixing to
        # maturity.
        forward_discount = forward_curve.discount(
            self._maturity_date
        ) / forward_curve.discount(forward_discount_start)
        prod /= forward_discount

        return (prod - 1.0) / day_counter.year_fraction(
            self._value_date, self._maturity_date
        )

    def rate(self) -> float:
        # C++ parity: ``OvernightIndexFuture::rate``.
        if self._averaging_method == RateAveraging.Simple:
            return self.averaged_rate()
        if self._averaging_method == RateAveraging.Compound:
            return self.compounded_rate()
        return qassert.fail(
            f"unknown compounding convention ({int(self._averaging_method)})"
        )

    # ---- Instrument interface ----------------------------------------------

    def is_expired(self) -> bool:
        # C++ parity: ``detail::simple_event(maturityDate_).hasOccurred()``.
        return self._maturity_date <= ObservableSettings().evaluation_date_or_today()

    def _perform_calculations(self) -> None:
        # C++ parity: ``OvernightIndexFuture::performCalculations``. No pricing
        # engine: the instrument prices itself, as in C++.
        r = self.convexity_adjustment() + self.rate()
        self._npv = 100.0 * (1.0 - r)
        self._error_estimate = 0.0
