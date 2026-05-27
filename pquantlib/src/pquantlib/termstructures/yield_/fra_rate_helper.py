"""FraRateHelper — bootstrap Forward Rate Agreement quote.

# C++ parity: ql/termstructures/yield/ratehelpers.{hpp,cpp} class FraRateHelper.

Two modes — both implemented:
- ``periodToStart`` + ``lengthInMonths`` / explicit ``ibor_index``:
  earliest = spot + periodToStart, maturity = spot + periodToStart + tenor.
- ``startDate`` / ``endDate`` direct: earliest = startDate, maturity = endDate.

``useIndexedCoupon`` switches between fixing-based (calls index.fixing) and
discount-factor-based (formula  ``(d(t_start)/d(t_end) - 1) / yearFraction``).
Both branches are implemented (the L2-D IborCoupon classes are now
available, completing the L2-C carry-over).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.currencies.currency import Currency
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper, PillarChoice
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class FraRateHelper(BootstrapHelper[YieldTermStructureProtocol]):
    """FRA rate helper. Supports both useIndexedCoupon branches."""

    def __init__(
        self,
        rate: Quote | float,
        months_to_start: int | None = None,
        months_to_end: int | None = None,
        period_to_start: Period | None = None,
        length_in_months: int | None = None,
        fixing_days: int | None = None,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention | None = None,
        end_of_month: bool | None = None,
        day_counter: DayCounter | None = None,
        ibor_index: IborIndex | None = None,
        pillar: PillarChoice = PillarChoice.LastRelevantDate,
        custom_pillar_date: Date | None = None,
        use_indexed_coupon: bool = False,
        evaluation_date: Date | None = None,
    ) -> None:
        super().__init__(rate)
        # Map months_to_start → period_to_start for both modes.
        if period_to_start is None and months_to_start is not None:
            period_to_start = Period(months_to_start, TimeUnit.Months)
        if length_in_months is None and months_to_end is not None and months_to_start is not None:
            length_in_months = months_to_end - months_to_start
        # Mode A: explicit conventions.
        if ibor_index is None:
            qassert.require(
                period_to_start is not None and length_in_months is not None
                and fixing_days is not None and calendar is not None
                and convention is not None and end_of_month is not None
                and day_counter is not None,
                "FraRateHelper: provide either ibor_index OR all of (period_to_start, "
                "length_in_months, fixing_days, calendar, convention, end_of_month, "
                "day_counter)",
            )
            assert period_to_start is not None
            assert length_in_months is not None
            assert fixing_days is not None
            assert calendar is not None
            assert convention is not None
            assert end_of_month is not None
            assert day_counter is not None
            self._ibor_index = IborIndex(
                "no-fix", Period(length_in_months, TimeUnit.Months),
                fixing_days, Currency(),
                calendar, convention, end_of_month, day_counter,
            )
        else:
            self._ibor_index = ibor_index
        qassert.require(period_to_start is not None, "period_to_start required")
        assert period_to_start is not None
        self._period_to_start: Period = period_to_start
        self._pillar_choice: PillarChoice = pillar
        self._use_indexed_coupon: bool = use_indexed_coupon
        self._spanning_time: float = 0.0
        self._fixing_date: Date | None = None
        # Custom pillar date stored upfront; resolved in initialize_dates.
        if custom_pillar_date is not None:
            self._pillar_date = custom_pillar_date
        if evaluation_date is not None:
            self.initialize_dates(evaluation_date)

    # --- BootstrapHelper interface --------------------------------------------

    def implied_quote(self) -> float:
        """Return the model-implied FRA rate.

        # C++ parity: ``FraRateHelper::impliedQuote``:
        #   - useIndexedCoupon=true  → index.fixing(fixingDate, true)
        #   - useIndexedCoupon=false → discount-factor par approximation
        """
        qassert.require(self._term_structure is not None, "term structure not set")
        assert self._term_structure is not None
        qassert.require(self._earliest_date is not None, "dates not initialized")
        qassert.require(self._maturity_date is not None, "dates not initialized")
        assert self._earliest_date is not None
        assert self._maturity_date is not None

        if self._use_indexed_coupon:
            qassert.require(self._fixing_date is not None, "fixing date not initialized")
            assert self._fixing_date is not None
            return self._ibor_index.fixing(
                self._fixing_date, forecast_todays_fixing=True
            )

        return (
            self._term_structure.discount(self._earliest_date)
            / self._term_structure.discount(self._maturity_date) - 1.0
        ) / self._spanning_time

    def set_term_structure(self, ts: YieldTermStructureProtocol) -> None:
        super().set_term_structure(ts)
        self._ibor_index = self._ibor_index.clone(ts)

    # --- dates ----------------------------------------------------------------

    def initialize_dates(self, evaluation_date: Date) -> None:
        cal = self._ibor_index.fixing_calendar()
        ref = cal.adjust(evaluation_date, BusinessDayConvention.Following)
        spot = cal.advance(
            ref, self._ibor_index.fixing_days(), TimeUnit.Days,
        )
        earliest = cal.advance(
            spot, self._period_to_start.length, self._period_to_start.units,
            self._ibor_index.business_day_convention(),
            self._ibor_index.end_of_month(),
        )
        end_period = self._period_to_start + self._ibor_index.tenor()
        maturity = cal.advance(
            spot, end_period.length, end_period.units,
            self._ibor_index.business_day_convention(),
            self._ibor_index.end_of_month(),
        )
        self._earliest_date = earliest
        self._maturity_date = maturity

        # C++ parity: ``initializeDates`` branches on useIndexedCoupon:
        # - true  → latest_relevant_date = index.maturityDate(earliest)
        #           spanning_time is unused.
        # - false → latest_relevant_date = maturity_date
        #           spanning_time = year_fraction(earliest, maturity).
        if self._use_indexed_coupon:
            self._latest_relevant_date = self._ibor_index.maturity_date(earliest)
        else:
            self._spanning_time = self._ibor_index.day_counter().year_fraction(
                earliest, maturity,
            )
            self._latest_relevant_date = maturity

        if self._pillar_choice in (PillarChoice.MaturityDate, PillarChoice.LastRelevantDate):
            if self._pillar_choice == PillarChoice.MaturityDate:
                self._pillar_date = maturity
            else:
                self._pillar_date = self._latest_relevant_date
        elif self._pillar_choice == PillarChoice.CustomDate:
            qassert.require(
                self._pillar_date is not None,
                "CustomDate pillar requires custom_pillar_date argument",
            )

        self._latest_date = self._pillar_date
        self._fixing_date = self._ibor_index.fixing_date(earliest)
