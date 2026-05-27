"""OISRateHelper — bootstrap from OIS rate quote.

# C++ parity: ql/termstructures/yield/oisratehelper.{hpp,cpp} class OISRateHelper.

C++ ``OISRateHelper`` builds an ``OvernightIndexedSwap`` via ``MakeOIS`` and
calls ``swap_->fairRate()`` for ``impliedQuote``. PQuantLib L2-C ports the
constructor + inspectors + ``initialize_dates``; ``implied_quote`` raises
until ``OvernightIndexedSwap`` + ``MakeOIS`` land in L3.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper, PillarChoice
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_ZERO_PERIOD = Period(0, TimeUnit.Days)


class OISRateHelper(BootstrapHelper[YieldTermStructureProtocol]):
    """OIS rate helper. Full ``implied_quote`` deferred to L3 (OvernightIndexedSwap)."""

    def __init__(
        self,
        settlement_days: int,
        tenor: Period,
        fixed_rate: Quote | float,
        overnight_index: OvernightIndex,
        discount_curve: YieldTermStructureProtocol | None = None,
        telescopic_value_dates: bool = False,
        payment_lag: int = 0,
        payment_convention: BusinessDayConvention = BusinessDayConvention.Following,
        forward_start: Period = _ZERO_PERIOD,
        pillar: PillarChoice = PillarChoice.LastRelevantDate,
        custom_pillar_date: Date | None = None,
        end_of_month: bool | None = None,
        evaluation_date: Date | None = None,
    ) -> None:
        super().__init__(fixed_rate)
        self._settlement_days: int = settlement_days
        self._tenor: Period = tenor
        self._overnight_index: OvernightIndex = overnight_index
        self._discount_curve: YieldTermStructureProtocol | None = discount_curve
        self._telescopic_value_dates: bool = telescopic_value_dates
        self._payment_lag: int = payment_lag
        self._payment_convention: BusinessDayConvention = payment_convention
        self._fwd_start: Period = forward_start
        self._end_of_month: bool | None = end_of_month
        self._pillar_choice: PillarChoice = pillar
        if custom_pillar_date is not None:
            self._pillar_date = custom_pillar_date
        if evaluation_date is not None:
            self.initialize_dates(evaluation_date)

    # --- BootstrapHelper interface --------------------------------------------

    def implied_quote(self) -> float:
        qassert.fail(
            "OISRateHelper.implied_quote requires L3 OvernightIndexedSwap + MakeOIS "
            "(deferred to L3).",
        )

    # --- dates ---------------------------------------------------------------

    def initialize_dates(self, evaluation_date: Date) -> None:
        cal = self._overnight_index.fixing_calendar()
        ref = cal.adjust(evaluation_date, BusinessDayConvention.Following)
        spot = cal.advance(ref, self._settlement_days, TimeUnit.Days)
        if self._fwd_start.length != 0:
            earliest = cal.advance(
                spot, self._fwd_start.length, self._fwd_start.units,
                self._payment_convention, self._end_of_month or False,
            )
        else:
            earliest = spot
        maturity = cal.advance(
            earliest, self._tenor.length, self._tenor.units,
            self._payment_convention, self._end_of_month or False,
        )
        self._earliest_date = earliest
        self._maturity_date = maturity
        self._latest_relevant_date = maturity
        if self._pillar_choice in (PillarChoice.MaturityDate, PillarChoice.LastRelevantDate):
            self._pillar_date = maturity
        self._latest_date = self._pillar_date

    # --- inspectors ---------------------------------------------------------

    def telescopic_value_dates(self) -> bool:
        return self._telescopic_value_dates
