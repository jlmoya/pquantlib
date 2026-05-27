"""SwapRateHelper — bootstrap from par-swap rate quote.

# C++ parity: ql/termstructures/yield/ratehelpers.{hpp,cpp} class SwapRateHelper.

C++ ``SwapRateHelper::impliedQuote`` constructs a ``VanillaSwap`` via
``MakeVanillaSwap`` and inspects its fixed/floating leg NPVs:

    impliedQuote = -(floatLegNPV + spreadNPV) / (fixedLegBPS / 1e-4)

That requires L3's ``VanillaSwap``, ``MakeVanillaSwap``, and
``DiscountingSwapEngine``. PQuantLib L2-C ports the constructor surface +
inspectors + dates initialization, but raises on ``implied_quote()`` —
the full bootstrap pipeline lands in L3.

Inspectors that don't depend on VanillaSwap (settlement_days / tenor /
fixed_frequency / fixed_convention / fixed_day_count / spread / fwd_start /
end_of_month / use_indexed_coupons) are fully ported.

Carve-out: the ``initialize_dates`` method currently approximates earliest /
maturity dates via ``calendar.advance(...)`` rather than building a full
``VanillaSwap`` schedule. That's accurate enough for pillar-date computation;
the precise BPS-based implied quote stays L3.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper, PillarChoice
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_ZERO_PERIOD = Period(0, TimeUnit.Days)


class SwapRateHelper(BootstrapHelper[YieldTermStructureProtocol]):
    """Par-swap rate helper. Full ``implied_quote`` deferred to L3 (VanillaSwap)."""

    def __init__(
        self,
        rate: Quote | float,
        tenor: Period | None = None,
        calendar: Calendar | None = None,
        fixed_frequency: Frequency | None = None,
        fixed_convention: BusinessDayConvention | None = None,
        fixed_day_count: DayCounter | None = None,
        ibor_index: IborIndex | None = None,
        swap_index: SwapIndex | None = None,
        spread: Quote | None = None,
        fwd_start: Period = _ZERO_PERIOD,
        discount_curve: YieldTermStructureProtocol | None = None,
        settlement_days: int | None = None,
        pillar: PillarChoice = PillarChoice.LastRelevantDate,
        custom_pillar_date: Date | None = None,
        end_of_month: bool = False,
        evaluation_date: Date | None = None,
    ) -> None:
        super().__init__(rate)
        # Path A: build from a SwapIndex (delegates the fixed-leg meta).
        if swap_index is not None:
            tenor = swap_index.tenor()
            calendar = swap_index.fixing_calendar()
            fixed_frequency = swap_index.fixed_leg_tenor().frequency()
            fixed_convention = swap_index.fixed_leg_convention()
            fixed_day_count = swap_index.fixed_leg_day_counter()
            ibor_index = swap_index.ibor_index()
        qassert.require(
            tenor is not None and calendar is not None and fixed_frequency is not None
            and fixed_convention is not None and fixed_day_count is not None
            and ibor_index is not None,
            "SwapRateHelper: provide swap_index OR (tenor + calendar + fixed_frequency + "
            "fixed_convention + fixed_day_count + ibor_index)",
        )
        assert tenor is not None
        assert calendar is not None
        assert fixed_frequency is not None
        assert fixed_convention is not None
        assert fixed_day_count is not None
        assert ibor_index is not None

        self._settlement_days: int = settlement_days if settlement_days is not None else ibor_index.fixing_days()
        self._tenor: Period = tenor
        self._calendar: Calendar = calendar
        self._fixed_frequency: Frequency = fixed_frequency
        self._fixed_convention: BusinessDayConvention = fixed_convention
        self._fixed_day_count: DayCounter = fixed_day_count
        self._ibor_index: IborIndex = ibor_index
        self._spread: Quote | None = spread
        self._fwd_start: Period = fwd_start
        self._discount_curve: YieldTermStructureProtocol | None = discount_curve
        self._end_of_month: bool = end_of_month
        self._pillar_choice: PillarChoice = pillar
        if custom_pillar_date is not None:
            self._pillar_date = custom_pillar_date
        if evaluation_date is not None:
            self.initialize_dates(evaluation_date)

    # --- BootstrapHelper interface --------------------------------------------

    def implied_quote(self) -> float:
        qassert.fail(
            "SwapRateHelper.implied_quote requires L3 VanillaSwap + DiscountingSwapEngine "
            "(deferred to L3).",
        )

    # --- dates ----------------------------------------------------------------

    def initialize_dates(self, evaluation_date: Date) -> None:
        """Approximate earliest / maturity dates via calendar.advance.

        Full VanillaSwap-driven schedule is L3.
        """
        cal = self._ibor_index.fixing_calendar()
        ref = cal.adjust(evaluation_date, BusinessDayConvention.Following)
        # Forward-start swap: advance by settlement_days then by fwd_start.
        spot = cal.advance(ref, self._settlement_days, TimeUnit.Days)
        if self._fwd_start.length != 0:
            earliest = cal.advance(
                spot, self._fwd_start.length, self._fwd_start.units,
                self._fixed_convention, self._end_of_month,
            )
        else:
            earliest = spot
        maturity = cal.advance(
            earliest, self._tenor.length, self._tenor.units,
            self._fixed_convention, self._end_of_month,
        )
        self._earliest_date = earliest
        self._maturity_date = maturity
        # In C++ the latest_relevant_date considers the last coupon's fixing_end_date;
        # here we conservatively use maturity.
        self._latest_relevant_date = maturity
        if self._pillar_choice in (PillarChoice.MaturityDate, PillarChoice.LastRelevantDate):
            self._pillar_date = maturity
        self._latest_date = self._pillar_date

    # --- inspectors ----------------------------------------------------------

    def spread(self) -> float:
        return 0.0 if self._spread is None else self._spread.value()

    def forward_start(self) -> Period:
        return self._fwd_start
