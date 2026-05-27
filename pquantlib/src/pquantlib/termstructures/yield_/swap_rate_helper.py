"""SwapRateHelper — bootstrap from par-swap rate quote.

# C++ parity: ql/termstructures/yield/ratehelpers.{hpp,cpp} class SwapRateHelper.

C++ ``SwapRateHelper::impliedQuote`` constructs a ``VanillaSwap`` via
``MakeVanillaSwap`` and inspects its fixed/floating leg NPVs:

    impliedQuote = -(floatLegNPV + spreadNPV) / (fixedLegBPS / 1e-4)

L3-C closes the carry-over: ``implied_quote`` now builds the underlying
swap via ``make_vanilla_swap`` and reads ``fair_rate()``.

Carve-out: the ``initialize_dates`` method still approximates earliest /
maturity dates via ``calendar.advance(...)`` rather than building a full
``VanillaSwap`` schedule. That's accurate enough for pillar-date computation;
a full schedule-driven implementation is deferred.
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
        """Implied par-swap rate from the underlying VanillaSwap.

        # C++ parity: ``SwapRateHelper::impliedQuote`` (ratehelpers.cpp).
        # We delegate to ``swap.fair_rate()``: the C++ formula
        # ``-(floatLegNPV + spreadNPV) / (fixedLegBPS / 1e-4)`` is exactly
        # what ``FixedVsFloatingSwap::fairRate`` computes via its result-fetch
        # fallback when the engine doesn't supply fair_rate directly.
        """
        # Local import: termstructures/ should not depend on instruments/.
        from pquantlib.instruments.make_vanilla_swap import make_vanilla_swap  # noqa: PLC0415

        # The bootstrap loop calls ``set_term_structure`` before each
        # ``implied_quote`` evaluation; use that curve as the discount.
        qassert.require(
            self._term_structure is not None,
            "SwapRateHelper: term structure not set yet",
        )
        ts = self._term_structure
        assert ts is not None
        # Build a swap whose fixed-rate is solved-for (None -> fair-rate path
        # in make_vanilla_swap). Use the helper's tenor + the index + curve;
        # MakeVanillaSwap's currency-driven defaults will reconstruct the
        # same schedule the constructor described.
        idx = self._ibor_index.clone(ts) if hasattr(self._ibor_index, "clone") else self._ibor_index
        swap = make_vanilla_swap(
            swap_tenor=self._tenor,
            ibor_index=idx,
            fixed_rate=None,
            forward_start=self._fwd_start,
            fixed_leg_tenor=Period.from_frequency(self._fixed_frequency),
            fixed_leg_day_count=self._fixed_day_count,
            fixed_leg_convention=self._fixed_convention,
            fixed_leg_termination_convention=self._fixed_convention,
            fixed_leg_end_of_month=self._end_of_month,
            floating_leg_spread=self.spread(),
            discount_curve=self._discount_curve if self._discount_curve is not None else ts,
            evaluation_date=ts.reference_date(),
            settlement_days=self._settlement_days,
        )
        return swap.fair_rate()

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
