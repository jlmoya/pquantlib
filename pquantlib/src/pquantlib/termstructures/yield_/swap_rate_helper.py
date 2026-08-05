"""SwapRateHelper — bootstrap from par-swap rate quote.

# C++ parity: ql/termstructures/yield/ratehelpers.{hpp,cpp} class SwapRateHelper (v1.43).

C++ ``SwapRateHelper::impliedQuote`` constructs a ``VanillaSwap`` via
``MakeVanillaSwap`` and inspects its fixed/floating leg NPVs:

    impliedQuote = -(floatLegNPV + spreadNPV) / (fixedLegBPS / 1e-4)

``initialize_dates`` builds the same swap and reads its schedule:

    earliest = swap.start_date()
    maturity = swap.maturity_date()
    latest_relevant = max(maturity, last_float_coupon.fixing_end_date())

An earlier revision approximated the schedule with ``calendar.advance``
and dropped the ``fixing_end_date`` term, noting that under a regular
schedule the max collapses to ``maturity``. It does — but only when the
last accrual end date is a business day on the *index's fixing calendar*,
which is not the calendar the schedule was rolled on. Whenever the two
differ (helper ``calendar`` != index fixing calendar) the par-coupon
round trip overshoots and ``latest_relevant`` lands a business day past
maturity; since ``LastRelevantDate`` is the default pillar, that moved
the curve node. See ``migration-harness/cpp/probes/v143_ts_swaphelper``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.ibor_coupon import IborCoupon
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

if TYPE_CHECKING:
    from pquantlib.instruments.vanilla_swap import VanillaSwap

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
        use_indexed_coupons: bool | None = None,
        float_convention: BusinessDayConvention | None = None,
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

        # ``None`` is C++'s ``Null<Natural>()``: MakeVanillaSwap then derives
        # the spot date as ``index->valueDate(fixingCalendar.adjust(today))``
        # instead of advancing ``settlementDays`` on the FLOATING-leg calendar.
        # Those two agree only while the helper's calendar and the index's
        # fixing calendar coincide, so the distinction has to survive.
        self._settlement_days: int | None = settlement_days
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
        self._use_indexed_coupons: bool | None = use_indexed_coupons
        self._float_convention: BusinessDayConvention | None = float_convention
        self._pillar_choice: PillarChoice = pillar
        if custom_pillar_date is not None:
            self._pillar_date = custom_pillar_date
        if evaluation_date is not None:
            self.initialize_dates(evaluation_date)

    # --- swap construction ----------------------------------------------------

    def _make_swap(
        self,
        evaluation_date: Date,
        ibor_index: IborIndex,
        fixed_rate: float | None,
        discount_curve: YieldTermStructureProtocol | None,
    ) -> VanillaSwap:
        """Build the underlying swap the way C++ ``initializeDates`` does.

        # C++ parity: ratehelpers.cpp:559-580 — the single MakeVanillaSwap
        # chain that both ``initializeDates`` and ``impliedQuote`` observe.
        """
        # Local import: termstructures/ should not depend on instruments/.
        from pquantlib.instruments.make_vanilla_swap import make_vanilla_swap  # noqa: PLC0415

        # C++: withFixedLegTenor(fixedFrequency_ == Once ? tenor_ : Period(fixedFrequency_)).
        fixed_tenor = (
            self._tenor
            if self._fixed_frequency == Frequency.Once
            else Period.from_frequency(self._fixed_frequency)
        )
        return make_vanilla_swap(
            swap_tenor=self._tenor,
            ibor_index=ibor_index,
            fixed_rate=fixed_rate,
            forward_start=self._fwd_start,
            fixed_leg_tenor=fixed_tenor,
            fixed_leg_calendar=self._calendar,
            fixed_leg_day_count=self._fixed_day_count,
            fixed_leg_convention=self._fixed_convention,
            fixed_leg_termination_convention=self._fixed_convention,
            fixed_leg_end_of_month=self._end_of_month,
            floating_leg_calendar=self._calendar,
            floating_leg_convention=self._float_convention,
            floating_leg_termination_convention=self._float_convention,
            floating_leg_end_of_month=self._end_of_month,
            floating_leg_spread=self.spread(),
            discount_curve=discount_curve,
            use_indexed_coupons=self._use_indexed_coupons,
            evaluation_date=evaluation_date,
            settlement_days=self._settlement_days,
        )

    # --- BootstrapHelper interface --------------------------------------------

    def implied_quote(self) -> float:
        """Implied par-swap rate from the underlying VanillaSwap.

        # C++ parity: ``SwapRateHelper::impliedQuote`` (ratehelpers.cpp).
        # We delegate to ``swap.fair_rate()``: the C++ formula
        # ``-(floatLegNPV + spreadNPV) / (fixedLegBPS / 1e-4)`` is exactly
        # what ``FixedVsFloatingSwap::fairRate`` computes via its result-fetch
        # fallback when the engine doesn't supply fair_rate directly.
        """
        # The bootstrap loop calls ``set_term_structure`` before each
        # ``implied_quote`` evaluation; use that curve as the discount.
        qassert.require(
            self._term_structure is not None,
            "SwapRateHelper: term structure not set yet",
        )
        ts = self._term_structure
        assert ts is not None
        idx = self._ibor_index.clone(ts) if hasattr(self._ibor_index, "clone") else self._ibor_index
        swap = self._make_swap(
            ts.reference_date(),
            idx,
            None,
            self._discount_curve if self._discount_curve is not None else ts,
        )
        return swap.fair_rate()

    # --- dates ----------------------------------------------------------------

    def initialize_dates(self, evaluation_date: Date) -> None:
        """Read earliest / maturity / latest-relevant off the real swap schedule.

        # C++ parity: ``SwapRateHelper::initializeDates`` (ratehelpers.cpp:557-616).
        """
        # A fixed rate of 0.0 (C++'s choice too) keeps this curve-free: the
        # fair-rate fallback would need a forecast curve, and none of the
        # dates depend on one.
        swap = self._make_swap(evaluation_date, self._ibor_index, 0.0, None)
        earliest = swap.start_date()
        maturity = swap.maturity_date()
        self._earliest_date = earliest
        self._maturity_date = maturity

        # C++ parity: ratehelpers.cpp:589-591 —
        #   latestRelevantDate_ = max(maturityDate_, lastCoupon->fixingEndDate())
        last_coupon = swap.floating_leg()[-1]
        qassert.require(
            isinstance(last_coupon, IborCoupon),
            "SwapRateHelper: last floating cashflow is not an IborCoupon",
        )
        assert isinstance(last_coupon, IborCoupon)
        self._latest_relevant_date = max(maturity, last_coupon.fixing_end_date())

        # C++ parity: ratehelpers.cpp:591-611.
        if self._pillar_choice == PillarChoice.MaturityDate:
            self._pillar_date = maturity
        elif self._pillar_choice == PillarChoice.LastRelevantDate:
            self._pillar_date = self._latest_relevant_date
        elif self._pillar_choice == PillarChoice.CustomDate:
            # pillar_date already assigned at construction time
            qassert.require(
                self._pillar_date is not None,
                "CustomDate pillar requires custom_pillar_date argument",
            )
            assert self._pillar_date is not None
            qassert.require(
                self._pillar_date >= earliest,
                f"pillar date ({self._pillar_date}) must be later than or equal "
                f"to the instrument's earliest date ({earliest})",
            )
            qassert.require(
                self._pillar_date <= self._latest_relevant_date,
                f"pillar date ({self._pillar_date}) must be before or equal to "
                f"the instrument's latest relevant date "
                f"({self._latest_relevant_date})",
            )
        else:
            qassert.fail(f"unknown Pillar.Choice({int(self._pillar_choice)})")

        # C++ parity: ratehelpers.cpp:613 — latestDate_ = pillarDate_, kept for
        # backward compatibility. It follows the PILLAR, not the latest
        # relevant date, so under Pillar::CustomDate the two differ.
        self._latest_date = self._pillar_date

    # --- inspectors ----------------------------------------------------------

    def spread(self) -> float:
        return 0.0 if self._spread is None else self._spread.value()

    def forward_start(self) -> Period:
        return self._fwd_start
