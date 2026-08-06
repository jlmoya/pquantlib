"""MakeCms — chained builder for a standard market constant-maturity swap.

# C++ parity: ql/instruments/makecms.{hpp,cpp} (v1.43).

A CMS swap is a CMS leg against a plain IBOR leg. Both legs get an independent
set of schedule settings (tenor / calendar / conventions / rule / end-of-month /
stub dates / day count), which is where a dropped setter hides most easily: the
CMS leg's own defaults (3M tenor, Act/360, ModifiedFollowing) differ from the
float leg's (index tenor, index day counter, index convention), so a setter
routed to the wrong leg produces a plausible-looking swap.

Both C++ constructors are collapsed into one Python signature: passing
``ibor_index=None`` takes the index off the swap index, which is exactly what
the 3-argument C++ constructor does (makecms.cpp:62-78).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.cms_coupon import CmsLeg
from pquantlib.cashflows.coupon_pricer import set_coupon_pricer
from pquantlib.cashflows.ibor_leg import ibor_leg
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.instruments.swap import Swap
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.cashflows.cash_flow import CashFlow
    from pquantlib.cashflows.cms_coupon_pricer import CmsCouponPricer
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.ibor_index import IborIndex
    from pquantlib.indexes.swap_index import SwapIndex
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.time.calendar import Calendar

_ZERO_PERIOD = Period(0, TimeUnit.Days)
_NULL_DATE = Date()
_BASIS_POINT = 1.0e-4


class MakeCms:
    """Chained builder for a CMS-vs-IBOR :class:`~pquantlib.instruments.swap.Swap`.

    # C++ parity: ``MakeCms`` (makecms.hpp:40-113). Every C++ ``withXxx``
    # setter is present as ``with_xxx`` and returns ``self``; C++'s
    # ``operator ext::shared_ptr<Swap>()`` is :meth:`build`.
    """

    def __init__(
        self,
        swap_tenor: Period,
        swap_index: SwapIndex,
        ibor_index: IborIndex | None = None,
        ibor_spread: float = 0.0,
        forward_start: Period = _ZERO_PERIOD,
    ) -> None:
        # # C++ parity: both MakeCms constructors (makecms.cpp:34-78).
        resolved_ibor = ibor_index if ibor_index is not None else swap_index.ibor_index()
        self._swap_tenor: Period = swap_tenor
        self._swap_index: SwapIndex = swap_index
        self._ibor_index: IborIndex = resolved_ibor
        self._ibor_spread: float = ibor_spread
        self._use_atm_spread: bool = False
        self._forward_start: Period = forward_start

        self._cms_spread: float = 0.0
        self._cms_gearing: float = 1.0
        self._cms_cap: float | None = None
        self._cms_floor: float | None = None

        self._effective_date: Date | None = None
        self._cms_calendar: Calendar = swap_index.fixing_calendar()
        self._float_calendar: Calendar = resolved_ibor.fixing_calendar()

        self._pay_cms: bool = True
        self._nominal: float = 1.0
        self._cms_tenor: Period = Period(3, TimeUnit.Months)
        self._float_tenor: Period = resolved_ibor.tenor()
        self._cms_convention: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing
        self._cms_termination_date_convention: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing
        self._float_convention: BusinessDayConvention = resolved_ibor.business_day_convention()
        self._float_termination_date_convention: BusinessDayConvention = (
            resolved_ibor.business_day_convention()
        )
        self._cms_rule: DateGeneration = DateGeneration.Backward
        self._float_rule: DateGeneration = DateGeneration.Backward
        self._cms_end_of_month: bool = False
        self._float_end_of_month: bool = False
        self._cms_first_date: Date | None = None
        self._cms_next_to_last_date: Date | None = None
        self._float_first_date: Date | None = None
        self._float_next_to_last_date: Date | None = None
        self._cms_day_count: DayCounter = Actual360()
        self._float_day_count: DayCounter = resolved_ibor.day_counter()

        # Arbitrary but documented C++ choice: discount on the swap index's
        # forecast curve unless withDiscountingTermStructure says otherwise.
        self._discount_curve: YieldTermStructureProtocol | None = swap_index.forwarding_term_structure()
        self._coupon_pricer: CmsCouponPricer | None = None

    # --- chained setters ---------------------------------------------------

    def receive_cms(self, flag: bool = True) -> MakeCms:
        """# C++ parity: ``receiveCms`` (makecms.cpp:172-175)."""
        self._pay_cms = not flag
        return self

    def with_nominal(self, n: float) -> MakeCms:
        """# C++ parity: ``withNominal`` (makecms.cpp:177-180)."""
        self._nominal = n
        return self

    def with_effective_date(self, effective_date: Date) -> MakeCms:
        """# C++ parity: ``withEffectiveDate`` (makecms.cpp:182-186)."""
        self._effective_date = effective_date
        return self

    # CMS leg ---------------------------------------------------------------

    def with_cms_leg_tenor(self, t: Period) -> MakeCms:
        """# C++ parity: ``withCmsLegTenor`` (makecms.cpp:199-202)."""
        self._cms_tenor = t
        return self

    def with_cms_leg_calendar(self, cal: Calendar) -> MakeCms:
        """# C++ parity: ``withCmsLegCalendar`` (makecms.cpp:204-208)."""
        self._cms_calendar = cal
        return self

    def with_cms_leg_convention(self, bdc: BusinessDayConvention) -> MakeCms:
        """# C++ parity: ``withCmsLegConvention`` (makecms.cpp:210-214)."""
        self._cms_convention = bdc
        return self

    def with_cms_leg_termination_date_convention(self, bdc: BusinessDayConvention) -> MakeCms:
        """# C++ parity: ``withCmsLegTerminationDateConvention`` (makecms.cpp:216-220)."""
        self._cms_termination_date_convention = bdc
        return self

    def with_cms_leg_rule(self, r: DateGeneration) -> MakeCms:
        """# C++ parity: ``withCmsLegRule`` (makecms.cpp:222-225)."""
        self._cms_rule = r
        return self

    def with_cms_leg_end_of_month(self, flag: bool = True) -> MakeCms:
        """# C++ parity: ``withCmsLegEndOfMonth`` (makecms.cpp:227-230)."""
        self._cms_end_of_month = flag
        return self

    def with_cms_leg_first_date(self, d: Date) -> MakeCms:
        """# C++ parity: ``withCmsLegFirstDate`` (makecms.cpp:232-235)."""
        self._cms_first_date = d
        return self

    def with_cms_leg_next_to_last_date(self, d: Date) -> MakeCms:
        """# C++ parity: ``withCmsLegNextToLastDate`` (makecms.cpp:237-241)."""
        self._cms_next_to_last_date = d
        return self

    def with_cms_leg_day_count(self, dc: DayCounter) -> MakeCms:
        """# C++ parity: ``withCmsLegDayCount`` (makecms.cpp:243-247)."""
        self._cms_day_count = dc
        return self

    # floating leg ----------------------------------------------------------

    def with_floating_leg_tenor(self, t: Period) -> MakeCms:
        """# C++ parity: ``withFloatingLegTenor`` (makecms.cpp:249-252)."""
        self._float_tenor = t
        return self

    def with_floating_leg_calendar(self, cal: Calendar) -> MakeCms:
        """# C++ parity: ``withFloatingLegCalendar`` (makecms.cpp:254-258)."""
        self._float_calendar = cal
        return self

    def with_floating_leg_convention(self, bdc: BusinessDayConvention) -> MakeCms:
        """# C++ parity: ``withFloatingLegConvention`` (makecms.cpp:260-264)."""
        self._float_convention = bdc
        return self

    def with_floating_leg_termination_date_convention(self, bdc: BusinessDayConvention) -> MakeCms:
        """# C++ parity: ``withFloatingLegTerminationDateConvention`` (makecms.cpp:266-270)."""
        self._float_termination_date_convention = bdc
        return self

    def with_floating_leg_rule(self, r: DateGeneration) -> MakeCms:
        """# C++ parity: ``withFloatingLegRule`` (makecms.cpp:272-275)."""
        self._float_rule = r
        return self

    def with_floating_leg_end_of_month(self, flag: bool = True) -> MakeCms:
        """# C++ parity: ``withFloatingLegEndOfMonth`` (makecms.cpp:277-280)."""
        self._float_end_of_month = flag
        return self

    def with_floating_leg_first_date(self, d: Date) -> MakeCms:
        """# C++ parity: ``withFloatingLegFirstDate`` (makecms.cpp:282-286)."""
        self._float_first_date = d
        return self

    def with_floating_leg_next_to_last_date(self, d: Date) -> MakeCms:
        """# C++ parity: ``withFloatingLegNextToLastDate`` (makecms.cpp:288-292)."""
        self._float_next_to_last_date = d
        return self

    def with_floating_leg_day_count(self, dc: DayCounter) -> MakeCms:
        """# C++ parity: ``withFloatingLegDayCount`` (makecms.cpp:294-298)."""
        self._float_day_count = dc
        return self

    # pricing ---------------------------------------------------------------

    def with_atm_spread(self, flag: bool = True) -> MakeCms:
        """Solve the float spread that zeroes the swap NPV.

        # C++ parity: ``withAtmSpread`` (makecms.cpp:300-303).
        """
        self._use_atm_spread = flag
        return self

    def with_discounting_term_structure(
        self, discounting_term_structure: YieldTermStructureProtocol
    ) -> MakeCms:
        """# C++ parity: ``withDiscountingTermStructure`` (makecms.cpp:188-193)."""
        self._discount_curve = discounting_term_structure
        return self

    def with_cms_coupon_pricer(self, coupon_pricer: CmsCouponPricer) -> MakeCms:
        """# C++ parity: ``withCmsCouponPricer`` (makecms.cpp:195-199)."""
        self._coupon_pricer = coupon_pricer
        return self

    # --- construction ------------------------------------------------------

    def _engine(self) -> DiscountingSwapEngine:
        qassert.require(
            self._discount_curve is not None,
            "MakeCms: no discounting term structure — the swap index carries no "
            "forecast curve and withDiscountingTermStructure was not called",
        )
        assert self._discount_curve is not None
        return DiscountingSwapEngine(self._discount_curve)

    def build(self) -> Swap:
        """Build the CMS swap.

        # C++ parity: ``MakeCms::operator ext::shared_ptr<Swap>()``
        # (makecms.cpp:86-170).
        """
        if self._effective_date is not None:
            start_date = self._effective_date
        else:
            fixing_days = self._ibor_index.fixing_days()
            # If the evaluation date is not a business day then move to the
            # next business day.
            ref_date = self._float_calendar.adjust(ObservableSettings().evaluation_date_or_today())
            spot_date = self._float_calendar.advance(ref_date, fixing_days, TimeUnit.Days)
            start_date = spot_date + self._forward_start

        termination_date = start_date + self._swap_tenor

        cms_schedule = Schedule.from_rule(
            start_date,
            termination_date,
            self._cms_tenor,
            self._cms_calendar,
            self._cms_convention,
            self._cms_termination_date_convention,
            self._cms_rule,
            self._cms_end_of_month,
            _or_null(self._cms_first_date),
            _or_null(self._cms_next_to_last_date),
        )
        float_schedule = Schedule.from_rule(
            start_date,
            termination_date,
            self._float_tenor,
            self._float_calendar,
            self._float_convention,
            self._float_termination_date_convention,
            self._float_rule,
            self._float_end_of_month,
            _or_null(self._float_first_date),
            _or_null(self._float_next_to_last_date),
        )

        cms_builder = (
            CmsLeg(cms_schedule, self._swap_index)
            .with_notionals(self._nominal)
            .with_payment_day_counter(self._cms_day_count)
            .with_payment_adjustment(self._cms_convention)
            .with_fixing_days(self._swap_index.fixing_days())
            .with_gearings(self._cms_gearing)
            .with_spreads(self._cms_spread)
        )
        if self._cms_cap is not None:
            cms_builder = cms_builder.with_caps(self._cms_cap)
        if self._cms_floor is not None:
            cms_builder = cms_builder.with_floors(self._cms_floor)
        cms_leg: list[CashFlow] = cms_builder.build()
        if self._coupon_pricer is not None:
            set_coupon_pricer(cms_leg, self._coupon_pricer)

        used_spread = self._ibor_spread
        if self._use_atm_spread:
            qassert.require(
                self._ibor_index.forecast_term_structure() is not None,
                f"null term structure set to this instance of {self._ibor_index.name()}",
            )
            qassert.require(
                self._swap_index.forwarding_term_structure() is not None,
                f"null term structure set to this instance of {self._swap_index.name()}",
            )
            qassert.require(self._coupon_pricer is not None, "no CmsCouponPricer set (yet)")
            probe_float_leg = self._build_float_leg(float_schedule, None)
            temp = Swap.from_legs(cms_leg, probe_float_leg)
            temp.set_pricing_engine(self._engine())
            npv = temp.leg_npv(0) + temp.leg_npv(1)
            used_spread = -npv / temp.leg_bps(1) * _BASIS_POINT

        float_leg = self._build_float_leg(float_schedule, used_spread)

        swap = Swap.from_legs(cms_leg, float_leg) if self._pay_cms else Swap.from_legs(float_leg, cms_leg)
        swap.set_pricing_engine(self._engine())
        return swap

    def _build_float_leg(self, schedule: Schedule, spread: float | None) -> list[CashFlow]:
        """The IBOR leg; ``spread=None`` is the un-spread probe leg C++ builds
        inside the ATM-spread solve (makecms.cpp:133-138)."""
        return ibor_leg(
            schedule,
            self._ibor_index,
            [self._nominal],
            payment_day_counter=self._float_day_count,
            payment_adjustment=self._float_convention,
            fixing_days=self._ibor_index.fixing_days(),
            spreads=0.0 if spread is None else spread,
        )


def _or_null(d: Date | None) -> Date:
    """``Schedule.from_rule`` takes the C++ null-``Date()`` sentinel, not ``None``."""
    return d if d is not None else _NULL_DATE


__all__ = ["MakeCms"]
