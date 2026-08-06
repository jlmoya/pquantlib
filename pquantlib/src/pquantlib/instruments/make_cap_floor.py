"""MakeCapFloor — chained builder for a standard market cap/floor.

# C++ parity: ql/instruments/makecapfloor.{hpp,cpp} (v1.43).

The C++ class holds a ``MakeVanillaSwap`` member and forwards almost every
setter onto its *floating* leg; only the floating leg of that swap is ever
used. PQuantLib renders ``MakeVanillaSwap`` as the
:func:`~pquantlib.instruments.make_vanilla_swap.make_vanilla_swap` keyword-arg
factory, so this builder accumulates the same settings as keyword arguments and
makes one call at :meth:`build` time — which keeps exactly one implementation of
the swap-construction logic.

Two constructor-time details are easy to lose and are pinned by tests:

* the fixed leg tenor is forced to 1Y and its day counter to Act/365F purely so
  that the currency-driven default lookup inside ``make_vanilla_swap`` can never
  fail (makecapfloor.cpp:37-40) — the fixed leg is discarded;
* a **zero** ``forward_start`` sets ``first_caplet_excluded`` to ``True``, so
  the default 5Y semi-annual cap has nine caplets, not ten
  (makecapfloor.cpp:33).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.coupon import Coupon
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.instruments.cap_floor import CapFloor, CapFloorType
from pquantlib.instruments.make_vanilla_swap import make_vanilla_swap
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.capfloor.black_capfloor_engine import (
    BlackStyleCapFloorEngine,
)
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pquantlib.cashflows.cash_flow import CashFlow
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.ibor_index import IborIndex
    from pquantlib.pricingengines.pricing_engine import PricingEngine
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.time.business_day_convention import BusinessDayConvention
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.date import Date

_ZERO_PERIOD = Period(0, TimeUnit.Days)


def atm_rate_of_leg(
    leg: Sequence[CashFlow],
    discount_curve: YieldTermStructureProtocol,
    include_settlement_date_flows: bool = False,
    settlement_date: Date | None = None,
    npv_date: Date | None = None,
    target_npv: float | None = None,
) -> float:
    """Par rate of ``leg`` against ``discount_curve``.

    # C++ parity: ``CashFlows::atmRate`` (ql/cashflows/cashflows.cpp) together
    # with the anonymous-namespace ``BPSCalculator`` visitor it drives. The
    # visitor accumulates ``nominal * accrualPeriod * df`` over Coupons (the
    # "bps" term) and ``amount * df`` over every *non*-Coupon cashflow (the
    # rate-insensitive "nonSensNPV" term); the ATM rate is
    # ``(npv - nonSensNPV) / bps``.
    #
    # Home note: in C++ this is a ``CashFlows`` static. PQuantLib's
    # ``CashFlows`` aggregator does not carry it yet, and that module is owned
    # elsewhere this session, so it lives here — the only consumers today are
    # ``MakeCapFloor`` and ``MakeYoYInflationCapFloor``. It should be hoisted
    # to ``CashFlows.atm_rate`` when that module is next touched.
    """
    if not leg:
        return 0.0
    settle = (
        settlement_date if settlement_date is not None else ObservableSettings().evaluation_date_or_today()
    )
    npv_d = npv_date if npv_date is not None else settle

    npv = 0.0
    bps = 0.0
    non_sens_npv = 0.0
    for cf in leg:
        if cf.has_occurred(settle, include_settlement_date_flows) or cf.trading_ex_coupon(settle):
            continue
        df = discount_curve.discount(cf.date())
        npv += cf.amount() * df
        if isinstance(cf, Coupon):
            bps += cf.nominal() * cf.accrual_period() * df
        else:
            non_sens_npv += cf.amount() * df

    if target_npv is None:
        used_target = npv - non_sens_npv
    else:
        used_target = target_npv * discount_curve.discount(npv_d) - non_sens_npv

    if used_target == 0.0:
        return 0.0
    qassert.require(bps != 0.0, "null bps: impossible atm rate")
    return used_target / bps


class MakeCapFloor:
    """Chained builder for :class:`~pquantlib.instruments.cap_floor.CapFloor`.

    # C++ parity: ``MakeCapFloor`` (makecapfloor.hpp:38-74). Every C++
    # ``withXxx`` setter is present as ``with_xxx`` and returns ``self``;
    # C++'s ``operator ext::shared_ptr<CapFloor>()`` is :meth:`build`.
    """

    def __init__(
        self,
        cap_floor_type: CapFloorType,
        cap_floor_tenor: Period,
        ibor_index: IborIndex,
        strike: float | None = None,
        forward_start: Period = _ZERO_PERIOD,
    ) -> None:
        # # C++ parity: MakeCapFloor::MakeCapFloor (makecapfloor.cpp:29-40).
        self._cap_floor_type: CapFloorType = cap_floor_type
        self._strike: float | None = strike
        self._first_caplet_excluded: bool = forward_start == _ZERO_PERIOD
        self._as_optionlet: bool = False
        self._engine: PricingEngine | None = None

        self._tenor: Period = cap_floor_tenor
        self._ibor_index: IborIndex = ibor_index
        self._forward_start: Period = forward_start
        # Settings forwarded to make_vanilla_swap. The fixed-leg tenor and day
        # count are pinned here (not left to the currency default) exactly as
        # C++ does, so an index in a currency without a default cannot throw;
        # only the floating leg of the swap is ever used.
        self._effective_date: Date | None = None
        self._floating_leg_tenor: Period | None = None
        self._floating_leg_calendar: Calendar | None = None
        self._floating_leg_convention: BusinessDayConvention | None = None
        self._floating_leg_termination_convention: BusinessDayConvention | None = None
        self._floating_leg_rule: DateGeneration = DateGeneration.Backward
        self._floating_leg_end_of_month: bool = False
        self._floating_leg_first_date: Date | None = None
        self._floating_leg_next_to_last_date: Date | None = None
        self._floating_leg_day_count: DayCounter | None = None
        self._nominal: float = 1.0

    # --- chained setters ---------------------------------------------------

    def with_nominal(self, n: float) -> MakeCapFloor:
        """# C++ parity: ``withNominal`` (makecapfloor.cpp:94-97)."""
        self._nominal = n
        return self

    def with_effective_date(self, effective_date: Date, first_caplet_excluded: bool) -> MakeCapFloor:
        """# C++ parity: ``withEffectiveDate`` (makecapfloor.cpp:99-104).

        Note that ``first_caplet_excluded`` is **not** optional in C++ either:
        setting an explicit effective date always restates it.
        """
        self._effective_date = effective_date
        self._first_caplet_excluded = first_caplet_excluded
        return self

    def with_tenor(self, t: Period) -> MakeCapFloor:
        """# C++ parity: ``withTenor`` → ``withFloatingLegTenor`` (.cpp:106-109)."""
        self._floating_leg_tenor = t
        return self

    def with_calendar(self, cal: Calendar) -> MakeCapFloor:
        """# C++ parity: ``withCalendar`` → ``withFloatingLegCalendar`` (.cpp:112-115)."""
        self._floating_leg_calendar = cal
        return self

    def with_convention(self, bdc: BusinessDayConvention) -> MakeCapFloor:
        """# C++ parity: ``withConvention`` → ``withFloatingLegConvention`` (.cpp:118-121)."""
        self._floating_leg_convention = bdc
        return self

    def with_termination_date_convention(self, bdc: BusinessDayConvention) -> MakeCapFloor:
        """# C++ parity: ``withTerminationDateConvention`` (.cpp:124-128)."""
        self._floating_leg_termination_convention = bdc
        return self

    def with_rule(self, r: DateGeneration) -> MakeCapFloor:
        """# C++ parity: ``withRule`` → ``withFloatingLegRule`` (.cpp:131-134)."""
        self._floating_leg_rule = r
        return self

    def with_end_of_month(self, flag: bool = True) -> MakeCapFloor:
        """# C++ parity: ``withEndOfMonth`` → ``withFloatingLegEndOfMonth`` (.cpp:136-139)."""
        self._floating_leg_end_of_month = flag
        return self

    def with_first_date(self, d: Date) -> MakeCapFloor:
        """# C++ parity: ``withFirstDate`` → ``withFloatingLegFirstDate`` (.cpp:142-145)."""
        self._floating_leg_first_date = d
        return self

    def with_next_to_last_date(self, d: Date) -> MakeCapFloor:
        """# C++ parity: ``withNextToLastDate`` (.cpp:147-150)."""
        self._floating_leg_next_to_last_date = d
        return self

    def with_day_count(self, dc: DayCounter) -> MakeCapFloor:
        """# C++ parity: ``withDayCount`` → ``withFloatingLegDayCount`` (.cpp:152-155)."""
        self._floating_leg_day_count = dc
        return self

    def as_optionlet(self, b: bool = True) -> MakeCapFloor:
        """Keep only the last coupon. # C++ parity: ``asOptionlet`` (.cpp:157-160)."""
        self._as_optionlet = b
        return self

    def with_pricing_engine(self, engine: PricingEngine) -> MakeCapFloor:
        """# C++ parity: ``withPricingEngine`` (.cpp:162-166)."""
        self._engine = engine
        return self

    # --- construction ------------------------------------------------------

    def build(self) -> CapFloor:
        """Build the cap/floor.

        # C++ parity: ``MakeCapFloor::operator ext::shared_ptr<CapFloor>()``
        # (makecapfloor.cpp:47-92).
        """
        swap = make_vanilla_swap(
            self._tenor,
            self._ibor_index,
            # C++ passes fixedRate 0.0, so no fair-rate solve happens and the
            # (discarded) fixed leg never needs a forecast curve.
            0.0,
            self._forward_start,
            nominal=self._nominal,
            effective_date=self._effective_date,
            evaluation_date=ObservableSettings().evaluation_date_or_today(),
            fixed_leg_tenor=Period(1, TimeUnit.Years),
            fixed_leg_day_count=Actual365Fixed(),
            floating_leg_tenor=self._floating_leg_tenor,
            floating_leg_calendar=self._floating_leg_calendar,
            floating_leg_convention=self._floating_leg_convention,
            floating_leg_termination_convention=self._floating_leg_termination_convention,
            floating_leg_rule=self._floating_leg_rule,
            floating_leg_end_of_month=self._floating_leg_end_of_month,
            floating_leg_first_date=self._floating_leg_first_date,
            floating_leg_next_to_last_date=self._floating_leg_next_to_last_date,
            floating_leg_day_count=self._floating_leg_day_count,
        )

        leg: list[CashFlow] = list(swap.floating_leg())
        if self._first_caplet_excluded:
            leg = leg[1:]
        # Only leaves the last coupon.
        if self._as_optionlet and len(leg) > 1:
            leg = leg[-1:]

        strike = self._strike
        if strike is None:
            # Temporary patch in C++ too: the ATM strike needs a discount
            # curve, and the only place to get one is the engine.
            qassert.require(
                isinstance(self._engine, BlackStyleCapFloorEngine),
                "cannot calculate ATM without a BlackCapFloorEngine or BachelierCapFloorEngine",
            )
            assert isinstance(self._engine, BlackStyleCapFloorEngine)
            discount_curve = self._engine.term_structure()
            strike = atm_rate_of_leg(
                leg,
                discount_curve,
                False,
                discount_curve.reference_date(),
            )

        cap_floor = CapFloor.from_strikes(self._cap_floor_type, leg, [strike])
        if self._engine is not None:
            cap_floor.set_pricing_engine(self._engine)
        return cap_floor


__all__ = ["MakeCapFloor", "atm_rate_of_leg"]
