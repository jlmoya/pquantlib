"""MakeYoYInflationCapFloor — chained builder for a standard YoY cap/floor.

# C++ parity: ql/instruments/makeyoyinflationcapfloor.{hpp,cpp} +
   ql/cashflows/yoyinflationcoupon.cpp (``yoyInflationLeg``) (v1.43).

The leg is a plain YoY swaplet leg (one ``YoYInflationCoupon`` per annual
period) — the cap/floor option payoff is applied by the pricing engine, not by
the coupons. That is the ``noOption`` branch of
``yoyInflationLeg::operator Leg()``.

Two C++ details worth spelling out, because both are easy to get wrong:

* ``withFixingDays`` moves the **spot date only**. C++ builds the leg with
  ``.withPaymentAdjustment().withPaymentDayCounter().withNotionals()`` and
  nothing else (makeyoyinflationcapfloor.cpp:63-67), so the coupons keep
  ``yoyInflationLeg``'s own default of 0 fixing days no matter what
  ``fixingDays_`` is. The C++ probe confirms it: with ``withFixingDays(2)``
  the schedule shifts two business days but every coupon's fixing date stays
  ``accrual_end - observation_lag``.
* ``withFirstCapletExcluded()`` is **declared but never defined** in v1.43
  (makeyoyinflationcapfloor.hpp:49 has no matching definition in the .cpp), so
  a C++ caller cannot link a program that uses it. The member it would set,
  and the leg-erasing code that reads it, both exist. PQuantLib supplies the
  obvious one-line body; see :meth:`MakeYoYInflationCapFloor.with_first_caplet_excluded`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pquantlib import qassert
from pquantlib.cashflows.yoy_inflation_coupon import YoYInflationCoupon
from pquantlib.cashflows.yoy_inflation_coupon_pricer import YoYInflationCouponPricer
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.instruments.make_cap_floor import atm_rate_of_leg
from pquantlib.instruments.yoy_inflation_capfloor import (
    YoYInflationCapFloor,
    YoYInflationCapFloorType,
    YoYInflationCouponLike,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pquantlib.cashflows.cash_flow import CashFlow
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.inflation.cpi import InterpolationType
    from pquantlib.indexes.inflation.inflation_index import YoYInflationIndex
    from pquantlib.pricingengines.pricing_engine import PricingEngine
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.date import Date


def yoy_inflation_leg(
    schedule: Schedule,
    calendar: Calendar,
    index: YoYInflationIndex,
    observation_lag: Period,
    interpolation: InterpolationType,
    *,
    notional: float,
    payment_day_counter: DayCounter,
    payment_adjustment: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing,
    fixing_days: int = 0,
    gearing: float = 1.0,
    spread: float = 0.0,
) -> list[YoYInflationCouponLike]:
    """Build a YoY swaplet leg over ``schedule``.

    # C++ parity: ``yoyInflationLeg::operator Leg()`` (yoyinflationcoupon.cpp,
    # noOption swaplet branch). The cap/floor, gearing-zero-becomes-fixed-rate
    # and per-period vector overloads of the C++ builder are not ported here —
    # ``MakeYoYInflationCapFloor`` never reaches them, and the full builder
    # belongs in ``cashflows/yoy_inflation_coupon.py`` when something needs it.
    """
    n = schedule.size() - 1
    qassert.require(n > 0, "schedule too short for a YoY leg")
    leg: list[YoYInflationCouponLike] = []
    for i in range(n):
        start = schedule.date(i)
        end = schedule.date(i + 1)
        payment_date = calendar.adjust(end, payment_adjustment)
        coupon = YoYInflationCoupon(
            payment_date=payment_date,
            nominal=notional,
            accrual_start_date=start,
            accrual_end_date=end,
            fixing_days=fixing_days,
            index=index,
            observation_lag=observation_lag,
            interpolation=interpolation,
            day_counter=payment_day_counter,
            gearing=gearing,
            spread=spread,
            ref_period_start=start,
            ref_period_end=end,
        )
        # YoYInflationCoupon structurally satisfies YoYInflationCouponLike;
        # the Protocol types register_with(observer: object) (broader) while
        # the concrete narrows it to Observer — an accepted contravariance
        # mismatch under pyright, so we cast.
        leg.append(cast(YoYInflationCouponLike, coupon))
        # C++ parity: with no caps or floors the builder attaches a plain
        # YoYInflationCouponPricer to the whole leg (yoyinflationcoupon.cpp,
        # tail of ``operator Leg()``). Without it ``rate()`` — and therefore
        # ``amount()`` and any atmRate over the leg — has no pricer to call.
        coupon.set_pricer(YoYInflationCouponPricer())
    return leg


class MakeYoYInflationCapFloor:
    """Chained builder for :class:`YoYInflationCapFloor`.

    # C++ parity: ``MakeYoYInflationCapFloor``
    # (makeyoyinflationcapfloor.hpp:36-82). Every C++ ``withXxx`` setter is
    # present as ``with_xxx`` and returns ``self``; C++'s
    # ``operator ext::shared_ptr<YoYInflationCapFloor>()`` is :meth:`build`.
    """

    def __init__(
        self,
        cap_floor_type: YoYInflationCapFloorType,
        index: YoYInflationIndex,
        length: int,
        cal: Calendar,
        observation_lag: Period,
        interpolation: InterpolationType,
    ) -> None:
        # # C++ parity: MakeYoYInflationCapFloor::MakeYoYInflationCapFloor
        # # (makeyoyinflationcapfloor.cpp:29-39).
        self._cap_floor_type: YoYInflationCapFloorType = cap_floor_type
        self._length: int = length
        self._calendar: Calendar = cal
        self._index: YoYInflationIndex = index
        self._observation_lag: Period = observation_lag
        self._interpolation: InterpolationType = interpolation
        self._strike: float | None = None
        self._first_caplet_excluded: bool = False
        self._as_optionlet: bool = False
        self._effective_date: Date | None = None
        self._forward_start: Period | None = None
        self._day_counter: DayCounter = Thirty360(Thirty360Convention.BondBasis)
        self._roll: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing
        self._fixing_days: int = 0
        self._nominal: float = 1_000_000.0
        self._nominal_term_structure: YieldTermStructureProtocol | None = None
        self._engine: PricingEngine | None = None

    # --- chained setters ---------------------------------------------------

    def with_nominal(self, n: float) -> MakeYoYInflationCapFloor:
        """# C++ parity: ``withNominal`` (makeyoyinflationcapfloor.cpp:91-94)."""
        self._nominal = n
        return self

    def with_effective_date(self, effective_date: Date) -> MakeYoYInflationCapFloor:
        """# C++ parity: ``withEffectiveDate`` (makeyoyinflationcapfloor.cpp:96-100)."""
        self._effective_date = effective_date
        return self

    def with_first_caplet_excluded(self) -> MakeYoYInflationCapFloor:
        """Drop the first coupon from the leg.

        # C++ parity: ``withFirstCapletExcluded`` is *declared* at
        # makeyoyinflationcapfloor.hpp:49 but has **no definition** anywhere in
        # v1.43, so no C++ program can call it — while ``firstCapletExcluded_``
        # and the ``leg.erase(leg.begin())`` that consumes it both exist
        # (makeyoyinflationcapfloor.cpp:69-70). This supplies the only body
        # consistent with the name and with the sibling ``MakeCapFloor``, whose
        # ``withEffectiveDate(d, firstCapletExcluded)`` sets the same flag. The
        # resulting leg is cross-validated as the default leg minus its first
        # coupon.
        """
        self._first_caplet_excluded = True
        return self

    def with_payment_day_counter(self, dc: DayCounter) -> MakeYoYInflationCapFloor:
        """# C++ parity: ``withPaymentDayCounter`` (makeyoyinflationcapfloor.cpp:109-113)."""
        self._day_counter = dc
        return self

    def with_payment_adjustment(self, bdc: BusinessDayConvention) -> MakeYoYInflationCapFloor:
        """# C++ parity: ``withPaymentAdjustment`` (makeyoyinflationcapfloor.cpp:102-107)."""
        self._roll = bdc
        return self

    def with_fixing_days(self, fixing_days: int) -> MakeYoYInflationCapFloor:
        """Set the spot-date offset (business days from the evaluation date).

        # C++ parity: ``withFixingDays`` (makeyoyinflationcapfloor.cpp:115-119).
        # This affects the **schedule start only** — C++ never forwards it to
        # ``yoyInflationLeg``, so the coupons keep 0 fixing days.
        """
        self._fixing_days = fixing_days
        return self

    def with_pricing_engine(self, engine: PricingEngine) -> MakeYoYInflationCapFloor:
        """# C++ parity: ``withPricingEngine`` (makeyoyinflationcapfloor.cpp:126-130)."""
        self._engine = engine
        return self

    def as_optionlet(self, b: bool = True) -> MakeYoYInflationCapFloor:
        """Keep only the last coupon.

        # C++ parity: ``asOptionlet`` (makeyoyinflationcapfloor.cpp:121-124).
        """
        self._as_optionlet = b
        return self

    def with_strike(self, strike: float) -> MakeYoYInflationCapFloor:
        """# C++ parity: ``withStrike`` (makeyoyinflationcapfloor.cpp:132-137)."""
        qassert.require(self._nominal_term_structure is None, "ATM strike already given")
        self._strike = strike
        return self

    def with_atm_strike(self, nominal_term_structure: YieldTermStructureProtocol) -> MakeYoYInflationCapFloor:
        """Strike the cap/floor at the leg's par rate on ``nominal_term_structure``.

        # C++ parity: ``withAtmStrike`` (makeyoyinflationcapfloor.cpp:139-146).
        """
        qassert.require(self._strike is None, "explicit strike already given")
        self._nominal_term_structure = nominal_term_structure
        return self

    def with_forward_start(self, forward_start: Period) -> MakeYoYInflationCapFloor:
        """# C++ parity: ``withForwardStart`` (makeyoyinflationcapfloor.cpp:148-152)."""
        self._forward_start = forward_start
        return self

    # --- construction ------------------------------------------------------

    def build(self) -> YoYInflationCapFloor:
        """Build the YoY cap/floor.

        # C++ parity:
        # ``MakeYoYInflationCapFloor::operator ext::shared_ptr<YoYInflationCapFloor>()``
        # (makeyoyinflationcapfloor.cpp:44-89).
        """
        if self._effective_date is not None:
            start_date = self._effective_date
        else:
            reference_date = ObservableSettings().evaluation_date_or_today()
            spot = self._calendar.advance(reference_date, self._fixing_days, TimeUnit.Days)
            start_date = spot + self._forward_start if self._forward_start is not None else spot

        end_date = self._calendar.advance(
            start_date,
            self._length,
            TimeUnit.Years,
            BusinessDayConvention.Unadjusted,
        )
        schedule = Schedule.from_rule(
            start_date,
            end_date,
            Period.from_frequency(Frequency.Annual),
            self._calendar,
            # Reference periods and accrual periods are both unadjusted; only
            # the payment date is rolled, by ``roll_``.
            BusinessDayConvention.Unadjusted,
            BusinessDayConvention.Unadjusted,
            DateGeneration.Forward,
            False,
        )
        leg = yoy_inflation_leg(
            schedule,
            self._calendar,
            self._index,
            self._observation_lag,
            self._interpolation,
            notional=self._nominal,
            payment_day_counter=self._day_counter,
            payment_adjustment=self._roll,
            # C++ does NOT forward fixingDays_ here — see the class docstring.
            fixing_days=0,
        )

        if self._first_caplet_excluded:
            leg = leg[1:]
        # Only leaves the last coupon.
        if self._as_optionlet and len(leg) > 1:
            leg = leg[-1:]

        strike = self._strike
        if strike is None:
            # ATM on the forecasting curve.
            qassert.require(
                self._nominal_term_structure is not None,
                "no strike given: call with_strike or with_atm_strike",
            )
            assert self._nominal_term_structure is not None
            ts = self._nominal_term_structure
            strike = atm_rate_of_leg(
                cast("Sequence[CashFlow]", leg),
                ts,
                False,
                ts.reference_date(),
            )

        cap_floor = YoYInflationCapFloor.from_strikes(self._cap_floor_type, leg, [strike])
        if self._engine is not None:
            cap_floor.set_pricing_engine(self._engine)
        return cap_floor


def make_yoy_inflation_cap_floor(
    cap_floor_type: YoYInflationCapFloorType,
    index: YoYInflationIndex,
    length: int,
    calendar: Calendar,
    observation_lag: Period,
    interpolation: InterpolationType,
    *,
    strike: float | None = None,
    nominal: float = 1_000_000.0,
    effective_date: Date | None = None,
    fixing_days: int = 0,
    forward_start: Period | None = None,
    payment_day_counter: DayCounter | None = None,
    payment_adjustment: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing,
    first_caplet_excluded: bool = False,
    as_optionlet: bool = False,
    atm_nominal_term_structure: YieldTermStructureProtocol | None = None,
    pricing_engine: PricingEngine | None = None,
) -> YoYInflationCapFloor:
    """Keyword-argument façade over :class:`MakeYoYInflationCapFloor`.

    The builder holds the only implementation; this is a convenience shim for
    the call sites that predate it.
    """
    builder = MakeYoYInflationCapFloor(
        cap_floor_type, index, length, calendar, observation_lag, interpolation
    ).with_nominal(nominal)
    if strike is not None:
        builder = builder.with_strike(strike)
    if atm_nominal_term_structure is not None:
        builder = builder.with_atm_strike(atm_nominal_term_structure)
    if effective_date is not None:
        builder = builder.with_effective_date(effective_date)
    if fixing_days != 0:
        builder = builder.with_fixing_days(fixing_days)
    if forward_start is not None:
        builder = builder.with_forward_start(forward_start)
    if payment_day_counter is not None:
        builder = builder.with_payment_day_counter(payment_day_counter)
    builder = builder.with_payment_adjustment(payment_adjustment)
    if first_caplet_excluded:
        builder = builder.with_first_caplet_excluded()
    if as_optionlet:
        builder = builder.as_optionlet(True)
    if pricing_engine is not None:
        builder = builder.with_pricing_engine(pricing_engine)
    return builder.build()


__all__ = [
    "MakeYoYInflationCapFloor",
    "make_yoy_inflation_cap_floor",
    "yoy_inflation_leg",
]
