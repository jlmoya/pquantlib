"""CmsCoupon — a coupon paying a CMS (constant-maturity-swap) rate.

# C++ parity: ql/cashflows/cmscoupon.hpp + .cpp (v1.42.1, 099987f0).

A :class:`~pquantlib.cashflows.floating_rate_coupon.FloatingRateCoupon` whose
underlying index is a :class:`~pquantlib.indexes.swap_index.SwapIndex`. The
coupon's rate is the (convexity-adjusted) par swap rate, set by a
:class:`~pquantlib.cashflows.cms_coupon_pricer.CmsCouponPricer` (Hagan /
conundrum replication) attached via ``set_pricer``.

# C++ parity divergence (Visitability): the C++ ``accept`` / ``AcyclicVisitor``
# dispatch is not ported (PQuantLib does not port the coupon visitor pattern —
# consistent with FloatingRateCoupon, which has no ``accept``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib.cashflows import cash_flow_vectors as cfv
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.schedule import Schedule

if TYPE_CHECKING:
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.period import Period


class CmsCoupon(FloatingRateCoupon):
    """A coupon paying a (gearing/spread-adjusted) CMS rate.

    .. warning:: As in C++, no date adjustment is performed: the start and end
       dates passed should already be rolled to business days.
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        start_date: Date,
        end_date: Date,
        fixing_days: int,
        swap_index: SwapIndex,
        gearing: float = 1.0,
        spread: float = 0.0,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
        day_counter: DayCounter | None = None,
        is_in_arrears: bool = False,
        ex_coupon_date: Date | None = None,
        fixing_convention: BusinessDayConvention = BusinessDayConvention.Preceding,
    ) -> None:
        super().__init__(
            payment_date,
            nominal,
            start_date,
            end_date,
            fixing_days,
            # C++ parity: FloatingRateCoupon takes an InterestRateIndex;
            # SwapIndex *is* one. PQuantLib's FloatingRateCoupon narrows the
            # static type to the Ibor/Overnight protocols, so we suppress the
            # mismatch here — the coupon only touches index methods
            # (fixing/day_counter/fixing_calendar) that SwapIndex provides.
            swap_index,  # type: ignore[arg-type]
            gearing,
            spread,
            ref_period_start,
            ref_period_end,
            day_counter,
            is_in_arrears,
            ex_coupon_date,
            fixing_convention,
        )
        self._swap_index: SwapIndex = swap_index

    def swap_index(self) -> SwapIndex:
        return self._swap_index


class CmsLeg:
    """Chained builder for a sequence of capped/floored CMS coupons.

    # C++ parity: ``CmsLeg`` (cmscoupon.hpp:69-111, .cpp:59-176).

    Every C++ ``withXxx`` setter is present as ``with_xxx`` and returns
    ``self``; C++'s ``operator Leg()`` is :meth:`build`.

    Note that ``CmsLeg`` has **no** payment-lag or payment-calendar setter:
    C++ forwards a hard-coded ``paymentLag = 0`` and an empty payment
    calendar to ``FloatingLeg`` (cmscoupon.cpp:170-176).
    """

    def __init__(self, schedule: Schedule, swap_index: SwapIndex) -> None:
        # C++ ``QL_REQUIRE(swapIndex_, "no index provided")`` (.cpp:61) is a
        # null-pointer guard; the Python signature is non-optional, so the
        # check is enforced statically instead.
        self._schedule: Schedule = schedule
        self._swap_index: SwapIndex = swap_index
        self._notionals: list[float] = []
        self._payment_day_counter: DayCounter | None = None
        self._payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following
        self._fixing_days: list[int] = []
        self._gearings: list[float] = []
        self._spreads: list[float] = []
        self._caps: list[float] = []
        self._floors: list[float] = []
        self._in_arrears: bool = False
        self._zero_payments: bool = False
        self._fixing_convention: BusinessDayConvention = BusinessDayConvention.Preceding
        self._ex_coupon_period: Period | None = None
        self._ex_coupon_calendar: Calendar | None = None
        self._ex_coupon_adjustment: BusinessDayConvention = BusinessDayConvention.Following
        self._ex_coupon_end_of_month: bool = False

    # --- chained setters -----------------------------------------------

    def with_notionals(self, notionals: float | Sequence[float]) -> CmsLeg:
        """# C++ parity: ``withNotionals`` (.cpp:64-72)."""
        self._notionals = cfv.as_float_list(notionals)
        return self

    def with_payment_day_counter(self, day_counter: DayCounter) -> CmsLeg:
        """# C++ parity: ``withPaymentDayCounter`` (.cpp:74-77)."""
        self._payment_day_counter = day_counter
        return self

    def with_payment_adjustment(self, convention: BusinessDayConvention) -> CmsLeg:
        """# C++ parity: ``withPaymentAdjustment`` (.cpp:79-82)."""
        self._payment_adjustment = convention
        return self

    def with_fixing_days(self, fixing_days: int | Sequence[int]) -> CmsLeg:
        """# C++ parity: ``withFixingDays`` (.cpp:84-92)."""
        self._fixing_days = cfv.as_int_list(fixing_days)
        return self

    def with_gearings(self, gearings: float | Sequence[float]) -> CmsLeg:
        """# C++ parity: ``withGearings`` (.cpp:94-102)."""
        self._gearings = cfv.as_float_list(gearings)
        return self

    def with_spreads(self, spreads: float | Sequence[float]) -> CmsLeg:
        """# C++ parity: ``withSpreads`` (.cpp:104-112)."""
        self._spreads = cfv.as_float_list(spreads)
        return self

    def with_caps(self, caps: float | Sequence[float]) -> CmsLeg:
        """# C++ parity: ``withCaps`` (.cpp:114-122)."""
        self._caps = cfv.as_float_list(caps)
        return self

    def with_floors(self, floors: float | Sequence[float]) -> CmsLeg:
        """# C++ parity: ``withFloors`` (.cpp:124-132)."""
        self._floors = cfv.as_float_list(floors)
        return self

    def in_arrears(self, flag: bool = True) -> CmsLeg:
        """# C++ parity: ``inArrears`` (.cpp:134-137)."""
        self._in_arrears = flag
        return self

    def with_zero_payments(self, flag: bool = True) -> CmsLeg:
        """# C++ parity: ``withZeroPayments`` (.cpp:139-142)."""
        self._zero_payments = flag
        return self

    def with_fixing_convention(self, convention: BusinessDayConvention) -> CmsLeg:
        """# C++ parity: ``withFixingConvention`` (.cpp:144-147)."""
        self._fixing_convention = convention
        return self

    def with_ex_coupon_period(
        self,
        period: Period,
        calendar: Calendar,
        convention: BusinessDayConvention,
        end_of_month: bool,
    ) -> CmsLeg:
        """# C++ parity: ``withExCouponPeriod`` (.cpp:149-160)."""
        self._ex_coupon_period = period
        self._ex_coupon_calendar = calendar
        self._ex_coupon_adjustment = convention
        self._ex_coupon_end_of_month = end_of_month
        return self

    # --- operator Leg() -------------------------------------------------

    def _make_coupon(self, spec: cfv.FloatingCouponSpec) -> CashFlow:
        # Local import: capped_floored_coupon imports CmsCoupon lazily, so a
        # module-level import here would still be safe, but this keeps the
        # cap/floor dependency confined to the branch that needs it.
        from pquantlib.cashflows.capped_floored_coupon import (  # noqa: PLC0415
            CappedFlooredCmsCoupon,
        )

        day_counter = (
            self._payment_day_counter
            if self._payment_day_counter is not None
            else self._swap_index.day_counter()
        )
        if spec.cap is None and spec.floor is None:
            return CmsCoupon(
                spec.payment_date,
                spec.nominal,
                spec.accrual_start_date,
                spec.accrual_end_date,
                spec.fixing_days,
                self._swap_index,
                spec.gearing,
                spec.spread,
                spec.ref_period_start,
                spec.ref_period_end,
                day_counter,
                self._in_arrears,
                spec.ex_coupon_date,
                self._fixing_convention,
            )
        return CappedFlooredCmsCoupon(
            spec.payment_date,
            spec.nominal,
            spec.accrual_start_date,
            spec.accrual_end_date,
            spec.fixing_days,
            self._swap_index,
            spec.gearing,
            spec.spread,
            spec.cap,
            spec.floor,
            spec.ref_period_start,
            spec.ref_period_end,
            day_counter,
            self._in_arrears,
            spec.ex_coupon_date,
            self._fixing_convention,
        )

    def build(self) -> list[CashFlow]:
        """Build the leg.

        # C++ parity: ``CmsLeg::operator Leg()`` (.cpp:162-176).
        """
        return cfv.floating_leg(
            self._schedule,
            self._notionals,
            self._swap_index.fixing_days(),
            self._payment_day_counter,
            self._payment_adjustment,
            self._fixing_days,
            self._gearings,
            self._spreads,
            self._caps,
            self._floors,
            self._in_arrears,
            self._zero_payments,
            self._make_coupon,
            payment_lag=0,
            payment_calendar=None,
            ex_coupon_period=self._ex_coupon_period,
            ex_coupon_calendar=self._ex_coupon_calendar,
            ex_coupon_adjustment=self._ex_coupon_adjustment,
            ex_coupon_end_of_month=self._ex_coupon_end_of_month,
        )


def cms_leg(
    schedule: Schedule,
    swap_index: SwapIndex,
    nominals: float | Sequence[float],
    *,
    payment_day_counter: DayCounter | None = None,
    payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following,
    fixing_days: int | None = None,
    gearings: float | Sequence[float] = 1.0,
    spreads: float | Sequence[float] = 0.0,
    caps: float | Sequence[float] | None = None,
    floors: float | Sequence[float] | None = None,
    in_arrears: bool = False,
    fixing_convention: BusinessDayConvention = BusinessDayConvention.Preceding,
) -> list[CashFlow]:
    """Build a leg of CmsCoupons from a schedule + swap index.

    Keyword-argument façade over :class:`CmsLeg`; the leg logic itself lives
    there, so there is exactly one implementation.

    ``fixing_days`` defaults to ``swap_index.fixing_days()``.
    ``payment_day_counter`` defaults to ``swap_index.day_counter()``.

    Periods carrying a cap or a floor become
    :class:`~pquantlib.cashflows.capped_floored_coupon.CappedFlooredCmsCoupon`,
    as in C++. Pricing such a coupon needs a CMS optionlet pricer (Hagan
    replication caplet/floorlet); constructing the leg does not.
    """
    builder = (
        CmsLeg(schedule, swap_index)
        .with_notionals(nominals)
        .with_payment_adjustment(payment_adjustment)
        .with_gearings(gearings)
        .with_spreads(spreads)
        .in_arrears(in_arrears)
        .with_fixing_convention(fixing_convention)
    )
    if payment_day_counter is not None:
        builder = builder.with_payment_day_counter(payment_day_counter)
    if fixing_days is not None:
        builder = builder.with_fixing_days(fixing_days)
    if caps is not None:
        builder = builder.with_caps(caps)
    if floors is not None:
        builder = builder.with_floors(floors)
    return builder.build()


__all__ = ["CmsCoupon", "CmsLeg", "cms_leg"]
