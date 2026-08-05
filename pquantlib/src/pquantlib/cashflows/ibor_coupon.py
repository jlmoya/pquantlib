"""IborCoupon + IborLeg — Libor-type coupon and its leg builder.

# C++ parity: ql/cashflows/iborcoupon.hpp + .cpp (v1.43).

Simplifications vs C++:
- ``IborCoupon::Settings`` global toggle (par vs indexed coupons) is NOT
  ported — coupons here behave as "indexed coupons" (forecast the
  fixing for the actual accrual period, not for the index's natural
  tenor period). This matches QL_USE_INDEXED_COUPON=OFF behaviour,
  which is the C++ default. The per-leg override is ported, though:
  ``IborLeg.with_indexed_coupons`` / ``with_at_par_coupons`` thread the
  flag into the default ``BlackIborCouponPricer``, exactly as C++ does.
- ``fixingValueDate`` / ``fixingMaturityDate`` / ``fixingEndDate`` /
  ``spanningTime`` cached-data accessors used by the C++ pricer are
  not exposed (no pricer in this port consults them).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib.cashflows import cash_flow_vectors as cfv
from pquantlib.cashflows.capped_floored_coupon import CappedFlooredIborCoupon
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.coupon_pricer import BlackIborCouponPricer, set_coupon_pricer
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.termstructures.protocols import IborIndexProtocol
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.period import Period
    from pquantlib.time.schedule import Schedule


class IborCoupon(FloatingRateCoupon):
    """Coupon paying a Libor-type index fixing.

    Inherits all behaviour from FloatingRateCoupon; the type narrows the
    index slot to ``IborIndexProtocol`` (strictly: the C++ ``IborIndex``).
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        accrual_start_date: Date,
        accrual_end_date: Date,
        fixing_days: int,
        index: IborIndexProtocol,
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
            accrual_start_date,
            accrual_end_date,
            fixing_days,
            index,
            gearing,
            spread,
            ref_period_start,
            ref_period_end,
            day_counter,
            is_in_arrears,
            ex_coupon_date,
            fixing_convention,
        )
        # Cache the fixing date (C++ computes once in ctor).
        self._fixing_date_cached: Date = super().fixing_date()

    def ibor_index(self) -> IborIndexProtocol:
        """Narrowed accessor returning the IborIndexProtocol-typed index.

        C++ parity: ql/cashflows/iborcoupon.hpp:59 ``iborIndex() const``.
        """
        # Already narrowed at construction via the typed parameter.
        return self._index  # type: ignore[return-value]

    def fixing_date(self) -> Date:
        """Return the cached fixing date (computed once at construction).

        C++ parity: ql/cashflows/iborcoupon.cpp:89-91.
        """
        return self._fixing_date_cached


class IborLeg:
    """Chained builder for a sequence of capped/floored Ibor coupons.

    # C++ parity: ``IborLeg`` (iborcoupon.hpp:134-183, .cpp:156-295).

    Every C++ ``withXxx`` setter is present as ``with_xxx`` and returns
    ``self``; C++'s ``operator Leg()`` is :meth:`build`.
    """

    def __init__(self, schedule: Schedule, index: IborIndexProtocol) -> None:
        self._schedule: Schedule = schedule
        self._index: IborIndexProtocol = index
        self._notionals: list[float] = []
        self._payment_day_counter: DayCounter | None = None
        self._payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following
        self._payment_lag: int = 0
        self._payment_calendar: Calendar | None = None
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
        self._ex_coupon_adjustment: BusinessDayConvention = BusinessDayConvention.Unadjusted
        self._ex_coupon_end_of_month: bool = False
        self._use_indexed_coupons: bool | None = None

    # --- chained setters -----------------------------------------------

    def with_notionals(self, notionals: float | Sequence[float]) -> IborLeg:
        """# C++ parity: ``withNotionals`` (.cpp:167-175)."""
        self._notionals = cfv.as_float_list(notionals)
        return self

    def with_payment_day_counter(self, day_counter: DayCounter) -> IborLeg:
        """# C++ parity: ``withPaymentDayCounter`` (.cpp:177-180)."""
        self._payment_day_counter = day_counter
        return self

    def with_payment_adjustment(self, convention: BusinessDayConvention) -> IborLeg:
        """# C++ parity: ``withPaymentAdjustment`` (.cpp:182-185)."""
        self._payment_adjustment = convention
        return self

    def with_payment_lag(self, lag: int) -> IborLeg:
        """# C++ parity: ``withPaymentLag`` (.cpp:187-190)."""
        self._payment_lag = lag
        return self

    def with_payment_calendar(self, calendar: Calendar) -> IborLeg:
        """# C++ parity: ``withPaymentCalendar`` (.cpp:192-195)."""
        self._payment_calendar = calendar
        return self

    def with_fixing_days(self, fixing_days: int | Sequence[int]) -> IborLeg:
        """# C++ parity: ``withFixingDays`` (.cpp:197-205)."""
        self._fixing_days = cfv.as_int_list(fixing_days)
        return self

    def with_gearings(self, gearings: float | Sequence[float]) -> IborLeg:
        """# C++ parity: ``withGearings`` (.cpp:207-215)."""
        self._gearings = cfv.as_float_list(gearings)
        return self

    def with_spreads(self, spreads: float | Sequence[float]) -> IborLeg:
        """# C++ parity: ``withSpreads`` (.cpp:217-225)."""
        self._spreads = cfv.as_float_list(spreads)
        return self

    def with_caps(self, caps: float | Sequence[float]) -> IborLeg:
        """# C++ parity: ``withCaps`` (.cpp:227-235)."""
        self._caps = cfv.as_float_list(caps)
        return self

    def with_floors(self, floors: float | Sequence[float]) -> IborLeg:
        """# C++ parity: ``withFloors`` (.cpp:237-245)."""
        self._floors = cfv.as_float_list(floors)
        return self

    def in_arrears(self, flag: bool = True) -> IborLeg:
        """# C++ parity: ``inArrears`` (.cpp:247-250)."""
        self._in_arrears = flag
        return self

    def with_zero_payments(self, flag: bool = True) -> IborLeg:
        """# C++ parity: ``withZeroPayments`` (.cpp:252-255).

        All coupons then pay on the leg's final payment date.
        """
        self._zero_payments = flag
        return self

    def with_ex_coupon_period(
        self,
        period: Period,
        calendar: Calendar,
        convention: BusinessDayConvention,
        end_of_month: bool = False,
    ) -> IborLeg:
        """# C++ parity: ``withExCouponPeriod`` (.cpp:257-266)."""
        self._ex_coupon_period = period
        self._ex_coupon_calendar = calendar
        self._ex_coupon_adjustment = convention
        self._ex_coupon_end_of_month = end_of_month
        return self

    def with_fixing_convention(self, convention: BusinessDayConvention) -> IborLeg:
        """# C++ parity: ``withFixingConvention`` (.cpp:268-271)."""
        self._fixing_convention = convention
        return self

    def with_indexed_coupons(self, flag: bool | None = True) -> IborLeg:
        """# C++ parity: ``withIndexedCoupons`` (.cpp:273-276).

        ``None`` restores "follow the library default" (C++ ``ext::nullopt``).
        The flag reaches the default ``BlackIborCouponPricer`` attached by
        :meth:`build`; it has no effect on a leg that carries caps, floors or
        in-arrears fixing, because C++ attaches no default pricer there.
        """
        self._use_indexed_coupons = flag
        return self

    def with_at_par_coupons(self, flag: bool = True) -> IborLeg:
        """# C++ parity: ``withAtParCoupons`` (.cpp:278-281) — the negation."""
        self._use_indexed_coupons = not flag
        return self

    # --- operator Leg() -------------------------------------------------

    def _make_coupon(self, spec: cfv.FloatingCouponSpec) -> CashFlow:
        day_counter = (
            self._payment_day_counter
            if self._payment_day_counter is not None
            else self._index.day_counter()
        )
        if spec.cap is None and spec.floor is None:
            return IborCoupon(
                spec.payment_date,
                spec.nominal,
                spec.accrual_start_date,
                spec.accrual_end_date,
                spec.fixing_days,
                self._index,
                spec.gearing,
                spec.spread,
                spec.ref_period_start,
                spec.ref_period_end,
                day_counter,
                self._in_arrears,
                spec.ex_coupon_date,
                self._fixing_convention,
            )
        return CappedFlooredIborCoupon(
            spec.payment_date,
            spec.nominal,
            spec.accrual_start_date,
            spec.accrual_end_date,
            spec.fixing_days,
            self._index,
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

        # C++ parity: ``IborLeg::operator Leg()`` (.cpp:283-295).

        As in C++, a default ``BlackIborCouponPricer`` is attached only when
        the leg has no caps, no floors and is not in arrears — the cases where
        the plain par-coupon forecast is the whole answer.
        """
        leg = cfv.floating_leg(
            self._schedule,
            self._notionals,
            self._index.fixing_days(),
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
            payment_lag=self._payment_lag,
            payment_calendar=self._payment_calendar,
            ex_coupon_period=self._ex_coupon_period,
            ex_coupon_calendar=self._ex_coupon_calendar,
            ex_coupon_adjustment=self._ex_coupon_adjustment,
            ex_coupon_end_of_month=self._ex_coupon_end_of_month,
        )
        if not self._caps and not self._floors and not self._in_arrears:
            set_coupon_pricer(
                leg,
                BlackIborCouponPricer(
                    None,
                    # C++ ``ext::nullopt`` means "use the library default",
                    # which in this port is at-par coupons (see module docstring).
                    use_indexed_coupons=bool(self._use_indexed_coupons),
                ),
            )
        return leg


__all__ = ["IborCoupon", "IborLeg"]
