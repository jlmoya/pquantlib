"""cash_flow_vectors — the shared leg-building machinery.

# C++ parity: ql/cashflows/cashflowvectors.{hpp,cpp} (v1.43).

C++ builds every floating leg through two function templates,
``FloatingLeg<Index, Coupon, CappedFlooredCoupon>`` and
``FloatingDigitalLeg<Index, Coupon, DigitalCoupon>``, parameterised on the
coupon types. This is where the gearing / spread / cap / floor /
in-arrears / zero-payment / ex-coupon / payment-lag logic actually lives —
``IborLeg``, ``CmsLeg``, ``DigitalIborLeg`` and ``DigitalCmsLeg`` are thin
wrappers that only collect the settings and forward them here.

Python has no templates, so the coupon types are injected as factory
callables taking a :class:`FloatingCouponSpec` (the per-period data the C++
template computes before dispatching on ``gearing == 0`` / ``noOption``).
Everything else — the loop, the stub reference periods, the payment-date
roll and the fixed-coupon degeneracy — is shared, exactly as in C++.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.date import Date
    from pquantlib.time.period import Period
    from pquantlib.time.schedule import Schedule


def get[T](values: Sequence[T], i: int, default: T) -> T:
    """``detail::get`` — element ``i``, else the last element, else ``default``.

    # C++ parity: ql/utilities/vectors.hpp:36-45.
    """
    if len(values) == 0:
        return default
    if i < len(values):
        return values[i]
    return values[-1]


def as_float_list(value: float | Sequence[float] | None) -> list[float]:
    """Collapse the C++ scalar / vector ``withXxx`` overload pair.

    Every ``withGearings(Real)`` / ``withGearings(const vector<Real>&)`` pair
    in the leg builders reduces to one Python setter taking either shape;
    the scalar form is stored as a one-element vector, exactly as C++ does.
    """
    if value is None:
        return []
    if isinstance(value, int | float):
        return [float(value)]
    return [float(x) for x in value]


def as_int_list(value: int | Sequence[int] | None) -> list[int]:
    """:func:`as_float_list` for the ``withFixingDays`` overload pair."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return list(value)


def get_or_none(values: Sequence[float], i: int) -> float | None:
    """``detail::get(v, i, Null<Rate>())`` — ``None`` for an empty vector.

    C++ spells "absent" as ``Null<Rate>()``; Python spells it ``None``, so
    the optional-valued vectors (caps, floors, digital strikes/payoffs) need
    their own accessor rather than :func:`get`'s typed default.
    """
    if len(values) == 0:
        return None
    if i < len(values):
        return values[i]
    return values[-1]


def effective_fixed_rate(
    spreads: Sequence[float],
    caps: Sequence[float],
    floors: Sequence[float],
    i: int,
) -> float:
    """Spread clamped by the period's floor and cap.

    # C++ parity: ``detail::effectiveFixedRate`` (cashflowvectors.cpp:34-46).

    Used when a period's gearing is zero: the coupon degenerates to a fixed
    rate whose value is the spread, floored then capped.
    """
    result = get(spreads, i, 0.0)
    floor = get_or_none(floors, i)
    if floor is not None:
        result = max(floor, result)
    cap = get_or_none(caps, i)
    if cap is not None:
        result = min(cap, result)
    return result


def no_option(caps: Sequence[float], floors: Sequence[float], i: int) -> bool:
    """True when period ``i`` carries neither a cap nor a floor.

    # C++ parity: ``detail::noOption`` (cashflowvectors.cpp:48-53).
    """
    return get_or_none(caps, i) is None and get_or_none(floors, i) is None


@dataclass(frozen=True)
class FloatingCouponSpec:
    """Everything ``FloatingLeg`` computes for one period before dispatch.

    ``cap`` / ``floor`` are ``None`` when the period carries no optionality
    (C++ ``Null<Rate>()``); a factory receiving both as ``None`` must build
    the plain floating coupon type, otherwise the capped/floored one.
    """

    payment_date: Date
    nominal: float
    accrual_start_date: Date
    accrual_end_date: Date
    fixing_days: int
    gearing: float
    spread: float
    ref_period_start: Date
    ref_period_end: Date
    ex_coupon_date: Date | None
    cap: float | None
    floor: float | None


FloatingCouponFactory = Callable[[FloatingCouponSpec], CashFlow]
"""Builds the plain or capped/floored coupon for one period."""

UnderlyingCouponFactory = Callable[[FloatingCouponSpec], "FloatingRateCoupon"]
"""Builds the *underlying* floating coupon of a digital coupon."""

DigitalCouponFactory = Callable[
    ["FloatingRateCoupon", "float | None", "float | None", "float | None", "float | None"],
    CashFlow,
]
"""``(underlying, call_strike, call_payoff, put_strike, put_payoff) -> coupon``."""


def _stub_reference_dates(
    schedule: Schedule,
    n: int,
    i: int,
    start: Date,
    end: Date,
    convention: BusinessDayConvention,
) -> tuple[Date, Date]:
    """Widen the first/last reference period when the stub is irregular.

    # C++ parity: cashflowvectors.hpp:117-124 (the two ``schedule.isRegular``
    # blocks inside ``FloatingLeg``).
    """
    calendar = schedule.calendar
    ref_start, ref_end = start, end
    has_stub_info = schedule.has_is_regular() and schedule.has_tenor()
    if i == 0 and has_stub_info and not schedule.is_regular_at(i + 1):
        ref_start = calendar.adjust(end - schedule.tenor, convention)
    if i == n - 1 and has_stub_info and not schedule.is_regular_at(i + 1):
        ref_end = calendar.adjust(start + schedule.tenor, convention)
    return ref_start, ref_end


def floating_leg(
    schedule: Schedule,
    nominals: Sequence[float],
    default_fixing_days: int,
    payment_day_counter: DayCounter | None,
    payment_adjustment: BusinessDayConvention,
    fixing_days: Sequence[int],
    gearings: Sequence[float],
    spreads: Sequence[float],
    caps: Sequence[float],
    floors: Sequence[float],
    is_in_arrears: bool,
    is_zero: bool,
    make_coupon: FloatingCouponFactory,
    *,
    payment_lag: int = 0,
    payment_calendar: Calendar | None = None,
    ex_coupon_period: Period | None = None,
    ex_coupon_calendar: Calendar | None = None,
    ex_coupon_adjustment: BusinessDayConvention = BusinessDayConvention.Unadjusted,
    ex_coupon_end_of_month: bool = False,
) -> list[CashFlow]:
    """Build a leg of floating coupons.

    # C++ parity: ``FloatingLeg`` (cashflowvectors.hpp:59-215).

    ``default_fixing_days`` stands in for the C++ ``index->fixingDays()``
    fallback; ``payment_day_counter`` is threaded to the factory by the
    caller's closure rather than passed per-period.
    """
    n = len(schedule) - 1
    qassert.require(len(nominals) > 0, "no notional given")
    qassert.require(
        len(nominals) <= n, f"too many nominals ({len(nominals)}), only {n} required"
    )
    qassert.require(
        len(gearings) <= n, f"too many gearings ({len(gearings)}), only {n} required"
    )
    qassert.require(len(spreads) <= n, f"too many spreads ({len(spreads)}), only {n} required")
    qassert.require(len(caps) <= n, f"too many caps ({len(caps)}), only {n} required")
    qassert.require(len(floors) <= n, f"too many floors ({len(floors)}), only {n} required")
    qassert.require(
        not is_zero or not is_in_arrears, "in-arrears and zero features are not compatible"
    )

    calendar = schedule.calendar
    pay_cal = payment_calendar if payment_calendar is not None else calendar
    ex_cal = ex_coupon_calendar if ex_coupon_calendar is not None else calendar

    leg: list[CashFlow] = []
    last_payment_date = pay_cal.advance(
        schedule.date(n), payment_lag, TimeUnit.Days, payment_adjustment
    )

    for i in range(n):
        start = schedule.date(i)
        end = schedule.date(i + 1)
        payment_date = (
            last_payment_date
            if is_zero
            else pay_cal.advance(end, payment_lag, TimeUnit.Days, payment_adjustment)
        )
        ref_start, ref_end = _stub_reference_dates(
            schedule, n, i, start, end, schedule.business_day_convention
        )
        ex_coupon_date: Date | None = None
        if ex_coupon_period is not None:
            ex_coupon_date = ex_cal.advance(
                payment_date,
                -ex_coupon_period.length,
                ex_coupon_period.units,
                ex_coupon_adjustment,
                ex_coupon_end_of_month,
            )

        gearing = get(gearings, i, 1.0)
        nominal = get(nominals, i, 1.0)
        if gearing == 0.0:
            # C++ parity: cashflowvectors.hpp:133-141 — a zero gearing
            # degenerates to a fixed coupon paying the clamped spread.
            qassert.require(
                payment_day_counter is not None,
                "a payment day counter is required for zero-gearing periods",
            )
            assert payment_day_counter is not None
            leg.append(
                FixedRateCoupon.from_rate(
                    payment_date,
                    nominal,
                    effective_fixed_rate(spreads, caps, floors, i),
                    payment_day_counter,
                    start,
                    end,
                    ref_start,
                    ref_end,
                    ex_coupon_date,
                )
            )
            continue

        plain = no_option(caps, floors, i)
        leg.append(
            make_coupon(
                FloatingCouponSpec(
                    payment_date=payment_date,
                    nominal=nominal,
                    accrual_start_date=start,
                    accrual_end_date=end,
                    fixing_days=get(fixing_days, i, default_fixing_days),
                    gearing=gearing,
                    spread=get(spreads, i, 0.0),
                    ref_period_start=ref_start,
                    ref_period_end=ref_end,
                    ex_coupon_date=ex_coupon_date,
                    cap=None if plain else get_or_none(caps, i),
                    floor=None if plain else get_or_none(floors, i),
                )
            )
        )
    return leg


def floating_digital_leg(
    schedule: Schedule,
    nominals: Sequence[float],
    default_fixing_days: int,
    payment_day_counter: DayCounter | None,
    payment_adjustment: BusinessDayConvention,
    fixing_days: Sequence[int],
    gearings: Sequence[float],
    spreads: Sequence[float],
    call_strikes: Sequence[float],
    call_payoffs: Sequence[float],
    put_strikes: Sequence[float],
    put_payoffs: Sequence[float],
    make_underlying: UnderlyingCouponFactory,
    make_digital: DigitalCouponFactory,
) -> list[CashFlow]:
    """Build a leg of digital (call/put) floating coupons.

    # C++ parity: ``FloatingDigitalLeg`` (cashflowvectors.hpp:218-317).

    Note the two deliberate C++ asymmetries vs :func:`floating_leg`, both
    reproduced here: the payment date is a plain ``calendar.adjust(end)``
    (no payment lag, no separate payment calendar, no ex-coupon date), and
    the zero-gearing fixed coupon pays ``get(spreads, i, 1.0)`` — default
    **1.0**, not the clamped spread (cashflowvectors.hpp:293-299).
    """
    n = len(schedule) - 1
    qassert.require(len(nominals) > 0, "no notional given")
    qassert.require(
        len(nominals) <= n, f"too many nominals ({len(nominals)}), only {n} required"
    )
    qassert.require(
        len(gearings) <= n, f"too many gearings ({len(gearings)}), only {n} required"
    )
    qassert.require(len(spreads) <= n, f"too many spreads ({len(spreads)}), only {n} required")
    qassert.require(
        len(call_strikes) <= n,
        f"too many call rates ({len(call_strikes)}), only {n} required",
    )
    qassert.require(
        len(put_strikes) <= n, f"too many put rates ({len(put_strikes)}), only {n} required"
    )

    calendar = schedule.calendar
    leg: list[CashFlow] = []

    for i in range(n):
        start = schedule.date(i)
        end = schedule.date(i + 1)
        payment_date = calendar.adjust(end, payment_adjustment)
        ref_start, ref_end = _stub_reference_dates(
            schedule, n, i, start, end, schedule.business_day_convention
        )
        gearing = get(gearings, i, 1.0)
        nominal = get(nominals, i, 1.0)

        if gearing == 0.0:
            qassert.require(
                payment_day_counter is not None,
                "a payment day counter is required for zero-gearing periods",
            )
            assert payment_day_counter is not None
            leg.append(
                FixedRateCoupon.from_rate(
                    payment_date,
                    nominal,
                    get(spreads, i, 1.0),
                    payment_day_counter,
                    start,
                    end,
                    ref_start,
                    ref_end,
                )
            )
            continue

        underlying = make_underlying(
            FloatingCouponSpec(
                payment_date=payment_date,
                nominal=nominal,
                accrual_start_date=start,
                accrual_end_date=end,
                fixing_days=get(fixing_days, i, default_fixing_days),
                gearing=gearing,
                spread=get(spreads, i, 0.0),
                ref_period_start=ref_start,
                ref_period_end=ref_end,
                ex_coupon_date=None,
                cap=None,
                floor=None,
            )
        )
        leg.append(
            make_digital(
                underlying,
                get_or_none(call_strikes, i),
                get_or_none(call_payoffs, i),
                get_or_none(put_strikes, i),
                get_or_none(put_payoffs, i),
            )
        )
    return leg


__all__ = [
    "DigitalCouponFactory",
    "FloatingCouponFactory",
    "FloatingCouponSpec",
    "UnderlyingCouponFactory",
    "as_float_list",
    "as_int_list",
    "effective_fixed_rate",
    "floating_digital_leg",
    "floating_leg",
    "get",
    "get_or_none",
    "no_option",
]
