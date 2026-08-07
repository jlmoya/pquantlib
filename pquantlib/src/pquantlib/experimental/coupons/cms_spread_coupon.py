"""CappedFlooredCmsSpreadCoupon + CmsSpreadLeg.

# C++ parity: ql/experimental/coupons/cmsspreadcoupon.{hpp,cpp} (v1.43).

:class:`CappedFlooredCmsSpreadCoupon` is a
:class:`~pquantlib.cashflows.capped_floored_coupon.CappedFlooredCoupon` over a
:class:`~pquantlib.cashflows.cms_spread_coupon.CmsSpreadCoupon` — the C++ class
is a single delegating constructor (cmsspreadcoupon.hpp:73-94) and nothing else.

:class:`CmsSpreadLeg` is the chained builder; its ``operator Leg()`` forwards to
the ``FloatingLeg`` template in ql/cashflows/cashflowvectors.hpp:60-219, which
is where all the actual behaviour lives (per-period coupon-type dispatch,
payment-date roll, irregular-stub reference dates, and the gearing-zero
fixed-coupon fallback). That template is reproduced here rather than referenced,
because PQuantLib has no shared ``FloatingLeg`` helper.

# C++ parity divergences:
# - ``accept`` / ``AcyclicVisitor`` Visitability is not ported (consistent with
#   the rest of the cashflows port).
# - The C++ builder is consumed via an implicit ``operator Leg()``
#   (``Leg leg = CmsSpreadLeg(...).withNotionals(...);``). Python has no
#   implicit conversion, so the chain is finished with :meth:`CmsSpreadLeg.leg`
#   (also available as ``__call__``, matching
#   :class:`~pquantlib.time.schedule.MakeSchedule`).
# - C++ ``CmsSpreadLeg``'s constructor does ``QL_REQUIRE(swapSpreadIndex_,
#   "no index provided")`` because ``ext::shared_ptr`` can be null. The Python
#   equivalent of that guarantee is the parameter's type annotation, so no
#   runtime check is duplicated here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.capped_floored_coupon import CappedFlooredCoupon
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.cms_spread_coupon import CmsSpreadCoupon
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.swap_spread_index import SwapSpreadIndex
    from pquantlib.time.date import Date
    from pquantlib.time.schedule import Schedule


# ---------------------------------------------------------------------------
# ql/utilities/vectors.hpp detail::get, and the two helpers in
# ql/cashflows/cashflowvectors.cpp that the leg templates call.
#
# These are shared by both leg builders in this package (CmsSpreadLeg and
# DigitalCmsSpreadLeg), which is why they live at module scope here.
# ---------------------------------------------------------------------------


def get(values: Sequence[float | None], i: int, default: float | None) -> float | None:
    """# C++ parity: ql/utilities/vectors.hpp:31-43 ``detail::get``.

    Empty sequence → the default; ``i`` in range → ``values[i]``; past the end →
    ``values[-1]`` (the **last entry repeats**; it does not fall back to the
    default).
    """
    if len(values) == 0:
        return default
    if i < len(values):
        return values[i]
    return values[-1]


def get_int(values: Sequence[int], i: int, default: int) -> int:
    """Integral :func:`get` — used for the per-period fixing-days vector."""
    if len(values) == 0:
        return default
    if i < len(values):
        return values[i]
    return values[-1]


def effective_fixed_rate(
    spreads: Sequence[float],
    caps: Sequence[float | None],
    floors: Sequence[float | None],
    i: int,
) -> float:
    """# C++ parity: ql/cashflows/cashflowvectors.cpp:34-46.

    The spread, raised to the floor and then clamped to the cap — in that
    order, so a floor above the cap wins on the floor and then loses to the cap.
    """
    result = get(spreads, i, 0.0)
    assert result is not None
    floor = get(floors, i, None)
    if floor is not None:
        result = max(floor, result)
    cap = get(caps, i, None)
    if cap is not None:
        result = min(cap, result)
    return result


def no_option(caps: Sequence[float | None], floors: Sequence[float | None], i: int) -> bool:
    """# C++ parity: ql/cashflows/cashflowvectors.cpp:48-53."""
    return get(caps, i, None) is None and get(floors, i, None) is None


def as_float_list(value: float | Sequence[float]) -> list[float]:
    """C++ ``withX(Real)`` stores ``std::vector<Real>(1, x)``; ``withX(vector)`` copies."""
    if isinstance(value, int | float):
        return [float(value)]
    return [float(v) for v in value]


def as_opt_float_list(value: float | None | Sequence[float | None]) -> list[float | None]:
    """As :func:`as_float_list`, with ``None`` standing in for C++ ``Null<Rate>()``."""
    if value is None or isinstance(value, int | float):
        return [None if value is None else float(value)]
    return [None if v is None else float(v) for v in value]


def as_int_list(value: int | Sequence[int]) -> list[int]:
    """Integral :func:`as_float_list` (fixing days)."""
    if isinstance(value, int):
        return [value]
    return list(value)


class CappedFlooredCmsSpreadCoupon(CappedFlooredCoupon):
    """Capped and/or floored CMS-spread coupon.

    # C++ parity: ql/experimental/coupons/cmsspreadcoupon.hpp:71-103 — builds a
    # ``CmsSpreadCoupon`` underlying, then wraps it with cap/floor.

    .. note:: The sign-aware cap/floor swap lives in the
       :class:`~pquantlib.cashflows.capped_floored_coupon.CappedFlooredCoupon`
       base: with ``gearing < 0`` the constructor's ``cap`` becomes the stored
       floor and vice versa, so ``cap()`` / ``floor()`` do not echo the
       constructor arguments back.
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        start_date: Date,
        end_date: Date,
        fixing_days: int,
        index: SwapSpreadIndex,
        gearing: float = 1.0,
        spread: float = 0.0,
        cap: float | None = None,
        floor: float | None = None,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
        day_counter: DayCounter | None = None,
        is_in_arrears: bool = False,
        ex_coupon_date: Date | None = None,
        fixing_convention: BusinessDayConvention = BusinessDayConvention.Preceding,
    ) -> None:
        underlying = CmsSpreadCoupon(
            payment_date,
            nominal,
            start_date,
            end_date,
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
        super().__init__(underlying, cap, floor)

    def swap_spread_index(self) -> SwapSpreadIndex:
        """The underlying coupon's ``SwapSpreadIndex``."""
        underlying = self._underlying
        assert isinstance(underlying, CmsSpreadCoupon)
        return underlying.swap_spread_index()


class CmsSpreadLeg:
    """Chained builder for a sequence of (capped/floored) CMS-spread coupons.

    # C++ parity: ql/experimental/coupons/cmsspreadcoupon.hpp:105-137 +
    # .cpp:48-145, whose ``operator Leg()`` calls
    # ``FloatingLeg<SwapSpreadIndex, CmsSpreadCoupon,
    # CappedFlooredCmsSpreadCoupon>`` (cashflowvectors.hpp:60-219) with
    # ``paymentLag = 0``, an empty payment calendar, no ex-coupon period and
    # ``fixingConvention = Preceding``.

    Usage mirrors the C++ chain, finished with :meth:`leg`::

        leg = (CmsSpreadLeg(schedule, index)
               .with_notionals(1.0e6)
               .with_payment_day_counter(Actual360())
               .with_caps(0.008)
               .leg())
    """

    def __init__(self, schedule: Schedule, swap_spread_index: SwapSpreadIndex) -> None:
        self._schedule: Schedule = schedule
        self._swap_spread_index: SwapSpreadIndex = swap_spread_index
        self._notionals: list[float] = []
        self._payment_day_counter: DayCounter | None = None
        self._payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following
        self._fixing_days: list[int] = []
        self._gearings: list[float] = []
        self._spreads: list[float] = []
        self._caps: list[float | None] = []
        self._floors: list[float | None] = []
        self._in_arrears: bool = False
        self._zero_payments: bool = False

    # --- chained setters (C++ cmsspreadcoupon.cpp:53-137) ------------------

    def with_notionals(self, notionals: float | Sequence[float]) -> CmsSpreadLeg:
        self._notionals = as_float_list(notionals)
        return self

    def with_payment_day_counter(self, day_counter: DayCounter) -> CmsSpreadLeg:
        self._payment_day_counter = day_counter
        return self

    def with_payment_adjustment(self, convention: BusinessDayConvention) -> CmsSpreadLeg:
        self._payment_adjustment = convention
        return self

    def with_fixing_days(self, fixing_days: int | Sequence[int]) -> CmsSpreadLeg:
        self._fixing_days = as_int_list(fixing_days)
        return self

    def with_gearings(self, gearings: float | Sequence[float]) -> CmsSpreadLeg:
        self._gearings = as_float_list(gearings)
        return self

    def with_spreads(self, spreads: float | Sequence[float]) -> CmsSpreadLeg:
        self._spreads = as_float_list(spreads)
        return self

    def with_caps(self, caps: float | None | Sequence[float | None]) -> CmsSpreadLeg:
        self._caps = as_opt_float_list(caps)
        return self

    def with_floors(self, floors: float | None | Sequence[float | None]) -> CmsSpreadLeg:
        self._floors = as_opt_float_list(floors)
        return self

    def in_arrears(self, flag: bool = True) -> CmsSpreadLeg:
        self._in_arrears = flag
        return self

    def with_zero_payments(self, flag: bool = True) -> CmsSpreadLeg:
        self._zero_payments = flag
        return self

    # --- operator Leg() ---------------------------------------------------

    def __call__(self) -> list[CashFlow]:
        """Sugar for :meth:`leg` — the C++ ``operator Leg() const``."""
        return self.leg()

    def _check_sizes(self, n: int) -> None:
        """# C++ parity: the QL_REQUIRE block at cashflowvectors.hpp:79-98."""
        qassert.require(len(self._notionals) > 0, "no notional given")
        qassert.require(
            len(self._notionals) <= n,
            f"too many nominals ({len(self._notionals)}), only {n} required",
        )
        qassert.require(
            len(self._gearings) <= n,
            f"too many gearings ({len(self._gearings)}), only {n} required",
        )
        qassert.require(
            len(self._spreads) <= n,
            f"too many spreads ({len(self._spreads)}), only {n} required",
        )
        qassert.require(
            len(self._caps) <= n, f"too many caps ({len(self._caps)}), only {n} required"
        )
        qassert.require(
            len(self._floors) <= n,
            f"too many floors ({len(self._floors)}), only {n} required",
        )
        qassert.require(
            not self._zero_payments or not self._in_arrears,
            "in-arrears and zero features are not compatible",
        )

    def leg(self) -> list[CashFlow]:
        """Build the leg.

        # C++ parity: ``CmsSpreadLeg::operator Leg()`` (cmsspreadcoupon.cpp:139-145)
        # → ``FloatingLeg`` (cashflowvectors.hpp:60-219).
        """
        schedule = self._schedule
        n = len(schedule) - 1
        self._check_sizes(n)

        calendar = schedule.calendar
        # C++ passes an empty Calendar, which FloatingLeg replaces with the
        # schedule's own (cashflowvectors.hpp:104-106).
        payment_calendar = calendar
        index = self._swap_spread_index
        default_fixing_days = index.fixing_days()
        day_counter = self._payment_day_counter

        # paymentLag == 0, so Calendar::advance collapses to adjust().
        last_payment_date = payment_calendar.advance(
            schedule.date(n), 0, TimeUnit.Days, self._payment_adjustment, False
        )

        leg: list[CashFlow] = []
        for i in range(n):
            ref_start = start = schedule.date(i)
            ref_end = end = schedule.date(i + 1)
            payment_date = (
                last_payment_date
                if self._zero_payments
                else payment_calendar.advance(
                    end, 0, TimeUnit.Days, self._payment_adjustment, False
                )
            )
            # Both C++ branches test isRegular(i+1) — the first period's own
            # regularity for i == 0, the last period's for i == n-1.
            irregular = (
                schedule.has_is_regular()
                and schedule.has_tenor()
                and not schedule.is_regular_at(i + 1)
            )
            if i == 0 and irregular:
                ref_start = calendar.adjust(end - schedule.tenor, schedule.business_day_convention)
            if i == n - 1 and irregular:
                ref_end = calendar.adjust(start + schedule.tenor, schedule.business_day_convention)

            nominal = get(self._notionals, i, 1.0)
            gearing = get(self._gearings, i, 1.0)
            spread = get(self._spreads, i, 0.0)
            assert nominal is not None
            assert gearing is not None
            assert spread is not None
            fixing_days = get_int(self._fixing_days, i, default_fixing_days)

            if gearing == 0.0:
                # C++ parity note: cashflowvectors.hpp:132-140 hands
                # paymentDayCounter to FixedRateCoupon with no fallback, so a
                # gearing-zero period on a leg with no withPaymentDayCounter()
                # yields a coupon whose empty DayCounter makes accrualPeriod()
                # throw. PQuantLib has no empty DayCounter, so the index's is
                # substituted; the only configuration that differs is one C++
                # cannot evaluate at all.
                leg.append(
                    FixedRateCoupon.from_rate(
                        payment_date,
                        nominal,
                        effective_fixed_rate(self._spreads, self._caps, self._floors, i),
                        day_counter if day_counter is not None else index.day_counter(),
                        start,
                        end,
                        ref_start,
                        ref_end,
                        None,
                    )
                )
            elif no_option(self._caps, self._floors, i):
                leg.append(
                    CmsSpreadCoupon(
                        payment_date,
                        nominal,
                        start,
                        end,
                        fixing_days,
                        index,
                        gearing,
                        spread,
                        ref_start,
                        ref_end,
                        day_counter,
                        self._in_arrears,
                        None,
                        BusinessDayConvention.Preceding,
                    )
                )
            else:
                leg.append(
                    CappedFlooredCmsSpreadCoupon(
                        payment_date,
                        nominal,
                        start,
                        end,
                        fixing_days,
                        index,
                        gearing,
                        spread,
                        get(self._caps, i, None),
                        get(self._floors, i, None),
                        ref_start,
                        ref_end,
                        day_counter,
                        self._in_arrears,
                        None,
                        BusinessDayConvention.Preceding,
                    )
                )
        return leg


__all__ = [
    "CappedFlooredCmsSpreadCoupon",
    "CmsSpreadLeg",
    "as_float_list",
    "as_int_list",
    "as_opt_float_list",
    "effective_fixed_rate",
    "get",
    "get_int",
    "no_option",
]
