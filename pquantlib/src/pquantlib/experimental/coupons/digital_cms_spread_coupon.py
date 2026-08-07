"""DigitalCmsSpreadCoupon + DigitalCmsSpreadLeg.

# C++ parity: ql/experimental/coupons/digitalcmsspreadcoupon.{hpp,cpp} (v1.43).

:class:`DigitalCmsSpreadCoupon` is a
:class:`~pquantlib.cashflows.digital_coupon.DigitalCoupon` over a
:class:`~pquantlib.cashflows.cms_spread_coupon.CmsSpreadCoupon`: a CMS-spread
coupon carrying a digital call and/or put on the spread, valued by call/put-
spread replication.

:class:`DigitalCmsSpreadLeg` is the chained builder; its ``operator Leg()``
forwards to the ``FloatingDigitalLeg`` template in
ql/cashflows/cashflowvectors.hpp:221-313, reproduced here.

# C++ parity divergences:
# - ``accept`` / ``AcyclicVisitor`` Visitability is not ported.
# - The C++ builder is consumed via an implicit ``operator Leg()``; Python
#   finishes the chain with :meth:`DigitalCmsSpreadLeg.leg` (or ``__call__``).
# - C++ ``DigitalCmsSpreadLeg`` performs no null-index check (unlike
#   ``CmsSpreadLeg``); the Python type annotation carries that guarantee.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.cms_spread_coupon import CmsSpreadCoupon
from pquantlib.cashflows.digital_coupon import DigitalCoupon
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.experimental.coupons.cms_spread_coupon import (
    as_float_list,
    as_int_list,
    as_opt_float_list,
    get,
    get_int,
)
from pquantlib.position import PositionType
from pquantlib.time.business_day_convention import BusinessDayConvention

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pquantlib.cashflows.replication import DigitalReplication
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.swap_spread_index import SwapSpreadIndex
    from pquantlib.time.schedule import Schedule


class DigitalCmsSpreadCoupon(DigitalCoupon):
    """CMS-spread coupon with an embedded digital call/put.

    # C++ parity: ql/experimental/coupons/digitalcmsspreadcoupon.hpp:34-53 +
    # .cpp:26-49 — a pure delegation to ``DigitalCoupon``.
    """

    def __init__(
        self,
        underlying: CmsSpreadCoupon,
        call_strike: float | None = None,
        call_position: PositionType = PositionType.Long,
        is_call_atm_included: bool = False,
        call_digital_payoff: float | None = None,
        put_strike: float | None = None,
        put_position: PositionType = PositionType.Long,
        is_put_atm_included: bool = False,
        put_digital_payoff: float | None = None,
        replication: DigitalReplication | None = None,
        naked_option: bool = False,
    ) -> None:
        super().__init__(
            underlying,
            call_strike,
            call_position,
            is_call_atm_included,
            call_digital_payoff,
            put_strike,
            put_position,
            is_put_atm_included,
            put_digital_payoff,
            replication,
            naked_option,
        )


class DigitalCmsSpreadLeg:
    """Chained builder for a sequence of digital CMS-spread coupons.

    # C++ parity: ql/experimental/coupons/digitalcmsspreadcoupon.hpp:57-105 +
    # .cpp:52-202, whose ``operator Leg()`` calls
    # ``FloatingDigitalLeg<SwapSpreadIndex, CmsSpreadCoupon,
    # DigitalCmsSpreadCoupon>`` (cashflowvectors.hpp:221-313).

    Usage mirrors the C++ chain, finished with :meth:`leg`::

        leg = (DigitalCmsSpreadLeg(schedule, index)
               .with_notionals(1.0e6)
               .with_call_strikes(0.005)
               .with_call_payoffs(0.03)
               .leg())
    """

    def __init__(self, schedule: Schedule, index: SwapSpreadIndex) -> None:
        self._schedule: Schedule = schedule
        self._index: SwapSpreadIndex = index
        self._notionals: list[float] = []
        self._payment_day_counter: DayCounter | None = None
        self._payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following
        self._fixing_days: list[int] = []
        self._gearings: list[float] = []
        self._spreads: list[float] = []
        self._in_arrears: bool = False
        self._call_strikes: list[float | None] = []
        self._call_payoffs: list[float | None] = []
        self._long_call_option: PositionType = PositionType.Long
        self._call_atm: bool = False
        self._put_strikes: list[float | None] = []
        self._put_payoffs: list[float | None] = []
        self._long_put_option: PositionType = PositionType.Long
        self._put_atm: bool = False
        self._replication: DigitalReplication | None = None
        # C++ parity note: ``DigitalCmsSpreadLeg::nakedOption_`` has NO default
        # member initialiser (digitalcmsspreadcoupon.hpp:104) and the
        # constructor does not assign it (.cpp:52-54), so a leg on which
        # ``withNakedOption()`` is never called reads an indeterminate bool —
        # undefined behaviour. Every other member of the class is initialised,
        # so the omission is plainly accidental and ``false`` is the intended
        # value (it is also the default of every ``withNakedOption`` overload
        # and of the ``DigitalCoupon`` constructor). The same defect is in
        # ``DigitalIborLeg`` (digitaliborcoupon.hpp:106) and ``DigitalCmsLeg``
        # (digitalcmscoupon.hpp:106). PQuantLib initialises it to False rather
        # than reproducing UB.
        self._naked_option: bool = False

    # --- chained setters (C++ digitalcmsspreadcoupon.cpp:56-190) ----------

    def with_notionals(self, notionals: float | Sequence[float]) -> DigitalCmsSpreadLeg:
        self._notionals = as_float_list(notionals)
        return self

    def with_payment_day_counter(self, day_counter: DayCounter) -> DigitalCmsSpreadLeg:
        self._payment_day_counter = day_counter
        return self

    def with_payment_adjustment(
        self, convention: BusinessDayConvention
    ) -> DigitalCmsSpreadLeg:
        self._payment_adjustment = convention
        return self

    def with_fixing_days(self, fixing_days: int | Sequence[int]) -> DigitalCmsSpreadLeg:
        self._fixing_days = as_int_list(fixing_days)
        return self

    def with_gearings(self, gearings: float | Sequence[float]) -> DigitalCmsSpreadLeg:
        self._gearings = as_float_list(gearings)
        return self

    def with_spreads(self, spreads: float | Sequence[float]) -> DigitalCmsSpreadLeg:
        self._spreads = as_float_list(spreads)
        return self

    def in_arrears(self, flag: bool = True) -> DigitalCmsSpreadLeg:
        self._in_arrears = flag
        return self

    def with_call_strikes(
        self, strikes: float | None | Sequence[float | None]
    ) -> DigitalCmsSpreadLeg:
        self._call_strikes = as_opt_float_list(strikes)
        return self

    def with_long_call_option(self, position: PositionType) -> DigitalCmsSpreadLeg:
        self._long_call_option = position
        return self

    def with_call_atm(self, flag: bool = True) -> DigitalCmsSpreadLeg:
        self._call_atm = flag
        return self

    def with_call_payoffs(
        self, payoffs: float | None | Sequence[float | None]
    ) -> DigitalCmsSpreadLeg:
        self._call_payoffs = as_opt_float_list(payoffs)
        return self

    def with_put_strikes(
        self, strikes: float | None | Sequence[float | None]
    ) -> DigitalCmsSpreadLeg:
        self._put_strikes = as_opt_float_list(strikes)
        return self

    def with_long_put_option(self, position: PositionType) -> DigitalCmsSpreadLeg:
        self._long_put_option = position
        return self

    def with_put_atm(self, flag: bool = True) -> DigitalCmsSpreadLeg:
        self._put_atm = flag
        return self

    def with_put_payoffs(
        self, payoffs: float | None | Sequence[float | None]
    ) -> DigitalCmsSpreadLeg:
        self._put_payoffs = as_opt_float_list(payoffs)
        return self

    def with_replication(self, replication: DigitalReplication) -> DigitalCmsSpreadLeg:
        self._replication = replication
        return self

    def with_naked_option(self, naked_option: bool = True) -> DigitalCmsSpreadLeg:
        self._naked_option = naked_option
        return self

    # --- operator Leg() ---------------------------------------------------

    def __call__(self) -> list[CashFlow]:
        """Sugar for :meth:`leg` — the C++ ``operator Leg() const``."""
        return self.leg()

    def _check_sizes(self, n: int) -> None:
        """# C++ parity: the QL_REQUIRE block at cashflowvectors.hpp:244-261."""
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
            len(self._call_strikes) <= n,
            f"too many call rates ({len(self._call_strikes)}), only {n} required",
        )
        qassert.require(
            len(self._put_strikes) <= n,
            f"too many put rates ({len(self._put_strikes)}), only {n} required",
        )

    def leg(self) -> list[CashFlow]:
        """Build the leg.

        # C++ parity: ``DigitalCmsSpreadLeg::operator Leg()``
        # (digitalcmsspreadcoupon.cpp:192-202) → ``FloatingDigitalLeg``
        # (cashflowvectors.hpp:221-313).
        """
        schedule = self._schedule
        n = len(schedule) - 1
        self._check_sizes(n)

        calendar = schedule.calendar
        index = self._index
        default_fixing_days = index.fixing_days()
        day_counter = self._payment_day_counter

        leg: list[CashFlow] = []
        for i in range(n):
            ref_start = start = schedule.date(i)
            ref_end = end = schedule.date(i + 1)
            # FloatingDigitalLeg has no payment lag and no payment calendar:
            # the schedule's own calendar adjusts the period end.
            payment_date = calendar.adjust(end, self._payment_adjustment)
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
            assert nominal is not None
            assert gearing is not None

            if gearing == 0.0:
                # C++ parity note: FloatingDigitalLeg's fixed-coupon fallback
                # (cashflowvectors.hpp:281-288) is NOT the FloatingLeg one. It
                # passes ``detail::get(spreads, i, 1.0)`` straight through —
                # default ONE, and no cap/floor clamping — so a gearing-zero
                # period on a leg with no spreads vector pays a 100% fixed
                # rate. Reproduced verbatim; see the coupons probe, key
                # ``G4_fixed_default_one_rates``.
                rate = get(self._spreads, i, 1.0)
                assert rate is not None
                leg.append(
                    FixedRateCoupon.from_rate(
                        payment_date,
                        nominal,
                        rate,
                        day_counter if day_counter is not None else index.day_counter(),
                        start,
                        end,
                        ref_start,
                        ref_end,
                        None,
                    )
                )
            else:
                spread = get(self._spreads, i, 0.0)
                assert spread is not None
                underlying = CmsSpreadCoupon(
                    payment_date,
                    nominal,
                    start,
                    end,
                    get_int(self._fixing_days, i, default_fixing_days),
                    index,
                    gearing,
                    spread,
                    ref_start,
                    ref_end,
                    day_counter,
                    self._in_arrears,
                )
                leg.append(
                    DigitalCmsSpreadCoupon(
                        underlying,
                        get(self._call_strikes, i, None),
                        self._long_call_option,
                        self._call_atm,
                        get(self._call_payoffs, i, None),
                        get(self._put_strikes, i, None),
                        self._long_put_option,
                        self._put_atm,
                        get(self._put_payoffs, i, None),
                        self._replication,
                        self._naked_option,
                    )
                )
        return leg


__all__ = ["DigitalCmsSpreadCoupon", "DigitalCmsSpreadLeg"]
