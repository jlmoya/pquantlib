"""FixedRateBond — bond with fixed-rate coupons from a Schedule.

# C++ parity: ql/instruments/bonds/fixedratebond.{hpp,cpp} (v1.42.1).

Carve-outs:
- ``firstPeriodDayCounter`` is accepted but only used if non-empty; the
  underlying ``fixed_rate_leg`` builder doesn't yet thread it (deferred
  in L2-D ``fixed_rate_leg`` carve-out list). We keep the field for
  parity but it's effectively unused.
- ``ex_coupon_period`` / ``ex_coupon_calendar`` are accepted as inputs
  but the ``fixed_rate_leg`` builder ignores them (L2-D carve-out).
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.cashflows.fixed_rate_leg import fixed_rate_leg
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.instruments.bond import Bond
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.schedule import Schedule


class FixedRateBond(Bond):
    """Fixed-rate bond.

    # C++ parity: fixedratebond.cpp:31-69.

    Constructor parameters mirror the C++ signature; ``redemption``
    defaults to 100 (par).
    """

    def __init__(
        self,
        settlement_days: int,
        face_amount: float,
        schedule: Schedule,
        coupons: Sequence[float],
        accrual_day_counter: DayCounter,
        payment_convention: BusinessDayConvention = BusinessDayConvention.Following,
        redemption: float = 100.0,
        issue_date: Date | None = None,
        payment_calendar: Calendar | None = None,
        first_period_day_counter: DayCounter | None = None,
    ) -> None:
        calendar = payment_calendar if payment_calendar is not None else schedule.calendar
        # Skip the base Bond ctor's coupon-sorting path — we control it.
        Bond.__init__(self, settlement_days, calendar, issue_date)

        self._frequency: Frequency = (
            schedule.tenor.frequency() if schedule.has_tenor() else Frequency.NoFrequency
        )
        self._day_counter: DayCounter = accrual_day_counter
        self._first_period_day_counter: DayCounter | None = first_period_day_counter

        self._maturity_date = schedule.end_date

        # Build the leg via the L2-D builder. The C++ builder uses the
        # bond's calendar as the payment calendar.
        self._cashflows = list(
            fixed_rate_leg(
                schedule,
                nominals=[face_amount],
                rates=list(coupons),
                day_counter=accrual_day_counter,
                compounding=Compounding.Simple,
                frequency=Frequency.Annual,
                payment_adjustment=payment_convention,
                payment_calendar=calendar,
            )
        )

        # Build redemptions on top of the coupons.
        self._add_redemptions_to_cashflows([redemption])

        # C++ QL_ENSURE: !cashflows().empty() && redemptions.size() == 1.
        qassert.require(len(self._cashflows) > 0, "bond with no cashflows!")
        qassert.require(len(self._redemptions) == 1, "multiple redemptions created")

        # Re-register on the (now built) cashflows.
        for cf in self._cashflows:
            cf.register_with(self)

    # ----- inspectors mirror C++ inline getters -----

    def frequency(self) -> Frequency:
        return self._frequency

    def day_counter(self) -> DayCounter:
        return self._day_counter

    def first_period_day_counter(self) -> DayCounter | None:
        return self._first_period_day_counter


__all__ = ["FixedRateBond"]
