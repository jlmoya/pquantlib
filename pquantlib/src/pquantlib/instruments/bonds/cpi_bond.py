"""CPIBond — zero-inflation-indexed-ratio-with-base bond.

# C++ parity: ql/instruments/bonds/cpibond.{hpp,cpp} (v1.43).

Each coupon pays ``fixedRate * I(t - lag) / baseCPI`` on the notional, and the
leg always ends with a :class:`CPICashFlow
<pquantlib.cashflows.cpi_coupon.CPICashFlow>` notional exchange — inflated, or
growth-only when ``growth_only`` is set. Unlike the other bond concretes the
redemption is *not* built by ``_add_redemptions_to_cashflows``: the CPI leg
already ends with the notional flow, so C++ derives the notional schedule from
the coupons and adopts that last flow as the redemption
(cpibond.cpp:97-101).

# C++ parity note — the ``growth_only`` argument:
# v1.43 has two constructors, one of which takes ``growthOnly`` as its third
# positional argument and is deprecated ("Use the overload without the
# growthOnly parameter", deprecated in 1.40); the non-deprecated one forwards
# ``growthOnly = false``. Python cannot overload, so the flag is a trailing
# keyword defaulting to ``False`` — omit it and you get the non-deprecated
# constructor's behaviour exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib.cashflows.cpi_coupon import CPILeg
from pquantlib.instruments.bond import Bond
from pquantlib.time.business_day_convention import BusinessDayConvention

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.inflation.cpi import InterpolationType
    from pquantlib.indexes.inflation.inflation_index import ZeroInflationIndex
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.date import Date
    from pquantlib.time.frequency import Frequency
    from pquantlib.time.period import Period
    from pquantlib.time.schedule import Schedule


class CPIBond(Bond):
    """CPI-indexed bond.

    # C++ parity: cpibond.cpp:62-108.

    If the schedule holds a single date the bond degenerates to a zero bond
    returning an inflated notional (the CPI leg still emits its final
    :class:`CPICashFlow <pquantlib.cashflows.cpi_coupon.CPICashFlow>`).
    """

    def __init__(
        self,
        settlement_days: int,
        face_amount: float,
        base_cpi: float,
        observation_lag: Period,
        cpi_index: ZeroInflationIndex,
        observation_interpolation: InterpolationType,
        schedule: Schedule,
        coupons: Sequence[float],
        accrual_day_counter: DayCounter,
        payment_convention: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing,
        issue_date: Date | None = None,
        payment_calendar: Calendar | None = None,
        ex_coupon_period: Period | None = None,
        ex_coupon_calendar: Calendar | None = None,
        ex_coupon_convention: BusinessDayConvention = BusinessDayConvention.Unadjusted,
        ex_coupon_end_of_month: bool = False,
        growth_only: bool = False,
    ) -> None:
        calendar = payment_calendar if payment_calendar is not None else schedule.calendar
        Bond.__init__(self, settlement_days, calendar, issue_date)

        self._frequency: Frequency = schedule.tenor.frequency()
        self._day_counter: DayCounter = accrual_day_counter
        self._growth_only: bool = growth_only
        self._base_cpi: float = base_cpi
        self._observation_lag: Period = observation_lag
        self._cpi_index: ZeroInflationIndex = cpi_index
        self._observation_interpolation: InterpolationType = observation_interpolation

        self._maturity_date = schedule.end_date

        builder = (
            CPILeg(schedule, cpi_index, base_cpi, observation_lag)
            .with_notionals(face_amount)
            .with_fixed_rates(coupons)
            .with_payment_day_counter(accrual_day_counter)
            .with_payment_adjustment(payment_convention)
            # C++ passes ``calendar_`` — the *bond's* calendar, i.e. the
            # payment calendar with the schedule's as fallback (cpibond.cpp:93).
            .with_payment_calendar(calendar)
            .with_observation_interpolation(observation_interpolation)
            .with_subtract_inflation_nominal(growth_only)
        )
        if ex_coupon_period is not None:
            ex_calendar = ex_coupon_calendar if ex_coupon_calendar is not None else calendar
            builder = builder.with_ex_coupon_period(
                ex_coupon_period,
                ex_calendar,
                ex_coupon_convention,
                ex_coupon_end_of_month,
            )
        self._cashflows = builder.build()

        self._calculate_notionals_from_cashflows()
        self._redemptions = [self._cashflows[-1]]

        cpi_index.register_with(self)
        for cf in self._cashflows:
            cf.register_with(self)

    # ----- inspectors mirror the C++ inline getters (cpibond.hpp:83-89) -----

    def frequency(self) -> Frequency:
        return self._frequency

    def day_counter(self) -> DayCounter:
        return self._day_counter

    def growth_only(self) -> bool:
        return self._growth_only

    def base_cpi(self) -> float:
        return self._base_cpi

    def observation_lag(self) -> Period:
        return self._observation_lag

    def cpi_index(self) -> ZeroInflationIndex:
        return self._cpi_index

    def observation_interpolation(self) -> InterpolationType:
        return self._observation_interpolation


__all__ = ["CPIBond"]
