"""YearOnYearInflationSwap — periodic YoY-rate vs fixed-rate inflation swap.

# C++ parity: ql/instruments/yearonyearinflationswap.{hpp,cpp} (v1.42.1).

The swap exchanges a periodic fixed leg (annual or other tenor) against a
YoY-rate leg whose coupons pay ``N * accrual * (gearing * YoY(T_i) + spread)``.

Cashflow seam: this cluster does not have L7-C's ``YoYInflationCoupon`` yet.
The constructor accepts a YoY leg of any ``YoYInflationCouponLike`` (Protocol
from ``yoy_inflation_capfloor``). Users can construct YoY coupons via a
scaffolding stub today; L7-C's production type will satisfy the same
Protocol at merge time.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.fixed_rate_leg import fixed_rate_leg
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.inflation.cpi import InterpolationType
from pquantlib.indexes.inflation.inflation_index import YoYInflationIndex
from pquantlib.instruments.swap import Swap, SwapType
from pquantlib.instruments.yoy_inflation_capfloor import YoYInflationCouponLike
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule


class YearOnYearInflationSwap(Swap):
    """Year-on-year inflation-indexed swap.

    # C++ parity: ``YearOnYearInflationSwap`` (yearonyearinflationswap.hpp:47-115).

    Constructor takes:
    - the fixed leg's schedule + rate + day-counter,
    - the YoY leg's pre-built coupon sequence (since L7-D doesn't yet have
      a YoY leg builder; L7-C will provide ``yoy_inflation_leg``),
    - the YoY index (forecast curve sits on it),
    - observation_lag, interpolation, spread, yoy_day_count, payment cal/conv.

    ``Type`` (Payer / Receiver) refers to the fixed leg.
    """

    def __init__(
        self,
        *,
        type_: SwapType,
        nominal: float,
        fixed_schedule: Schedule,
        fixed_rate: float,
        fixed_day_count: DayCounter,
        yoy_leg: Sequence[YoYInflationCouponLike],
        yoy_index: YoYInflationIndex,
        observation_lag: Period,
        interpolation: InterpolationType,
        spread: float,
        yoy_day_count: DayCounter,
        payment_calendar: Calendar,
        payment_convention: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing,
    ) -> None:
        # # C++ parity: YearOnYearInflationSwap::YearOnYearInflationSwap
        # # (yearonyearinflationswap.cpp:34-78).
        super().__init__(n_legs=2)
        self._type: SwapType = type_
        self._nominal: float = nominal
        self._fixed_schedule: Schedule = fixed_schedule
        self._fixed_rate: float = fixed_rate
        self._fixed_day_count: DayCounter = fixed_day_count
        self._yoy_index: YoYInflationIndex = yoy_index
        self._observation_lag: Period = observation_lag
        self._interpolation: InterpolationType = interpolation
        self._spread: float = spread
        self._yoy_day_count: DayCounter = yoy_day_count
        self._payment_calendar: Calendar = payment_calendar
        self._payment_convention: BusinessDayConvention = payment_convention

        # Build the fixed leg via the existing FixedRateLeg helper.
        fixed_leg: list[CashFlow] = fixed_rate_leg(
            fixed_schedule,
            nominals=[nominal],
            rates=[fixed_rate],
            day_counter=fixed_day_count,
            compounding=Compounding.Simple,
            frequency=Frequency.Annual,
            payment_adjustment=payment_convention,
            payment_calendar=payment_calendar,
        )

        # The YoY leg comes in pre-built (subagent-friendly seam: L7-C will
        # add a yoy_inflation_leg builder that matches the C++ signature).
        yoy_leg_list: list[CashFlow] = []
        for cf in yoy_leg:
            # We registered_with these as the engine reads them; we also
            # store as CashFlow for the Swap leg array.
            yoy_leg_list.append(cf)  # type: ignore[arg-type]

        self._legs = [fixed_leg, yoy_leg_list]
        # # C++ parity: payer multipliers (yearonyearinflationswap.cpp:71-77).
        if type_ == SwapType.Payer:
            self._payer = [-1.0, +1.0]
        else:
            self._payer = [+1.0, -1.0]

        # Register observers (cashflows + index updates).
        for cf in fixed_leg:
            cf.register_with(self)
        for cf in yoy_leg_list:
            cf.register_with(self)

    # ---- inspectors --------------------------------------------------

    def type(self) -> SwapType:
        return self._type

    def nominal(self) -> float:
        return self._nominal

    def fixed_schedule(self) -> Schedule:
        return self._fixed_schedule

    def fixed_rate(self) -> float:
        return self._fixed_rate

    def fixed_day_count(self) -> DayCounter:
        return self._fixed_day_count

    def yoy_inflation_index(self) -> YoYInflationIndex:
        return self._yoy_index

    def observation_lag(self) -> Period:
        return self._observation_lag

    def interpolation(self) -> InterpolationType:
        return self._interpolation

    def spread(self) -> float:
        return self._spread

    def yoy_day_count(self) -> DayCounter:
        return self._yoy_day_count

    def payment_calendar(self) -> Calendar:
        return self._payment_calendar

    def payment_convention(self) -> BusinessDayConvention:
        return self._payment_convention

    def fixed_leg(self) -> list[CashFlow]:
        return self._legs[0]

    def yoy_leg(self) -> list[CashFlow]:
        return self._legs[1]

    # ---- results -----------------------------------------------------

    def fixed_leg_npv(self) -> float:
        self.calculate()
        qassert.require(self._leg_npv[0] is not None, "result not available")
        assert self._leg_npv[0] is not None
        return self._leg_npv[0]

    def yoy_leg_npv(self) -> float:
        self.calculate()
        qassert.require(self._leg_npv[1] is not None, "result not available")
        assert self._leg_npv[1] is not None
        return self._leg_npv[1]

    def fair_rate(self) -> float:
        """``K`` such that NPV is zero given current curves.

        # C++ parity: YearOnYearInflationSwap::fairRate
        # (yearonyearinflationswap.cpp:136-140). Derived from NPV / BPS
        # when results carry a precomputed value; otherwise:
        #   fair_rate = fixed_rate - NPV / (legBPS[0] / basisPoint)
        """
        self.calculate()
        leg_bps = self._leg_bps[0]
        qassert.require(
            leg_bps is not None and not math.isclose(leg_bps, 0.0, abs_tol=1e-30),
            "result not available (legBPS[0] is None or zero)",
        )
        assert leg_bps is not None
        return self._fixed_rate - self.npv() / (leg_bps / 1e-4)

    def fair_spread(self) -> float:
        """Spread that zeroes out the swap NPV.

        # C++ parity: YearOnYearInflationSwap::fairSpread
        # (yearonyearinflationswap.cpp:142-145).
        """
        self.calculate()
        leg_bps = self._leg_bps[1]
        qassert.require(
            leg_bps is not None and not math.isclose(leg_bps, 0.0, abs_tol=1e-30),
            "result not available (legBPS[1] is None or zero)",
        )
        assert leg_bps is not None
        return self._spread - self.npv() / (leg_bps / 1e-4)


__all__ = ["YearOnYearInflationSwap"]
