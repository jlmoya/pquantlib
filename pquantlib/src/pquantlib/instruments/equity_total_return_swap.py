"""EquityTotalReturnSwap — equity total return against a floating leg.

# C++ parity: ql/instruments/equitytotalreturnswap.hpp + .cpp (v1.43).

The equity leg pays

.. math:: FV^{equity} = N \\left[ \\frac{I(t, T_M)}{I(T_0)} - 1 \\right]

as a single :class:`EquityCashFlow`; the interest leg is an Ibor leg or an
overnight leg. The swap ``type`` refers to the **equity** leg.

C++ has two public constructors differing only in the static type of the
interest-rate index — ``IborIndex`` selects ``IborLeg``, ``OvernightIndex``
selects ``OvernightLeg``. Python has no overload resolution on a runtime
type, and ``OvernightIndex`` is a subclass of ``IborIndex`` in C++ as well
(so even there the *static* type decides), which makes an ``isinstance``
dispatch a different rule from C++'s. The two constructors are therefore
kept apart as the explicit classmethods :meth:`with_ibor_index` and
:meth:`with_overnight_index`; the caller says which leg it wants, exactly as
the C++ caller does by choosing the argument's declared type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.equity_cashflow import EquityCashFlow
from pquantlib.cashflows.ibor_coupon import IborLeg
from pquantlib.cashflows.overnight_indexed_coupon import OvernightLeg
from pquantlib.instruments.swap import Swap, SwapType
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.equity_index import EquityIndex
    from pquantlib.indexes.interest_rate_index import InterestRateIndex
    from pquantlib.termstructures.protocols import (
        IborIndexProtocol,
        OvernightIndexProtocol,
    )
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.schedule import Schedule

_BASIS_POINT = 1.0e-4


class EquityTotalReturnSwap(Swap):
    """Total return of an equity index against a floating leg."""

    def __init__(
        self,
        equity_index: EquityIndex,
        interest_rate_index: InterestRateIndex,
        swap_type: SwapType,
        nominal: float,
        schedule: Schedule,
        day_counter: DayCounter,
        margin: float,
        gearing: float = 1.0,
        payment_calendar: Calendar | None = None,
        payment_convention: BusinessDayConvention = BusinessDayConvention.Unadjusted,
        payment_delay: int = 0,
    ) -> None:
        """Common part of both C++ constructors (the private one).

        C++ parity: equitytotalreturnswap.cpp:66-101. Builds the equity leg
        and sets the payer signs; the interest leg is filled in by the
        classmethod that selected it.
        """
        super().__init__(2)
        qassert.require(not nominal < 0.0, "Nominal cannot be negative")
        self._equity_index = equity_index
        self._interest_rate_index = interest_rate_index
        self._type = swap_type
        self._nominal = nominal
        self._schedule = schedule
        self._day_counter = day_counter
        self._margin = margin
        self._gearing = gearing
        self._payment_calendar = payment_calendar
        self._payment_convention = payment_convention
        self._payment_delay = payment_delay

        self._legs[0] = [self._create_equity_cash_flow()]
        for cf in self._legs[0]:
            cf.register_with(self)

        if swap_type == SwapType.Payer:
            self._payer = [-1.0, +1.0]
        elif swap_type == SwapType.Receiver:
            self._payer = [+1.0, -1.0]
        else:  # pragma: no cover - SwapType has exactly two members
            qassert.fail("unknown equity total return swap type")

    # --- construction --------------------------------------------------

    def _payment_cal(self) -> Calendar:
        """C++ parity: equitytotalreturnswap.cpp:34-38.

        An empty payment calendar falls back to the schedule's calendar.
        C++ additionally rejects a schedule whose calendar is empty; a
        Python ``Schedule.calendar`` is a non-optional ``Calendar``, so that
        branch is unreachable here and is deliberately not reproduced.
        """
        if self._payment_calendar is not None:
            return self._payment_calendar
        return self._schedule.calendar

    def _create_equity_cash_flow(self) -> EquityCashFlow:
        # C++ parity: equitytotalreturnswap.cpp:27-45.
        start_date = self._schedule.start_date
        end_date = self._schedule.end_date
        payment_date = self._payment_cal().advance(
            end_date,
            self._payment_delay,
            TimeUnit.Days,
            self._payment_convention,
            # C++ passes schedule.endOfMonth() unconditionally; the Python
            # Schedule raises if the flag was never set (a date-list
            # schedule), so fall back to C++'s own default of False.
            self._schedule.end_of_month if self._schedule.has_end_of_month() else False,
        )
        return EquityCashFlow(
            self._nominal, self._equity_index, start_date, end_date, payment_date
        )

    def _set_interest_leg(self, leg: list[CashFlow]) -> None:
        self._legs[1] = list(leg)
        for cf in self._legs[1]:
            cf.register_with(self)

    @classmethod
    def with_ibor_index(
        cls,
        swap_type: SwapType,
        nominal: float,
        schedule: Schedule,
        equity_index: EquityIndex,
        interest_rate_index: IborIndexProtocol,
        day_counter: DayCounter,
        margin: float,
        gearing: float = 1.0,
        payment_calendar: Calendar | None = None,
        payment_convention: BusinessDayConvention = BusinessDayConvention.Unadjusted,
        payment_delay: int = 0,
    ) -> EquityTotalReturnSwap:
        """C++ parity: equitytotalreturnswap.cpp:103-127 (the ``IborIndex`` ctor)."""
        swap = cls(
            equity_index,
            interest_rate_index,  # pyright: ignore[reportArgumentType]
            swap_type,
            nominal,
            schedule,
            day_counter,
            margin,
            gearing,
            payment_calendar,
            payment_convention,
            payment_delay,
        )
        # C++ parity: createInterestLeg<IborIndex, IborLeg>
        # (equitytotalreturnswap.cpp:47-64) — every one of these seven
        # setters is passed through, payment lag included.
        builder = (
            IborLeg(schedule, interest_rate_index)
            .with_notionals(nominal)
            .with_payment_day_counter(day_counter)
            .with_spreads(margin)
            .with_gearings(gearing)
            .with_payment_adjustment(payment_convention)
            .with_payment_lag(payment_delay)
        )
        if payment_calendar is not None:
            builder = builder.with_payment_calendar(payment_calendar)
        swap._set_interest_leg(builder.build())
        return swap

    @classmethod
    def with_overnight_index(
        cls,
        swap_type: SwapType,
        nominal: float,
        schedule: Schedule,
        equity_index: EquityIndex,
        interest_rate_index: OvernightIndexProtocol,
        day_counter: DayCounter,
        margin: float,
        gearing: float = 1.0,
        payment_calendar: Calendar | None = None,
        payment_convention: BusinessDayConvention = BusinessDayConvention.Unadjusted,
        payment_delay: int = 0,
    ) -> EquityTotalReturnSwap:
        """C++ parity: equitytotalreturnswap.cpp:129-153 (the ``OvernightIndex`` ctor)."""
        swap = cls(
            equity_index,
            interest_rate_index,  # pyright: ignore[reportArgumentType]
            swap_type,
            nominal,
            schedule,
            day_counter,
            margin,
            gearing,
            payment_calendar,
            payment_convention,
            payment_delay,
        )
        # C++ parity: createInterestLeg<OvernightIndex, OvernightLeg>.
        builder = (
            OvernightLeg(schedule, interest_rate_index)
            .with_notionals(nominal)
            .with_payment_day_counter(day_counter)
            .with_spreads(margin)
            .with_gearings(gearing)
            .with_payment_adjustment(payment_convention)
            .with_payment_lag(payment_delay)
        )
        if payment_calendar is not None:
            builder = builder.with_payment_calendar(payment_calendar)
        swap._set_interest_leg(builder.build())
        return swap

    # --- inspectors ----------------------------------------------------

    def type(self) -> SwapType:
        return self._type

    def nominal(self) -> float:
        return self._nominal

    def equity_index(self) -> EquityIndex:
        return self._equity_index

    def interest_rate_index(self) -> InterestRateIndex:
        return self._interest_rate_index

    def schedule(self) -> Schedule:
        return self._schedule

    def day_counter(self) -> DayCounter:
        return self._day_counter

    def margin(self) -> float:
        return self._margin

    def gearing(self) -> float:
        return self._gearing

    def payment_calendar(self) -> Calendar | None:
        return self._payment_calendar

    def payment_convention(self) -> BusinessDayConvention:
        return self._payment_convention

    def payment_delay(self) -> int:
        return self._payment_delay

    def equity_leg(self) -> list[CashFlow]:
        return self.leg(0)

    def interest_rate_leg(self) -> list[CashFlow]:
        return self.leg(1)

    # --- results -------------------------------------------------------

    def equity_leg_npv(self) -> float:
        return self.leg_npv(0)

    def interest_rate_leg_npv(self) -> float:
        return self.leg_npv(1)

    def fair_margin(self) -> float:
        """Margin that zeroes the NPV.

        C++ parity: equitytotalreturnswap.cpp:163-172.
        """
        interest_leg_bps = self.leg_bps(1) / _BASIS_POINT
        ex_margin_interest_leg_npv = self.interest_rate_leg_npv() - self.margin() * interest_leg_bps
        return -(self.equity_leg_npv() + ex_margin_interest_leg_npv) / interest_leg_bps


__all__ = ["EquityTotalReturnSwap"]
