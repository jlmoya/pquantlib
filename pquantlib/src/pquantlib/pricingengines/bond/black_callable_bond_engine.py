"""BlackCallableFixedRateBondEngine — Black engine for callable bonds.

# C++ parity: ql/experimental/callablebonds/blackcallablebondengine.{hpp,cpp}
#             (v1.42.1).

Prices the embedded *European* call/put on the forward bond price using
the Black formula (Hull, Fourth Edition, Chapter 20 "European bond
option").  Requires exactly one call/put date.

The quoted volatility is a forward *yield* volatility; the engine
converts it to a forward *price* volatility via the bond's forward
modified duration:

    fwdPriceVol = yieldVol * fwdModifiedDuration * fwdYtm

then prices a Black option on the forward cash price struck at the cash
call price.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.cash_flows import CashFlows
from pquantlib.cashflows.duration import Duration
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.instruments.callability import CallabilityType
from pquantlib.instruments.callable_bond import (
    CallableBondArguments,
    CallableBondResults,
)
from pquantlib.interest_rate import InterestRate
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.callable_bond_constant_vol import (
    CallableBondConstantVolatility,
)
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency

if TYPE_CHECKING:
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.termstructures.volatility.callable_bond_vol_structure import (
        CallableBondVolatilityStructure,
    )


class BlackCallableFixedRateBondEngine(
    GenericEngine[CallableBondArguments, CallableBondResults]
):
    """Black-formula callable fixed-rate bond engine.

    # C++ parity: ``class BlackCallableFixedRateBondEngine``
    # (blackcallablebondengine.{hpp,cpp}, v1.42.1).

    The first ctor accepts a forward-yield ``Quote`` (wrapped in a flat
    :class:`CallableBondConstantVolatility`); the second accepts a full
    :class:`CallableBondVolatilityStructure`.
    """

    def __init__(
        self,
        fwd_yield_vol: Quote | CallableBondVolatilityStructure,
        discount_curve: YieldTermStructureProtocol,
    ) -> None:
        super().__init__(CallableBondArguments(), CallableBondResults())
        vol: CallableBondVolatilityStructure
        if isinstance(fwd_yield_vol, Quote):
            vol = CallableBondConstantVolatility(
                fwd_yield_vol,
                Actual365Fixed(),
                settlement_days=0,
                calendar=NullCalendar(),
            )
        else:
            vol = fwd_yield_vol
        self._volatility: CallableBondVolatilityStructure = vol
        self._discount_curve: YieldTermStructureProtocol = discount_curve

    # ------------------------------------------------------------------

    def _spot_income(self) -> float:
        # C++ parity: blackcallablebondengine.cpp:50-72.
        settlement = self._arguments.settlement_date
        cf = self._arguments.cashflows
        option_maturity = self._arguments.put_call_schedule[0].date()

        income = 0.0
        for c in cf[:-1]:
            if not c.has_occurred(settlement, False):
                if c.has_occurred(option_maturity, False):
                    income += c.amount() * self._discount_curve.discount(c.date())
                else:
                    break
        return income / self._discount_curve.discount(settlement)

    def _forward_price_volatility(self) -> float:
        # C++ parity: blackcallablebondengine.cpp:75-122.
        bond_maturity = self._arguments.redemption_date
        exercise_date = self._arguments.callability_dates[0]
        fixed_leg = self._arguments.cashflows

        fwd_npv = CashFlows.npv_curve(
            fixed_leg, self._discount_curve, False, exercise_date
        )

        day_counter = self._arguments.payment_day_counter
        assert day_counter is not None
        frequency = self._arguments.frequency
        if frequency in (Frequency.NoFrequency, Frequency.Once):
            frequency = Frequency.Annual

        fwd_ytm = CashFlows.irr(
            fixed_leg,
            fwd_npv,
            day_counter,
            Compounding.Compounded,
            frequency,
            False,
            exercise_date,
        )

        fwd_rate = InterestRate(fwd_ytm, day_counter, Compounding.Compounded, frequency)

        fwd_dur = CashFlows.duration(
            fixed_leg,
            fwd_rate,
            Duration.Modified,
            False,
            exercise_date,
        )

        cash_strike = (
            self._arguments.callability_prices[0] * self._arguments.face_amount / 100.0
        )
        vol_dc = self._volatility.day_counter()
        reference_date = self._volatility.reference_date()
        exercise_time = vol_dc.year_fraction(reference_date, exercise_date)
        maturity_time = vol_dc.year_fraction(reference_date, bond_maturity)
        yield_vol = self._volatility.volatility(
            exercise_time, maturity_time - exercise_time, cash_strike
        )
        return yield_vol * fwd_dur * fwd_ytm

    # ------------------------------------------------------------------

    def calculate(self) -> None:
        # C++ parity: blackcallablebondengine.cpp:125-175.
        results = self._results
        results.reset()
        args = self._arguments

        qassert.require(
            len(args.put_call_schedule) == 1,
            "Must have exactly one call/put date to use Black Engine",
        )
        settle = args.settlement_date
        exercise_date = args.callability_dates[0]
        qassert.require(exercise_date >= settle, "must have exercise Date >= settlement Date")

        fixed_leg = args.cashflows

        value = CashFlows.npv_curve(fixed_leg, self._discount_curve, False, settle)
        npv = CashFlows.npv_curve(
            fixed_leg, self._discount_curve, False, self._discount_curve.reference_date()
        )

        fwd_cash_price = (value - self._spot_income()) / self._discount_curve.discount(
            exercise_date
        )
        cash_strike = args.callability_prices[0] * args.face_amount / 100.0

        opt_type = (
            OptionType.Call
            if args.put_call_schedule[0].type() == CallabilityType.Call
            else OptionType.Put
        )

        price_vol = self._forward_price_volatility()
        exercise_time = self._volatility.day_counter().year_fraction(
            self._volatility.reference_date(), exercise_date
        )

        discount = self._discount_curve.discount(exercise_date)
        discount_to_settlement = discount / self._discount_curve.discount(settle)

        embedded_option_value = black_formula(
            opt_type,
            cash_strike,
            fwd_cash_price,
            price_vol * math.sqrt(exercise_time),
        )

        if opt_type == OptionType.Call:
            results.value = npv - embedded_option_value * discount
            results.settlement_value = value - embedded_option_value * discount_to_settlement
        else:
            results.value = npv + embedded_option_value * discount
            results.settlement_value = value + embedded_option_value * discount_to_settlement


class BlackCallableZeroCouponBondEngine(BlackCallableFixedRateBondEngine):
    """Black-formula callable zero-coupon bond engine.

    # C++ parity: ``class BlackCallableZeroCouponBondEngine``
    # (blackcallablebondengine.hpp:74-88) — identical behaviour, distinct type.
    """


__all__ = [
    "BlackCallableFixedRateBondEngine",
    "BlackCallableZeroCouponBondEngine",
]
