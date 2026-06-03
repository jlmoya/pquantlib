"""Swap sample — minimal vanilla interest-rate swap valuation.

Port of ``org.jquantlib.samples.Swap`` (itself modelled on the swap-pricing
tail of QuantLib's MulticurveBootstrapping example). A flat-forward 3% Euribor
6M curve is built, a 5-year payer vanilla swap (4% fixed vs Euribor 6M + 0 bp)
is constructed, priced with :class:`DiscountingSwapEngine`, and the per-leg
NPVs together with the fair fixed rate and fair spread are printed. A
consistency check re-prices the swap at the fair fixed rate and confirms its
NPV collapses to ~0.

PQuantLib has no ``RelinkableHandle``; the flat-forward term structure is passed
directly to both the index and the engine (a single shared object), which is the
idiomatic Python equivalent of linking a handle once.
"""

from __future__ import annotations

from dataclasses import dataclass

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.vanilla_swap import VanillaSwap
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit


@dataclass(frozen=True, slots=True)
class SwapResult:
    """Computed quantities a :func:`run` would print — for cross-checking."""

    evaluation_date: Date
    settlement_date: Date
    maturity_date: Date
    nominal: float
    input_fixed_rate: float
    floating_spread: float
    npv: float
    fixed_leg_npv: float
    floating_leg_npv: float
    fair_rate: float
    fair_spread: float
    fair_npv_check: float


def compute() -> SwapResult:
    today = Date.from_ymd(15, Month.February, 2010)
    ObservableSettings().evaluation_date = today

    calendar = TARGET()
    nominal = 1_000_000.0
    fixed_rate = 0.04
    floating_spread = 0.0

    fixed_day_count = Thirty360(Convention.BondBasis)
    float_day_count = Actual360()

    # Flat 3% curve, shared by the floating index and the discount engine.
    discount_curve = FlatForward.from_rate(today, 0.03, Actual360())
    euribor6m = Euribor.six_months(discount_curve)

    # Spot start, 5-year tenor.
    settlement = calendar.advance_period(today, Period(2, TimeUnit.Days))
    maturity = calendar.advance_period(settlement, Period(5, TimeUnit.Years))

    fixed_schedule = Schedule.from_rule(
        settlement,
        maturity,
        Period.from_frequency(Frequency.Annual),
        calendar,
        BusinessDayConvention.Unadjusted,
        BusinessDayConvention.Unadjusted,
        DateGeneration.Forward,
        False,
    )
    float_schedule = Schedule.from_rule(
        settlement,
        maturity,
        Period.from_frequency(Frequency.Semiannual),
        calendar,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Forward,
        False,
    )

    swap = VanillaSwap(
        SwapType.Payer,
        nominal,
        fixed_schedule,
        fixed_rate,
        fixed_day_count,
        float_schedule,
        euribor6m,
        floating_spread,
        float_day_count,
    )
    swap.set_pricing_engine(DiscountingSwapEngine(discount_curve))

    fair_rate = swap.fair_rate()

    # Consistency check: re-price at the fair fixed rate; NPV should vanish.
    fair_swap = VanillaSwap(
        SwapType.Payer,
        nominal,
        fixed_schedule,
        fair_rate,
        fixed_day_count,
        float_schedule,
        euribor6m,
        0.0,
        float_day_count,
    )
    fair_swap.set_pricing_engine(DiscountingSwapEngine(discount_curve))

    return SwapResult(
        evaluation_date=today,
        settlement_date=settlement,
        maturity_date=maturity,
        nominal=nominal,
        input_fixed_rate=fixed_rate,
        floating_spread=floating_spread,
        npv=swap.npv(),
        fixed_leg_npv=swap.fixed_leg_npv(),
        floating_leg_npv=swap.floating_leg_npv(),
        fair_rate=fair_rate,
        fair_spread=swap.fair_spread(),
        fair_npv_check=fair_swap.npv(),
    )


def run() -> None:
    print("::::: Swap :::::")
    r = compute()

    print(f"Evaluation date     : {r.evaluation_date}")
    print(f"Settlement date     : {r.settlement_date}")
    print(f"Maturity date       : {r.maturity_date}")
    print(f"Nominal             : {r.nominal:,.2f}")
    print(f"Fixed coupon (input): {r.input_fixed_rate:.6f}")
    print(f"Floating spread     : {r.floating_spread:.6f}")
    print()
    print(f"NPV                 : {r.npv:,.6f}")
    print(f"Fixed leg NPV       : {r.fixed_leg_npv:,.6f}")
    print(f"Floating leg NPV    : {r.floating_leg_npv:,.6f}")
    print(f"Fair fixed rate     : {r.fair_rate:.8f}")
    print(f"Fair float spread   : {r.fair_spread:.8f}")
    print()
    print(f"Check - NPV at fairRate: {r.fair_npv_check:,.10f}")


if __name__ == "__main__":
    run()
