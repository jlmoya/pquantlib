"""End-to-end vanilla swap pricing under a bootstrapped EUR yield curve.

Builds a 5y flat-forward curve, prices a 5y receiver swap on Euribor3M at
3.5% fixed, prints NPV + fair rate + BPS.

Run: ``uv run python -m pquantlib_samples.vanilla_swap_pricing``
"""

from __future__ import annotations

from pquantlib.cashflows.fixed_rate_leg import fixed_rate_leg
from pquantlib.cashflows.ibor_leg import ibor_leg
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.vanilla_swap import VanillaSwap
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit


def main() -> None:
    today = Date.from_ymd(15, Month.January, 2026)
    settlement_days = 2
    calendar = TARGET()
    settlement = calendar.advance(today, settlement_days, TimeUnit.Days)

    discount_curve = FlatForward.from_rate(
        reference_date=settlement,
        rate=0.035,
        day_counter=Actual365Fixed(),
    )

    maturity = calendar.advance(settlement, 5, TimeUnit.Years)
    fixed_schedule = Schedule.from_dates(
        effective_date=settlement,
        termination_date=maturity,
        tenor=Period(1, TimeUnit.Years),
        calendar=calendar,
        convention=BusinessDayConvention.ModifiedFollowing,
        end_of_month=False,
    )
    floating_schedule = Schedule.from_dates(
        effective_date=settlement,
        termination_date=maturity,
        tenor=Period(3, TimeUnit.Months),
        calendar=calendar,
        convention=BusinessDayConvention.ModifiedFollowing,
        end_of_month=False,
    )

    index = Euribor.three_months(h=discount_curve)
    notional = 1_000_000.0
    fixed_rate = 0.035

    fixed = fixed_rate_leg(
        schedule=fixed_schedule,
        notionals=[notional],
        rates=[fixed_rate],
        day_counter=Thirty360(),
    )
    floating = ibor_leg(
        schedule=floating_schedule,
        notionals=[notional],
        index=index,
        day_counter=Actual360(),
        fixing_days=2,
    )

    swap = VanillaSwap(
        type_=VanillaSwap.Type.Receiver,
        nominal=notional,
        fixed_schedule=fixed_schedule,
        fixed_rate=fixed_rate,
        fixed_day_counter=Thirty360(),
        floating_schedule=floating_schedule,
        ibor_index=index,
        spread=0.0,
        floating_day_counter=Actual360(),
        payment_convention=BusinessDayConvention.ModifiedFollowing,
    )
    swap.set_pricing_engine(DiscountingSwapEngine(discount_curve))

    print(f"Today        : {today}")
    print(f"Settlement   : {settlement}")
    print(f"Maturity     : {maturity}")
    print(f"Curve rate   : 3.50% (flat, Act/365Fixed, Continuous)")
    print(f"Index        : Euribor 3M")
    print(f"Notional     : {notional:,.0f} EUR")
    print(f"Fixed rate   : {fixed_rate * 100:.2f}%")
    print(f"---")
    print(f"NPV          : {swap.npv():,.4f} EUR")
    print(f"Fair rate    : {swap.fair_rate() * 100:.4f}%")
    print(f"BPS          : {swap.bps():,.4f}")
    del fixed, floating  # leg lists also built for inspection if desired


if __name__ == "__main__":
    main()
