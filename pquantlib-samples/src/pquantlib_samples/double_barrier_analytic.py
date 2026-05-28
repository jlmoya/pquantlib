"""Double-barrier analytic pricing under Black-Scholes (sample program).

Prices a double-barrier knock-out call via AnalyticDoubleBarrierEngine
(L6-C). Verifies KO + KI = European (in-out parity).

Run: ``uv run python -m pquantlib_samples.double_barrier_analytic``
"""

from __future__ import annotations

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.double_barrier_option import (
    DoubleBarrierOption,
    DoubleBarrierType,
)
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.barrier.analytic_double_barrier_engine import (
    AnalyticDoubleBarrierEngine,
)
from pquantlib.pricingengines.vanilla.analytic_european_engine import (
    AnalyticEuropeanEngine,
)
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def main() -> None:
    today = Date.from_ymd(15, Month.January, 2026)
    expiry = Date.from_ymd(15, Month.January, 2027)
    spot = SimpleQuote(100.0)
    strike = 100.0
    barrier_lo = 80.0
    barrier_hi = 120.0
    sigma = 0.20
    rate = 0.05
    div = 0.0

    process = BlackScholesMertonProcess(
        s0=spot,
        dividend_yield=FlatForward.from_rate(today, div, Actual365Fixed()),
        risk_free_rate=FlatForward.from_rate(today, rate, Actual365Fixed()),
        black_vol=BlackConstantVol(
            reference_date=today,
            volatility=SimpleQuote(sigma),
            day_counter=Actual365Fixed(),
        ),
    )

    payoff = PlainVanillaPayoff(OptionType.Call, strike)
    exercise = EuropeanExercise(expiry)

    ko = DoubleBarrierOption(
        DoubleBarrierType.KnockOut, barrier_lo, barrier_hi, 0.0, payoff, exercise
    )
    ki = DoubleBarrierOption(
        DoubleBarrierType.KnockIn, barrier_lo, barrier_hi, 0.0, payoff, exercise
    )
    euro = VanillaOption(payoff, exercise)

    barrier_engine = AnalyticDoubleBarrierEngine(process)
    ko.set_pricing_engine(barrier_engine)
    ki.set_pricing_engine(barrier_engine)
    euro.set_pricing_engine(AnalyticEuropeanEngine(process))

    print(f"S={spot.value()}, K={strike}, L={barrier_lo}, U={barrier_hi}")
    print(f"T=1y, r={rate}, q={div}, σ={sigma}")
    print(f"---")
    print(f"Knock-out call NPV : {ko.npv():.4f}")
    print(f"Knock-in  call NPV : {ki.npv():.4f}")
    print(f"European  call NPV : {euro.npv():.4f}")
    print(f"KO + KI            : {ko.npv() + ki.npv():.4f}  (should match European)")


if __name__ == "__main__":
    main()
