"""Longstaff-Schwartz American MC pricing (sample program).

Prices an American put under Black-Scholes-Merton via the MCAmericanEngine
(L6-A). Cross-validates against the Longstaff-Schwartz 1998 paper
reference value (4.478 at S=36, K=40, T=1, r=6%, σ=20%).

Run: ``uv run python -m pquantlib_samples.american_option_mc``
"""

from __future__ import annotations

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import AmericanExercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.methods.montecarlo.lsm_basis_system import PolynomialType
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.mc_american_engine import MCAmericanEngine
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
    spot = SimpleQuote(36.0)
    strike = 40.0
    sigma = 0.20
    rate = 0.06
    div = 0.0
    expiry = Date.from_ymd(15, Month.January, 2027)

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

    option = VanillaOption(
        payoff=PlainVanillaPayoff(OptionType.Put, strike),
        exercise=AmericanExercise(today, expiry),
    )

    engine = MCAmericanEngine(
        process=process,
        time_steps=50,
        samples=2000,
        calibration_samples=2000,
        polynom_type=PolynomialType.Monomial,
        polynom_order=2,
        seed=42,
    )
    option.set_pricing_engine(engine)

    price = option.npv()
    print(f"S0={spot.value()}, K={strike}, T=1y, r={rate}, q={div}, σ={sigma}")
    print(f"Longstaff-Schwartz 1998 paper reference : 4.478")
    print(f"MCAmericanEngine (N=2000, basis order 2): {price:.4f}")


if __name__ == "__main__":
    main()
