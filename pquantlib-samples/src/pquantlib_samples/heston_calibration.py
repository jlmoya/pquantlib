"""Heston model calibration to a vanilla-option surface (sample program).

Calibrates HestonModel parameters (kappa, theta, sigma, rho, v0) to a
synthetic 5-strike × 4-tenor implied-vol surface via AnalyticHestonEngine.

Run: ``uv run python -m pquantlib_samples.heston_calibration``
"""

from __future__ import annotations

import math

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.math.optimization.constraint import PositiveConstraint
from pquantlib.math.optimization.end_criteria import EndCriteria
from pquantlib.math.optimization.levenberg_marquardt import LevenbergMarquardt
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.models.equity.heston_model_helper import HestonModelHelper
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_heston_engine import (
    AnalyticHestonEngine,
)
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def main() -> None:
    today = Date.from_ymd(15, Month.January, 2026)

    risk_free = FlatForward.from_rate(today, 0.05, Actual365Fixed())
    dividend = FlatForward.from_rate(today, 0.02, Actual365Fixed())
    spot = SimpleQuote(100.0)

    # True process for synthetic data
    true_process = HestonProcess(
        risk_free_rate=risk_free,
        dividend_yield=dividend,
        s0=spot,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
    )

    # Initial guess for calibration (intentionally off)
    init_process = HestonProcess(
        risk_free_rate=risk_free,
        dividend_yield=dividend,
        s0=spot,
        v0=0.05,
        kappa=1.5,
        theta=0.05,
        sigma=0.4,
        rho=-0.5,
    )
    model = HestonModel(init_process)
    engine = AnalyticHestonEngine(model)

    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    maturities_months = [3, 6, 12, 24]
    day_counter = Actual365Fixed()

    helpers: list[HestonModelHelper] = []
    for months in maturities_months:
        expiry = Date.from_ymd(15, Month(((months + 0) % 12) or 12), 2026 + months // 12)
        for k in strikes:
            true_engine = AnalyticHestonEngine(HestonModel(true_process))
            opt = VanillaOption(
                payoff=PlainVanillaPayoff(OptionType.Call, k),
                exercise=EuropeanExercise(expiry),
            )
            opt.set_pricing_engine(true_engine)
            true_price = opt.npv()
            vol_quote = SimpleQuote(0.20)
            helper = HestonModelHelper(
                maturity_date=expiry,
                calendar=None,
                s0=spot,
                strike=k,
                volatility=vol_quote,
                risk_free_rate=risk_free,
                dividend_yield=dividend,
                error_type=HestonModelHelper.ErrorType.PriceError,
            )
            helper.set_pricing_engine(engine)
            helper.set_market_value(true_price)
            helpers.append(helper)

    optimizer = LevenbergMarquardt(epsfcn=1e-8, xtol=1e-8, gtol=1e-8)
    end_criteria = EndCriteria(
        max_iterations=400, max_stationary_state_iterations=50,
        root_epsilon=1e-8, function_epsilon=1e-8, gradient_norm_epsilon=1e-8,
    )

    model.calibrate(
        instruments=helpers,
        method=optimizer,
        end_criteria=end_criteria,
        constraint=PositiveConstraint(),
    )

    p = model.params()
    print(f"Calibrated params:")
    print(f"  v0    : {p[0]:.4f} (true 0.0400)")
    print(f"  kappa : {p[1]:.4f} (true 2.0000)")
    print(f"  theta : {p[2]:.4f} (true 0.0400)")
    print(f"  sigma : {p[3]:.4f} (true 0.3000)")
    print(f"  rho   : {p[4]:.4f} (true -0.7000)")

    errors = [h.calibration_error() for h in helpers]
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    print(f"  RMSE  : {rmse:.6f}")


if __name__ == "__main__":
    main()
