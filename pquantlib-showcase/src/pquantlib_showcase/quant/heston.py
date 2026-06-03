"""Heston stochastic-volatility: pricing, the implied-vol smile, calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.european_option import EuropeanOption
from pquantlib.math.optimization.end_criteria import EndCriteria
from pquantlib.math.optimization.levenberg_marquardt import LevenbergMarquardt
from pquantlib.models.calibration_helper import CalibrationErrorType
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.models.equity.heston_model_helper import HestonModelHelper
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_heston_engine import AnalyticHestonEngine
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

from .common import expiry_from_years, flat_curve, heston_process, reference_date
from .options import implied_vol


@dataclass(frozen=True, slots=True)
class HestonSmile:
    strikes: list[float]
    prices: list[float]
    implied_vols: list[float]
    feller_satisfied: bool


def heston_smile(
    spot: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    t_years: float,
    strikes: list[float],
) -> HestonSmile:
    """Price calls across strikes under Heston and back out the vol smile."""
    ref = reference_date()
    model = HestonModel(heston_process(spot, r, q, v0, kappa, theta, sigma, rho, ref))
    engine = AnalyticHestonEngine(model)
    expiry = expiry_from_years(ref, t_years)
    prices: list[float] = []
    ivols: list[float] = []
    for k in strikes:
        opt = EuropeanOption(PlainVanillaPayoff(OptionType.Call, k), EuropeanExercise(expiry))
        opt.set_pricing_engine(engine)
        px = opt.npv()
        prices.append(px)
        try:
            ivols.append(implied_vol("Call", spot, k, r, q, t_years, px))
        except Exception:  # noqa: BLE001 — deep wings can defeat the inverter
            ivols.append(math.nan)
    return HestonSmile(strikes, prices, ivols, feller_satisfied=2.0 * kappa * theta > sigma * sigma)


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float
    strikes: list[float]
    market_vols: list[float]
    model_vols: list[float]
    rmse_bps: float
    feller_satisfied: bool


def calibrate_heston(
    spot: float,
    r: float,
    q: float,
    t_years: float,
    strikes: list[float],
    market_vols: list[float],
) -> CalibrationResult:
    """Fit the 5 Heston parameters to a market vol smile (Levenberg-Marquardt)."""
    ref = reference_date()
    rf = flat_curve(r, ref)
    div = flat_curve(q, ref)
    s0 = SimpleQuote(spot)
    # Seed the optimiser from a generic, Feller-respecting starting point.
    model = HestonModel(heston_process(spot, r, q, 0.04, 1.5, 0.04, 0.4, -0.5, ref))
    engine = AnalyticHestonEngine(model)
    maturity = Period(max(1, round(t_years * 12)), TimeUnit.Months)
    helpers: list[HestonModelHelper] = []
    for k, ivol in zip(strikes, market_vols, strict=True):
        h = HestonModelHelper(
            maturity=maturity,
            calendar=NullCalendar(),
            s0=s0,
            strike_price=k,
            volatility=SimpleQuote(ivol),
            risk_free_rate=rf,
            dividend_yield=div,
            calibration_error_type=CalibrationErrorType.RelativePriceError,
        )
        h.set_pricing_engine(engine)
        helpers.append(h)
    model.calibrate(helpers, LevenbergMarquardt(), EndCriteria(1000, 200, 1e-8, 1e-8, 1e-8))

    smile = heston_smile(
        spot, r, q, model.v0(), model.kappa(), model.theta(), model.sigma(), model.rho(), t_years, strikes
    )
    resid = [(m - mk) for m, mk in zip(smile.implied_vols, market_vols, strict=True) if not math.isnan(m)]
    rmse = math.sqrt(sum(d * d for d in resid) / len(resid)) if resid else math.nan
    return CalibrationResult(
        v0=model.v0(),
        kappa=model.kappa(),
        theta=model.theta(),
        sigma=model.sigma(),
        rho=model.rho(),
        strikes=strikes,
        market_vols=market_vols,
        model_vols=smile.implied_vols,
        rmse_bps=rmse * 1e4,
        feller_satisfied=2.0 * model.kappa() * model.theta() > model.sigma() ** 2,
    )


def synthetic_market_smile(
    atm: float = 0.20, skew: float = 0.0008, smile: float = 0.00002
) -> tuple[list[float], list[float]]:
    """A plausible equity-index smile (downward skew + convexity) for the demo."""
    strikes = list(np.linspace(70.0, 130.0, 9))
    vols = [atm - skew * (k - 100.0) + smile * (k - 100.0) ** 2 for k in strikes]
    return strikes, vols
