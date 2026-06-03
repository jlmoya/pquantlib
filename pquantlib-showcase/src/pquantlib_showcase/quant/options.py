"""Vanilla European options priced four ways, plus Greeks and implied vol.

The four engines — closed-form, binomial tree, Monte Carlo, and finite
differences — all price the *same* contract, which is the headline
demonstration that PQuantLib's pricing stack is internally consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.european_option import EuropeanOption
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_european_engine import AnalyticEuropeanEngine
from pquantlib.pricingengines.vanilla.binomial_engine import BinomialVanillaEngine, TreeBuilder
from pquantlib.pricingengines.vanilla.fd_black_scholes_vanilla_engine import (
    FdBlackScholesVanillaEngine,
)
from pquantlib.pricingengines.vanilla.mc_european_engine import MCEuropeanEngine

from .common import bsm_process, expiry_from_years, reference_date

_TREES = {
    "Cox-Ross-Rubinstein": TreeBuilder.CoxRossRubinstein,
    "Jarrow-Rudd": TreeBuilder.JarrowRudd,
    "Tian": TreeBuilder.Tian,
}


def option_type(kind: str) -> OptionType:
    return OptionType.Call if kind.lower().startswith("c") else OptionType.Put


@dataclass(frozen=True, slots=True)
class VanillaResult:
    analytic: float
    binomial: float
    mc: float
    mc_error: float
    fd: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def price_vanilla(
    kind: str,
    spot: float,
    strike: float,
    r: float,
    q: float,
    vol: float,
    t_years: float,
    binomial_steps: int = 500,
    mc_samples: int = 20000,
    tree: str = "Cox-Ross-Rubinstein",
) -> VanillaResult:
    """Price one European option with all four engines and read the Greeks."""
    ref = reference_date()
    proc = bsm_process(spot, r, q, vol, ref)
    payoff = PlainVanillaPayoff(option_type(kind), strike)
    exercise = EuropeanExercise(expiry_from_years(ref, t_years))

    an = EuropeanOption(payoff, exercise)
    an.set_pricing_engine(AnalyticEuropeanEngine(proc))
    analytic = an.npv()

    bn = EuropeanOption(payoff, exercise)
    bn.set_pricing_engine(BinomialVanillaEngine(proc, binomial_steps, _TREES[tree]))

    mc = EuropeanOption(payoff, exercise)
    mc.set_pricing_engine(
        MCEuropeanEngine(proc, time_steps=1, antithetic_variate=True, required_samples=mc_samples, seed=42)
    )

    fd = VanillaOption(payoff, exercise)
    fd.set_pricing_engine(
        FdBlackScholesVanillaEngine(
            proc, t_grid=200, x_grid=200, damping_steps=0, scheme_desc=FdmSchemeDesc.crank_nicolson()
        )
    )

    return VanillaResult(
        analytic=analytic,
        binomial=bn.npv(),
        mc=mc.npv(),
        mc_error=mc.error_estimate(),
        fd=fd.npv(),
        delta=an.delta(),
        gamma=an.gamma(),
        vega=an.vega(),
        theta=an.theta(),
        rho=an.rho(),
    )


def binomial_convergence(
    kind: str,
    spot: float,
    strike: float,
    r: float,
    q: float,
    vol: float,
    t_years: float,
    max_steps: int = 200,
) -> tuple[list[int], list[float], float]:
    """Binomial price vs step count, converging onto the closed-form value."""
    ref = reference_date()
    proc = bsm_process(spot, r, q, vol, ref)
    payoff = PlainVanillaPayoff(option_type(kind), strike)
    exercise = EuropeanExercise(expiry_from_years(ref, t_years))

    ref_opt = EuropeanOption(payoff, exercise)
    ref_opt.set_pricing_engine(AnalyticEuropeanEngine(proc))
    analytic = ref_opt.npv()

    steps = list(range(5, max_steps + 1, 5))
    prices: list[float] = []
    for n in steps:
        opt = EuropeanOption(payoff, exercise)
        opt.set_pricing_engine(BinomialVanillaEngine(proc, n, TreeBuilder.CoxRossRubinstein))
        prices.append(opt.npv())
    return steps, prices, analytic


@dataclass(frozen=True, slots=True)
class GreekProfiles:
    spots: list[float]
    price: list[float]
    delta: list[float]
    gamma: list[float]
    vega: list[float]
    theta: list[float]


def greeks_vs_spot(
    kind: str,
    strike: float,
    r: float,
    q: float,
    vol: float,
    t_years: float,
    spot_lo: float,
    spot_hi: float,
    n: int = 70,
) -> GreekProfiles:
    """Sweep spot and record price + Greeks — the risk-profile picture."""
    ref = reference_date()
    payoff = PlainVanillaPayoff(option_type(kind), strike)
    exercise = EuropeanExercise(expiry_from_years(ref, t_years))
    spots = list(np.linspace(spot_lo, spot_hi, n))
    price: list[float] = []
    delta: list[float] = []
    gamma: list[float] = []
    vega: list[float] = []
    theta: list[float] = []
    for s in spots:
        opt = EuropeanOption(payoff, exercise)
        opt.set_pricing_engine(AnalyticEuropeanEngine(bsm_process(s, r, q, vol, ref)))
        price.append(opt.npv())
        delta.append(opt.delta())
        gamma.append(opt.gamma())
        vega.append(opt.vega())
        theta.append(opt.theta())
    return GreekProfiles(spots, price, delta, gamma, vega, theta)


def implied_vol(
    kind: str, spot: float, strike: float, r: float, q: float, t_years: float, price: float
) -> float:
    """Invert a market price back to Black-Scholes implied volatility."""
    ref = reference_date()
    proc = bsm_process(spot, r, q, 0.20, ref)  # seed vol; inversion is vol-independent
    payoff = PlainVanillaPayoff(option_type(kind), strike)
    opt = VanillaOption(payoff, EuropeanExercise(expiry_from_years(ref, t_years)))
    return opt.implied_volatility(price, proc, accuracy=1e-7, max_evaluations=200, min_vol=0.001, max_vol=5.0)


def payoff_at_expiry(kind: str, strike: float, spots: list[float]) -> list[float]:
    sign = 1.0 if option_type(kind) == OptionType.Call else -1.0
    return [max(sign * (s - strike), 0.0) for s in spots]
