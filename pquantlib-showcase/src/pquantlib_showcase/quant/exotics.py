"""Exotic options: single barrier, double barrier, and discrete Asian."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.asian_option import DiscreteAveragingAsianOption
from pquantlib.instruments.average_type import AverageType
from pquantlib.instruments.barrier_option import BarrierOption, BarrierType
from pquantlib.instruments.double_barrier_option import DoubleBarrierOption, DoubleBarrierType
from pquantlib.instruments.european_option import EuropeanOption
from pquantlib.payoffs import PlainVanillaPayoff
from pquantlib.pricingengines.asian.analytic_discrete_geometric_average_price_engine import (
    AnalyticDiscreteGeometricAveragePriceAsianEngine,
)
from pquantlib.pricingengines.barrier.analytic_barrier_engine import AnalyticBarrierEngine
from pquantlib.pricingengines.barrier.analytic_double_barrier_engine import (
    AnalyticDoubleBarrierEngine,
)
from pquantlib.pricingengines.vanilla.analytic_european_engine import AnalyticEuropeanEngine

from .common import bsm_process, expiry_from_years, reference_date
from .options import option_type

BARRIER_TYPES = {
    "Down-and-Out": BarrierType.DownOut,
    "Down-and-In": BarrierType.DownIn,
    "Up-and-Out": BarrierType.UpOut,
    "Up-and-In": BarrierType.UpIn,
}
DOUBLE_BARRIER_TYPES = {
    "Knock-Out": DoubleBarrierType.KnockOut,
    "Knock-In": DoubleBarrierType.KnockIn,
}


@dataclass(frozen=True, slots=True)
class ExoticResult:
    price: float
    vanilla_price: float
    triggered: bool = False
    """True when the barrier is already breached at inception (option is dead/vanilla)."""


def _vanilla_ref(kind: str, spot: float, strike: float, r: float, q: float, vol: float, t: float) -> float:
    ref = reference_date()
    opt = EuropeanOption(
        PlainVanillaPayoff(option_type(kind), strike), EuropeanExercise(expiry_from_years(ref, t))
    )
    opt.set_pricing_engine(AnalyticEuropeanEngine(bsm_process(spot, r, q, vol, ref)))
    return opt.npv()


def price_barrier(
    kind: str,
    barrier_kind: str,
    spot: float,
    strike: float,
    barrier: float,
    rebate: float,
    r: float,
    q: float,
    vol: float,
    t_years: float,
) -> ExoticResult:
    """Reiner-Rubinstein analytic single-barrier price."""
    ref = reference_date()
    proc = bsm_process(spot, r, q, vol, ref)
    opt = BarrierOption(
        BARRIER_TYPES[barrier_kind],
        barrier,
        rebate,
        PlainVanillaPayoff(option_type(kind), strike),
        EuropeanExercise(expiry_from_years(ref, t_years)),
    )
    opt.set_pricing_engine(AnalyticBarrierEngine(proc))
    vanilla = _vanilla_ref(kind, spot, strike, r, q, vol, t_years)
    try:
        return ExoticResult(opt.npv(), vanilla)
    except LibraryException:
        # Barrier already breached at inception — the analytic engine declines it.
        return ExoticResult(math.nan, vanilla, triggered=True)


def price_double_barrier(
    kind: str,
    barrier_kind: str,
    spot: float,
    strike: float,
    low: float,
    high: float,
    rebate: float,
    r: float,
    q: float,
    vol: float,
    t_years: float,
) -> ExoticResult:
    """Ikeda-Kunitomo analytic double-barrier price."""
    ref = reference_date()
    proc = bsm_process(spot, r, q, vol, ref)
    opt = DoubleBarrierOption(
        DOUBLE_BARRIER_TYPES[barrier_kind],
        low,
        high,
        rebate,
        PlainVanillaPayoff(option_type(kind), strike),
        EuropeanExercise(expiry_from_years(ref, t_years)),
    )
    opt.set_pricing_engine(AnalyticDoubleBarrierEngine(proc))
    vanilla = _vanilla_ref(kind, spot, strike, r, q, vol, t_years)
    try:
        return ExoticResult(opt.npv(), vanilla)
    except LibraryException:
        return ExoticResult(math.nan, vanilla, triggered=True)


def price_asian(
    kind: str, spot: float, strike: float, r: float, q: float, vol: float, t_years: float, n_fixings: int = 12
) -> ExoticResult:
    """Analytic discrete *geometric*-average Asian price."""
    ref = reference_date()
    proc = bsm_process(spot, r, q, vol, ref)
    expiry = expiry_from_years(ref, t_years)
    total_days = max(1, round(t_years * 365.0))
    fixings = [ref + round(total_days * (i + 1) / n_fixings) for i in range(n_fixings)]
    opt = DiscreteAveragingAsianOption(
        AverageType.Geometric,
        0.0,
        0,
        fixings,
        PlainVanillaPayoff(option_type(kind), strike),
        EuropeanExercise(expiry),
    )
    opt.set_pricing_engine(AnalyticDiscreteGeometricAveragePriceAsianEngine(proc))
    return ExoticResult(opt.npv(), _vanilla_ref(kind, spot, strike, r, q, vol, t_years))


def barrier_vs_level(
    kind: str,
    barrier_kind: str,
    spot: float,
    strike: float,
    rebate: float,
    r: float,
    q: float,
    vol: float,
    t_years: float,
    n: int = 60,
) -> tuple[list[float], list[float], float]:
    """Barrier price as the barrier level sweeps the valid (un-breached) region.

    Bounds are derived from spot: a down-barrier must sit strictly below spot, an
    up-barrier strictly above, so the analytic engine is never handed a breached
    contract. Any residual triggered level is dropped from the curve.
    """
    if barrier_kind.startswith("Down"):
        levels = list(np.linspace(spot * 0.5, spot * 0.97, n))
    else:
        levels = list(np.linspace(spot * 1.03, spot * 1.5, n))
    kept_levels: list[float] = []
    prices: list[float] = []
    for b in levels:
        res = price_barrier(kind, barrier_kind, spot, strike, b, rebate, r, q, vol, t_years)
        if not res.triggered:
            kept_levels.append(b)
            prices.append(res.price)
    vanilla = _vanilla_ref(kind, spot, strike, r, q, vol, t_years)
    return kept_levels, prices, vanilla
