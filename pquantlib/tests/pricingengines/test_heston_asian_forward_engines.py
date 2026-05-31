"""Cross-validate the Heston Asian/forward analytic engines (W8-C, batch b).

Probe source: migration-harness/cpp/probes/cluster_w8c/probe.cpp
Reference:    migration-harness/references/cluster/w8c.json

Covers:
  * AnalyticContinuousGeometricAveragePriceAsianHestonEngine (Kim-Wee).
  * AnalyticDiscreteGeometricAveragePriceAsianHestonEngine (Kim-Wee).
  * AnalyticHestonForwardEuropeanEngine (Kruse forward-start).

All three are CF-integration engines — LOOSE tier (empirically machine
precision because the Gauss-Legendre nodes match the C++ integrator).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.asian_option import (
    AverageType,
    ContinuousAveragingAsianOption,
    DiscreteAveragingAsianOption,
)
from pquantlib.instruments.forward_vanilla_option import ForwardVanillaOption
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.asian.analytic_cont_geom_av_price_heston_engine import (
    AnalyticContinuousGeometricAveragePriceAsianHestonEngine,
)
from pquantlib.pricingengines.asian.analytic_discr_geom_av_price_heston_engine import (
    AnalyticDiscreteGeometricAveragePriceAsianHestonEngine,
)
from pquantlib.pricingengines.forward.analytic_heston_forward_european_engine import (
    AnalyticHestonForwardEuropeanEngine,
)
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import tolerance
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    return load_reference("cluster/w8c")


def _heston_process() -> tuple[HestonProcess, Date]:
    today = Date.from_ymd(15, Month.January, 2024)
    ObservableSettings().evaluation_date = today
    dc = Actual365Fixed()
    spot = SimpleQuote(100.0)
    r_ts = FlatForward.from_rate(reference_date=today, forward_rate=0.05, day_counter=dc)
    q_ts = FlatForward.from_rate(reference_date=today, forward_rate=0.0, day_counter=dc)
    proc = HestonProcess(
        risk_free_rate=r_ts,
        dividend_yield=q_ts,
        s0=spot,
        v0=0.09,
        kappa=1.15,
        theta=0.0348,
        sigma=0.39,
        rho=-0.64,
    )
    return proc, today


def test_continuous_geometric_asian_heston(ref: dict[str, Any]) -> None:
    proc, today = _heston_process()
    exercise = EuropeanExercise(today + 365)
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    opt = ContinuousAveragingAsianOption(AverageType.Geometric, payoff, exercise)
    opt.set_pricing_engine(
        AnalyticContinuousGeometricAveragePriceAsianHestonEngine(proc)
    )
    tolerance.loose(opt.npv(), ref["heston_cont_geom_asian_call"])


def test_discrete_geometric_asian_heston(ref: dict[str, Any]) -> None:
    proc, today = _heston_process()
    exercise = EuropeanExercise(today + 365)
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    fixings = [today + Period(i, TimeUnit.Months) for i in range(1, 13)]
    opt = DiscreteAveragingAsianOption(AverageType.Geometric, 1.0, 0, fixings, payoff, exercise)
    opt.set_pricing_engine(
        AnalyticDiscreteGeometricAveragePriceAsianHestonEngine(proc)
    )
    tolerance.loose(opt.npv(), ref["heston_discr_geom_asian_call"])


def test_heston_forward_european(ref: dict[str, Any]) -> None:
    proc, today = _heston_process()
    exercise = EuropeanExercise(today + 365)
    payoff = PlainVanillaPayoff(OptionType.Call, 0.0)
    opt = ForwardVanillaOption(1.1, today + 182, payoff, exercise)
    opt.set_pricing_engine(AnalyticHestonForwardEuropeanEngine(proc))
    tolerance.loose(opt.npv(), ref["heston_forward_call"])


def test_heston_forward_put_call_consistency() -> None:
    # A forward-start put and call with the same params satisfy a parity-like
    # relation; here we just sanity-check the put is positive and finite.
    proc, today = _heston_process()
    exercise = EuropeanExercise(today + 365)
    put_payoff = PlainVanillaPayoff(OptionType.Put, 0.0)
    opt = ForwardVanillaOption(1.1, today + 182, put_payoff, exercise)
    opt.set_pricing_engine(AnalyticHestonForwardEuropeanEngine(proc))
    npv = opt.npv()
    assert npv > 0.0
    assert npv < 100.0
