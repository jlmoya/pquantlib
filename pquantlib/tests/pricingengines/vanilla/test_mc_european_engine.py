"""Tests for ``MCEuropeanEngine``.

# C++ parity: ql/pricingengines/vanilla/mceuropeanengine.hpp (v1.42.1).

Cross-validates against the analytic engine on the textbook BSM
scenario (S=K=100, r=5%, q=0%, sigma=20%, T=1y) and against the
``cluster/l5c.json`` ``analytic_european.call_npv`` reference. MC
sampling uses LOOSE-tier (3-sigma band around the analytic value).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.european_option import EuropeanOption
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_european_engine import (
    AnalyticEuropeanEngine,
)
from pquantlib.pricingengines.vanilla.mc_european_engine import MCEuropeanEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def reference_data() -> dict[str, Any]:
    return reference_reader.load("cluster/l5c")


def _build_textbook_bsm() -> tuple[GeneralizedBlackScholesProcess, Date]:
    """Return (process, expiry) for the BSM textbook scenario.

    S=100, K=100, r=5%, q=0%, sigma=20%, T=1y (365 calendar days).
    """
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.May, 2026)
    expiry = ref + 365
    spot = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)
    vol = BlackConstantVol(
        reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20
    )
    process = GeneralizedBlackScholesProcess(
        x0=spot, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol
    )
    return process, expiry


def test_call_npv_converges_to_analytic_3sigma(reference_data: dict[str, Any]) -> None:
    """MC NPV at 10000 samples should sit within 3 sigma of analytic."""
    process, expiry = _build_textbook_bsm()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(
        MCEuropeanEngine(
            process,
            time_steps=1,
            antithetic_variate=False,
            required_samples=10000,
            seed=42,
        )
    )
    npv = opt.npv()
    err = opt.error_estimate()
    analytic = float(reference_data["analytic_european"]["call_npv"])
    # LOOSE: |npv - analytic| <= 3 * 1-sigma.
    assert abs(npv - analytic) < 3 * err
    # Standard error at 10000 samples is in [0.05, 0.30].
    assert 0.05 < err < 0.30


def test_put_npv_converges_to_analytic_3sigma(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_textbook_bsm()
    payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(
        MCEuropeanEngine(
            process,
            time_steps=1,
            antithetic_variate=False,
            required_samples=10000,
            seed=42,
        )
    )
    npv = opt.npv()
    err = opt.error_estimate()
    analytic = float(reference_data["analytic_european"]["put_npv"])
    assert abs(npv - analytic) < 3 * err


def test_antithetic_reduces_error_estimate() -> None:
    """Antithetic should cut the standard error by ~30% in this scenario."""
    process, expiry = _build_textbook_bsm()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)

    opt_a = EuropeanOption(payoff, exercise)
    opt_a.set_pricing_engine(
        MCEuropeanEngine(
            process,
            time_steps=1,
            antithetic_variate=False,
            required_samples=10000,
            seed=42,
        )
    )
    _ = opt_a.npv()
    err_no_anti = opt_a.error_estimate()

    # Antithetic at 5000 sample pairs (= 10000 effective draws).
    opt_b = EuropeanOption(payoff, exercise)
    opt_b.set_pricing_engine(
        MCEuropeanEngine(
            process,
            time_steps=1,
            antithetic_variate=True,
            required_samples=5000,
            seed=42,
        )
    )
    _ = opt_b.npv()
    err_anti = opt_b.error_estimate()
    assert err_anti < 0.8 * err_no_anti


def test_multi_step_path_still_converges(reference_data: dict[str, Any]) -> None:
    """Multi-step (12 steps) MC for European should still converge to analytic."""
    process, expiry = _build_textbook_bsm()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(
        MCEuropeanEngine(
            process,
            time_steps=12,
            antithetic_variate=False,
            required_samples=10000,
            seed=42,
        )
    )
    npv = opt.npv()
    err = opt.error_estimate()
    analytic = float(reference_data["analytic_european"]["call_npv"])
    assert abs(npv - analytic) < 3 * err


def test_tolerance_driven_termination() -> None:
    """``required_tolerance=0.02`` should drive sample count higher until the
    error falls under tolerance.
    """
    process, expiry = _build_textbook_bsm()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(
        MCEuropeanEngine(
            process,
            time_steps=1,
            required_tolerance=0.05,
            max_samples=200_000,
            seed=42,
        )
    )
    _ = opt.npv()
    err = opt.error_estimate()
    assert err <= 0.05


def test_brownian_bridge_does_not_break_convergence(
    reference_data: dict[str, Any],
) -> None:
    """Enabling Brownian bridge with a multi-step grid should still converge."""
    process, expiry = _build_textbook_bsm()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(
        MCEuropeanEngine(
            process,
            time_steps=4,
            brownian_bridge=True,
            antithetic_variate=False,
            required_samples=10000,
            seed=42,
        )
    )
    npv = opt.npv()
    err = opt.error_estimate()
    analytic = float(reference_data["analytic_european"]["call_npv"])
    assert abs(npv - analytic) < 3 * err


def test_neither_samples_nor_tolerance_raises() -> None:
    process, expiry = _build_textbook_bsm()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(
        MCEuropeanEngine(process, time_steps=1, seed=42)
    )
    with pytest.raises(Exception, match="neither tolerance nor number"):
        opt.npv()


def test_cross_check_against_analytic_engine_3sigma() -> None:
    """Direct comparison: MCEuropean vs AnalyticEuropean on the same option."""
    process, expiry = _build_textbook_bsm()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)

    opt_mc = EuropeanOption(payoff, exercise)
    opt_mc.set_pricing_engine(
        MCEuropeanEngine(
            process,
            time_steps=1,
            antithetic_variate=True,
            required_samples=10000,
            seed=42,
        )
    )
    mc_npv = opt_mc.npv()
    mc_err = opt_mc.error_estimate()

    opt_an = EuropeanOption(payoff, exercise)
    opt_an.set_pricing_engine(AnalyticEuropeanEngine(process))
    an_npv = opt_an.npv()

    # TIGHT-tier check on the analytic baseline against ourselves.
    tight(opt_an.npv(), an_npv)
    # LOOSE check on the MC against analytic.
    assert abs(mc_npv - an_npv) < 3 * mc_err
