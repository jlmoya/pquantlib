"""Tests for VanillaOption.implied_volatility (closes Phase 3 carve-out).

# C++ parity: ql/instruments/vanillaoption.cpp ``impliedVolatility`` +
# ql/instruments/impliedvolatility.cpp ``ImpliedVolatilityHelper``
# @ v1.42.1.

Cross-validates against ``implied_vol_european`` section of
``migration-harness/references/cluster/l5d.json``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_european_engine import (
    AnalyticEuropeanEngine,
)
from pquantlib.pricingengines.vanilla.fd_black_scholes_vanilla_engine import (
    FdBlackScholesVanillaEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import custom, loose
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/l5d")


def _build_process_and_expiry() -> tuple[GeneralizedBlackScholesProcess, Date]:
    """Textbook BSM: S=K=100, r=5%, q=0%, sigma=20%, T=1y."""
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    expiry = ref + 365
    spot_q = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20)
    process = GeneralizedBlackScholesProcess(x0=spot_q, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol)
    return process, expiry


def test_european_call_recovers_input_vol(reference_data: dict[str, Any]) -> None:
    """Round-trip: build a European Call at sigma=0.20, get NPV,
    invert via implied_volatility — must return 0.20.

    LOOSE-equivalent: the C++ probe uses ``accuracy=1e-6`` and gets
    0.19999995701 (~5e-8 from 0.20). We allow LOOSE here to absorb
    Brent's iteration tail.
    """
    process, expiry = _build_process_and_expiry()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = VanillaOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process))
    target_npv = opt.npv()

    # Sanity vs C++ reference for the target NPV.
    cpp_target = float(reference_data["implied_vol_european"]["npv_at_vol20"])
    loose(target_npv, cpp_target)

    recovered = opt.implied_volatility(
        target_npv,
        process,
        accuracy=1e-6,
        max_evaluations=100,
        min_vol=0.001,
        max_vol=4.0,
    )
    # Tight enough — Brent at 1e-6 accuracy reliably converges to 1e-7.
    custom(
        recovered,
        0.20,
        abs_tol=1e-5,
        rel_tol=1e-5,
        reason="Brent set with accuracy=1e-6 on a smooth-monotone vega curve "
        "typically converges to ~5e-8; LOOSE tier is sufficient.",
    )


def test_european_put_recovers_input_vol() -> None:
    """Put implied vol round-trip."""
    process, expiry = _build_process_and_expiry()
    payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = VanillaOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process))
    target_npv = opt.npv()

    recovered = opt.implied_volatility(
        target_npv,
        process,
        accuracy=1e-6,
        max_evaluations=100,
        min_vol=0.001,
        max_vol=4.0,
    )
    custom(
        recovered,
        0.20,
        abs_tol=1e-5,
        rel_tol=1e-5,
        reason="Brent at 1e-6 accuracy on a smooth-monotone vega curve.",
    )


def test_implied_vol_at_different_strike() -> None:
    """ATM call gets correctly inverted for a different strike too."""
    process, expiry = _build_process_and_expiry()
    payoff = PlainVanillaPayoff(OptionType.Call, 120.0)
    exercise = EuropeanExercise(expiry)
    opt = VanillaOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process))
    target_npv = opt.npv()

    recovered = opt.implied_volatility(
        target_npv,
        process,
        accuracy=1e-6,
        max_evaluations=100,
        min_vol=0.001,
        max_vol=4.0,
    )
    custom(
        recovered,
        0.20,
        abs_tol=1e-5,
        rel_tol=1e-5,
        reason="OTM call vega is monotonic in vol — Brent converges as fast.",
    )


def test_implied_vol_american_uses_fd_engine() -> None:
    """For American exercise the helper switches to the FD engine.

    Round-trip: price an American put at sigma=0.20 with the FD
    engine to get its NPV, then ask implied_volatility (which builds
    its own FD engine internally) to recover sigma. Tolerance here
    is **dominated by FD discretization** (`xGrid=100, tGrid=100`
    defaults — 50x slower per axis than the convergence test, but
    accurate to ~1e-3 absolute price), so the recovered vol is
    only accurate to ~5e-3.
    """
    process, expiry = _build_process_and_expiry()
    payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    ref_date = process.risk_free_rate().reference_date()
    exercise = AmericanExercise(ref_date, expiry)

    # Step 1: build the option, price it once with the FD engine to
    # obtain the target NPV.
    opt = VanillaOption(payoff, exercise)
    opt.set_pricing_engine(FdBlackScholesVanillaEngine(process))
    target_npv = opt.npv()

    # Step 2: invert. The helper builds its own FD engine internally —
    # so the function and its inversion both use the same numerical
    # method (and the same FD error cancels out).
    recovered = opt.implied_volatility(
        target_npv,
        process,
        accuracy=1e-4,
        max_evaluations=100,
        min_vol=0.001,
        max_vol=4.0,
    )
    custom(
        recovered,
        0.20,
        abs_tol=5e-3,
        rel_tol=5e-3,
        reason="American implied vol uses the FD engine; FD-on-FD inversion limits accuracy to ~5e-3.",
    )


def test_implied_vol_max_evaluations_exhausted() -> None:
    """If max_evaluations is too small, Brent should fail."""
    process, expiry = _build_process_and_expiry()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = VanillaOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process))
    target_npv = opt.npv()

    with pytest.raises(LibraryException):
        # Hard-limit evaluations to a tiny number.
        opt.implied_volatility(
            target_npv,
            process,
            accuracy=1e-12,
            max_evaluations=3,
            min_vol=0.001,
            max_vol=4.0,
        )
