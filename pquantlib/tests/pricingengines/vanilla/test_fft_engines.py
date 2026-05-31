"""Tests for the Carr-Madan FFT engines (vanilla + variance-gamma).

# C++ parity: ql/experimental/variancegamma/{fftengine,fftvanillaengine,
# fftvariancegammaengine}.hpp.

Cross-validates against ``migration-harness/references/cluster/w7a.json``:
FFTVanillaEngine reproduces AnalyticEuropeanEngine and FFTVarianceGammaEngine
reproduces the analytic VG engine, both to LOOSE (FFT-grid discretization).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.experimental.variancegamma.variance_gamma_process import (
    VarianceGammaProcess,
)
from pquantlib.instruments.european_option import EuropeanOption
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_variance_gamma_engine import (
    VarianceGammaEngine,
)
from pquantlib.pricingengines.vanilla.fft_vanilla_engine import FFTVanillaEngine
from pquantlib.pricingengines.vanilla.fft_variance_gamma_engine import (
    FFTVarianceGammaEngine,
)
from pquantlib.processes.black_scholes_merton_process import (
    BlackScholesMertonProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/w7a")


# --- FFT vanilla (Black-Scholes) --------------------------------------------


def _build_bs() -> tuple[BlackScholesMertonProcess, Date]:
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.January, 2024)
    spot = SimpleQuote(100.0)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.02, day_counter=dc)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.25)
    process = BlackScholesMertonProcess(
        x0=spot, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol
    )
    return process, ref + 365


def test_fft_vanilla_calls_match_analytic(reference_data: dict[str, Any]) -> None:
    """Precalculated FFT call NPVs reproduce the analytic European engine.

    LOOSE: Carr-Madan FFT grid discretization.
    """
    process, expiry = _build_bs()
    exercise = EuropeanExercise(expiry)
    strikes = [90.0, 100.0, 110.0]
    names = ["fft_vanilla_call_90", "fft_vanilla_call_100", "fft_vanilla_call_110"]
    options = [
        EuropeanOption(PlainVanillaPayoff(OptionType.Call, k), exercise) for k in strikes
    ]
    engine = FFTVanillaEngine(process)
    for opt in options:
        opt.set_pricing_engine(engine)
    engine.precalculate(list(options))
    for opt, name in zip(options, names, strict=True):
        loose(opt.npv(), float(reference_data[name]))


def test_fft_vanilla_put_matches_analytic(reference_data: dict[str, Any]) -> None:
    """Precalculated FFT put NPV (put-call parity) reproduces the analytic value.

    LOOSE: Carr-Madan FFT grid discretization.
    """
    process, expiry = _build_bs()
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(PlainVanillaPayoff(OptionType.Put, 100.0), exercise)
    engine = FFTVanillaEngine(process)
    opt.set_pricing_engine(engine)
    engine.precalculate([opt])
    loose(opt.npv(), float(reference_data["fft_vanilla_put_100"]))


def test_fft_vanilla_uncached_single_option(reference_data: dict[str, Any]) -> None:
    """A single option priced WITHOUT precalculate triggers calculateUncached.

    LOOSE: the engine clones itself, precalculates a 1-option list, reprices.
    """
    process, expiry = _build_bs()
    exercise = EuropeanExercise(expiry)
    opt = VanillaOption(PlainVanillaPayoff(OptionType.Call, 100.0), exercise)
    engine = FFTVanillaEngine(process)
    opt.set_pricing_engine(engine)
    # No precalculate — calculate() must fall back to the uncached path.
    loose(opt.npv(), float(reference_data["fft_vanilla_call_100"]))


# --- FFT variance-gamma ------------------------------------------------------


def _build_vg() -> tuple[VarianceGammaProcess, Date]:
    dc = Actual360()
    ref = Date.from_ymd(15, Month.January, 2024)
    spot = SimpleQuote(6000.0)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.00, day_counter=dc)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    return VarianceGammaProcess(spot, div, rf, 0.20, 0.05, -0.50), ref + 360


def test_fft_vg_calls_match_analytic(reference_data: dict[str, Any]) -> None:
    """Precalculated FFT VG call NPVs reproduce the analytic VG engine.

    LOOSE: FFT discretization + VG characteristic function.
    """
    process, expiry = _build_vg()
    exercise = EuropeanExercise(expiry)
    strikes = [5550.0, 6000.0, 6500.0]
    names = ["vg_fft_call_5550", "vg_fft_call_6000", "vg_fft_call_6500"]
    options = [
        EuropeanOption(PlainVanillaPayoff(OptionType.Call, k), exercise) for k in strikes
    ]
    engine = FFTVarianceGammaEngine(process)
    for opt in options:
        opt.set_pricing_engine(engine)
    engine.precalculate(list(options))
    for opt, name in zip(options, names, strict=True):
        loose(opt.npv(), float(reference_data[name]))


def test_fft_vg_put_matches_analytic(reference_data: dict[str, Any]) -> None:
    """Precalculated FFT VG put NPV reproduces the analytic VG value.

    LOOSE: FFT discretization + put-call parity.
    """
    process, expiry = _build_vg()
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(PlainVanillaPayoff(OptionType.Put, 5550.0), exercise)
    engine = FFTVarianceGammaEngine(process)
    opt.set_pricing_engine(engine)
    engine.precalculate([opt])
    loose(opt.npv(), float(reference_data["vg_fft_put_5550"]))


def test_fft_vg_matches_analytic_vg_directly() -> None:
    """FFT VG call agrees with the analytic VG engine on the same option.

    Tolerance exception (NOT LOOSE): this compares two *different*
    numerical methods on the same price — the Carr-Madan FFT grid vs the
    Madan-Carr-Chang exact quadrature. The C++ test-suite (variancegamma.cpp
    line 127) uses an absolute tolerance of 0.01 for exactly this
    cross-engine check; the FFT-grid discretization error (~1.2e-3 here)
    exceeds the LOOSE rel_tol=1e-8 but is well inside the C++ 0.01 bound.
    The tight FFT-Python-vs-FFT-C++ agreement is covered separately by the
    probe-based LOOSE tests above.
    """
    process, expiry = _build_vg()
    exercise = EuropeanExercise(expiry)
    payoff = PlainVanillaPayoff(OptionType.Call, 6000.0)

    an_opt = EuropeanOption(payoff, exercise)
    an_opt.set_pricing_engine(VarianceGammaEngine(process))
    an_value = an_opt.npv()

    fft_opt = EuropeanOption(payoff, exercise)
    fft_engine = FFTVarianceGammaEngine(process)
    fft_opt.set_pricing_engine(fft_engine)
    fft_engine.precalculate([fft_opt])

    assert abs(fft_opt.npv() - an_value) < 0.01
