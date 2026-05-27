"""AnalyticHestonEngine tests + HestonModelHelper round-trip via the engine.

Cross-validates against ``migration-harness/references/cluster/l4c.json``.

C++ parity: ql/pricingengines/vanilla/analytichestonengine.{hpp,cpp}
            @ v1.42.1 (099987f0) — Gatheral form, gauss-Laguerre order 144.

Tolerance choice:

* Engine NPV vs C++ reference: **LOOSE** (abs_tol=1e-8, rel_tol=1e-8).
  pquantlib uses scipy.integrate.quad (QUADPACK adaptive) while C++
  uses Gauss-Laguerre quadrature. Both converge to ~1e-8 absolute
  accuracy on the standard Heston regime but on different node grids
  — diverging at the 7th-8th significant figure. The L4-C design
  explicitly accepts this trade-off (see analytic_heston_engine.py
  docstring).
* Put-Call parity: TIGHT (algebraic identity; should hold to float64
  noise plus 2x the quad noise → still well inside tight).
* Helper round-trip ``model_value`` vs engine NPV: tight (both go
  through the same engine code path).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.european_option import EuropeanOption
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.models.calibration_helper import CalibrationErrorType
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.models.equity.heston_model_helper import HestonModelHelper
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_heston_engine import (
    AnalyticHestonEngine,
)
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose, tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

# Albrecher-Mayer-Schoutens-Tistaert canonical testbed.
_S = 100.0
_R = 0.05
_Q = 0.0
_V0 = 0.04
_KAPPA = 2.0
_THETA = 0.04
_SIGMA = 0.3
_RHO = -0.7
_T = 1.0


@pytest.fixture
def cpp_refs() -> dict[str, Any]:
    return load_reference("cluster/l4c")


@pytest.fixture
def setup() -> tuple[HestonModel, Date]:
    """Build the standard Heston model + expiry date."""
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    expiry = ref + 365  # exactly T=1.0 under Actual/365 Fixed
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=_R, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=_Q, day_counter=dc)
    process = HestonProcess(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=SimpleQuote(_S),
        v0=_V0,
        kappa=_KAPPA,
        theta=_THETA,
        sigma=_SIGMA,
        rho=_RHO,
    )
    return HestonModel(process), expiry


def _price(model: HestonModel, expiry: Date, strike: float, option_type: OptionType) -> float:
    """Price a single European vanilla under Heston via the engine."""
    engine = AnalyticHestonEngine(model)
    payoff = PlainVanillaPayoff(option_type, strike)
    exercise = EuropeanExercise(expiry)
    option = EuropeanOption(payoff, exercise)
    option.set_pricing_engine(engine)
    return option.npv()


def test_call_atm(setup: tuple[HestonModel, Date], cpp_refs: dict[str, Any]) -> None:
    """ATM Call price vs C++ Gauss-Laguerre reference (LOOSE)."""
    model, expiry = setup
    e = cpp_refs["heston_engine"]
    loose(_price(model, expiry, 100.0, OptionType.Call), e["call_atm"])


def test_put_atm(setup: tuple[HestonModel, Date], cpp_refs: dict[str, Any]) -> None:
    """ATM Put price vs C++ reference (LOOSE)."""
    model, expiry = setup
    e = cpp_refs["heston_engine"]
    loose(_price(model, expiry, 100.0, OptionType.Put), e["put_atm"])


def test_call_otm_low(setup: tuple[HestonModel, Date], cpp_refs: dict[str, Any]) -> None:
    """Deep-ITM Call (K=80) price (LOOSE)."""
    model, expiry = setup
    e = cpp_refs["heston_engine"]
    loose(_price(model, expiry, 80.0, OptionType.Call), e["call_otm_low"])


def test_call_otm_high(
    setup: tuple[HestonModel, Date], cpp_refs: dict[str, Any]
) -> None:
    """OTM Call (K=120) price (LOOSE)."""
    model, expiry = setup
    e = cpp_refs["heston_engine"]
    loose(_price(model, expiry, 120.0, OptionType.Call), e["call_otm_high"])


def test_put_otm_low(setup: tuple[HestonModel, Date], cpp_refs: dict[str, Any]) -> None:
    """OTM Put (K=80) price (LOOSE)."""
    model, expiry = setup
    e = cpp_refs["heston_engine"]
    loose(_price(model, expiry, 80.0, OptionType.Put), e["put_otm_low"])


def test_put_otm_high(setup: tuple[HestonModel, Date], cpp_refs: dict[str, Any]) -> None:
    """Deep-ITM Put (K=120) price (LOOSE)."""
    model, expiry = setup
    e = cpp_refs["heston_engine"]
    loose(_price(model, expiry, 120.0, OptionType.Put), e["put_otm_high"])


def test_put_call_parity_atm(setup: tuple[HestonModel, Date]) -> None:
    """C - P = S*df_q - K*df_r at ATM.

    Algebraic identity that holds for any consistent stochastic-vol model.
    Should match to TIGHT — two engine integrations times ~1e-10 quad noise.
    """
    model, expiry = setup
    call = _price(model, expiry, 100.0, OptionType.Call)
    put = _price(model, expiry, 100.0, OptionType.Put)
    # S * df_q - K * df_r with r=5%, q=0, T=1, S=K=100:
    #   = 100*1 - 100*exp(-0.05) = 100 * (1 - exp(-0.05))
    expected = _S - 100.0 * math.exp(-_R * _T)
    tight(
        call - put,
        expected,
        reason="put-call parity: C - P = S*df_q - K*df_r at q=0",
    )


def test_helper_model_value_matches_engine_npv(
    setup: tuple[HestonModel, Date],
) -> None:
    """HestonModelHelper.model_value() routes through the engine.

    # C++ parity: hestonmodelhelper.cpp:80-84 — modelValue() sets the
    # engine on the option and returns option.NPV().

    Verifies the helper observes the engine attachment correctly.
    """
    model, expiry = setup
    del expiry  # the helper builds its own from the period

    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=_R, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=_Q, day_counter=dc)

    helper = HestonModelHelper(
        maturity=Period(12, TimeUnit.Months),
        calendar=NullCalendar(),
        s0=SimpleQuote(_S),
        strike_price=100.0,
        volatility=SimpleQuote(0.20),
        risk_free_rate=rf,
        dividend_yield=div,
        calibration_error_type=CalibrationErrorType.RelativePriceError,
    )

    engine = AnalyticHestonEngine(model)
    helper.set_pricing_engine(engine)

    # Helper picks Put at ATM (C++ probe confirmed earlier).
    helper.calculate()
    model_val_via_helper = helper.model_value()
    direct = _price(model, ref + 365, 100.0, OptionType.Put)
    tight(
        model_val_via_helper,
        direct,
        reason="helper just wraps the same engine call; should agree",
    )


def test_helper_calibration_error_matches_cpp(
    cpp_refs: dict[str, Any],
) -> None:
    """End-to-end: market value (Black 20%) vs model value (Heston engine).

    The C++ probe emits the resulting relative price error at the
    canonical testbed; we cross-validate it through the full chain.
    """
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=_R, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=_Q, day_counter=dc)
    process = HestonProcess(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=SimpleQuote(_S),
        v0=_V0,
        kappa=_KAPPA,
        theta=_THETA,
        sigma=_SIGMA,
        rho=_RHO,
    )
    model = HestonModel(process)
    engine = AnalyticHestonEngine(model)

    helper = HestonModelHelper(
        maturity=Period(12, TimeUnit.Months),
        calendar=NullCalendar(),
        s0=SimpleQuote(_S),
        strike_price=100.0,
        volatility=SimpleQuote(0.20),
        risk_free_rate=rf,
        dividend_yield=div,
        calibration_error_type=CalibrationErrorType.RelativePriceError,
    )
    helper.set_pricing_engine(engine)

    h = cpp_refs["heston_helper"]
    loose(
        helper.model_value(),
        h["model_value"],
        reason="engine NPV via helper; LOOSE for quad-vs-Laguerre",
    )
    loose(
        helper.calibration_error(),
        h["calibration_error"],
        reason="(market - model)/market; LOOSE because model_value is LOOSE",
    )


def test_engine_number_of_evaluations_nonzero(
    setup: tuple[HestonModel, Date],
) -> None:
    """The engine's evaluation counter is incremented during calculate.

    # C++ parity: analytichestonengine.cpp:721-723 — numberOfEvaluations()
    # reads the accumulated count. scipy.quad provides ``neval`` in the
    # info dict.
    """
    model, expiry = setup
    engine = AnalyticHestonEngine(model)
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    option = EuropeanOption(payoff, exercise)
    option.set_pricing_engine(engine)
    option.npv()
    assert engine.number_of_evaluations() > 0


def test_engine_rejects_non_european(setup: tuple[HestonModel, Date]) -> None:
    """The engine raises on non-European exercise."""
    model, expiry = setup
    engine = AnalyticHestonEngine(model)
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    # AmericanExercise spans [earliest_date, expiry].
    exercise = AmericanExercise(
        earliest_date=Date.from_ymd(15, Month.June, 2026),
        latest_date=expiry,
    )
    option = VanillaOption(payoff, exercise)
    option.set_pricing_engine(engine)
    with pytest.raises(LibraryException, match="not a European option"):
        option.npv()


def test_engine_holds_model_reference(setup: tuple[HestonModel, Date]) -> None:
    """``model()`` returns the model passed at construction."""
    model, _expiry = setup
    engine = AnalyticHestonEngine(model)
    assert engine.model() is model
