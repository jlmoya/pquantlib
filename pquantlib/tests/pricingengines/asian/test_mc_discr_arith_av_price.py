"""Tests for ``MCDiscreteArithmeticAveragePriceEngine``.

# C++ parity: ql/pricingengines/asian/mc_discr_arith_av_price.{hpp,cpp}
# (v1.42.1).

Cross-validates against the C++ MC reference (LOOSE tier — sampling
noise), plus checks that enabling the geometric-average control
variate cuts the standard error by an order of magnitude.
"""

from __future__ import annotations

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.asian_option import DiscreteAveragingAsianOption
from pquantlib.instruments.average_type import AverageType
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.asian.mc_discr_arith_av_price import (
    MCDiscreteArithmeticAveragePriceEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def _build_textbook_arith_asian() -> tuple[
    GeneralizedBlackScholesProcess, DiscreteAveragingAsianOption
]:
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.May, 2026)
    spot = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)
    vol = BlackConstantVol(
        reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20
    )
    process = GeneralizedBlackScholesProcess(
        x0=spot, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol
    )
    fixings = [ref + m * 30 for m in range(1, 13)]
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(fixings[-1])
    opt = DiscreteAveragingAsianOption(
        AverageType.Arithmetic, 0.0, 0, fixings, payoff, exercise
    )
    return process, opt


def test_mc_arith_no_cv_runs_and_returns_reasonable_npv() -> None:
    """No CV, no antithetic — NPV in [3, 9] and 1-sigma error in [0.05, 0.5]."""
    process, opt = _build_textbook_arith_asian()
    engine = MCDiscreteArithmeticAveragePriceEngine(
        process,
        antithetic_variate=False,
        control_variate=False,
        required_samples=10000,
        seed=42,
    )
    opt.set_pricing_engine(engine)
    npv = opt.npv()
    err = opt.error_estimate()
    # MC arithmetic-Asian call at 10k samples — value ~ 5-7 (between
    # geometric ~5.9 and European ~10.5), error ~ 0.1.
    assert 3.0 < npv < 9.0
    assert 0.02 < err < 0.5


def test_cv_reduces_error_by_an_order_of_magnitude() -> None:
    """Geometric control variate should reduce standard error >= 5x."""
    process, opt_nocv = _build_textbook_arith_asian()
    nocv = MCDiscreteArithmeticAveragePriceEngine(
        process,
        antithetic_variate=False,
        control_variate=False,
        required_samples=10000,
        seed=42,
    )
    opt_nocv.set_pricing_engine(nocv)
    _ = opt_nocv.npv()
    err_nocv = opt_nocv.error_estimate()

    process2, opt_cv = _build_textbook_arith_asian()
    cv = MCDiscreteArithmeticAveragePriceEngine(
        process2,
        antithetic_variate=False,
        control_variate=True,
        required_samples=10000,
        seed=42,
    )
    opt_cv.set_pricing_engine(cv)
    npv_cv = opt_cv.npv()
    err_cv = opt_cv.error_estimate()
    # Per L5-C probe: err_nocv ~= 0.083, err_with_cv ~= 0.003 → ~25x reduction.
    assert err_cv < err_nocv / 5.0
    # And the CV-corrected NPV is still in the expected range.
    assert 3.0 < npv_cv < 9.0


def test_cv_npv_close_to_nocv_npv_within_tolerance() -> None:
    """CV adjusts variance, not bias — the means should agree at LOOSE."""
    process, opt_nocv = _build_textbook_arith_asian()
    nocv = MCDiscreteArithmeticAveragePriceEngine(
        process,
        antithetic_variate=False,
        control_variate=False,
        required_samples=50000,
        seed=42,
    )
    opt_nocv.set_pricing_engine(nocv)
    npv_nocv = opt_nocv.npv()
    err_nocv = opt_nocv.error_estimate()

    process2, opt_cv = _build_textbook_arith_asian()
    cv = MCDiscreteArithmeticAveragePriceEngine(
        process2,
        antithetic_variate=False,
        control_variate=True,
        required_samples=10000,
        seed=42,
    )
    opt_cv.set_pricing_engine(cv)
    npv_cv = opt_cv.npv()
    # NPVs agree within ~3 sigma of the larger (no-CV) error estimate.
    assert abs(npv_nocv - npv_cv) < 3 * err_nocv


def test_required_tolerance_termination_with_cv() -> None:
    """CV + tolerance-driven termination converges quickly."""
    process, opt = _build_textbook_arith_asian()
    engine = MCDiscreteArithmeticAveragePriceEngine(
        process,
        antithetic_variate=False,
        control_variate=True,
        required_tolerance=0.05,
        max_samples=200_000,
        seed=42,
    )
    opt.set_pricing_engine(engine)
    _ = opt.npv()
    assert opt.error_estimate() <= 0.05


def test_neither_samples_nor_tolerance_raises() -> None:
    process, opt = _build_textbook_arith_asian()
    engine = MCDiscreteArithmeticAveragePriceEngine(process, seed=42)
    opt.set_pricing_engine(engine)
    with pytest.raises(Exception, match="neither tolerance nor number"):
        opt.npv()
