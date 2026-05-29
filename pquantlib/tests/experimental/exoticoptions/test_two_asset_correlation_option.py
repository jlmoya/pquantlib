"""Cross-validate TwoAssetCorrelationOption + AnalyticTwoAssetCorrelationEngine.

Probe source: migration-harness/cpp/probes/cluster_w4a/probe.cpp
Reference:    migration-harness/references/cluster/w4a.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.two_asset_correlation_option import (
    TwoAssetCorrelationOption,
    TwoAssetCorrelationOptionArguments,
)
from pquantlib.payoffs import NullPayoff, OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.exoticoptions.analytic_two_asset_correlation_engine import (
    AnalyticTwoAssetCorrelationEngine,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.time_unit import TimeUnit

from .conftest import make_bsm_for_two_asset


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w4a")


@pytest.fixture(scope="module")
def exercise(today: Date, calendar: Calendar) -> EuropeanExercise:
    return EuropeanExercise(calendar.advance(today, 1, TimeUnit.Years))


def test_two_asset_correlation_construction_round_trip(
    exercise: EuropeanExercise,
) -> None:
    """Construction + inspectors."""
    opt = TwoAssetCorrelationOption(OptionType.Call, 100.0, 110.0, exercise)
    assert opt.strike2() == 110.0
    payoff = opt.payoff()
    # C++ wraps strike1 in a PlainVanillaPayoff(option_type, strike1).
    assert isinstance(payoff, PlainVanillaPayoff)
    assert payoff.strike() == 100.0
    assert payoff.option_type() == OptionType.Call
    assert opt.exercise() is exercise


def test_two_asset_correlation_arguments_validate(
    exercise: EuropeanExercise,
) -> None:
    """``validate()`` requires X2."""
    args = TwoAssetCorrelationOptionArguments()
    args.payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    args.exercise = exercise
    args.x2 = None
    with pytest.raises(LibraryException):
        args.validate()
    args.x2 = 100.0
    args.validate()  # no raise


def test_two_asset_correlation_call_npv(
    cpp_ref: dict[str, Any],
    today: Date,
    exercise: EuropeanExercise,
) -> None:
    """Call NPV matches Zhang closed form."""
    ref = cpp_ref["two_asset_correlation"]
    p1 = make_bsm_for_two_asset(today, 100.0, 0.05, 0.02, 0.20)
    p2 = make_bsm_for_two_asset(today, 105.0, 0.05, 0.02, 0.30)
    opt = TwoAssetCorrelationOption(
        OptionType.Call, ref["strike1"], ref["strike2"], exercise
    )
    rho_quote = SimpleQuote(ref["rho"])
    opt.set_pricing_engine(
        AnalyticTwoAssetCorrelationEngine(p1, p2, rho_quote)
    )
    py_npv = opt.npv()
    tolerance.custom(
        py_npv,
        ref["npv_call"],
        abs_tol=1e-4,
        rel_tol=1e-5,
        reason=(
            "scipy.multivariate_normal.cdf (Genz-Bretz) vs C++ "
            "BivariateCumulativeNormalDistributionDr78 (Drezner 1978 6dp) "
            "— both at 6dp design precision."
        ),
    )


def test_two_asset_correlation_put_npv(
    cpp_ref: dict[str, Any],
    today: Date,
    exercise: EuropeanExercise,
) -> None:
    """Put NPV matches Zhang closed form."""
    ref = cpp_ref["two_asset_correlation"]
    p1 = make_bsm_for_two_asset(today, 100.0, 0.05, 0.02, 0.20)
    p2 = make_bsm_for_two_asset(today, 105.0, 0.05, 0.02, 0.30)
    opt = TwoAssetCorrelationOption(
        OptionType.Put, ref["strike1"], ref["strike2"], exercise
    )
    rho_quote = SimpleQuote(ref["rho"])
    opt.set_pricing_engine(
        AnalyticTwoAssetCorrelationEngine(p1, p2, rho_quote)
    )
    py_npv = opt.npv()
    tolerance.custom(
        py_npv,
        ref["npv_put"],
        abs_tol=1e-4,
        rel_tol=1e-5,
        reason=(
            "scipy vs C++ Drezner 1978 6dp bivariate CDF (see "
            "test_two_asset_correlation_call_npv for justification)."
        ),
    )


def test_two_asset_correlation_invalid_payoff_rejected(
    today: Date, exercise: EuropeanExercise
) -> None:
    """Engine rejects non-PlainVanilla payoffs.

    We can't construct ``TwoAssetCorrelationOption`` with a NullPayoff
    directly (its constructor builds a PlainVanillaPayoff), so we mutate
    the args after setup_arguments.
    """
    p1 = make_bsm_for_two_asset(today, 100.0, 0.05, 0.02, 0.20)
    p2 = make_bsm_for_two_asset(today, 105.0, 0.05, 0.02, 0.30)
    engine = AnalyticTwoAssetCorrelationEngine(p1, p2, SimpleQuote(0.4))
    args = engine.get_arguments()
    args.payoff = NullPayoff()
    args.exercise = exercise
    args.x2 = 100.0
    with pytest.raises(LibraryException):
        engine.calculate()
