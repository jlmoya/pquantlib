"""Cross-validate TwoAssetBarrierOption + AnalyticTwoAssetBarrierEngine.

Probe source: migration-harness/cpp/probes/cluster_w4a/probe.cpp
Reference:    migration-harness/references/cluster/w4a.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.barrier_option import BarrierType
from pquantlib.instruments.two_asset_barrier_option import (
    TwoAssetBarrierOption,
    TwoAssetBarrierOptionArguments,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.exoticoptions.analytic_two_asset_barrier_engine import (
    AnalyticTwoAssetBarrierEngine,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.time_unit import TimeUnit

# conftest helper for building the GBSM processes (rate=0.05, div=0.02).
from .conftest import make_bsm_for_two_asset


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w4a")


@pytest.fixture(scope="module")
def exercise(today: Date, calendar: Calendar) -> EuropeanExercise:
    return EuropeanExercise(calendar.advance(today, 1, TimeUnit.Years))


def test_two_asset_barrier_construction_round_trip(
    today: Date, exercise: EuropeanExercise
) -> None:
    """Construction + inspector round-trip."""
    po = PlainVanillaPayoff(OptionType.Call, 100.0)
    opt = TwoAssetBarrierOption(BarrierType.UpOut, 130.0, po, exercise)
    assert opt.barrier_type() == BarrierType.UpOut
    assert opt.barrier() == 130.0
    payoff = opt.payoff()
    assert isinstance(payoff, PlainVanillaPayoff)
    assert payoff.strike() == 100.0
    assert opt.exercise() is exercise
    assert opt.is_expired() is False


def test_two_asset_barrier_arguments_validate(exercise: EuropeanExercise) -> None:
    """``validate()`` requires barrier_type and barrier."""
    args = TwoAssetBarrierOptionArguments()
    args.payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    args.exercise = exercise
    args.barrier_type = None
    args.barrier = 130.0
    with pytest.raises(LibraryException):
        args.validate()
    args.barrier_type = BarrierType.UpOut
    args.barrier = None
    with pytest.raises(LibraryException):
        args.validate()
    args.barrier = 130.0
    args.validate()  # no raise


def test_two_asset_barrier_upout_call_npv(
    cpp_ref: dict[str, Any],
    today: Date,
    exercise: EuropeanExercise,
) -> None:
    """UpOut call NPV matches Heynen-Kat closed form.

    Tolerance: 1e-4 absolute, 1e-5 relative — scipy's bivariate CDF
    (Genz-Bretz) and the C++ Drezner 1978 quadrature both target
    6-decimal-place precision; their final-bit disagreement
    accumulates through the Heynen-Kat formula's two ``M(.,.)``
    calls per branch + power-law drifts.
    """
    ref = cpp_ref["two_asset_barrier"]
    p1 = make_bsm_for_two_asset(today, 100.0, 0.05, 0.02, 0.20)
    p2 = make_bsm_for_two_asset(today, 110.0, 0.05, 0.02, 0.25)
    po = PlainVanillaPayoff(OptionType.Call, ref["strike"])
    opt = TwoAssetBarrierOption(
        BarrierType.UpOut, ref["barrier_upout"], po, exercise
    )
    rho_quote = SimpleQuote(ref["rho"])
    opt.set_pricing_engine(
        AnalyticTwoAssetBarrierEngine(p1, p2, rho_quote)
    )
    py_npv = opt.npv()
    tolerance.custom(
        py_npv,
        ref["npv_upout_call"],
        abs_tol=1e-4,
        rel_tol=1e-5,
        reason=(
            "scipy.multivariate_normal.cdf (Genz-Bretz) vs C++ "
            "BivariateCumulativeNormalDistributionDr78 (Drezner 1978 6dp) — "
            "both at 6dp design precision; final-bit disagreement accumulates "
            "through Heynen-Kat's two bivariate-CDF calls per branch."
        ),
    )


def test_two_asset_barrier_downout_put_npv(
    cpp_ref: dict[str, Any],
    today: Date,
    exercise: EuropeanExercise,
) -> None:
    """DownOut put NPV matches Heynen-Kat closed form (~1e-5 abs)."""
    ref = cpp_ref["two_asset_barrier"]
    p1 = make_bsm_for_two_asset(today, 100.0, 0.05, 0.02, 0.20)
    p2 = make_bsm_for_two_asset(today, 110.0, 0.05, 0.02, 0.25)
    po = PlainVanillaPayoff(OptionType.Put, ref["strike"])
    opt = TwoAssetBarrierOption(BarrierType.DownOut, 90.0, po, exercise)
    rho_quote = SimpleQuote(ref["rho"])
    opt.set_pricing_engine(
        AnalyticTwoAssetBarrierEngine(p1, p2, rho_quote)
    )
    py_npv = opt.npv()
    tolerance.custom(
        py_npv,
        ref["npv_downout_put"],
        abs_tol=1e-4,
        rel_tol=1e-5,
        reason=(
            "scipy Genz-Bretz vs C++ Drezner 1978 6dp bivariate CDF (see "
            "test_two_asset_barrier_upout_call_npv for justification)."
        ),
    )


def test_two_asset_barrier_triggered_raises(
    today: Date, exercise: EuropeanExercise
) -> None:
    """If barrier already touched (UpOut with S2 > barrier), engine raises."""
    p1 = make_bsm_for_two_asset(today, 100.0, 0.05, 0.02, 0.20)
    # S2 = 140 already above the UpOut barrier of 130 — triggered.
    p2 = make_bsm_for_two_asset(today, 140.0, 0.05, 0.02, 0.25)
    po = PlainVanillaPayoff(OptionType.Call, 100.0)
    opt = TwoAssetBarrierOption(BarrierType.UpOut, 130.0, po, exercise)
    opt.set_pricing_engine(
        AnalyticTwoAssetBarrierEngine(p1, p2, SimpleQuote(0.5))
    )
    with pytest.raises(LibraryException):
        opt.npv()
