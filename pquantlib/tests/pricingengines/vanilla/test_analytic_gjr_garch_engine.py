"""AnalyticGjrGarchEngine tests.

Cross-validates against ``migration-harness/references/cluster/w1d.json``.

C++ parity: ql/pricingengines/vanilla/analyticgjrgarchengine.{hpp,cpp}
            @ v1.42.1 (099987f0) — Edgeworth-expansion (Duan et al. 2006).

Tolerance choice:

* Engine NPV vs C++ reference: **LOOSE** (abs_tol=1e-8, rel_tol=1e-8).
  The Edgeworth expansion involves a triple-loop accumulation of
  moments over T (= 252 days for 1y); float64 round-off of all the
  product+division chains diverges from C++ at ~1e-10 absolute. Loose
  is safe.
* Put-call parity: TIGHT (algebraic identity using both NPV results).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.models.equity.gjr_garch_model import GjrGarchModel
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_gjr_garch_engine import (
    AnalyticGjrGarchEngine,
)
from pquantlib.processes.gjr_garch_process import GjrGarchProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose, tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_S = 100.0
_R = 0.05
_Q = 0.0


@pytest.fixture
def cpp_refs() -> dict[str, Any]:
    return load_reference("cluster/w1d")


@pytest.fixture
def setup() -> tuple[GjrGarchModel, Date]:
    """Build the canonical GJR-GARCH model + 1y expiry date."""
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    expiry = ref + 365
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=_R, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=_Q, day_counter=dc)
    process = GjrGarchProcess(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=SimpleQuote(_S),
        v0=0.000160,
        omega=0.000002,
        alpha=0.024,
        beta=0.93,
        gamma=0.059,
        lambda_=0.2,
        days_per_year=252.0,
    )
    return GjrGarchModel(process), expiry


def test_atm_call_matches_cpp(
    setup: tuple[GjrGarchModel, Date], cpp_refs: dict[str, Any]
) -> None:
    """ATM 1y call NPV matches the C++ AnalyticGJRGARCHEngine.

    # C++ parity: analyticgjrgarchengine.cpp:38-298.
    """
    model, expiry = setup
    engine = AnalyticGjrGarchEngine(model)
    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, _S), EuropeanExercise(expiry)
    )
    option.set_pricing_engine(engine)
    expected = cpp_refs["analytic_gjr_garch_engine"]["call_atm_1y"]
    loose(
        option.npv(),
        expected,
        reason="Edgeworth-expansion triple loop diverges from C++ at ~1e-10",
    )


def test_atm_put_matches_cpp(
    setup: tuple[GjrGarchModel, Date], cpp_refs: dict[str, Any]
) -> None:
    """ATM 1y put NPV matches the C++ AnalyticGJRGARCHEngine.

    # C++ parity: analyticgjrgarchengine.cpp:283-291 — put via parity.
    """
    model, expiry = setup
    engine = AnalyticGjrGarchEngine(model)
    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Put, _S), EuropeanExercise(expiry)
    )
    option.set_pricing_engine(engine)
    expected = cpp_refs["analytic_gjr_garch_engine"]["put_atm_1y"]
    loose(option.npv(), expected, reason="put via parity → same precision as call")


def test_put_call_parity(setup: tuple[GjrGarchModel, Date]) -> None:
    """Put-call parity: C - P ~ S * Df_div - K * Df_rf.

    # C++ parity: implicit in the put branch using
    # P = C + K*Df_rf/Df_div - S.
    """
    model, expiry = setup
    engine = AnalyticGjrGarchEngine(model)
    call = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, _S), EuropeanExercise(expiry)
    )
    call.set_pricing_engine(engine)
    put = VanillaOption(
        PlainVanillaPayoff(OptionType.Put, _S), EuropeanExercise(expiry)
    )
    put.set_pricing_engine(engine)
    process = model.process()
    df_rf = process.risk_free_rate().discount(expiry)
    df_div = process.dividend_yield().discount(expiry)
    # P = C + K*Df_rf/Df_div - S (form used by the engine's put branch).
    lhs = call.npv() + _S * df_rf / df_div - _S
    rhs = put.npv()
    tight(
        lhs,
        rhs,
        reason="algebraic identity — engine put = call + K*Df_rf/Df_div - S",
    )


def test_engine_rejects_non_european_exercise(
    setup: tuple[GjrGarchModel, Date],
) -> None:
    """The Edgeworth engine is European-only.

    # C++ parity: analyticgjrgarchengine.cpp:40-41.
    """
    model, expiry = setup
    engine = AnalyticGjrGarchEngine(model)
    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, _S),
        AmericanExercise(expiry, expiry),
    )
    option.set_pricing_engine(engine)
    with pytest.raises(LibraryException, match="not an European option"):
        option.npv()


def test_engine_inspector_returns_model(
    setup: tuple[GjrGarchModel, Date],
) -> None:
    """``engine.model()`` returns the supplied GjrGarchModel."""
    model, _ = setup
    engine = AnalyticGjrGarchEngine(model)
    assert engine.model() is model
