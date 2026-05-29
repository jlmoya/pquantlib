"""Tests for ContinuousArithmeticAsianVecerEngine.

Cross-validates against ``migration-harness/references/cluster/w4b.json``.

C++ parity:
ql/experimental/exoticoptions/continuousarithmeticasianvecerengine.{hpp,cpp}
@ v1.42.1 (099987f0).

Tolerance: **CUSTOM** (abs_tol=5e-3) — PDE engine on a finite
100x100 grid. Both implementations use Crank-Nicolson with the
same tridiagonal structure, but small differences arise from:
* boundary-condition application order (C++ applies BCs both
  before applying the explicit step and after; we apply them after).
* scipy.linalg.solve_banded vs the in-house Thomas algorithm in
  QuantLib::TridiagonalOperator.

Empirically the agreement is ~1e-3 on the standard testbed, well
inside the PDE discretization error itself (~1% for 100 steps).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.asian_option import (
    AverageType,
    ContinuousAveragingAsianOption,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.asian.continuous_arithmetic_asian_vecer_engine import (
    ContinuousArithmeticAsianVecerEngine,
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
from pquantlib.testing.tolerance import custom
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def refs() -> dict[str, Any]:
    return load_reference("cluster/w4b")


def _setup() -> tuple[
    GeneralizedBlackScholesProcess, Date, Date, SimpleQuote
]:
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    start = ref
    expiry = ref + 365

    spot = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.02, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20)
    process = GeneralizedBlackScholesProcess(
        x0=spot, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol
    )
    return process, start, expiry, SimpleQuote(100.0)


_REASON = (
    "Vecer PDE on 100x100 Crank-Nicolson grid; ~1e-3 agreement vs C++ "
    "from boundary-condition application order + Thomas-vs-banded "
    "solver differences. Well inside PDE discretization error."
)


def test_call_matches_cpp(refs: dict[str, Any]) -> None:
    process, start, expiry, avg = _setup()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    opt = ContinuousAveragingAsianOption(
        AverageType.Arithmetic, payoff, EuropeanExercise(expiry)
    )
    opt.set_pricing_engine(
        ContinuousArithmeticAsianVecerEngine(
            process, avg, start, 100, 100, -1.0, 1.0
        )
    )

    custom(
        opt.npv(),
        float(refs["asian_vecer"]["call_npv"]),
        abs_tol=5e-3,
        rel_tol=1e-3,
        reason=_REASON,
    )


def test_put_matches_cpp(refs: dict[str, Any]) -> None:
    process, start, expiry, avg = _setup()
    payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    opt = ContinuousAveragingAsianOption(
        AverageType.Arithmetic, payoff, EuropeanExercise(expiry)
    )
    opt.set_pricing_engine(
        ContinuousArithmeticAsianVecerEngine(
            process, avg, start, 100, 100, -1.0, 1.0
        )
    )

    custom(
        opt.npv(),
        float(refs["asian_vecer"]["put_npv"]),
        abs_tol=5e-3,
        rel_tol=1e-3,
        reason=_REASON,
    )
