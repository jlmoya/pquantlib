"""Tests for AnalyticDoubleBarrierEngine (Ikeda-Kunitomo 1992).

# C++ parity:
# ql/pricingengines/barrier/analyticdoublebarrierengine.{hpp,cpp} @
# v1.42.1.

Cross-validates against the ``textbook`` and ``asymmetric`` sections of
``migration-harness/references/cluster/l6c.json``. Tolerance: TIGHT
(rapidly-converging geometric series; series=5 is bit-identical to
series=10 for textbook barrier widths).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.double_barrier_option import (
    DoubleBarrierOption,
    DoubleBarrierType,
)
from pquantlib.payoffs import CashOrNothingPayoff, OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.barrier.analytic_double_barrier_engine import (
    AnalyticDoubleBarrierEngine,
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
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/l6c")


def _build_process(
    *, rate: float = 0.05, dividend: float = 0.00, volatility: float = 0.20
) -> tuple[GeneralizedBlackScholesProcess, Date]:
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    expiry = ref + 365
    spot_q = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=rate, day_counter=dc)
    div = FlatForward.from_rate(
        reference_date=ref, forward_rate=dividend, day_counter=dc
    )
    vol = BlackConstantVol(
        reference_date=ref, calendar=cal, day_counter=dc, volatility=volatility
    )
    process = GeneralizedBlackScholesProcess(
        x0=spot_q, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol
    )
    return process, expiry


def _price(
    process: GeneralizedBlackScholesProcess,
    expiry: Date,
    option_type: OptionType,
    barrier_type: DoubleBarrierType,
    barrier_lo: float,
    barrier_hi: float,
    rebate: float = 0.0,
    strike: float = 100.0,
    series: int = 5,
) -> float:
    payoff = PlainVanillaPayoff(option_type, strike)
    exercise = EuropeanExercise(expiry)
    opt = DoubleBarrierOption(
        barrier_type, barrier_lo, barrier_hi, rebate, payoff, exercise
    )
    opt.set_pricing_engine(AnalyticDoubleBarrierEngine(process, series))
    return opt.npv()


# --- textbook NPV cases (S=K=100, L=80, U=120, r=5%, q=0%, sigma=20%) -----


def test_textbook_ko_call_matches_ikeda_kunitomo(
    reference_data: dict[str, Any],
) -> None:
    process, expiry = _build_process()
    npv = _price(
        process, expiry, OptionType.Call, DoubleBarrierType.KnockOut, 80.0, 120.0
    )
    tight(npv, float(reference_data["textbook"]["ko_call"]))


def test_textbook_ki_call_matches_ikeda_kunitomo(
    reference_data: dict[str, Any],
) -> None:
    process, expiry = _build_process()
    npv = _price(
        process, expiry, OptionType.Call, DoubleBarrierType.KnockIn, 80.0, 120.0
    )
    tight(npv, float(reference_data["textbook"]["ki_call"]))


def test_textbook_ko_put_matches_ikeda_kunitomo(
    reference_data: dict[str, Any],
) -> None:
    process, expiry = _build_process()
    npv = _price(
        process, expiry, OptionType.Put, DoubleBarrierType.KnockOut, 80.0, 120.0
    )
    tight(npv, float(reference_data["textbook"]["ko_put"]))


def test_textbook_ki_put_matches_ikeda_kunitomo(
    reference_data: dict[str, Any],
) -> None:
    process, expiry = _build_process()
    npv = _price(
        process, expiry, OptionType.Put, DoubleBarrierType.KnockIn, 80.0, 120.0
    )
    tight(npv, float(reference_data["textbook"]["ki_put"]))


# --- in-out parity (Call + Put): KO + KI must equal European vanilla -----


def test_in_out_parity_call_against_european_vanilla(
    reference_data: dict[str, Any],
) -> None:
    """C++ ``callKI = max(0, vanilla - callKO)``: parity holds exactly
    when ``callKO < vanilla`` (always true for non-zero textbook
    barriers), so ``KI + KO == vanilla`` to TIGHT.
    """
    process, expiry = _build_process()
    ko = _price(
        process, expiry, OptionType.Call, DoubleBarrierType.KnockOut, 80.0, 120.0
    )
    ki = _price(
        process, expiry, OptionType.Call, DoubleBarrierType.KnockIn, 80.0, 120.0
    )
    vanilla = float(reference_data["textbook"]["vanilla_call"])
    tight(ki + ko, vanilla)


def test_in_out_parity_put_against_european_vanilla(
    reference_data: dict[str, Any],
) -> None:
    process, expiry = _build_process()
    ko = _price(
        process, expiry, OptionType.Put, DoubleBarrierType.KnockOut, 80.0, 120.0
    )
    ki = _price(process, expiry, OptionType.Put, DoubleBarrierType.KnockIn, 80.0, 120.0)
    vanilla = float(reference_data["textbook"]["vanilla_put"])
    tight(ki + ko, vanilla)


# --- asymmetric KO call (L=90, U=130, r=5%, q=2%, sigma=25%) -------------


def test_asymmetric_ko_call_matches_ikeda_kunitomo(
    reference_data: dict[str, Any],
) -> None:
    process, expiry = _build_process(rate=0.05, dividend=0.02, volatility=0.25)
    npv = _price(
        process, expiry, OptionType.Call, DoubleBarrierType.KnockOut, 90.0, 130.0
    )
    tight(npv, float(reference_data["asymmetric"]["ko_call"]))


# --- series-convergence sanity ------------------------------------------


def test_series_convergence_5_vs_10_tight(reference_data: dict[str, Any]) -> None:
    """The Ikeda-Kunitomo series converges geometrically; for textbook
    barriers (80 / 120) series=5 and series=10 are bit-identical in
    double precision.
    """
    process, expiry = _build_process()
    npv5 = _price(
        process,
        expiry,
        OptionType.Call,
        DoubleBarrierType.KnockOut,
        80.0,
        120.0,
        series=5,
    )
    npv10 = _price(
        process,
        expiry,
        OptionType.Call,
        DoubleBarrierType.KnockOut,
        80.0,
        120.0,
        series=10,
    )
    tight(npv5, npv10)
    # Also matches the C++ probe's series=10 reference exactly.
    tight(npv10, float(reference_data["textbook"]["ko_call_s10"]))


# --- error paths ---------------------------------------------------------


def test_engine_rejects_american_exercise() -> None:
    process, expiry = _build_process()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    earliest = Date.from_ymd(15, Month.June, 2026)
    exercise = AmericanExercise(earliest, expiry)
    opt = DoubleBarrierOption(
        DoubleBarrierType.KnockOut, 80.0, 120.0, 0.0, payoff, exercise
    )
    opt.set_pricing_engine(AnalyticDoubleBarrierEngine(process))
    with pytest.raises(LibraryException, match="european"):
        opt.npv()


def test_engine_rejects_kiko() -> None:
    """KIKO is a valid instrument type but unsupported by the
    Ikeda-Kunitomo series; the engine raises with the C++ wording.
    """
    process, expiry = _build_process()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = DoubleBarrierOption(
        DoubleBarrierType.KIKO, 80.0, 120.0, 0.0, payoff, exercise
    )
    opt.set_pricing_engine(AnalyticDoubleBarrierEngine(process))
    with pytest.raises(LibraryException, match="unsupported double-barrier type"):
        opt.npv()


def test_engine_rejects_koki() -> None:
    process, expiry = _build_process()
    payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = DoubleBarrierOption(
        DoubleBarrierType.KOKI, 80.0, 120.0, 0.0, payoff, exercise
    )
    opt.set_pricing_engine(AnalyticDoubleBarrierEngine(process))
    with pytest.raises(LibraryException, match="unsupported double-barrier type"):
        opt.npv()


def test_engine_rejects_triggered_barrier_lo() -> None:
    """Spot=100 with L=110 means the lower barrier is already touched."""
    process, expiry = _build_process()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = DoubleBarrierOption(
        DoubleBarrierType.KnockOut, 110.0, 120.0, 0.0, payoff, exercise
    )
    opt.set_pricing_engine(AnalyticDoubleBarrierEngine(process))
    with pytest.raises(LibraryException, match="touched"):
        opt.npv()


def test_engine_rejects_non_plain_payoff() -> None:
    """CashOrNothingPayoff is not supported by the analytic engine."""
    process, expiry = _build_process()
    payoff = CashOrNothingPayoff(OptionType.Call, 100.0, 10.0)
    exercise = EuropeanExercise(expiry)
    opt = DoubleBarrierOption(
        DoubleBarrierType.KnockOut, 80.0, 120.0, 0.0, payoff, exercise
    )
    opt.set_pricing_engine(AnalyticDoubleBarrierEngine(process))
    with pytest.raises(LibraryException, match="non-plain payoff"):
        opt.npv()
