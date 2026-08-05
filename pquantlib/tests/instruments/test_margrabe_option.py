"""Cross-validate MargrabeOption + its two analytic engines.

Probe source: migration-harness/cpp/probes/v143_inst_margrabechooser/probe.cpp
Reference:    migration-harness/references/v143/inst/margrabechooser.json

The reference grid perturbs every constructor / engine argument one at a
time from a common base case (both quantities, both dividend yields, both
volatilities, the correlation, both spots and the maturity), so an
argument that is accepted and then dropped changes no output and fails
here.

Tolerance: TIGHT everywhere except one deep-out-of-the-money American
case — see ``_AMERICAN_CANCELLATION_*`` below for the derivation.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.margrabe_option import (
    MargrabeOption,
    MargrabeOptionArguments,
    MargrabeOptionResults,
)
from pquantlib.instruments.multi_asset_option import MultiAssetOption
from pquantlib.payoffs import NullPayoff
from pquantlib.pricingengines.exotic.analytic_american_margrabe_engine import (
    AnalyticAmericanMargrabeEngine,
)
from pquantlib.pricingengines.exotic.analytic_european_margrabe_engine import (
    AnalyticEuropeanMargrabeEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date

# The Bjerksund-Stensland value is the algebraic sum of five terms of order
# max(spot, strike). For "quantity2_3" (spot 100, strike 3*105 = 315) the
# terms are +1.3e-10, +98.01, -98.01, -305.67, +305.66 -- sum of absolute
# values 807.4 against a result of 1.23e-4, i.e. a cancellation ratio of
# 6.57e6. One ulp of disagreement per term between libm and CPython's math
# module therefore surfaces at DBL_EPSILON * 6.57e6 = 1.5e-9 relative; the
# measured gap is 6.5e-10. The bound below carries a small multiple of the
# single-rounding estimate to cover the several exp/log/pow calls inside
# each term. Every other case in the grid holds at TIGHT.
_AMERICAN_CANCELLATION_CASES = frozenset({"quantity2_3"})
_AMERICAN_CANCELLATION_REL = 5e-9
_AMERICAN_CANCELLATION_REASON = (
    "deep-OTM Bjerksund-Stensland value: five O(315) terms cancel to 1.23e-4 "
    "(ratio 6.6e6), so DBL_EPSILON-level libm differences show at ~1e-9 relative"
)

# The engines fill only part of the Greek surface; the rest must raise.
_EUROPEAN_FILLED = ("value", "delta1", "delta2", "gamma1", "gamma2", "theta", "rho")
_EUROPEAN_UNFILLED = ("delta", "gamma", "vega", "dividend_rho")
_ALL_GREEKS = (
    "delta1",
    "delta2",
    "gamma1",
    "gamma2",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "dividend_rho",
)


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("v143/inst/margrabechooser")


@pytest.fixture(scope="module")
def day_counter() -> DayCounter:
    return Actual365Fixed()


@pytest.fixture(scope="module")
def today(cpp_ref: dict[str, Any]) -> Date:
    return Date(int(cpp_ref["meta"]["today_serial"]))


def _make_process(
    today: Date,
    day_counter: DayCounter,
    spot: float,
    r: float,
    q: float,
    vol: float,
) -> GeneralizedBlackScholesProcess:
    return GeneralizedBlackScholesProcess(
        x0=SimpleQuote(spot),
        dividend_ts=FlatForward.from_rate(today, q, day_counter),
        risk_free_ts=FlatForward.from_rate(today, r, day_counter),
        black_vol_ts=BlackConstantVol(
            reference_date=today,
            calendar=NullCalendar(),
            day_counter=day_counter,
            volatility=vol,
        ),
    )


def _processes(
    inputs: dict[str, Any], today: Date, day_counter: DayCounter
) -> tuple[GeneralizedBlackScholesProcess, GeneralizedBlackScholesProcess]:
    p1 = _make_process(today, day_counter, inputs["s1"], inputs["r"], inputs["q1"], inputs["v1"])
    p2 = _make_process(today, day_counter, inputs["s2"], inputs["r"], inputs["q2"], inputs["v2"])
    return p1, p2


def _accessor(opt: MargrabeOption, name: str) -> Any:
    return getattr(opt, "npv" if name == "value" else name)


def _case_names(cpp_ref: dict[str, Any]) -> list[str]:
    return list(cpp_ref["margrabe"].keys())


# ---------------------------------------------------------------- instrument --


def test_margrabe_is_a_multi_asset_option_with_null_payoff(today: Date) -> None:
    """C++ builds the base MultiAssetOption from a NullPayoff."""
    exercise = EuropeanExercise(today + 365)
    opt = MargrabeOption(2, 3, exercise)
    assert isinstance(opt, MultiAssetOption)
    assert isinstance(opt.payoff(), NullPayoff)
    assert opt.exercise() is exercise


def test_margrabe_setup_arguments_carries_both_quantities(today: Date) -> None:
    """Both quantities must reach the engine arguments (non-default values)."""
    opt = MargrabeOption(5, 4, EuropeanExercise(today + 365))
    args = MargrabeOptionArguments()
    opt.setup_arguments(args)
    assert args.q1 == 5
    assert args.q2 == 4
    assert isinstance(args.payoff, NullPayoff)
    assert args.exercise is not None
    args.validate()  # no raise


@pytest.mark.parametrize(
    ("q1", "q2", "message"),
    [
        (None, 1, "unspecified quantity for asset 1"),
        (1, None, "unspecified quantity for asset 2"),
        (0, 1, "quantity of asset 1 must be positive"),
        (-2, 1, "quantity of asset 1 must be positive"),
        (1, 0, "quantity of asset 2 must be positive"),
        (1, -2, "quantity of asset 2 must be positive"),
    ],
)
def test_margrabe_arguments_validate_rejects(
    today: Date, q1: int | None, q2: int | None, message: str
) -> None:
    """# C++ parity: ``MargrabeOption::arguments::validate``."""
    args = MargrabeOptionArguments()
    args.payoff = NullPayoff()
    args.exercise = EuropeanExercise(today + 365)
    args.q1 = q1
    args.q2 = q2
    with pytest.raises(LibraryException, match=message):
        args.validate()


def test_margrabe_results_reset_clears_every_field() -> None:
    """``reset()`` must clear the four per-asset Greeks and the base Greeks."""
    results = MargrabeOptionResults()
    results.value = 1.0
    results.delta1 = 2.0
    results.delta2 = 3.0
    results.gamma1 = 4.0
    results.gamma2 = 5.0
    results.theta = 6.0
    results.rho = 7.0
    results.reset()
    for field in ("value", "delta1", "delta2", "gamma1", "gamma2", "theta", "rho"):
        assert getattr(results, field) is None


# ------------------------------------------------------------------- pricing --


def test_margrabe_european_matches_cpp(cpp_ref: dict[str, Any], today: Date, day_counter: DayCounter) -> None:
    """Every filled European result matches C++ at TIGHT, over the whole grid."""
    for name in _case_names(cpp_ref):
        case = cpp_ref["margrabe"][name]
        inputs = case["inputs"]
        p1, p2 = _processes(inputs, today, day_counter)
        opt = MargrabeOption(
            int(inputs["Q1"]),
            int(inputs["Q2"]),
            EuropeanExercise(Date(int(inputs["maturity_serial"]))),
        )
        opt.set_pricing_engine(AnalyticEuropeanMargrabeEngine(p1, p2, inputs["rho"]))
        for field in _EUROPEAN_FILLED:
            expected = case["european"][field]
            assert not isinstance(expected, dict), f"{name}/{field} unexpectedly raises"
            tolerance.tight(_accessor(opt, field)(), expected, reason=f"{name}/european/{field}")


def test_margrabe_american_matches_cpp(cpp_ref: dict[str, Any], today: Date, day_counter: DayCounter) -> None:
    """American NPV matches C++ over the whole grid."""
    for name in _case_names(cpp_ref):
        case = cpp_ref["margrabe"][name]
        inputs = case["inputs"]
        p1, p2 = _processes(inputs, today, day_counter)
        opt = MargrabeOption(
            int(inputs["Q1"]),
            int(inputs["Q2"]),
            AmericanExercise(today, Date(int(inputs["maturity_serial"]))),
        )
        opt.set_pricing_engine(AnalyticAmericanMargrabeEngine(p1, p2, inputs["rho"]))
        expected = case["american"]["value"]
        assert not isinstance(expected, dict)
        if name in _AMERICAN_CANCELLATION_CASES:
            tolerance.custom(
                opt.npv(),
                expected,
                abs_tol=1e-14,
                rel_tol=_AMERICAN_CANCELLATION_REL,
                reason=f"{name}: {_AMERICAN_CANCELLATION_REASON}",
            )
        else:
            tolerance.tight(opt.npv(), expected, reason=f"{name}/american/value")


def test_margrabe_european_and_american_differ(
    cpp_ref: dict[str, Any], today: Date, day_counter: DayCounter
) -> None:
    """The two engines must not be interchangeable.

    ``base`` prices the early-exercise premium; ``american_immediate_branch``
    is deep in the money with a large asset-1 yield, so the American engine
    exercises immediately at ``Q1*S1 - Q2*S2`` while the European one does
    not. If the American engine silently fell through to the European
    formula, this test would fail.
    """
    for name in ("base", "american_immediate_branch"):
        case = cpp_ref["margrabe"][name]
        assert case["american"]["value"] > case["european"]["value"]
    immediate = cpp_ref["margrabe"]["american_immediate_branch"]
    inputs = immediate["inputs"]
    p1, p2 = _processes(inputs, today, day_counter)
    opt = MargrabeOption(
        int(inputs["Q1"]),
        int(inputs["Q2"]),
        AmericanExercise(today, Date(int(inputs["maturity_serial"]))),
    )
    opt.set_pricing_engine(AnalyticAmericanMargrabeEngine(p1, p2, inputs["rho"]))
    tolerance.tight(opt.npv(), inputs["Q1"] * inputs["s1"] - inputs["Q2"] * inputs["s2"])


@pytest.mark.parametrize("field", _EUROPEAN_UNFILLED)
def test_margrabe_european_leaves_greeks_unset(
    cpp_ref: dict[str, Any], today: Date, day_counter: DayCounter, field: str
) -> None:
    """delta/gamma/vega/dividend_rho stay Null in C++, so they must raise."""
    inputs = cpp_ref["margrabe"]["base"]["inputs"]
    assert cpp_ref["margrabe"]["base"]["european"][field] == {"raises": True}
    p1, p2 = _processes(inputs, today, day_counter)
    opt = MargrabeOption(
        int(inputs["Q1"]),
        int(inputs["Q2"]),
        EuropeanExercise(Date(int(inputs["maturity_serial"]))),
    )
    opt.set_pricing_engine(AnalyticEuropeanMargrabeEngine(p1, p2, inputs["rho"]))
    with pytest.raises(LibraryException, match="not provided"):
        _accessor(opt, field)()


@pytest.mark.parametrize("field", _ALL_GREEKS)
def test_margrabe_american_leaves_every_greek_unset(
    cpp_ref: dict[str, Any], today: Date, day_counter: DayCounter, field: str
) -> None:
    """The American engine fills ``value`` only, exactly as in C++."""
    inputs = cpp_ref["margrabe"]["base"]["inputs"]
    assert cpp_ref["margrabe"]["base"]["american"][field] == {"raises": True}
    p1, p2 = _processes(inputs, today, day_counter)
    opt = MargrabeOption(
        int(inputs["Q1"]),
        int(inputs["Q2"]),
        AmericanExercise(today, Date(int(inputs["maturity_serial"]))),
    )
    opt.set_pricing_engine(AnalyticAmericanMargrabeEngine(p1, p2, inputs["rho"]))
    with pytest.raises(LibraryException, match="not provided"):
        _accessor(opt, field)()


def test_margrabe_rho_is_identically_zero(
    cpp_ref: dict[str, Any], today: Date, day_counter: DayCounter
) -> None:
    """C++ sets ``results_.rho = 0.0`` on every case — pin the constant."""
    for name, case in cpp_ref["margrabe"].items():
        assert case["european"]["rho"] == 0.0, name
    inputs = cpp_ref["margrabe"]["base"]["inputs"]
    p1, p2 = _processes(inputs, today, day_counter)
    opt = MargrabeOption(1, 1, EuropeanExercise(Date(int(inputs["maturity_serial"]))))
    opt.set_pricing_engine(AnalyticEuropeanMargrabeEngine(p1, p2, inputs["rho"]))
    tolerance.exact(opt.rho(), 0.0)


# -------------------------------------------------------------- error branches --


def test_margrabe_engines_reject_wrong_exercise(
    cpp_ref: dict[str, Any], today: Date, day_counter: DayCounter
) -> None:
    """Each engine rejects the other's exercise style (pinned as raising)."""
    errors = cpp_ref["margrabe_errors"]
    assert errors["european_engine_american_exercise"] == {"raises": True}
    assert errors["american_engine_european_exercise"] == {"raises": True}

    inputs = cpp_ref["margrabe"]["base"]["inputs"]
    p1, p2 = _processes(inputs, today, day_counter)
    maturity = Date(int(inputs["maturity_serial"]))

    opt = MargrabeOption(1, 1, AmericanExercise(today, maturity))
    opt.set_pricing_engine(AnalyticEuropeanMargrabeEngine(p1, p2, 0.3))
    with pytest.raises(LibraryException, match="not an European Option"):
        opt.npv()

    opt = MargrabeOption(1, 1, EuropeanExercise(maturity))
    opt.set_pricing_engine(AnalyticAmericanMargrabeEngine(p1, p2, 0.3))
    with pytest.raises(LibraryException, match="not an American option"):
        opt.npv()


@pytest.mark.parametrize(("key", "q1", "q2"), [("q1_zero", 0, 1), ("q2_negative", 1, -2)])
def test_margrabe_non_positive_quantities_reject(
    cpp_ref: dict[str, Any],
    today: Date,
    day_counter: DayCounter,
    key: str,
    q1: int,
    q2: int,
) -> None:
    """# C++ parity: ``margrabe_errors`` in the probe."""
    assert cpp_ref["margrabe_errors"][key] == {"raises": True}
    inputs = cpp_ref["margrabe"]["base"]["inputs"]
    p1, p2 = _processes(inputs, today, day_counter)
    opt = MargrabeOption(q1, q2, EuropeanExercise(Date(int(inputs["maturity_serial"]))))
    opt.set_pricing_engine(AnalyticEuropeanMargrabeEngine(p1, p2, 0.3))
    with pytest.raises(LibraryException, match="must be positive"):
        opt.npv()
