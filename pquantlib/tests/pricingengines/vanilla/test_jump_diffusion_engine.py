"""Cross-validate :class:`JumpDiffusionEngine` and :class:`Merton76Process`.

Reference: ``migration-harness/references/v143/pe/american`` (produced by
``migration-harness/cpp/probes/v143_pe_american/probe.cpp``), cases prefixed
``jd_`` and ``merton76_``.

The engine sums Merton's Poisson series of Black-Scholes prices.  Three
things make it easy to port *almost* right, and each has its own case:

1. **The loop condition contains an ``or``**::

       for (i = 0; (lastContribution > relativeAccuracy_ && i < maxIterations_)
                   || i < Size(lambda*t); i++)

   so the series always runs at least ``floor(lambda t)`` terms regardless of
   convergence.  The ``jd_high_intensity_*`` cases sit at ``lambda t >= 1``
   where a plain while-not-converged loop stops early and disagrees.

2. **The per-term curves are rebuilt with mismatched conventions** — the
   *volatility* day counter and the *risk-free* reference date::

       new FlatForward(rateRefDate, r, voldc)
       new BlackConstantVol(rateRefDate, volcal, v, voldc)

3. **``theta`` telescopes**: each term adds ``lambda * value_i`` and then
   subtracts ``p(i-1) * lambda * value_i`` for ``i > 0``, on top of a
   correction mixing that term's vega and rho.

Coverage: ITM/ATM/OTM calls and puts, jump intensities 0 / 1 / 2 / 5 / 10,
jump volatilities from three ``gamma`` splits of the total variance, with and
without a dividend yield; ``relative_accuracy`` at 1e-2 / 1e-4 / 1e-10 (the
knob must be exposed, not hardcoded); ``max_iterations = 1`` pinned as the
``QL_ENSURE`` failure; a zero-intensity case that degenerates to plain
Black-Scholes; and an American exercise, which the inner
``AnalyticEuropeanEngine`` refuses.

The engine fills value, delta, gamma, theta, vega, rho and dividend_rho —
and nothing else.  ``theta_per_day``, ``strike_sensitivity``,
``itm_cash_probability``, ``delta_forward`` and ``elasticity`` must all
raise, and that is asserted per case.

:class:`Merton76Process` is covered directly: ``x0`` and ``time`` delegate to
the inner :class:`BlackScholesMertonProcess`, the seven inspectors return
what was handed in, and ``drift`` / ``diffusion`` / ``apply`` all raise —
they are ``QL_FAIL`` in C++, so a port that "helpfully" implements them is
wrong.

Tolerance is TIGHT.  The series is a finite sum of closed-form Black prices
with identical term counts on both sides; agreement is ~1e-15 relative.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.jump_diffusion_engine import JumpDiffusionEngine
from pquantlib.processes.merton76_process import Merton76Process
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import BlackConstantVol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar

from ._american_v143 import DC, TODAY, expect_number, option_type

CPP: dict[str, Any] = reference_reader.load("v143/pe/american")

_JD_CASES = [name for name in CPP if name.startswith("jd_")]

_FILLED_KEYS = ["delta", "gamma", "theta", "vega", "rho", "dividend_rho"]
_UNSET_KEYS = [
    "theta_per_day",
    "strike_sensitivity",
    "itm_cash_probability",
    "delta_forward",
    "elasticity",
]


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    # probe.cpp — ``Settings::instance().evaluationDate() = TODAY;`` in main(),
    # with ``const Date TODAY(1, March, 2025);``.
    settings.evaluation_date = TODAY
    yield
    settings.evaluation_date = saved


def _build_process(inputs: dict[str, Any]) -> Merton76Process:
    return Merton76Process(
        state_variable=SimpleQuote(float(inputs["spot"])),
        dividend_ts=FlatForward.from_rate(
            reference_date=TODAY, forward_rate=float(inputs["q"]), day_counter=DC
        ),
        risk_free_ts=FlatForward.from_rate(
            reference_date=TODAY, forward_rate=float(inputs["r"]), day_counter=DC
        ),
        black_vol_ts=BlackConstantVol(
            reference_date=TODAY,
            calendar=NullCalendar(),
            day_counter=DC,
            volatility=float(inputs["vol"]),
        ),
        jump_int=SimpleQuote(float(inputs["jump_intensity"])),
        log_j_mean=SimpleQuote(float(inputs["log_mean_jump"])),
        log_j_vol=SimpleQuote(float(inputs["log_jump_volatility"])),
    )


def test_case_table_is_populated() -> None:
    """Guard against a reference that silently lost its cases."""
    assert len(_JD_CASES) >= 12


@pytest.mark.tight
@pytest.mark.parametrize("case", _JD_CASES)
def test_jump_diffusion_engine(case: str) -> None:
    """Reproduce the Poisson series, its greeks, and the greeks it omits."""
    record = CPP[case]
    inputs = record["inputs"]
    expected = record["expected"]

    process = _build_process(inputs)
    ex_date = TODAY + int(inputs["maturity_days"])
    exercise = (
        AmericanExercise(TODAY, ex_date)
        if inputs["american_exercise"]
        else EuropeanExercise(ex_date)
    )
    option = VanillaOption(
        PlainVanillaPayoff(option_type(inputs["type"]), float(inputs["strike"])), exercise
    )
    option.set_pricing_engine(
        JumpDiffusionEngine(
            process,
            float(inputs["relative_accuracy"]),
            int(inputs["max_iterations"]),
        )
    )

    if expected["throws"]:
        with pytest.raises(LibraryException):
            option.npv()
        return

    expect_number(option.npv, expected["npv"], f"{case} npv")
    for key in _FILLED_KEYS:
        assert expected[key] != "unset", f"{case}: probe says {key} is unset"
        expect_number(getattr(option, key), expected[key], f"{case} {key}")
    for key in _UNSET_KEYS:
        assert expected[key] == "unset", f"{case}: probe says {key} is filled"
        expect_number(getattr(option, key), expected[key], f"{case} {key}")

    tolerance.tight(process.x0(), float(expected["process_x0"]), reason=f"{case} x0")
    tolerance.tight(
        process.time(ex_date),
        float(expected["process_time_at_maturity"]),
        reason=f"{case} time",
    )


def test_relative_accuracy_changes_the_answer() -> None:
    """A port that ignores ``relative_accuracy`` returns the same number thrice."""
    coarse = CPP["jd_accuracy_1e2"]["expected"]["npv"]
    default = CPP["jd_accuracy_1e4"]["expected"]["npv"]
    fine = CPP["jd_accuracy_1e10"]["expected"]["npv"]
    assert coarse != default
    assert default != fine


def test_too_few_iterations_is_an_error() -> None:
    """``QL_ENSURE(i < maxIterations_, ...)`` after the loop."""
    assert CPP["jd_rejects_too_few_iterations"]["expected"]["throws"] is True

    inputs = CPP["jd_rejects_too_few_iterations"]["inputs"]
    option = VanillaOption(
        PlainVanillaPayoff(option_type(inputs["type"]), float(inputs["strike"])),
        EuropeanExercise(TODAY + int(inputs["maturity_days"])),
    )
    option.set_pricing_engine(
        JumpDiffusionEngine(_build_process(inputs), float(inputs["relative_accuracy"]), 1)
    )
    with pytest.raises(LibraryException):
        option.npv()


def test_zero_jump_intensity_degenerates_to_black_scholes() -> None:
    """With ``lambda == 0`` only the i = 0 term carries weight."""
    expected = CPP["jd_zero_intensity_is_black_scholes"]["expected"]
    assert expected["throws"] is False
    assert expected["npv"] > 0.0

    inputs = CPP["jd_zero_intensity_is_black_scholes"]["inputs"]
    option = VanillaOption(
        PlainVanillaPayoff(option_type(inputs["type"]), float(inputs["strike"])),
        EuropeanExercise(TODAY + int(inputs["maturity_days"])),
    )
    option.set_pricing_engine(JumpDiffusionEngine(_build_process(inputs)))
    expect_number(option.npv, expected["npv"], "zero intensity npv")


def test_engine_defaults_match_the_cpp_signature() -> None:
    """``relativeAccuracy_ = 1e-4``, ``maxIterations = 100``."""
    inputs = CPP["jd_accuracy_1e4"]["inputs"]
    assert float(inputs["relative_accuracy"]) == 1e-4
    assert int(inputs["max_iterations"]) == 100

    option = VanillaOption(
        PlainVanillaPayoff(option_type(inputs["type"]), float(inputs["strike"])),
        EuropeanExercise(TODAY + int(inputs["maturity_days"])),
    )
    # No accuracy / iteration arguments: exercise the defaults at the call site.
    option.set_pricing_engine(JumpDiffusionEngine(_build_process(inputs)))
    expect_number(option.npv, CPP["jd_accuracy_1e4"]["expected"]["npv"], "default knobs")


# --- Merton76Process -------------------------------------------------------


def _inspector_process() -> Merton76Process:
    inputs = CPP["merton76_inspectors"]["inputs"]
    return Merton76Process(
        state_variable=SimpleQuote(float(inputs["spot"])),
        dividend_ts=FlatForward.from_rate(
            reference_date=TODAY, forward_rate=float(inputs["q"]), day_counter=Actual365Fixed()
        ),
        risk_free_ts=FlatForward.from_rate(
            reference_date=TODAY, forward_rate=float(inputs["r"]), day_counter=Actual365Fixed()
        ),
        black_vol_ts=BlackConstantVol(
            reference_date=TODAY,
            calendar=NullCalendar(),
            day_counter=Actual365Fixed(),
            volatility=float(inputs["vol"]),
        ),
        jump_int=SimpleQuote(1.0),
        log_j_mean=SimpleQuote(-0.03125),
        log_j_vol=SimpleQuote(0.25),
    )


@pytest.mark.tight
def test_merton76_process_inspectors() -> None:
    """Every inspector delegates to the inner Black-Scholes-Merton process."""
    record = CPP["merton76_inspectors"]
    expected = record["expected"]
    process = _inspector_process()
    ex_date = TODAY + int(record["inputs"]["maturity_days"])

    tolerance.tight(process.x0(), float(expected["x0"]), reason="x0")
    tolerance.tight(
        process.time(ex_date), float(expected["time_at_maturity"]), reason="time"
    )
    tolerance.tight(
        process.jump_intensity().value(),
        float(expected["jump_intensity"]),
        reason="jump intensity",
    )
    tolerance.tight(
        process.log_mean_jump().value(),
        float(expected["log_mean_jump"]),
        reason="log mean jump",
    )
    tolerance.tight(
        process.log_jump_volatility().value(),
        float(expected["log_jump_volatility"]),
        reason="log jump volatility",
    )
    tolerance.tight(
        process.state_variable().value(),
        float(expected["state_variable"]),
        reason="state variable",
    )
    tolerance.tight(
        process.dividend_yield().discount(ex_date),
        float(expected["dividend_discount"]),
        reason="dividend discount",
    )
    tolerance.tight(
        process.risk_free_rate().discount(ex_date),
        float(expected["risk_free_discount"]),
        reason="risk-free discount",
    )
    tolerance.tight(
        process.black_volatility().black_variance(ex_date, 100.0),
        float(expected["black_variance"]),
        reason="black variance",
    )


def _call_drift(process: Merton76Process) -> float:
    return process.drift_1d(0.5, 100.0)


def _call_diffusion(process: Merton76Process) -> float:
    return process.diffusion_1d(0.5, 100.0)


def _call_apply(process: Merton76Process) -> float:
    return process.apply_1d(100.0, 0.01)


@pytest.mark.parametrize(
    ("case", "call"),
    [
        ("merton76_drift_unsupported", _call_drift),
        ("merton76_diffusion_unsupported", _call_diffusion),
        ("merton76_apply_unsupported", _call_apply),
    ],
)
def test_merton76_process_refuses_to_be_simulated(
    case: str, call: Callable[[Merton76Process], float]
) -> None:
    """``drift`` / ``diffusion`` / ``apply`` are ``QL_FAIL`` in C++."""
    assert CPP[case]["expected"]["throws"] is True
    process = _inspector_process()
    with pytest.raises(LibraryException):
        call(process)
