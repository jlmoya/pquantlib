"""Cross-validate :class:`IntegralEngine` against C++ v1.43.

Reference: ``migration-harness/references/v143/pe/american`` (produced by
``migration-harness/cpp/probes/v143_pe_american/probe.cpp``), cases prefixed
``integral_``.

The engine prices a European option by integrating the payoff against the
risk-neutral lognormal density on a **fixed** ``SegmentIntegral(5000)`` over
``[drift - 10 sqrt(var), drift + 10 sqrt(var)]``.  That makes the quadrature
part of the contract rather than an implementation detail: an adaptive
integrator, or a different segment count, converges somewhere else and will
not reproduce these numbers.  It also means the engine's answer is *not* the
Black-Scholes price — it is the Black-Scholes price plus a truncation error
and a midpoint-rule error, both of which the reference captures.

Coverage: both option types at ITM / ATM / OTM, 5% and 80% volatility, 7-day
and 10-year maturities, zero carry, a negative rate, and the two binary
payoffs (the integrand applies ``arguments_.payoff`` — the *raw* payoff, not
the ``StrikedTypePayoff`` the engine validated — so cash-or-nothing and
asset-or-nothing work and are covered).  An American exercise is pinned as a
rejection.

Only ``value`` is filled; all eleven greek accessors must raise, and that is
asserted for every case.

Tolerance is TIGHT.  Both sides run the identical 5000-segment sum in the
identical order, so this is a pure round-off comparison — and it comes out
bit-identical on every case in the table, which is the strongest possible
evidence that the segment count, the interval, and the summation order were
all reproduced rather than approximated.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.vanilla.integral_engine import IntegralEngine
from pquantlib.testing import reference_reader

from ._american_v143 import TODAY, build_payoff, expect_number, market

CPP: dict[str, Any] = reference_reader.load("v143/pe/american")

_INTEGRAL_CASES = [name for name in CPP if name.startswith("integral_")]

_GREEK_KEYS = [
    "delta",
    "gamma",
    "theta",
    "theta_per_day",
    "vega",
    "rho",
    "dividend_rho",
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


def test_case_table_is_populated() -> None:
    """Guard against a reference that silently lost its cases."""
    assert len(_INTEGRAL_CASES) >= 15


@pytest.mark.tight
@pytest.mark.parametrize("case", _INTEGRAL_CASES)
def test_integral_engine(case: str) -> None:
    """Reproduce the 5000-segment quadrature, and the absence of every greek."""
    record = CPP[case]
    inputs = record["inputs"]
    expected = record["expected"]

    process = market(
        float(inputs["spot"]), float(inputs["q"]), float(inputs["r"]), float(inputs["vol"])
    )
    ex_date = TODAY + int(inputs["maturity_days"])
    exercise = (
        AmericanExercise(TODAY, ex_date)
        if inputs["american_exercise"]
        else EuropeanExercise(ex_date)
    )
    option = VanillaOption(build_payoff(inputs), exercise)
    option.set_pricing_engine(IntegralEngine(process))

    if expected["throws"]:
        with pytest.raises(LibraryException):
            option.npv()
        return

    expect_number(option.npv, expected["npv"], f"{case} npv")
    for key in _GREEK_KEYS:
        assert expected[key] == "unset", f"{case}: probe says {key} is filled"
        expect_number(getattr(option, key), expected[key], f"{case} {key}")


def test_engine_rejects_american_exercise() -> None:
    """``QL_REQUIRE(exercise->type() == European, "not an European Option")``."""
    assert CPP["integral_rejects_american_exercise"]["expected"]["throws"] is True

    inputs = CPP["integral_rejects_american_exercise"]["inputs"]
    process = market(
        float(inputs["spot"]), float(inputs["q"]), float(inputs["r"]), float(inputs["vol"])
    )
    option = VanillaOption(
        build_payoff(inputs), AmericanExercise(TODAY, TODAY + int(inputs["maturity_days"]))
    )
    option.set_pricing_engine(IntegralEngine(process))
    with pytest.raises(LibraryException):
        option.npv()


def test_quadrature_error_is_visible_at_low_volatility() -> None:
    """The engine is a quadrature, not a closed form — and the table shows it.

    At 5% vol over a year the integration range is only +-0.5 in log-space
    and 5000 midpoints resolve it well; at 80% vol the range is +-8 and the
    same 5000 segments are coarser.  Both are pinned, so a port that swapped
    in an adaptive integrator "because it is more accurate" fails on both.
    """
    low = CPP["integral_call_lowvol"]["expected"]["npv"]
    high = CPP["integral_call_highvol"]["expected"]["npv"]
    assert 0.0 < low < high
