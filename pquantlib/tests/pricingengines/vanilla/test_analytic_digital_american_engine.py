"""Cross-validate the two American digital engines against C++ v1.43.

Reference: ``migration-harness/references/v143/pe/american`` (produced by
``migration-harness/cpp/probes/v143_pe_american/probe.cpp``), cases prefixed
``digital_``.

Covers :class:`AnalyticDigitalAmericanEngine` and
:class:`AnalyticDigitalAmericanKOEngine` end to end through a
:class:`VanillaOption` — payoff in, NPV and greeks out — for:

* cash-or-nothing and asset-or-nothing, Call and Put;
* the barrier above spot, below spot, and already breached at t = 0;
* ``payoff_at_expiry`` both ways, which is what selects
  :class:`AmericanPayoffAtExpiry` over :class:`AmericanPayoffAtHit`;
* the knock-in engine and the knock-out subclass.

The point of the exercise is as much *which* results exist as what they are.
The at-hit path fills value / delta / gamma / rho and leaves vega, theta,
theta-per-day, dividend rho, strike sensitivity, ITM cash probability, delta
forward and elasticity unset; the at-expiry path fills value alone.  Every
one of those absences is pinned — the probe emits ``"unset"`` and the
assertion requires the accessor to raise, so a port that invents a greek
fails just as loudly as one that miscomputes it.

The trap this file exists to catch
----------------------------------
:class:`AmericanPayoffAtHit` takes no knock-in argument, so with
``payoff_at_expiry() == False`` the KO engine returns *exactly* the knock-in
price — ``knock_in()`` is consulted only on the at-expiry path.
:func:`test_knock_out_at_hit_equals_knock_in_at_hit` asserts that identity
directly, and the at-expiry cases assert that KI and KO *do* differ there.

Tolerance is TIGHT: this is the closed form of
``test_american_payoffs_v143.py`` with two term-structure lookups in front of
it, and agreement is ~1e-15 relative.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.vanilla.analytic_digital_american_engine import (
    AnalyticDigitalAmericanEngine,
    AnalyticDigitalAmericanKOEngine,
)
from pquantlib.testing import reference_reader

from ._american_v143 import TODAY, build_payoff, expect_number, market

CPP: dict[str, Any] = reference_reader.load("v143/pe/american")

_DIGITAL_CASES = [name for name in CPP if name.startswith("digital_")]

_GREEK_SLOTS: list[tuple[str, str]] = [
    ("delta", "delta"),
    ("gamma", "gamma"),
    ("theta", "theta"),
    ("theta_per_day", "theta_per_day"),
    ("vega", "vega"),
    ("rho", "rho"),
    ("dividend_rho", "dividend_rho"),
    ("strike_sensitivity", "strike_sensitivity"),
    ("itm_cash_probability", "itm_cash_probability"),
    ("delta_forward", "delta_forward"),
    ("elasticity", "elasticity"),
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


def _build_option(inputs: dict[str, Any]) -> VanillaOption:
    payoff = build_payoff(inputs)
    ex_date = TODAY + int(inputs["maturity_days"])
    if inputs.get("exercise") == "European":
        return VanillaOption(payoff, EuropeanExercise(ex_date))
    earliest = TODAY + int(inputs.get("earliest_offset_days", 0))
    exercise = AmericanExercise(earliest, ex_date, bool(inputs["payoff_at_expiry"]))
    return VanillaOption(payoff, exercise)


def test_case_table_is_populated() -> None:
    """Guard against a reference that silently lost its cases."""
    assert len(_DIGITAL_CASES) >= 18


@pytest.mark.tight
@pytest.mark.parametrize("case", _DIGITAL_CASES)
def test_digital_american_engine(case: str) -> None:
    """Reproduce NPV and the exact set of greeks C++ fills, case by case."""
    record = CPP[case]
    inputs = record["inputs"]
    expected = record["expected"]

    process = market(
        float(inputs["spot"]), float(inputs["q"]), float(inputs["r"]), float(inputs["vol"])
    )
    option = _build_option(inputs)
    knock_out = inputs.get("engine") == "AnalyticDigitalAmericanKOEngine"
    engine = (
        AnalyticDigitalAmericanKOEngine(process)
        if knock_out
        else AnalyticDigitalAmericanEngine(process)
    )
    option.set_pricing_engine(engine)

    if expected["throws"]:
        with pytest.raises(LibraryException):
            option.npv()
        return

    expect_number(option.npv, expected["npv"], f"{case} npv")
    for attr, key in _GREEK_SLOTS:
        expect_number(getattr(option, attr), expected[key], f"{case} {key}")


def test_knock_out_at_hit_equals_knock_in_at_hit() -> None:
    """The KO engine is a no-op on the at-hit path — pinned, not assumed.

    ``AmericanPayoffAtHit`` has no knock-in parameter, so ``knock_in()`` is
    never consulted when ``payoff_at_expiry()`` is false.  Both engines
    therefore return the identical price *and* the identical greeks.
    """
    for ki_case, ko_case in (
        ("digital_hit_cash_call_ki", "digital_hit_cash_call_ko_equals_ki"),
        ("digital_hit_cash_put_ki", "digital_hit_cash_put_ko_equals_ki"),
    ):
        ki = CPP[ki_case]["expected"]
        ko = CPP[ko_case]["expected"]
        assert ki["npv"] == ko["npv"]
        assert ki["delta"] == ko["delta"]
        assert ki["gamma"] == ko["gamma"]
        assert ki["rho"] == ko["rho"]

        inputs = CPP[ko_case]["inputs"]
        process = market(
            float(inputs["spot"]), float(inputs["q"]), float(inputs["r"]), float(inputs["vol"])
        )
        option = _build_option(inputs)
        option.set_pricing_engine(AnalyticDigitalAmericanKOEngine(process))
        expect_number(option.npv, ko["npv"], f"{ko_case} npv")


def test_knock_out_differs_from_knock_in_at_expiry() -> None:
    """On the at-expiry path ``knock_in()`` genuinely changes the answer."""
    for ki_case, ko_case in (
        ("digital_expiry_cash_call_ki", "digital_expiry_cash_call_ko"),
        ("digital_expiry_cash_put_ki", "digital_expiry_cash_put_ko"),
        ("digital_expiry_asset_call_ki", "digital_expiry_asset_call_ko"),
    ):
        assert CPP[ki_case]["expected"]["npv"] != CPP[ko_case]["expected"]["npv"]


def test_knock_out_is_worthless_once_the_barrier_is_breached() -> None:
    """A breached knock-out pays nothing; the knock-in pays the full binary."""
    assert CPP["digital_expiry_cash_call_breached_ko"]["expected"]["npv"] == 0.0
    assert CPP["digital_expiry_cash_put_breached_ko"]["expected"]["npv"] == 0.0
    assert CPP["digital_expiry_cash_call_breached_ki"]["expected"]["npv"] > 0.0


def test_engines_report_their_knock_in_flag() -> None:
    """The whole KO subclass is one overridden predicate."""
    process = market(100.0, 0.04, 0.05, 0.25)
    assert AnalyticDigitalAmericanEngine(process).knock_in() is True
    assert AnalyticDigitalAmericanKOEngine(process).knock_in() is False
    assert isinstance(AnalyticDigitalAmericanKOEngine(process), AnalyticDigitalAmericanEngine)
