"""BachelierCalculator cross-validation against C++ QuantLib v1.43.

Reference: ``migration-harness/references/v143/pe/bondswap.json`` (probe
``migration-harness/cpp/probes/v143_pe_bondswap/probe.cpp``, "PART 2").

The grid deliberately includes negative forwards and negative strikes — the
entire reason the normal model exists, and where a port that reuses Black's
logarithms breaks — plus the zero-volatility, zero-maturity and QL_REQUIRE
boundaries and the three exotic payoffs whose visitor overrides change the
greeks but not the value.

No evaluation date is involved: ``BachelierCalculator`` is a pure function of
``(payoff, forward, stdDev, discount)``, so this module needs no Settings
fixture.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.payoffs import (
    AssetOrNothingPayoff,
    CashOrNothingPayoff,
    GapPayoff,
    OptionType,
    PercentageStrikePayoff,
    PlainVanillaPayoff,
    StrikedTypePayoff,
)
from pquantlib.pricingengines.bachelier_calculator import BachelierCalculator
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.testing import reference_reader, tolerance

CPP: dict[str, Any] = reference_reader.load("v143/pe/bondswap")

_PLAIN_CASES = sorted(
    k
    for k in CPP
    if k.startswith("bachelier_")
    and CPP[k]["inputs"].get("payoff") == "PlainVanillaPayoff"
)

_EXOTIC_PAYOFFS: dict[str, Callable[[], StrikedTypePayoff]] = {
    "bachelier_con_call": lambda: CashOrNothingPayoff(OptionType.Call, 100.0, 7.0),
    "bachelier_con_put": lambda: CashOrNothingPayoff(OptionType.Put, 100.0, 7.0),
    "bachelier_aon_call": lambda: AssetOrNothingPayoff(OptionType.Call, 100.0),
    "bachelier_aon_put": lambda: AssetOrNothingPayoff(OptionType.Put, 100.0),
    "bachelier_gap_call": lambda: GapPayoff(OptionType.Call, 100.0, 105.0),
    "bachelier_unsupported_payoff_throws": lambda: PercentageStrikePayoff(OptionType.Call, 0.9),
}


def _option_type(name: str) -> OptionType:
    return OptionType.Call if name == "Call" else OptionType.Put


@pytest.mark.parametrize("case", _PLAIN_CASES)
def test_plain_vanilla_surface(case: str) -> None:
    inputs = CPP[case]["inputs"]
    expected = CPP[case]["expected"]
    payoff = PlainVanillaPayoff(_option_type(inputs["option_type"]), float(inputs["strike"]))
    forward = float(inputs["forward"])
    std_dev = float(inputs["std_dev"])
    discount = float(inputs["discount"])
    spot = float(inputs["spot"])
    maturity = float(inputs["maturity"])

    if expected.get("throws") is True:
        with pytest.raises(LibraryException):
            BachelierCalculator(payoff, forward, std_dev, discount)
        return

    calc = BachelierCalculator(payoff, forward, std_dev, discount)

    tolerance.tight(calc.value(), expected["value"])
    tolerance.tight(calc.delta_forward(), expected["delta_forward"])
    tolerance.tight(calc.delta(spot), expected["delta"])
    tolerance.tight(calc.elasticity_forward(), expected["elasticity_forward"])
    tolerance.tight(calc.elasticity(spot), expected["elasticity"])
    tolerance.tight(calc.gamma_forward(), expected["gamma_forward"])
    tolerance.tight(calc.gamma(spot), expected["gamma"])
    tolerance.tight(calc.vega(maturity), expected["vega"])
    tolerance.tight(calc.rho(maturity), expected["rho"])
    tolerance.tight(calc.dividend_rho(maturity), expected["dividend_rho"])
    tolerance.tight(calc.itm_cash_probability(), expected["itm_cash_probability"])
    tolerance.tight(calc.itm_asset_probability(), expected["itm_asset_probability"])
    tolerance.tight(calc.strike_sensitivity(), expected["strike_sensitivity"])
    tolerance.tight(calc.strike_gamma(), expected["strike_gamma"])
    tolerance.tight(calc.vanna(maturity), expected["vanna"])
    tolerance.tight(calc.volga(maturity), expected["volga"])
    tolerance.tight(calc.alpha(), expected["alpha"])
    tolerance.tight(calc.beta(), expected["beta"])

    # theta is only pinned where log(forward/spot) is finite.
    if "theta" in expected:
        tolerance.tight(calc.theta(spot, maturity), expected["theta"])
        tolerance.tight(calc.theta_per_day(spot, maturity), expected["theta_per_day"])

    # The (Option::Type, strike) constructor wraps a PlainVanillaPayoff, so it
    # must agree BIT-for-bit with the payoff constructor (a same-language claim,
    # hence the EXACT tier) and to TIGHT with C++ (cross-language, so the last
    # ulp of the normal CDF is not comparable).
    from_type = BachelierCalculator.from_type_strike(
        _option_type(inputs["option_type"]), float(inputs["strike"]), forward, std_dev, discount
    )
    tolerance.exact(from_type.value(), calc.value())
    tolerance.exact(from_type.delta_forward(), calc.delta_forward())
    tolerance.tight(from_type.value(), expected["value_from_type_ctor"])
    tolerance.tight(from_type.delta_forward(), expected["delta_forward_from_type_ctor"])


@pytest.mark.parametrize("case", sorted(_EXOTIC_PAYOFFS))
def test_exotic_payoffs(case: str) -> None:
    inputs = CPP[case]["inputs"]
    expected = CPP[case]["expected"]
    payoff = _EXOTIC_PAYOFFS[case]()
    forward = float(inputs["forward"])
    std_dev = float(inputs["std_dev"])
    discount = float(inputs["discount"])
    maturity = float(inputs["maturity"])

    if expected.get("throws") is True:
        with pytest.raises(LibraryException, match="unsupported payoff type"):
            BachelierCalculator(payoff, forward, std_dev, discount)
        return

    calc = BachelierCalculator(payoff, forward, std_dev, discount)
    tolerance.tight(calc.value(), expected["value"])
    tolerance.tight(calc.alpha(), expected["alpha"])
    tolerance.tight(calc.beta(), expected["beta"])
    tolerance.tight(calc.delta_forward(), expected["delta_forward"])
    tolerance.tight(calc.gamma_forward(), expected["gamma_forward"])
    tolerance.tight(calc.itm_cash_probability(), expected["itm_cash_probability"])
    tolerance.tight(calc.itm_asset_probability(), expected["itm_asset_probability"])
    tolerance.tight(calc.strike_sensitivity(), expected["strike_sensitivity"])
    tolerance.tight(calc.vega(maturity), expected["vega"])
    tolerance.tight(calc.rho(maturity), expected["rho"])
    tolerance.tight(calc.dividend_rho(maturity), expected["dividend_rho"])


def test_value_ignores_the_payoff_dispatch() -> None:
    """C++ ``value()`` recomputes the plain-vanilla payoff whatever the payoff is."""
    plain = CPP["bachelier_call_atm"]["expected"]["value"]
    for case in ("bachelier_con_call", "bachelier_aon_call", "bachelier_gap_call"):
        tolerance.exact(CPP[case]["expected"]["value"], plain)


def test_asset_or_nothing_put_takes_the_call_branch() -> None:
    """``alpha_ = 1 - N(d) >= 0`` makes every ``alpha_ >= 0`` test read "Call"."""
    aon_put = CPP["bachelier_aon_put"]["expected"]
    aon_call = CPP["bachelier_aon_call"]["expected"]
    # ATM, so N(d) == 1 - N(d) == 0.5 and the two coincide exactly.
    tolerance.exact(aon_put["alpha"], aon_call["alpha"])
    tolerance.exact(aon_put["delta_forward"], aon_call["delta_forward"])
    calc = BachelierCalculator(AssetOrNothingPayoff(OptionType.Put, 100.0), 100.0, 20.0, 0.95)
    assert calc.alpha() >= 0.0
    tolerance.tight(calc.delta_forward(), aon_put["delta_forward"])


def test_negative_forward_and_strike_are_supported() -> None:
    """The Black formula is undefined here; the Bachelier one is not."""
    case = CPP["bachelier_call_neg_fwd_neg_strike"]
    inputs, expected = case["inputs"], case["expected"]
    calc = BachelierCalculator(
        PlainVanillaPayoff(OptionType.Call, float(inputs["strike"])),
        float(inputs["forward"]),
        float(inputs["std_dev"]),
        float(inputs["discount"]),
    )
    tolerance.tight(calc.value(), expected["value"])
    # And the Black calculator refuses the same inputs, as C++ does — its
    # ``QL_REQUIRE(strike >= 0)`` fires before the forward is even looked at.
    with pytest.raises(LibraryException, match="must be non-negative"):
        BlackCalculator(
            PlainVanillaPayoff(OptionType.Call, float(inputs["strike"])),
            float(inputs["forward"]),
            float(inputs["std_dev"]),
            float(inputs["discount"]),
        )


def test_black_calculator_ql_min_real_arm() -> None:
    """Regression: ``QL_MIN_REAL`` is ``-DBL_MAX``, not the smallest positive double.

    ``black_calculator.py`` previously defined ``_QL_MIN_REAL`` as
    ``sys.float_info.min`` (+2.2e-308), so ``elasticity`` / ``elasticity_forward``
    returned the wrong sign AND the wrong magnitude in the arm C++ reserves for
    "value is zero and delta is negative".
    """
    case = CPP["black_calculator_elasticity_min_real_arm"]
    inputs, expected = case["inputs"], case["expected"]
    calc = BlackCalculator(
        PlainVanillaPayoff(OptionType.Put, float(inputs["strike"])),
        float(inputs["forward"]),
        float(inputs["std_dev"]),
        float(inputs["discount"]),
    )
    tolerance.exact(calc.value(), expected["value"])
    tolerance.exact(calc.delta_forward(), expected["delta_forward"])
    tolerance.exact(calc.elasticity_forward(), expected["elasticity_forward"])
    tolerance.exact(calc.elasticity(float(inputs["spot"])), expected["elasticity"])
    tolerance.exact(expected["elasticity_forward"], expected["ql_min_real"])


def test_bachelier_ql_min_real_arm() -> None:
    """The same arm in ``BachelierCalculator``: a zero-vol ATM put."""
    case = CPP["bachelier_zero_stddev_atm_put"]
    inputs, expected = case["inputs"], case["expected"]
    calc = BachelierCalculator(
        PlainVanillaPayoff(OptionType.Put, float(inputs["strike"])),
        float(inputs["forward"]),
        float(inputs["std_dev"]),
        float(inputs["discount"]),
    )
    tolerance.exact(calc.elasticity_forward(), expected["elasticity_forward"])
    tolerance.exact(calc.elasticity(float(inputs["spot"])), expected["elasticity"])
