"""Cross-validate VanillaForwardPayoff against C++ v1.43.

Probe source: migration-harness/cpp/probes/v143_inst_margrabechooser/probe.cpp
Reference:    migration-harness/references/v143/inst/margrabechooser.json

The payoff is the *unclamped* forward payoff -- ``price - strike`` (Call)
and ``strike - price`` (Put) with no ``max(., 0)``. The reference price
grid straddles both strikes, so a port that floors at zero (i.e. copied
``PlainVanillaPayoff``) fails on every out-of-the-money price.

Placement note: C++ declares this payoff in
``ql/instruments/vanillaswingoption.hpp``; this port keeps it in the same
module as ``VanillaSwingOption``, which lives under
``experimental/finitedifferences/``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.experimental.finitedifferences.vanilla_swing_option import (
    VanillaForwardPayoff,
)
from pquantlib.payoffs import OptionType, StrikedTypePayoff
from pquantlib.testing import reference_reader, tolerance

_CASES = {
    "call_87_5": OptionType.Call,
    "put_87_5": OptionType.Put,
    "call_100": OptionType.Call,
    "put_100": OptionType.Put,
}


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("v143/inst/margrabechooser")["vanilla_forward_payoff"]


def test_vanilla_forward_payoff_is_a_striked_type_payoff() -> None:
    payoff = VanillaForwardPayoff(OptionType.Call, 87.5)
    assert isinstance(payoff, StrikedTypePayoff)
    assert payoff.option_type() == OptionType.Call
    assert payoff.strike() == 87.5


@pytest.mark.parametrize("case", list(_CASES))
def test_vanilla_forward_payoff_name(cpp_ref: dict[str, Any], case: str) -> None:
    """# C++ parity: ``name()`` returns "ForwardTypePayoff"."""
    ref = cpp_ref[case]
    payoff = VanillaForwardPayoff(_CASES[case], ref["strike"])
    assert payoff.name() == ref["name"] == "ForwardTypePayoff"


@pytest.mark.parametrize("case", ["call_87_5", "put_87_5"])
def test_vanilla_forward_payoff_description_matches_cpp(cpp_ref: dict[str, Any], case: str) -> None:
    """Verbatim match against the C++ ``description()``.

    Restricted to the 87.5-strike cases: C++ renders a Real through
    ``std::ostream`` at default precision, so an integral strike prints as
    "100" while Python's f-string prints "100.0". 87.5 renders identically
    under both, which lets the string be compared exactly. The integral
    strikes are covered by ``..._description_components`` below.
    """
    ref = cpp_ref[case]
    payoff = VanillaForwardPayoff(_CASES[case], ref["strike"])
    assert payoff.description() == ref["description"]


@pytest.mark.parametrize("case", ["call_100", "put_100"])
def test_vanilla_forward_payoff_description_components(cpp_ref: dict[str, Any], case: str) -> None:
    """Integral strike: compare the parts C++ and Python agree on."""
    ref = cpp_ref[case]
    payoff = VanillaForwardPayoff(_CASES[case], ref["strike"])
    description = payoff.description()
    assert description.startswith("ForwardTypePayoff ")
    assert str(_CASES[case]) in description
    assert "100" in description
    assert description.endswith(" strike")


@pytest.mark.parametrize("case", list(_CASES))
def test_vanilla_forward_payoff_values_match_cpp(cpp_ref: dict[str, Any], case: str) -> None:
    """``operator()(price)`` over the whole price grid, both option types."""
    ref = cpp_ref[case]
    payoff = VanillaForwardPayoff(_CASES[case], ref["strike"])
    prices: list[float] = cpp_ref["prices"]
    expected: list[float] = ref["values"]
    assert len(prices) == len(expected)
    for price, want in zip(prices, expected, strict=True):
        tolerance.exact(payoff(price), want, reason=f"{case} @ {price}")


@pytest.mark.parametrize("case", list(_CASES))
def test_vanilla_forward_payoff_is_not_floored_at_zero(cpp_ref: dict[str, Any], case: str) -> None:
    """The defining difference from PlainVanillaPayoff: negatives are kept."""
    ref = cpp_ref[case]
    assert any(value < 0.0 for value in ref["values"]), case
    payoff = VanillaForwardPayoff(_CASES[case], ref["strike"])
    prices: list[float] = cpp_ref["prices"]
    assert any(payoff(price) < 0.0 for price in prices)


def test_vanilla_forward_payoff_call_and_put_are_mirror_images(
    cpp_ref: dict[str, Any],
) -> None:
    """Call(p) + Put(p) == 0 for a common strike.

    Stated as a sum rather than ``put == -call`` so the at-the-money price
    compares ``+0.0`` against ``+0.0`` instead of ``+0.0`` against ``-0.0``.
    """
    call = VanillaForwardPayoff(OptionType.Call, 87.5)
    put = VanillaForwardPayoff(OptionType.Put, 87.5)
    for price in cpp_ref["prices"]:
        tolerance.exact(call(price) + put(price), 0.0, reason=f"@ {price}")
