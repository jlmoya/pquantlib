"""Cross-validate the two American digital payoff calculators against C++ v1.43.

Reference: ``migration-harness/references/v143/pe/american`` (produced by
``migration-harness/cpp/probes/v143_pe_american/probe.cpp``).

Covers :class:`AmericanPayoffAtHit` and :class:`AmericanPayoffAtExpiry`, the
closed forms behind :class:`AnalyticDigitalAmericanEngine`.  Both are pure
calculators over ``(spot, discount, dividend_discount, variance, payoff)``, so
every case is reconstructed straight from the probe's ``inputs`` block — no
term structures and, deliberately, no evaluation date to pin: neither class
reads ``Settings``.

Coverage follows the branch structure rather than the price surface:

* both option types crossed with the barrier above / below / exactly at spot
  (the last of these is where the non-strict ``alpha``/``beta`` test and the
  strict ``in_the_money`` test disagree);
* cash-or-nothing and asset-or-nothing payoffs (asset-or-nothing shifts
  ``mu`` by 1 and swaps ``K`` for the forward);
* knock-in and knock-out for the at-expiry form, including the already
  breached cases where the knock-in collapses to 0.5/0.5 and the knock-out
  to 0/0;
* the ``variance < QL_EPSILON`` path, both with a representable tiny variance
  (where the branch's real content is visible) and with variance exactly 0
  (where C++ divides 0 by 0 and every field goes NaN — pinned so a port
  cannot "helpfully" special-case it);
* all four ``QL_REQUIRE`` guards on each class, plus ``rho``'s negative
  maturity check.

Tolerance is TIGHT throughout.  These are closed forms — a handful of
``exp``/``log``/``pow``/``N(x)`` calls with no cancellation-driven loss — and
the measured agreement is ~1e-15 relative, two to three orders inside the
1e-12 relative bound.

One slot is deliberately **not** asserted: ``gamma`` on the
``variance < QL_EPSILON`` path.  ``AmericanPayoffAtHit::gamma()`` reads
``D1_``/``D2_``, which that path of the C++ constructor never assigns — an
indeterminate read, flagged by the probe as ``gamma_undefined``.  There is no
correct value to compare against.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.payoffs import AssetOrNothingPayoff, CashOrNothingPayoff, OptionType
from pquantlib.pricingengines.american_payoff_at_expiry import AmericanPayoffAtExpiry
from pquantlib.pricingengines.american_payoff_at_hit import AmericanPayoffAtHit
from pquantlib.testing import reference_reader, tolerance

CPP: dict[str, Any] = reference_reader.load("v143/pe/american")

_AT_HIT_CASES = [name for name in CPP if name.startswith("hit_")]
_AT_EXPIRY_CASES = [name for name in CPP if name.startswith("expiry_")]


def _build_payoff(inputs: dict[str, Any]) -> CashOrNothingPayoff | AssetOrNothingPayoff:
    otype = OptionType[inputs["type"]]
    strike = float(inputs["strike"])
    if inputs["payoff"] == "CashOrNothing":
        return CashOrNothingPayoff(otype, strike, float(inputs["cash"]))
    return AssetOrNothingPayoff(otype, strike)


def _discounts(inputs: dict[str, Any]) -> tuple[float, float, float]:
    """``(discount, dividend_discount, variance)``, however the probe framed it."""
    return (
        float(inputs["discount"]),
        float(inputs["dividend_discount"]),
        float(inputs["variance"]),
    )


def test_at_hit_case_table_is_not_empty() -> None:
    """Guard against a reference that silently lost its cases."""
    assert len(_AT_HIT_CASES) >= 20
    assert len(_AT_EXPIRY_CASES) >= 30


# --- AmericanPayoffAtHit ---------------------------------------------------


@pytest.mark.tight
@pytest.mark.parametrize("case", _AT_HIT_CASES)
def test_american_payoff_at_hit(case: str) -> None:
    """Reproduce value / delta / gamma / rho for every at-hit branch."""
    record = CPP[case]
    inputs = record["inputs"]
    expected = record["expected"]

    spot = float(inputs["spot"])
    discount, dividend_discount, variance = _discounts(inputs)
    payoff = _build_payoff(inputs)

    if expected["throws"]:
        # ``rho`` is the only accessor with its own precondition, so the
        # ``hit_rho_rejects_negative_maturity`` case constructs successfully
        # and fails on the accessor instead of in the constructor.
        def _construct_and_read() -> float:
            pricer = AmericanPayoffAtHit(spot, discount, dividend_discount, variance, payoff)
            return pricer.rho(float(inputs.get("T", 1.0)))

        with pytest.raises(LibraryException):
            _construct_and_read()
        return

    pricer = AmericanPayoffAtHit(spot, discount, dividend_discount, variance, payoff)
    maturity = float(inputs["T"])

    _assert_maybe_nan(pricer.value(), expected["value"], f"{case} value")
    _assert_maybe_nan(pricer.delta(), expected["delta"], f"{case} delta")
    if not expected["gamma_undefined"]:
        _assert_maybe_nan(pricer.gamma(), expected["gamma"], f"{case} gamma")
    _assert_maybe_nan(pricer.rho(maturity), expected["rho"], f"{case} rho")


def _assert_maybe_nan(actual: float, expected: Any, label: str) -> None:
    if isinstance(expected, str):
        assert expected == "nan", f"{label}: unexpected encoding {expected!r}"
        assert actual != actual, f"{label}: expected NaN, got {actual!r}"  # noqa: PLR0124
        return
    tolerance.tight(actual, float(expected), reason=label)


def test_at_hit_value_is_nan_when_variance_is_exactly_zero() -> None:
    """The zero-variance corner is a 0/0 in ``mu``; pinned, not smoothed over."""
    record = CPP["hit_cash_call_zero_variance_nan"]
    assert record["expected"]["value"] == "nan"
    inputs = record["inputs"]
    pricer = AmericanPayoffAtHit(
        float(inputs["spot"]),
        float(inputs["discount"]),
        float(inputs["dividend_discount"]),
        float(inputs["variance"]),
        _build_payoff(inputs),
    )
    value = pricer.value()
    assert value != value  # noqa: PLR0124 - NaN check without importing math


def test_at_hit_gamma_is_flagged_undefined_on_the_small_variance_path() -> None:
    """The probe must mark, and only mark, the sub-epsilon-variance cases."""
    flagged = {
        name
        for name in _AT_HIT_CASES
        if not CPP[name]["expected"]["throws"] and CPP[name]["expected"]["gamma_undefined"]
    }
    assert flagged, "no case exercises the variance < QL_EPSILON path"
    for name in flagged:
        assert float(CPP[name]["inputs"]["variance"]) < 2.220446049250313e-16


# --- AmericanPayoffAtExpiry ------------------------------------------------


@pytest.mark.tight
@pytest.mark.parametrize("case", _AT_EXPIRY_CASES)
def test_american_payoff_at_expiry(case: str) -> None:
    """Reproduce the value for every (type x knock_in x breached) combination."""
    record = CPP[case]
    inputs = record["inputs"]
    expected = record["expected"]

    spot = float(inputs["spot"])
    discount, dividend_discount, variance = _discounts(inputs)
    payoff = _build_payoff(inputs)
    knock_in = bool(inputs["knock_in"])

    if expected["throws"]:
        with pytest.raises(LibraryException):
            AmericanPayoffAtExpiry(
                spot, discount, dividend_discount, variance, payoff, knock_in
            )
        return

    pricer = AmericanPayoffAtExpiry(
        spot, discount, dividend_discount, variance, payoff, knock_in
    )
    _assert_maybe_nan(pricer.value(), expected["value"], f"{case} value")


def test_at_expiry_knock_out_negates_the_second_term() -> None:
    """Knock-in and knock-out differ by more than a sign on the whole price.

    The C++ construction negates ``Y`` only, so KI + KO is *not* the
    unconditional binary — this asserts the two are genuinely different
    numbers rather than accidentally equal, which is the cheapest way to
    catch a port that drops the ``knock_in`` flag entirely.
    """
    ki = CPP["expiry_cash_call_barrier_above_ki"]["expected"]["value"]
    ko = CPP["expiry_cash_call_barrier_above_ko"]["expected"]["value"]
    assert ki != ko

    inputs = CPP["expiry_cash_call_barrier_above_ki"]["inputs"]
    discount, dividend_discount, variance = _discounts(inputs)
    payoff = _build_payoff(inputs)
    spot = float(inputs["spot"])
    got_ki = AmericanPayoffAtExpiry(
        spot, discount, dividend_discount, variance, payoff, True
    ).value()
    got_ko = AmericanPayoffAtExpiry(
        spot, discount, dividend_discount, variance, payoff, False
    ).value()
    tolerance.tight(got_ki, float(ki), reason="knock-in")
    tolerance.tight(got_ko, float(ko), reason="knock-out")
