"""Gaussian1dSwaptionEngine + Gaussian1dJamshidianSwaptionEngine vs C++ v1.43.

Reference: ``migration-harness/references/v143/pe/swaption.json``, produced by
``migration-harness/cpp/probes/v143_pe_swaption/probe.cpp``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import BermudanExercise, EuropeanExercise
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import (
    SettlementMethod,
    SettlementType,
    Swaption,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.swaption.gaussian1d_jamshidian_swaption_engine import (
    Gaussian1dJamshidianSwaptionEngine,
)
from pquantlib.pricingengines.swaption.gaussian1d_swaption_engine import (
    Gaussian1dSwaptionEngine,
    Probabilities,
)
from pquantlib.testing import reference_reader, tolerance

from . import _v143_swaption_market as mkt

_SWAP_TYPES = {"Payer": SwapType.Payer, "Receiver": SwapType.Receiver}
_SETTLEMENT_TYPES = {"Physical": SettlementType.Physical, "Cash": SettlementType.Cash}
_SETTLEMENT_METHODS = {
    "PhysicalOTC": SettlementMethod.PhysicalOTC,
    "PhysicalCleared": SettlementMethod.PhysicalCleared,
    "CollateralizedCashPrice": SettlementMethod.CollateralizedCashPrice,
    "ParYieldCurve": SettlementMethod.ParYieldCurve,
}
_PROBABILITIES = {
    "None": Probabilities.None_,
    "Naive": Probabilities.Naive,
    "Digital": Probabilities.Digital,
}

# Integration-vs-C++ tolerance for the Gaussian1d engines.
#
# Derivation: the engine builds a cubic interpolant on a (2n+1)-point grid,
# evaluates it at a second interpolated grid, then integrates each of the 2n
# segments in closed form against the normal density. Every segment
# contributes a difference of erf() and exp() terms of order 1, and the
# results are summed, so the accumulated rounding is ~2n * eps in the sum
# plus the conditioning of the tridiagonal spline solve (n = 64 here, and
# the C++ tridiagonal elimination is transcribed step for step, so only the
# order of the floating-point sums differs). 2 * 64 * 2^-52 ~ 3e-14 absolute
# on a value of order 3e-2 is ~1e-12 relative; the largest observed
# deviation across the 25 pinned cases is 3.0e-11 relative, which is the
# same quantity amplified by the max() in the backward induction picking a
# different branch within rounding on a handful of grid nodes. LOOSE (1e-8)
# covers that with four orders of margin and would NOT hide a real
# algorithmic difference: swapping the C++ interpolant for a natural cubic
# spline moves these numbers by ~2e-7 relative, which LOOSE rejects.
_INTEGRATION_TOL = tolerance.loose


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/pe/swaption")


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    # probe.cpp:293 — Settings::instance().evaluationDate() = Date(3, March, 2025)
    settings.evaluation_date = mkt.TODAY
    yield
    settings.evaluation_date = saved


_G1D_CASES = sorted(
    k
    for k, v in reference_reader.load("v143/pe/swaption").items()
    if v["inputs"].get("engine") == "Gaussian1dSwaptionEngine"
    and "exercise_date_override" not in v["inputs"]
)
_JAM_CASES = sorted(
    k
    for k, v in reference_reader.load("v143/pe/swaption").items()
    if v["inputs"].get("engine") == "Gaussian1dJamshidianSwaptionEngine"
)
_CROSS_CASES = sorted(k for k in reference_reader.load("v143/pe/swaption") if k.startswith("g1d_vs_jamshidian_"))


def test_case_sets_are_populated() -> None:
    assert len(_G1D_CASES) == 24
    assert len(_JAM_CASES) == 8
    assert len(_CROSS_CASES) == 6


def _build(inputs: dict[str, Any]) -> Swaption:
    swap = mkt.make_swap(
        mkt.forwarding_curve(),
        _SWAP_TYPES[inputs["swap_type"]],
        inputs["fixed_rate"],
        inputs.get("float_spread", 0.0),
    )
    exercise = (
        BermudanExercise(mkt.bermudan_dates())
        if inputs["bermudan"]
        else EuropeanExercise(mkt.exercise_date())
    )
    return Swaption(
        swap,
        exercise,
        _SETTLEMENT_TYPES[inputs["settlement_type"]],
        _SETTLEMENT_METHODS[inputs["settlement_method"]],
    )


@pytest.mark.parametrize("name", _G1D_CASES)
def test_gaussian1d_swaption_engine(cpp: dict[str, Any], name: str) -> None:
    case = cpp[name]
    inputs, expected = case["inputs"], case["expected"]
    if inputs["bermudan"]:
        assert [str(d) for d in mkt.bermudan_dates()] == inputs["exercise_dates"]

    swaption = _build(inputs)
    swaption.set_pricing_engine(
        Gaussian1dSwaptionEngine(
            mkt.make_gsr(mkt.forwarding_curve()),
            inputs["integration_points"],
            inputs["stddevs"],
            inputs["extrapolate_payoff"],
            inputs["flat_payoff_extrapolation"],
            mkt.discount_curve() if inputs["discount_curve_override"] else None,
            _PROBABILITIES[inputs["probabilities"]],
        )
    )

    if expected["throws"]:
        with pytest.raises(LibraryException):
            swaption.npv()
        return

    _INTEGRATION_TOL(swaption.npv(), expected["npv"])
    extras = swaption.additional_results()
    assert len(extras) == expected["additional_results_count"]
    if "probabilities" in expected:
        probs = extras["probabilities"]
        assert len(probs) == len(expected["probabilities"])
        for got, want in zip(probs, expected["probabilities"], strict=True):
            _INTEGRATION_TOL(got, want)


def test_expired_swaption_returns_zero(cpp: dict[str, Any]) -> None:
    """Last exercise date <= the model's reference date short-circuits to 0."""
    case = cpp["g1d_expired"]
    assert str(mkt.TODAY) == case["inputs"]["exercise_date_override"]
    swap = mkt.make_swap(mkt.forwarding_curve(), SwapType.Payer, 0.03)
    swaption = Swaption(swap, EuropeanExercise(mkt.TODAY))
    swaption.set_pricing_engine(
        Gaussian1dSwaptionEngine(mkt.make_gsr(mkt.forwarding_curve()), 64, 7.0)
    )
    tolerance.exact(swaption.npv(), case["expected"]["npv"])


def test_knobs_actually_change_the_price(cpp: dict[str, Any]) -> None:
    """Every tuning knob must move the answer, or a port could ignore it."""
    base = cpp["g1d_payer_k030_n64_s7"]["expected"]["npv"]
    for other in (
        "g1d_payer_k030_n16_s7",
        "g1d_payer_k030_n32_s7",
        "g1d_payer_k030_n64_s4",
        "g1d_payer_k030_n64_s2",
        "g1d_payer_k030_noextrap",
        "g1d_payer_k030_flatextrap",
        "g1d_payer_k030_disc_override",
    ):
        assert cpp[other]["expected"]["npv"] != base, other


@pytest.mark.parametrize("name", _JAM_CASES)
def test_gaussian1d_jamshidian_swaption_engine(cpp: dict[str, Any], name: str) -> None:
    case = cpp[name]
    inputs, expected = case["inputs"], case["expected"]
    swaption = _build(inputs)
    swaption.set_pricing_engine(
        Gaussian1dJamshidianSwaptionEngine(mkt.make_gsr(mkt.forwarding_curve()))
    )
    if expected["throws"]:
        with pytest.raises(LibraryException):
            swaption.npv()
        return
    _INTEGRATION_TOL(swaption.npv(), expected["npv"])


@pytest.mark.parametrize("name", _CROSS_CASES)
def test_integration_and_jamshidian_agree(cpp: dict[str, Any], name: str) -> None:
    """The two engines price the same European swaption two different ways.

    This is the strongest single check on either port: the numerical
    integration on the y-grid and the Jamshidian bond-option decomposition
    share almost no code, so agreeing to the integration error means both
    are right. The absolute difference C++ itself shows is pinned so the
    Python agreement can be required to be no worse.
    """
    case = cpp[name]
    inputs, expected = case["inputs"], case["expected"]
    swap_type = _SWAP_TYPES[inputs["swap_type"]]
    rate = inputs["fixed_rate"]

    integ = Swaption(
        mkt.make_swap(mkt.forwarding_curve(), swap_type, rate),
        EuropeanExercise(mkt.exercise_date()),
    )
    integ.set_pricing_engine(
        Gaussian1dSwaptionEngine(
            mkt.make_gsr(mkt.forwarding_curve()),
            inputs["integration_points"],
            inputs["stddevs"],
            True,
            False,
        )
    )
    jam = Swaption(
        mkt.make_swap(mkt.forwarding_curve(), swap_type, rate),
        EuropeanExercise(mkt.exercise_date()),
    )
    jam.set_pricing_engine(
        Gaussian1dJamshidianSwaptionEngine(mkt.make_gsr(mkt.forwarding_curve()))
    )

    _INTEGRATION_TOL(integ.npv(), expected["integration_npv"])
    _INTEGRATION_TOL(jam.npv(), expected["jamshidian_npv"])
    # The two engines must agree at least as well as C++'s own pair does.
    # C++'s |difference| is between 4e-7 and 2.6e-6 on these markets; allow
    # a 10% slack on that bound rather than a fresh magic number.
    cpp_difference = expected["abs_difference"]
    assert abs(integ.npv() - jam.npv()) <= cpp_difference * 1.1
