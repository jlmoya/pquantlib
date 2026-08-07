"""Cross-validate VannaVolgaInterpolation against C++ QuantLib v1.43.

Probe source: migration-harness/cpp/probes/v143_experimental_interp/probe.cpp
Reference:    migration-harness/references/v143/experimental/interp.json

Covers ``detail::VannaVolgaInterpolationImpl``
(ql/experimental/barrieroption/vannavolgainterpolation.hpp:82), which C++
exposes only through the ``VannaVolgaInterpolation`` facade (line 39) and
the ``VannaVolga`` factory (line 58). Until this file existed the class was
exercised only transitively, through the two vanna-volga barrier engines,
so a compensating error in the interpolation and the engine would have gone
unnoticed.

The probe pins the intermediates (forward, ATM vol, the two premium vectors
and the vegas) as well as ``value``, so a red test bisects instead of
merely reporting a wrong number.

Tolerance
---------
Intermediates: TIGHT — closed-form Black premia and a normal density.

``value``: LOOSE. It inverts the reproduced call premium with
``blackFormulaImpliedStdDev``, whose C++ default accuracy is 1e-6 **on the
std dev**; the returned quantity is ``stdDev / sqrt(T)``. Two faithful
implementations of the same Brent solve stop at different points inside
that 1e-6 window, so the achievable agreement is bounded by the solver's
own accuracy, not by the arithmetic. The observed agreement is far tighter
than the bound (see ``test_value_agreement_is_far_inside_the_solver_bound``,
which pins the actual worst-case gap so a future regression that merely
"stays inside LOOSE" still shows up).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.barrieroption.vanna_volga_interpolation import (
    VannaVolga,
    VannaVolgaInterpolation,
)
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import loose, tight

_CASES = ["vv_a", "vv_b", "vv_c"]


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("v143/experimental/interp")


def _build(cpp_ref: dict[str, Any], tag: str) -> VannaVolgaInterpolation:
    return VannaVolgaInterpolation(
        cpp_ref[f"{tag}_strikes"],
        cpp_ref[f"{tag}_vols"],
        cpp_ref[f"{tag}_spot"],
        cpp_ref[f"{tag}_dDiscount"],
        cpp_ref[f"{tag}_fDiscount"],
        cpp_ref[f"{tag}_T"],
    )


@pytest.mark.parametrize("tag", _CASES)
def test_intermediates_match_cpp(cpp_ref: dict[str, Any], tag: str) -> None:
    """``VannaVolgaInterpolationImpl::update`` (vannavolgainterpolation.hpp:97-106).

    Pins ``fwd_ = spot * fDiscount / dDiscount``, ``atmVol_ = y[1]`` and the
    three parallel vectors ``premiaBS`` / ``premiaMKT`` / ``vegas``.
    Ordering matters: a port that swapped premiaBS and premiaMKT would still
    reproduce the pillars, but not the wings.
    """
    f = _build(cpp_ref, tag)
    # White-box: the C++ members are private, so the port keeps them private
    # too; reading them here is what makes a failure bisectable.
    tight(f._fwd, cpp_ref[f"{tag}_fwd"])  # pyright: ignore[reportPrivateUsage]
    tight(f._atm_vol, cpp_ref[f"{tag}_atmVol"])  # pyright: ignore[reportPrivateUsage]
    premia_bs = f._premia_bs  # pyright: ignore[reportPrivateUsage]
    premia_mkt = f._premia_mkt  # pyright: ignore[reportPrivateUsage]
    vegas = f._vegas  # pyright: ignore[reportPrivateUsage]
    for got, expected in zip(premia_bs, cpp_ref[f"{tag}_premiaBS"], strict=True):
        tight(got, expected)
    for got, expected in zip(premia_mkt, cpp_ref[f"{tag}_premiaMKT"], strict=True):
        tight(got, expected)
    for got, expected in zip(vegas, cpp_ref[f"{tag}_vegas"], strict=True):
        tight(got, expected)


@pytest.mark.parametrize("tag", _CASES)
def test_value_matches_cpp(cpp_ref: dict[str, Any], tag: str) -> None:
    """``VannaVolgaInterpolationImpl::value`` (vannavolgainterpolation.hpp:107-122).

    Queried at the three pillars, between them and in both wings.
    """
    f = _build(cpp_ref, tag)
    for k, expected in zip(cpp_ref[f"{tag}_q"], cpp_ref[f"{tag}_value"], strict=True):
        loose(f.value(k), expected, reason=f"{tag} smile vol at k={k}")


@pytest.mark.parametrize("tag", _CASES)
def test_call_operator_is_value(cpp_ref: dict[str, Any], tag: str) -> None:
    f = _build(cpp_ref, tag)
    for k in cpp_ref[f"{tag}_q"]:
        assert f(k) == f.value(k)


def test_value_agreement_is_far_inside_the_solver_bound(cpp_ref: dict[str, Any]) -> None:
    """The LOOSE tier is the ceiling, not the achieved accuracy.

    ``blackFormulaImpliedStdDev`` targets 1e-6 on the std dev, so LOOSE
    (1e-8) is only defensible if the two implementations happen to land on
    the same root far more precisely than that bound. This pins the actual
    worst-case relative gap across every case, so a regression that
    degrades agreement to, say, 5e-9 — still "passing" LOOSE — fails here.
    """
    worst = 0.0
    for tag in _CASES:
        f = _build(cpp_ref, tag)
        for k, expected in zip(cpp_ref[f"{tag}_q"], cpp_ref[f"{tag}_value"], strict=True):
            worst = max(worst, abs(f.value(k) - expected) / abs(expected))
    assert worst < 1e-12, f"worst relative gap {worst!r} exceeded 1e-12"


@pytest.mark.parametrize("tag", _CASES)
def test_pillars_round_trip_to_quoted_vols(cpp_ref: dict[str, Any], tag: str) -> None:
    """At a pillar the hedge weights collapse and the smile returns the quote.

    Two of the three ``log(x_i/k)`` numerators vanish at ``k == x_i``, the
    survivor equals ``vega(k)/vegas[i]``, so the reproduced premium is
    exactly ``premiaMKT[i]``. Inverting it gives back the quoted vol up to
    the Brent accuracy of ``blackFormulaImpliedStdDev`` (1e-6 on the std
    dev, i.e. 1e-6/sqrt(T) on the vol) — the C++ reference itself is only
    that close to the quote, which is why this is asserted against a
    derived bound rather than a tier.
    """
    f = _build(cpp_ref, tag)
    t = cpp_ref[f"{tag}_T"]
    bound = 1e-6 / t**0.5
    for k, quoted in zip(
        cpp_ref[f"{tag}_strikes"], cpp_ref[f"{tag}_vols"], strict=True
    ):
        assert abs(f.value(k) - quoted) < bound


def test_requires_exactly_three_points(cpp_ref: dict[str, Any]) -> None:
    """C++ ctor: ``QL_REQUIRE(xEnd_-xBegin_ == 3, ...)``."""
    with pytest.raises(LibraryException, match="only interpolates 3 volatilities"):
        VannaVolgaInterpolation([1.0, 1.1], [0.1, 0.1], 1.0, 1.0, 1.0, 1.0)
    with pytest.raises(LibraryException, match="only interpolates 3 volatilities"):
        VannaVolgaInterpolation(
            [1.0, 1.1, 1.2, 1.3], [0.1, 0.1, 0.1, 0.1], 1.0, 1.0, 1.0, 1.0
        )


def test_unimplemented_overrides_raise(cpp_ref: dict[str, Any]) -> None:
    """C++ ``primitive`` / ``derivative`` / ``secondDerivative`` are QL_FAIL.

    vannavolgainterpolation.hpp:123-131. A port that silently returned 0.0
    would let a caller integrate a smile that QuantLib refuses to integrate.
    """
    f = _build(cpp_ref, "vv_a")
    with pytest.raises(LibraryException, match="primitive not implemented"):
        f.primitive(1.35)
    with pytest.raises(LibraryException, match="derivative not implemented"):
        f.derivative(1.35)
    with pytest.raises(LibraryException, match="secondDerivative not implemented"):
        f.second_derivative(1.35)


def test_traits_and_factory_match_cpp(cpp_ref: dict[str, Any]) -> None:
    """``VannaVolga::requiredPoints == 3`` and ``VannaVolga::interpolate``."""
    assert VannaVolga.required_points == cpp_ref["vv_required_points"]
    factory = VannaVolga(
        cpp_ref["vv_a_spot"],
        cpp_ref["vv_a_dDiscount"],
        cpp_ref["vv_a_fDiscount"],
        cpp_ref["vv_a_T"],
    )
    f = factory.interpolate(cpp_ref["vv_a_strikes"], cpp_ref["vv_a_vols"])
    for k, expected in zip(
        [1.30, 1.35, 1.40], cpp_ref["vv_factory_value"], strict=True
    ):
        loose(f.value(k), expected)
