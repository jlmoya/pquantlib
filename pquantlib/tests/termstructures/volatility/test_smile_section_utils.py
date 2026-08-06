"""SmileSectionUtils cross-validated against C++ v1.43.

Reference: migration-harness/references/v143/zabr/smilesectionutils.json
Probe:     migration-harness/cpp/probes/v143_zabr_smilesectionutils/probe.cpp

Nine sections chosen so that every branch of the constructor is reached:
both hard-coded default moneyness tables, the normal and shifted-lognormal
strike maps, the leading-zero prepend, the explicit-atm override, the
min/maxStrike endpoint insertion on a limited-range section, and an
arbitrageable smile with and without ``delete_arbitrage_points``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.termstructures.volatility.flat_smile_section import FlatSmileSection
from pquantlib.termstructures.volatility.interpolated_smile_section import (
    InterpolatedSmileSection,
)
from pquantlib.termstructures.volatility.sabr_smile_section import SabrSmileSection
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.smile_section_utils import SmileSectionUtils
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.testing import reference_reader, tolerance

# Same SABR parameters as the C++ ZabrTests consistency test.
_SABR_PARAMS = (0.08, 0.70, 0.20, -0.30)
_CUSTOM_MONEY = [0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0]


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/zabr/smilesectionutils")


def _sabr() -> SabrSmileSection:
    return SabrSmileSection(exercise_time=5.0, forward=0.03, sabr_params=_SABR_PARAMS)


def _sabr_shifted() -> SabrSmileSection:
    # Negative forward with a positive shift, so k = m*(f + shift) - shift
    # cannot be confused with the normal map k = f + m.
    return SabrSmileSection(
        exercise_time=5.0, forward=-0.005, sabr_params=_SABR_PARAMS, shift=0.02
    )


def _flat_normal() -> FlatSmileSection:
    return FlatSmileSection(
        exercise_time=3.0,
        volatility=0.0080,
        day_counter=Actual365Fixed(),
        atm_level=0.02,
        volatility_type=VolatilityType.Normal,
    )


def _interp_limited() -> InterpolatedSmileSection:
    # Pillars span only [0.01, 0.05] while the default grid runs to 20x the
    # forward, so both endpoint insertions fire.
    #
    # ``LinearInterpolation`` (not the pquantlib default cubic) because the
    # probe uses ``InterpolatedSmileSection<Linear>``. C++ interpolates
    # std-devs and divides by sqrt(T) where pquantlib interpolates vols
    # directly; for any interpolator linear in its ordinates — linear and
    # cubic spline both are — the two agree exactly, so the choice of
    # interpolator FAMILY is what has to match, not the ordinate convention.
    return InterpolatedSmileSection(
        strikes=[0.01, 0.02, 0.03, 0.04, 0.05],
        volatilities=[0.42, 0.36, 0.33, 0.34, 0.37],
        atm_level=0.03,
        exercise_time=2.0,
        interpolator=LinearInterpolation,
    )


def _sabr_arbitrageable() -> SabrSmileSection:
    # Long-dated, low beta, high vol-of-vol, strongly negative rho: Hagan's
    # expansion is arbitrageable in the wings.
    return SabrSmileSection(
        exercise_time=15.0, forward=0.02, sabr_params=(0.06, 0.10, 0.85, -0.65)
    )


def _utils(key: str) -> SmileSectionUtils:
    section: SmileSection
    if key == "sabr_default":
        return SmileSectionUtils(_sabr())
    if key == "sabr_custom_money":
        return SmileSectionUtils(_sabr(), _CUSTOM_MONEY)
    if key == "sabr_money_from_zero":
        return SmileSectionUtils(_sabr(), [0.0, *_CUSTOM_MONEY])
    if key == "sabr_atm_override":
        return SmileSectionUtils(_sabr(), None, 0.035)
    if key == "sabr_shifted":
        return SmileSectionUtils(_sabr_shifted())
    if key == "flat_normal":
        section = _flat_normal()
        return SmileSectionUtils(section)
    if key == "interp_limited_range":
        section = _interp_limited()
        return SmileSectionUtils(section)
    if key == "sabr_arbitrageable":
        return SmileSectionUtils(_sabr_arbitrageable())
    if key == "sabr_arbitrageable_deleted":
        return SmileSectionUtils(_sabr_arbitrageable(), None, None, True)
    raise AssertionError(f"unknown case {key}")


_CASES = [
    "sabr_default",
    "sabr_custom_money",
    "sabr_money_from_zero",
    "sabr_atm_override",
    "sabr_shifted",
    "flat_normal",
    "interp_limited_range",
    "sabr_arbitrageable",
    "sabr_arbitrageable_deleted",
]


@pytest.mark.parametrize("key", _CASES)
def test_grids_match_cpp(cpp: dict[str, Any], key: str) -> None:
    block = cpp[key]
    u = _utils(key)
    tolerance.tight(u.atm_level(), float(block["atm_level"]))
    for got, expected in zip(u.money_grid(), block["money_grid"], strict=True):
        tolerance.tight(got, float(expected))
    for got, expected in zip(u.strike_grid(), block["strike_grid"], strict=True):
        tolerance.tight(got, float(expected))
    for got, expected in zip(u.call_prices(), block["call_prices"], strict=True):
        tolerance.tight(got, float(expected))


@pytest.mark.parametrize("key", _CASES)
def test_arbitragefree_window_matches_cpp(cpp: dict[str, Any], key: str) -> None:
    block = cpp[key]
    u = _utils(key)
    low, high = u.arbitragefree_region()
    tolerance.tight(low, float(block["af_region_low"]))
    tolerance.tight(high, float(block["af_region_high"]))
    assert u.arbitragefree_indices() == (
        int(block["af_index_low"]),
        int(block["af_index_high"]),
    )


def test_default_lognormal_grid_is_the_hard_coded_table(cpp: dict[str, Any]) -> None:
    """The default grid is a fixed 21-entry table, not a sigma-scaled one."""
    assert cpp["sabr_default"]["money_grid"] == [
        0.0, 0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
        1.0, 1.25, 1.5, 1.75, 2.0, 5.0, 7.5, 10.0, 15.0, 20.0,
    ]  # fmt: skip
    assert _utils("sabr_default").money_grid() == cpp["sabr_default"]["money_grid"]


def test_default_normal_grid_is_a_different_table(cpp: dict[str, Any]) -> None:
    """Normal sections get their own 27-entry ABSOLUTE-moneyness table."""
    money = _utils("flat_normal").money_grid()
    assert len(money) == 27
    assert money[0] == -0.20
    assert money[13] == 0.0
    assert money == cpp["flat_normal"]["money_grid"]


def test_arbitrage_shrinks_the_window(cpp: dict[str, Any]) -> None:
    """The arbitrageable smile's window is a strict subset of its grid."""
    block = cpp["sabr_arbitrageable"]
    left, right = _utils("sabr_arbitrageable").arbitragefree_indices()
    assert (left, right) == (1, 12)
    assert right < len(block["strike_grid"]) - 1


def test_delete_arbitrage_points_shortens_the_grid(cpp: dict[str, Any]) -> None:
    """``delete_arbitrage_points`` erases points, it does not only re-index."""
    kept = _utils("sabr_arbitrageable").strike_grid()
    pruned = _utils("sabr_arbitrageable_deleted").strike_grid()
    assert len(kept) == 21
    assert len(pruned) == 13
    assert len(pruned) == len(cpp["sabr_arbitrageable_deleted"]["strike_grid"])


def test_explicit_atm_overrides_the_section(cpp: dict[str, Any]) -> None:
    """An explicit atm rescales every strike and moves the central index."""
    u = _utils("sabr_atm_override")
    tolerance.tight(u.atm_level(), 0.035)
    assert u.atm_level() != _sabr().atm_level()
    assert u.arbitragefree_indices()[0] == int(cpp["sabr_atm_override"]["af_index_low"])


def test_endpoint_insertion_on_limited_strike_range() -> None:
    """Out-of-range points collapse onto the section's own min/max strike."""
    section = _interp_limited()
    u = _utils("interp_limited_range")
    strikes = u.strike_grid()
    assert section.min_strike() in strikes
    assert section.max_strike() in strikes
    # Each endpoint is inserted at most once.
    assert strikes.count(section.min_strike()) == 1
    assert strikes.count(section.max_strike()) == 1
    assert all(section.min_strike() <= k <= section.max_strike() for k in strikes[1:])


def test_non_increasing_moneyness_grid_raises() -> None:
    from pquantlib.exceptions import LibraryException  # noqa: PLC0415

    with pytest.raises(LibraryException, match="strictly increasing"):
        SmileSectionUtils(_sabr(), [0.5, 1.0, 1.0, 1.5])


def test_negative_moneyness_rejected_for_lognormal() -> None:
    from pquantlib.exceptions import LibraryException  # noqa: PLC0415

    with pytest.raises(LibraryException, match="non negative"):
        SmileSectionUtils(_sabr(), [-0.5, 0.5, 1.0, 1.5, 2.0])
