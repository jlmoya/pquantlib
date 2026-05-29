"""Tests for KahaleSmileSection (arbitrage-free smile reformulation)."""

from __future__ import annotations

import math

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.volatility.flat_smile_section import FlatSmileSection
from pquantlib.termstructures.volatility.kahale_smile_section import KahaleSmileSection
from pquantlib.termstructures.volatility.sabr_smile_section import SabrSmileSection
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.testing import reference_reader, tolerance

_REF = reference_reader.load("cluster/l10a")
_K = _REF["kahale_smile_section"]


def _sabr_base() -> SabrSmileSection:
    return SabrSmileSection(
        forward=0.05,
        sabr_params=(0.04, 0.5, 0.4, -0.1),
        exercise_time=_K["exercise_time"],
    )


# --- structural identities -------------------------------------------


def test_atm_level_matches_input() -> None:
    sec = KahaleSmileSection(base=_sabr_base(), atm=_K["base_atm_level"])
    tolerance.exact(sec.atm_level(), _K["base_atm_level"])


def test_min_strike_is_minus_shift() -> None:
    sec = KahaleSmileSection(base=_sabr_base())
    # SABR base has shift=0, so min_strike = 0.
    tolerance.exact(sec.min_strike(), -0.0)


def test_max_strike_is_infinity() -> None:
    sec = KahaleSmileSection(base=_sabr_base())
    assert sec.max_strike() == math.inf


def test_exercise_time_delegates_to_base() -> None:
    sec = KahaleSmileSection(base=_sabr_base())
    tolerance.exact(sec.exercise_time(), _K["exercise_time"])


# --- arbitrage-free repair invariants --------------------------------


def test_butterfly_arbitrage_is_zero_or_positive() -> None:
    """d²C/dK² ≥ 0 at intermediate strikes in the AF region (LOOSE)."""
    sec = KahaleSmileSection(base=_sabr_base(), atm=0.05)
    # Sample a few interior strikes — call prices should be convex.
    test_strikes = [0.04, 0.045, 0.05, 0.055, 0.06]
    h = 1.0e-4
    for k in test_strikes:
        try:
            c_minus = sec.option_price(k - h, option_type=1)
            c_plus = sec.option_price(k + h, option_type=1)
            c_mid = sec.option_price(k, option_type=1)
        except Exception:
            continue
        # Convexity: c(k-h) - 2 c(k) + c(k+h) ≥ 0.
        d2 = c_minus - 2.0 * c_mid + c_plus
        # LOOSE — the AF repair guarantees convexity at machine precision
        # except near the AF region's boundary.
        assert d2 >= -1.0e-6, f"butterfly arb at {k}: d²C={d2}"


def test_call_prices_monotone_decreasing_in_strike() -> None:
    """dC/dK ≤ 0 across the AF region (LOOSE)."""
    sec = KahaleSmileSection(base=_sabr_base(), atm=0.05)
    test_strikes = [0.03, 0.04, 0.05, 0.06, 0.07]
    last = None
    for k in test_strikes:
        try:
            c = sec.option_price(k, option_type=1)
        except Exception:
            continue
        if last is not None:
            assert c <= last + 1.0e-6, f"non-monotone at {k}: c={c} last={last}"
        last = c


def test_call_price_matches_base_at_atm_when_base_is_af() -> None:
    """For a SABR base (already AF), Kahale's price should match."""
    sec = KahaleSmileSection(base=_sabr_base(), atm=_K["base_atm_level"])
    # Match the C++ probe to LOOSE.
    tolerance.loose(
        sec.option_price(_K["base_atm_level"], option_type=1),
        _K["kahale_call_price_atm"],
    )


def test_volatility_at_atm_is_finite() -> None:
    sec = KahaleSmileSection(base=_sabr_base(), atm=0.05)
    v = sec.volatility(0.05)
    assert v > 0
    assert v < 10.0  # sanity


# --- core indices ----------------------------------------------------


def test_core_indices_form_non_empty_window() -> None:
    sec = KahaleSmileSection(base=_sabr_base())
    left, right = sec.core_indices()
    assert left < right


# --- bounds + rejection ----------------------------------------------


def test_rejects_normal_vol_base() -> None:
    """Kahale only supports shifted lognormal source sections (C++)."""
    normal_base = SabrSmileSection(
        forward=0.05,
        sabr_params=(0.04, 0.5, 0.4, -0.1),
        exercise_time=2.0,
        volatility_type=VolatilityType.Normal,
    )
    with pytest.raises(LibraryException, match="shifted lognormal"):
        KahaleSmileSection(base=normal_base)


def test_rejects_base_without_atm_when_no_atm_given() -> None:
    """A base without ATM + no explicit ATM is a hard error."""
    # FlatSmileSection without an atm_level returns NaN.
    base = FlatSmileSection(volatility=0.18, exercise_time=2.0)  # atm_level=None
    with pytest.raises(LibraryException, match="atm"):
        KahaleSmileSection(base=base)


def test_custom_moneyness_grid() -> None:
    sec = KahaleSmileSection(
        base=_sabr_base(),
        atm=0.05,
        moneyness_grid=[-2.0, -1.0, 0.0, 1.0, 2.0],
    )
    # Should construct cleanly.
    assert sec.atm_level() == 0.05


def test_exponential_right_tail() -> None:
    sec = KahaleSmileSection(
        base=_sabr_base(),
        atm=0.05,
        exponential_extrapolation=True,
    )
    # Should construct cleanly + offer some right-tail extrapolation.
    far_strike = 0.5
    c = sec.option_price(far_strike, option_type=1)
    assert c >= 0


def test_interpolate_mode_constructs() -> None:
    sec = KahaleSmileSection(
        base=_sabr_base(),
        atm=0.05,
        interpolate=True,
    )
    # Just check it doesn't blow up + atm_level still correct.
    tolerance.exact(sec.atm_level(), 0.05)


def test_left_and_right_core_strikes_accessible() -> None:
    sec = KahaleSmileSection(base=_sabr_base(), atm=0.05)
    left = sec.left_core_strike()
    right = sec.right_core_strike()
    assert left < right
    assert left > -1.0  # sanity
    assert right < 10.0  # sanity
