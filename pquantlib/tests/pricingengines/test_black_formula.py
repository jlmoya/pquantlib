"""Cross-validate the BlackFormula family against the L3-A C++ probe.

Probe key: ``l3a/foundations`` →
  - ``black_formula``,
  - ``black_derivatives``,
  - ``black_implied_std_dev``,
  - ``bachelier_black_formula``,
  - ``bachelier_std_dev_derivative``,
  - ``bachelier_implied_vol``.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import (
    bachelier_black_formula,
    bachelier_black_formula_implied_vol,
    bachelier_black_formula_std_dev_derivative,
    black_formula,
    black_formula_implied_std_dev,
    black_formula_implied_std_dev_approximation,
    black_formula_std_dev_derivative,
    black_formula_vol_derivative,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("l3a/foundations")


# --- Black 1976 (lognormal) ---------------------------------------------


def test_black_formula_call_atm_matches_cpp(cpp: dict[str, Any]) -> None:
    bf = cpp["black_formula"]
    v = black_formula(OptionType.Call, 100.0, 100.0, 0.20, 0.95)
    tolerance.tight(v, bf["call_k100"])


def test_black_formula_put_atm_matches_cpp(cpp: dict[str, Any]) -> None:
    bf = cpp["black_formula"]
    v = black_formula(OptionType.Put, 100.0, 100.0, 0.20, 0.95)
    tolerance.tight(v, bf["put_k100"])


def test_black_formula_call_itm_matches_cpp(cpp: dict[str, Any]) -> None:
    bf = cpp["black_formula"]
    v = black_formula(OptionType.Call, 80.0, 100.0, 0.20, 0.95)
    tolerance.tight(v, bf["call_k80"])


def test_black_formula_call_otm_matches_cpp(cpp: dict[str, Any]) -> None:
    bf = cpp["black_formula"]
    v = black_formula(OptionType.Call, 120.0, 100.0, 0.20, 0.95)
    tolerance.tight(v, bf["call_k120"])


def test_black_formula_put_itm_matches_cpp(cpp: dict[str, Any]) -> None:
    bf = cpp["black_formula"]
    v = black_formula(OptionType.Put, 120.0, 100.0, 0.20, 0.95)
    tolerance.tight(v, bf["put_k120"])


def test_black_formula_zero_stddev_returns_intrinsic_discounted(cpp: dict[str, Any]) -> None:
    bf = cpp["black_formula"]
    v_call = black_formula(OptionType.Call, 80.0, 100.0, 0.0, 0.95)
    tolerance.tight(v_call, bf["call_zero_stddev"])
    v_put = black_formula(OptionType.Put, 120.0, 100.0, 0.0, 0.95)
    tolerance.tight(v_put, bf["put_zero_stddev"])


def test_black_formula_no_discount_matches_cpp(cpp: dict[str, Any]) -> None:
    bf = cpp["black_formula"]
    v = black_formula(OptionType.Call, 100.0, 100.0, 0.20, 1.0)
    tolerance.tight(v, bf["call_k100_no_df"])


def test_black_formula_with_displacement_matches_cpp(cpp: dict[str, Any]) -> None:
    bf = cpp["black_formula"]
    v = black_formula(OptionType.Call, 100.0, 100.0, 0.20, 1.0, 10.0)
    tolerance.tight(v, bf["call_k100_shift10"])


def test_black_formula_rejects_negative_std_dev() -> None:
    with pytest.raises(LibraryException, match="stdDev"):
        black_formula(OptionType.Call, 100.0, 100.0, -0.1, 1.0)


def test_black_formula_rejects_non_positive_discount() -> None:
    with pytest.raises(LibraryException, match="discount"):
        black_formula(OptionType.Call, 100.0, 100.0, 0.20, 0.0)


# --- Derivatives --------------------------------------------------------


def test_black_formula_std_dev_derivative_matches_cpp(cpp: dict[str, Any]) -> None:
    bd = cpp["black_derivatives"]
    d = black_formula_std_dev_derivative(100.0, 100.0, 0.20, 0.95)
    tolerance.tight(d, bd["std_dev_derivative"])


def test_black_formula_vol_derivative_matches_cpp(cpp: dict[str, Any]) -> None:
    bd = cpp["black_derivatives"]
    d = black_formula_vol_derivative(100.0, 100.0, 0.20, 1.0, 0.95)
    tolerance.tight(d, bd["vol_derivative"])


def test_black_formula_vol_derivative_equals_std_dev_derivative_at_ttm_1() -> None:
    """When ttm=1, vol_derivative = std_dev_derivative * sqrt(1) = std_dev_derivative."""
    sd = black_formula_std_dev_derivative(100.0, 100.0, 0.20, 0.95)
    vd = black_formula_vol_derivative(100.0, 100.0, 0.20, 1.0, 0.95)
    tolerance.tight(vd, sd)


# --- Implied std dev approximation + Newton-safe solver ----------------


def test_black_formula_implied_std_dev_roundtrip(cpp: dict[str, Any]) -> None:
    bi = cpp["black_implied_std_dev"]
    implied = black_formula_implied_std_dev(
        OptionType.Call,
        strike=100.0,
        forward=100.0,
        black_price=bi["price"],
        discount=0.95,
    )
    # Solver target accuracy is 1e-6 (LOOSE tier).
    tolerance.loose(implied, bi["implied"])


def test_black_formula_implied_std_dev_approximation_is_reasonable() -> None:
    """The approximation should be within ~10% of the true vol for ATM."""
    target = 0.20
    price = black_formula(OptionType.Call, 100.0, 100.0, target, 1.0)
    approx = black_formula_implied_std_dev_approximation(
        OptionType.Call, 100.0, 100.0, price, 1.0
    )
    assert abs(approx - target) < 0.05


def test_black_formula_implied_std_dev_with_explicit_guess() -> None:
    """Solver with a user-supplied initial guess."""
    target = 0.20
    price = black_formula(OptionType.Call, 100.0, 100.0, target, 1.0)
    implied = black_formula_implied_std_dev(
        OptionType.Call, 100.0, 100.0, price, 1.0, guess=0.15
    )
    tolerance.loose(implied, target)


# --- Bachelier (normal) -------------------------------------------------


def test_bachelier_black_formula_call_atm_matches_cpp(cpp: dict[str, Any]) -> None:
    bb = cpp["bachelier_black_formula"]
    v = bachelier_black_formula(OptionType.Call, 100.0, 100.0, 5.0, 0.95)
    tolerance.tight(v, bb["call_k100"])


def test_bachelier_black_formula_put_atm_matches_cpp(cpp: dict[str, Any]) -> None:
    bb = cpp["bachelier_black_formula"]
    v = bachelier_black_formula(OptionType.Put, 100.0, 100.0, 5.0, 0.95)
    tolerance.tight(v, bb["put_k100"])


def test_bachelier_black_formula_call_itm_matches_cpp(cpp: dict[str, Any]) -> None:
    bb = cpp["bachelier_black_formula"]
    v = bachelier_black_formula(OptionType.Call, 80.0, 100.0, 5.0, 0.95)
    tolerance.tight(v, bb["call_k80"])


def test_bachelier_black_formula_zero_stddev_returns_intrinsic_discounted(
    cpp: dict[str, Any],
) -> None:
    bb = cpp["bachelier_black_formula"]
    v = bachelier_black_formula(OptionType.Call, 80.0, 100.0, 0.0, 0.95)
    tolerance.tight(v, bb["call_zero_stddev"])


def test_bachelier_std_dev_derivative_matches_cpp(cpp: dict[str, Any]) -> None:
    d = bachelier_black_formula_std_dev_derivative(100.0, 100.0, 5.0, 0.95)
    tolerance.tight(d, cpp["bachelier_std_dev_derivative"])


# --- Bachelier implied vol (exact, Jaeckel 2017) ------------------------


def test_bachelier_implied_vol_roundtrip(cpp: dict[str, Any]) -> None:
    bi = cpp["bachelier_implied_vol"]
    # K = forward edge — closed-form path.
    implied = bachelier_black_formula_implied_vol(
        OptionType.Call,
        strike=100.0,
        forward=100.0,
        ttm=1.0,
        bachelier_price=bi["price"],
        discount=0.95,
    )
    tolerance.tight(implied, bi["implied_vol"])


def test_bachelier_implied_vol_off_atm_roundtrip() -> None:
    """Strike != forward exercises the Jaeckel _inverse_phi_tilde path."""
    target_vol = 5.0
    strike = 105.0
    forward = 100.0
    ttm = 1.0
    price = bachelier_black_formula(
        OptionType.Call, strike, forward, target_vol * math.sqrt(ttm), 1.0
    )
    implied = bachelier_black_formula_implied_vol(
        OptionType.Call, strike, forward, ttm, price, 1.0
    )
    tolerance.tight(implied, target_vol)
