"""Tests for SABR closed-form volatility (Hagan 2002 + Bachelier variant)."""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.sabr_formula import (
    sabr_normal_volatility,
    sabr_volatility,
    shifted_sabr_volatility,
    validate_sabr_parameters,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.testing import reference_reader, tolerance

_REF = reference_reader.load("cluster/l9c")
_SABR = _REF["sabr_formula"]


# --- validate_sabr_parameters ------------------------------------------


def test_validate_sabr_parameters_accepts_valid_set() -> None:
    validate_sabr_parameters(0.04, 0.5, 0.4, -0.1)


def test_validate_sabr_parameters_rejects_non_positive_alpha() -> None:
    with pytest.raises(LibraryException, match="alpha"):
        validate_sabr_parameters(0.0, 0.5, 0.4, -0.1)


def test_validate_sabr_parameters_rejects_beta_out_of_unit_interval() -> None:
    with pytest.raises(LibraryException, match="beta"):
        validate_sabr_parameters(0.04, 1.5, 0.4, -0.1)


def test_validate_sabr_parameters_rejects_negative_nu() -> None:
    with pytest.raises(LibraryException, match="nu"):
        validate_sabr_parameters(0.04, 0.5, -0.1, -0.1)


def test_validate_sabr_parameters_rejects_rho_at_unit_modulus() -> None:
    with pytest.raises(LibraryException, match="rho"):
        validate_sabr_parameters(0.04, 0.5, 0.4, 1.0)


# --- sabr_volatility (lognormal arm) -----------------------------------


def test_sabr_volatility_atm() -> None:
    """ATM (K=F) — exercises the Taylor-expansion branch."""
    actual = sabr_volatility(
        _SABR["forward"],
        _SABR["forward"],
        _SABR["expiry"],
        _SABR["alpha"],
        _SABR["beta"],
        _SABR["nu"],
        _SABR["rho"],
    )
    tolerance.tight(actual, _SABR["vol_atm"])


def test_sabr_volatility_strike_3pct() -> None:
    """OTM put — exercises the regular z/x(z) branch."""
    actual = sabr_volatility(
        0.03,
        _SABR["forward"],
        _SABR["expiry"],
        _SABR["alpha"],
        _SABR["beta"],
        _SABR["nu"],
        _SABR["rho"],
    )
    tolerance.tight(actual, _SABR["vol_strike_3pct"])


def test_sabr_volatility_strike_4pct() -> None:
    actual = sabr_volatility(
        0.04,
        _SABR["forward"],
        _SABR["expiry"],
        _SABR["alpha"],
        _SABR["beta"],
        _SABR["nu"],
        _SABR["rho"],
    )
    tolerance.tight(actual, _SABR["vol_strike_4pct"])


def test_sabr_volatility_strike_6pct() -> None:
    actual = sabr_volatility(
        0.06,
        _SABR["forward"],
        _SABR["expiry"],
        _SABR["alpha"],
        _SABR["beta"],
        _SABR["nu"],
        _SABR["rho"],
    )
    tolerance.tight(actual, _SABR["vol_strike_6pct"])


def test_sabr_volatility_strike_7pct() -> None:
    actual = sabr_volatility(
        0.07,
        _SABR["forward"],
        _SABR["expiry"],
        _SABR["alpha"],
        _SABR["beta"],
        _SABR["nu"],
        _SABR["rho"],
    )
    tolerance.tight(actual, _SABR["vol_strike_7pct"])


def test_sabr_volatility_explicit_lognormal_type_matches_default() -> None:
    """Explicit ShiftedLognormal == default arm."""
    a = sabr_volatility(
        0.05, _SABR["forward"], _SABR["expiry"],
        _SABR["alpha"], _SABR["beta"], _SABR["nu"], _SABR["rho"],
        VolatilityType.ShiftedLognormal,
    )
    b = sabr_volatility(
        0.05, _SABR["forward"], _SABR["expiry"],
        _SABR["alpha"], _SABR["beta"], _SABR["nu"], _SABR["rho"],
    )
    tolerance.exact(a, b)


# --- sabr_volatility (normal arm) --------------------------------------


def test_sabr_volatility_normal_atm() -> None:
    actual = sabr_volatility(
        _SABR["forward"],
        _SABR["forward"],
        _SABR["expiry"],
        _SABR["alpha"],
        _SABR["beta"],
        _SABR["nu"],
        _SABR["rho"],
        VolatilityType.Normal,
    )
    tolerance.tight(actual, _SABR["vol_atm_normal"])


def test_sabr_volatility_normal_strike_4pct() -> None:
    actual = sabr_volatility(
        0.04,
        _SABR["forward"],
        _SABR["expiry"],
        _SABR["alpha"],
        _SABR["beta"],
        _SABR["nu"],
        _SABR["rho"],
        VolatilityType.Normal,
    )
    tolerance.tight(actual, _SABR["vol_strike_4pct_normal"])


def test_sabr_normal_volatility_helper_matches_explicit_normal() -> None:
    a = sabr_normal_volatility(
        0.04, _SABR["forward"], _SABR["expiry"],
        _SABR["alpha"], _SABR["beta"], _SABR["nu"], _SABR["rho"],
    )
    b = sabr_volatility(
        0.04, _SABR["forward"], _SABR["expiry"],
        _SABR["alpha"], _SABR["beta"], _SABR["nu"], _SABR["rho"],
        VolatilityType.Normal,
    )
    tolerance.exact(a, b)


# --- input bounds ------------------------------------------------------


def test_sabr_volatility_rejects_non_positive_strike() -> None:
    with pytest.raises(LibraryException, match="strike"):
        sabr_volatility(
            0.0, _SABR["forward"], _SABR["expiry"],
            _SABR["alpha"], _SABR["beta"], _SABR["nu"], _SABR["rho"],
        )


def test_sabr_volatility_rejects_non_positive_forward() -> None:
    with pytest.raises(LibraryException, match="forward"):
        sabr_volatility(
            0.05, 0.0, _SABR["expiry"],
            _SABR["alpha"], _SABR["beta"], _SABR["nu"], _SABR["rho"],
        )


def test_sabr_volatility_rejects_negative_expiry() -> None:
    with pytest.raises(LibraryException, match="expiry"):
        sabr_volatility(
            0.05, _SABR["forward"], -1.0,
            _SABR["alpha"], _SABR["beta"], _SABR["nu"], _SABR["rho"],
        )


# --- shifted_sabr_volatility -------------------------------------------


def test_shifted_sabr_volatility_zero_shift_matches_unshifted() -> None:
    a = shifted_sabr_volatility(
        0.05, _SABR["forward"], _SABR["expiry"],
        _SABR["alpha"], _SABR["beta"], _SABR["nu"], _SABR["rho"],
        0.0,
    )
    b = sabr_volatility(
        0.05, _SABR["forward"], _SABR["expiry"],
        _SABR["alpha"], _SABR["beta"], _SABR["nu"], _SABR["rho"],
    )
    tolerance.exact(a, b)


def test_shifted_sabr_volatility_positive_shift_permits_negative_strike() -> None:
    """With shift = 0.01, a strike of -0.005 is admissible."""
    actual = shifted_sabr_volatility(
        -0.005, 0.005, 1.0,
        0.04, 0.5, 0.4, -0.1,
        shift=0.01,
    )
    # We just check the value is positive — no probe reference for this one.
    assert actual > 0.0
