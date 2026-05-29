"""Tests for SabrInterpolatedSmileSection (composition fit + wrap)."""

from __future__ import annotations

import math

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.termstructures.volatility.sabr_interpolated_smile_section import (
    SabrInterpolatedSmileSection,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_REF = reference_reader.load("cluster/l10a")
_SIS = _REF["sabr_interpolated_smile_section"]


def _option_date() -> Date:
    # A 1-year option on Actual/365F = 365 days.
    return Date.from_ymd(15, Month.June, 2027)


def _reference_date() -> Date:
    return Date.from_ymd(15, Month.June, 2026)


def _fixture() -> SabrInterpolatedSmileSection:
    return SabrInterpolatedSmileSection(
        option_date=_option_date(),
        forward=_SIS["forward"],
        strikes=_SIS["strikes"],
        vols=_SIS["input_vols"],
        day_counter=Actual365Fixed(),
        reference_date=_reference_date(),
    )


# --- construction / fit ----------------------------------------------


def test_constructor_runs_and_converges() -> None:
    sec = _fixture()
    assert sec.converged()


def test_atm_level_returns_input_forward() -> None:
    sec = _fixture()
    tolerance.exact(sec.atm_level(), _SIS["forward"])


def test_exercise_time_is_one_year() -> None:
    sec = _fixture()
    tolerance.tight(sec.exercise_time(), 1.0)


# --- fit quality -----------------------------------------------------


def test_round_trip_recovers_pillar_vols() -> None:
    """Synthetic input was generated from SABR — fit should recover it."""
    sec = _fixture()
    for k, v_in in zip(_SIS["strikes"], _SIS["input_vols"], strict=True):
        tolerance.loose(sec.volatility(k), v_in)


def test_rms_error_is_tiny() -> None:
    sec = _fixture()
    # Synthetic input — well below 1e-4.
    assert sec.rms_error() < 1.0e-4


def test_max_error_is_tiny() -> None:
    sec = _fixture()
    assert sec.max_error() < 1.0e-3


def test_fitted_alpha_close_to_truth() -> None:
    sec = _fixture()
    tolerance.loose(sec.alpha(), _SIS["true_alpha"])


def test_fitted_beta_close_to_truth() -> None:
    sec = _fixture()
    tolerance.loose(sec.beta(), _SIS["true_beta"])


def test_fitted_nu_close_to_truth() -> None:
    sec = _fixture()
    tolerance.loose(sec.nu(), _SIS["true_nu"])


def test_fitted_rho_close_to_truth() -> None:
    sec = _fixture()
    tolerance.loose(sec.rho(), _SIS["true_rho"])


# --- strike bounds + delegation --------------------------------------


def test_min_max_strike_match_input_grid() -> None:
    sec = _fixture()
    tolerance.exact(sec.min_strike(), _SIS["strikes"][0])
    tolerance.exact(sec.max_strike(), _SIS["strikes"][-1])


def test_volatility_at_pillars_matches_probe() -> None:
    sec = _fixture()
    # The probe emits sabr_vol_at_5pct (atm), 3pct, 6pct as the
    # *closed-form* SABR vol at the truth params. Our fit recovers
    # these.
    tolerance.loose(sec.volatility(0.05), _SIS["sabr_vol_at_5pct"])
    tolerance.loose(sec.volatility(0.03), _SIS["sabr_vol_at_3pct"])
    tolerance.loose(sec.volatility(0.06), _SIS["sabr_vol_at_6pct"])


def test_variance_at_atm() -> None:
    sec = _fixture()
    vol = sec.volatility(0.05)
    expected_variance = vol * vol * sec.exercise_time()
    tolerance.tight(sec.variance(0.05), expected_variance)


# --- default day counter ---------------------------------------------


def test_default_day_counter_is_actual_365_fixed() -> None:
    """When day_counter is None C++ defaults to Actual365Fixed."""
    sec = SabrInterpolatedSmileSection(
        option_date=_option_date(),
        forward=0.05,
        strikes=[0.03, 0.04, 0.05, 0.06, 0.07],
        vols=[0.22, 0.20, 0.18, 0.20, 0.22],
        reference_date=_reference_date(),
    )
    tolerance.tight(sec.exercise_time(), 1.0)


# --- input checks ----------------------------------------------------


def test_fewer_than_two_strikes_raises() -> None:
    from pquantlib.exceptions import LibraryException  # noqa: PLC0415
    with pytest.raises(LibraryException, match="at least 2"):
        SabrInterpolatedSmileSection(
            option_date=_option_date(),
            forward=0.05,
            strikes=[0.05],
            vols=[0.18],
            reference_date=_reference_date(),
        )


# --- Halton multi-start coverage -------------------------------------


def test_multi_start_returns_no_worse_fit() -> None:
    """max_guesses > 1 should never produce a worse RMS than single-start."""
    sec_single = _fixture()
    sec_multi = SabrInterpolatedSmileSection(
        option_date=_option_date(),
        forward=_SIS["forward"],
        strikes=_SIS["strikes"],
        vols=_SIS["input_vols"],
        day_counter=Actual365Fixed(),
        reference_date=_reference_date(),
        max_guesses=5,
    )
    # Both should converge to the true SABR params on synthetic input.
    assert sec_multi.rms_error() <= sec_single.rms_error() + 1.0e-9
    assert not math.isnan(sec_multi.rms_error())
