"""Tests for ZabrInterpolatedSmileSection (composition fit + wrap)."""

from __future__ import annotations

import math

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.math.interpolations.zabr_formula import (
    ZabrEvaluation,
    zabr_volatility,
)
from pquantlib.termstructures.volatility.zabr_interpolated_smile_section import (
    ZabrInterpolatedSmileSection,
)
from pquantlib.termstructures.volatility.zabr_smile_section import ZabrSmileSection
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_REF = reference_reader.load("cluster/w2a")
_ZIS = _REF["zabr_interpolated_smile_section"]
_GAMMA_FREE = _REF["zabr_interpolation_fit_gamma_free"]


def _option_date() -> Date:
    # Matches the C++ probe layout: today + 1095 days (3 * 365). This
    # avoids the leap-year offset that would arise from a calendar 3y
    # difference, keeping exercise_time exactly 3.0 on Actual/365F.
    return _reference_date() + 1095


def _reference_date() -> Date:
    return Date.from_ymd(15, Month.January, 2024)


def _fixture() -> ZabrInterpolatedSmileSection:
    return ZabrInterpolatedSmileSection(
        option_date=_option_date(),
        forward=_GAMMA_FREE["forward"],
        strikes=_GAMMA_FREE["strikes"],
        vols=_GAMMA_FREE["input_vols"],
        alpha=0.05,
        beta=0.5,
        nu=0.3,
        rho=0.0,
        gamma=1.0,
        beta_is_fixed=True,
        day_counter=Actual365Fixed(),
        reference_date=_reference_date(),
    )


# --- construction / fit ----------------------------------------------


def test_constructor_runs_and_converges() -> None:
    sec = _fixture()
    assert sec.converged()


def test_atm_level_returns_input_forward() -> None:
    sec = _fixture()
    tolerance.exact(sec.atm_level(), _GAMMA_FREE["forward"])


def test_exercise_time_is_three_years() -> None:
    sec = _fixture()
    tolerance.tight(sec.exercise_time(), 3.0)


def test_default_evaluation_is_short_maturity_lognormal() -> None:
    sec = _fixture()
    assert sec.evaluation() == ZabrEvaluation.ShortMaturityLognormal


# --- fit quality -----------------------------------------------------


def test_round_trip_recovers_pillar_vols() -> None:
    """Synthetic input was generated from ZABR — fit should recover it."""
    sec = _fixture()
    for k, v_in in zip(_GAMMA_FREE["strikes"], _GAMMA_FREE["input_vols"], strict=True):
        tolerance.loose(sec.volatility(k), v_in)


def test_rms_error_below_threshold() -> None:
    sec = _fixture()
    # Synthetic input — well below 1e-4.
    assert sec.rms_error() < 1.0e-4


def test_max_error_below_threshold() -> None:
    sec = _fixture()
    assert sec.max_error() < 1.0e-3


# --- delegation to ZabrSmileSection ----------------------------------


def test_volatility_matches_zabr_smile_section_with_fitted_params() -> None:
    """Section.volatility(K) routes through a ZabrSmileSection wrapper.

    TIGHT tier — the wrapper is identical machinery.
    """
    sec = _fixture()
    manual = ZabrSmileSection(
        forward=sec.atm_level(),
        zabr_params=(sec.alpha(), sec.beta(), sec.nu(), sec.rho(), sec.gamma()),
        exercise_time=sec.exercise_time(),
        evaluation=sec.evaluation(),
    )
    for k in (0.03, 0.04, 0.05, 0.06, 0.07):
        tolerance.tight(sec.volatility(k), manual.volatility(k))


def test_volatility_matches_zabr_formula_at_fitted_params() -> None:
    """Section.volatility(K) matches a direct zabr_volatility call.

    TIGHT tier — both call the same closed-form math.
    """
    sec = _fixture()
    for k in (0.03, 0.045, 0.055, 0.07):
        direct = zabr_volatility(
            k, sec.atm_level(), sec.exercise_time(),
            sec.alpha(), sec.beta(), sec.nu(), sec.rho(), sec.gamma(),
            mode=sec.evaluation(),
        )
        tolerance.tight(sec.volatility(k), direct)


def test_variance_routes_through_wrapper() -> None:
    sec = _fixture()
    for k in (0.03, 0.05, 0.07):
        vol = sec.volatility(k)
        expected = vol * vol * sec.exercise_time()
        tolerance.tight(sec.variance(k), expected)


# --- strike bounds ---------------------------------------------------


def test_min_max_strike_match_input_grid() -> None:
    sec = _fixture()
    tolerance.exact(sec.min_strike(), _GAMMA_FREE["strikes"][0])
    tolerance.exact(sec.max_strike(), _GAMMA_FREE["strikes"][-1])


# --- default day counter ---------------------------------------------


def test_default_day_counter_is_actual_365_fixed() -> None:
    """When day_counter is None C++ defaults to Actual365Fixed."""
    sec = ZabrInterpolatedSmileSection(
        option_date=_option_date(),
        forward=0.05,
        strikes=[0.03, 0.04, 0.05, 0.06, 0.07],
        vols=[0.22, 0.20, 0.18, 0.20, 0.22],
        reference_date=_reference_date(),
        beta_is_fixed=True,
    )
    # 3-year period yields exercise_time = 3.0.
    tolerance.tight(sec.exercise_time(), 3.0)


# --- inspector tests -------------------------------------------------


def test_alpha_beta_nu_rho_gamma_match_underlying_interp() -> None:
    sec = _fixture()
    # Beta was pinned at 0.5 — exact.
    tolerance.exact(sec.beta(), 0.5)
    # Other params live in the admissible region.
    assert sec.alpha() > 0.0
    assert sec.nu() > 0.0
    assert -1.0 < sec.rho() < 1.0
    assert sec.gamma() > 0.0


# --- C++ probe TIGHT — comparing against the C++ probe inspectors ---


def test_fitted_inspectors_match_cpp_probe() -> None:
    """Recovered params + exercise_time + atm_level match the C++ probe.

    LOOSE-with-extra-slack on the recovered params per the
    ZabrInterpolation divergence note (scipy TRF vs C++ LM).
    """
    sec = _fixture()
    tolerance.tight(sec.exercise_time(), _ZIS["exercise_time"])
    tolerance.tight(sec.atm_level(), _ZIS["atm_level"])
    # The C++ probe used the same synthetic slice and the same initial
    # guess; the recovered params should be within the same 1e-4 ball
    # documented in test_zabr_interpolation_recovers_known_params_gamma_free.
    param_tol = 1.0e-4
    assert abs(sec.alpha() - _ZIS["alpha"]) < param_tol
    assert abs(sec.nu() - _ZIS["nu"]) < param_tol
    assert abs(sec.rho() - _ZIS["rho"]) < param_tol
    assert abs(sec.gamma() - _ZIS["gamma"]) < param_tol
    # C++ probe emits 0.5000...01 due to a JSON 1-ulp rounding artifact;
    # the pinned beta is exactly 0.5 on both sides.
    tolerance.tight(sec.beta(), _ZIS["beta"])


# --- vol at specific probe strikes ----------------------------------


def test_volatility_at_probe_strikes_matches_cpp() -> None:
    """volatility(K) matches the C++ probe values at LOOSE."""
    sec = _fixture()
    tolerance.loose(sec.volatility(0.03), _ZIS["vol_strike_3pct"])
    tolerance.loose(sec.volatility(0.05), _ZIS["vol_strike_5pct"])
    tolerance.loose(sec.volatility(0.07), _ZIS["vol_strike_7pct"])


# --- input checks ----------------------------------------------------


def test_fewer_than_two_strikes_raises() -> None:
    from pquantlib.exceptions import LibraryException  # noqa: PLC0415
    with pytest.raises(LibraryException, match="at least 2"):
        ZabrInterpolatedSmileSection(
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
    sec_multi = ZabrInterpolatedSmileSection(
        option_date=_option_date(),
        forward=_GAMMA_FREE["forward"],
        strikes=_GAMMA_FREE["strikes"],
        vols=_GAMMA_FREE["input_vols"],
        alpha=0.05,
        beta=0.5,
        nu=0.3,
        rho=0.0,
        gamma=1.0,
        beta_is_fixed=True,
        day_counter=Actual365Fixed(),
        reference_date=_reference_date(),
        max_guesses=3,
    )
    assert sec_multi.rms_error() <= sec_single.rms_error() + 1.0e-9
    assert not math.isnan(sec_multi.rms_error())
