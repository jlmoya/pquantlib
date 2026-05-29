"""Cross-validate ZabrSmileSection against the L10-C C++ probe.

Reference: ``migration-harness/references/cluster/l10c.json`` —
``zabr_smile_section`` section.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.zabr_formula import ZabrEvaluation
from pquantlib.termstructures.volatility.zabr_smile_section import (
    ZabrSmileSection,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/l10c")


def _make_section() -> ZabrSmileSection:
    return ZabrSmileSection(
        exercise_time=5.0,
        forward=0.05,
        zabr_params=(0.04, 0.5, 0.4, -0.1, 0.75),
        evaluation=ZabrEvaluation.ShortMaturityLognormal,
    )


def test_exercise_time_and_atm(cpp: dict[str, Any]) -> None:
    block = cpp["zabr_smile_section"]
    section = _make_section()
    tolerance.tight(section.exercise_time(), float(block["exercise_time"]))
    tolerance.tight(section.atm_level(), float(block["atm_level"]))


def test_inspectors_return_zabr_params(cpp: dict[str, Any]) -> None:
    block = cpp["zabr_smile_section"]
    section = _make_section()
    tolerance.tight(section.alpha(), float(block["alpha"]))
    tolerance.tight(section.beta(), float(block["beta"]))
    tolerance.tight(section.nu(), float(block["nu"]))
    tolerance.tight(section.rho(), float(block["rho"]))
    tolerance.tight(section.gamma(), float(block["gamma"]))


def test_vol_atm_matches_cpp(cpp: dict[str, Any]) -> None:
    block = cpp["zabr_smile_section"]
    section = _make_section()
    tolerance.tight(section.volatility(0.05), float(block["vol_atm"]))


def test_vol_strikes_match_cpp(cpp: dict[str, Any]) -> None:
    """OTM strikes: 4% and 6%. Custom tier on the gamma=0.75 RK45 path."""
    block = cpp["zabr_smile_section"]
    section = _make_section()
    tolerance.custom(
        section.volatility(0.04), float(block["vol_strike_4pct"]),
        abs_tol=1.0e-5, rel_tol=1.0e-5,
        reason="RK45 rtol/atol drift between scipy.solve_ivp and C++ AdaptiveRungeKutta",
    )
    tolerance.custom(
        section.volatility(0.06), float(block["vol_strike_6pct"]),
        abs_tol=1.0e-5, rel_tol=1.0e-5,
        reason="RK45 rtol/atol drift between scipy.solve_ivp and C++ AdaptiveRungeKutta",
    )


def test_variance_atm(cpp: dict[str, Any]) -> None:
    """variance = vol^2 * T."""
    block = cpp["zabr_smile_section"]
    section = _make_section()
    tolerance.tight(section.variance(0.05), float(block["variance_atm"]))


def test_min_strike_is_zero() -> None:
    section = _make_section()
    tolerance.exact(section.min_strike(), 0.0)


def test_max_strike_is_inf() -> None:
    section = _make_section()
    assert section.max_strike() == float("inf")


def test_gamma_one_reduces_to_sabr_short_maturity() -> None:
    """A ZabrSmileSection with gamma=1 returns the SABR leading-order vol.

    At gamma=1 the ZABR x(K) transform has a closed form (no ODE).
    The vol returned is *not* the full Hagan SABR — it's the
    short-maturity expansion (i.e. no ``d`` correction factor).
    Verify by comparing ATM (where ATM-closed-form equals
    ``alpha * F^(beta-1)``).
    """
    section = ZabrSmileSection(
        exercise_time=5.0,
        forward=0.05,
        zabr_params=(0.04, 0.5, 0.4, -0.1, 1.0),
        evaluation=ZabrEvaluation.ShortMaturityLognormal,
    )
    expected_atm = 0.04 * (0.05 ** (0.5 - 1.0))
    tolerance.tight(section.volatility(0.05), expected_atm)


def test_normal_arm_atm() -> None:
    """Normal arm at ATM equals ``alpha * F^beta``."""
    section = ZabrSmileSection(
        exercise_time=5.0,
        forward=0.05,
        zabr_params=(0.04, 0.5, 0.4, -0.1, 0.75),
        evaluation=ZabrEvaluation.ShortMaturityNormal,
    )
    expected = 0.04 * (0.05**0.5)
    tolerance.tight(section.volatility(0.05), expected)


def test_zero_forward_raises() -> None:
    with pytest.raises(LibraryException):
        ZabrSmileSection(
            exercise_time=1.0,
            forward=0.0,
            zabr_params=(0.04, 0.5, 0.4, -0.1, 0.75),
        )


def test_negative_forward_raises() -> None:
    with pytest.raises(LibraryException):
        ZabrSmileSection(
            exercise_time=1.0,
            forward=-0.01,
            zabr_params=(0.04, 0.5, 0.4, -0.1, 0.75),
        )


def test_date_anchored_constructor() -> None:
    """Date-anchored construction uses Actual365Fixed by default."""
    ref = Date.from_ymd(15, Month.January, 2026)
    expiry = Date.from_ymd(15, Month.January, 2031)  # exactly 5 years.
    section = ZabrSmileSection(
        forward=0.05,
        zabr_params=(0.04, 0.5, 0.4, -0.1, 0.75),
        exercise_date=expiry,
        reference_date=ref,
    )
    dc = Actual365Fixed()
    expected_t = dc.year_fraction(ref, expiry)
    tolerance.tight(section.exercise_time(), expected_t)


def test_clamps_small_strike() -> None:
    """Strike <= 1e-6 is clamped to 1e-6 (matches C++ behavior)."""
    section = _make_section()
    v_clamped = section.volatility(1.0e-9)
    v_floor = section.volatility(1.0e-6)
    # Same value because the underlying computation gets the same
    # clamped input.
    tolerance.tight(v_clamped, v_floor)


def test_fd_evaluation_mode_raises() -> None:
    """Constructing a section with an FD evaluation mode is fine.

    The exception only fires when ``volatility(K)`` is called.
    """
    section = ZabrSmileSection(
        exercise_time=5.0,
        forward=0.05,
        zabr_params=(0.04, 0.5, 0.4, -0.1, 0.75),
        evaluation=ZabrEvaluation.LocalVolatility,
    )
    with pytest.raises(LibraryException, match="not implemented"):
        section.volatility(0.05)
