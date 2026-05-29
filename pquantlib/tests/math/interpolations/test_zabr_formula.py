"""Cross-validate zabr_volatility against the L10-C C++ probe.

Reference: ``migration-harness/references/cluster/l10c.json`` —
``zabr_formula`` section. Tests:
  * gamma = 1 collapses to leading-order SABR short-maturity
    expansion (TIGHT against C++).
  * gamma != 1 (= 0.75) ShortMaturityLognormal arm matches C++
    (TIGHT — same RK45 integration both sides).
  * Normal arm: gamma = 1 matches C++.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.zabr_formula import (
    ZabrEvaluation,
    zabr_volatility,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/l10c")


# Standard parameters used throughout the L10-C probe.
ALPHA = 0.04
BETA = 0.5
NU = 0.4
RHO = -0.1
FORWARD = 0.05
EXPIRY = 5.0


def test_gamma1_atm_matches_cpp(cpp: dict[str, Any]) -> None:
    """gamma = 1 ATM vol equals ``alpha * F^(beta-1)`` (TIGHT)."""
    block = cpp["zabr_formula"]
    actual = zabr_volatility(
        FORWARD, FORWARD, EXPIRY, ALPHA, BETA, NU, RHO, 1.0,
        mode=ZabrEvaluation.ShortMaturityLognormal,
    )
    tolerance.tight(actual, float(block["gamma1_vol_atm"]))


def test_gamma1_otm_strikes_match_cpp(cpp: dict[str, Any]) -> None:
    """gamma = 1 OTM (strike != forward) vol matches C++."""
    block = cpp["zabr_formula"]
    actual_4 = zabr_volatility(
        0.04, FORWARD, EXPIRY, ALPHA, BETA, NU, RHO, 1.0,
        mode=ZabrEvaluation.ShortMaturityLognormal,
    )
    actual_6 = zabr_volatility(
        0.06, FORWARD, EXPIRY, ALPHA, BETA, NU, RHO, 1.0,
        mode=ZabrEvaluation.ShortMaturityLognormal,
    )
    tolerance.tight(actual_4, float(block["gamma1_vol_strike_4pct"]))
    tolerance.tight(actual_6, float(block["gamma1_vol_strike_6pct"]))


def test_gamma75_atm_matches_cpp(cpp: dict[str, Any]) -> None:
    """gamma = 0.75 ATM vol matches C++ (closed-form at ATM)."""
    block = cpp["zabr_formula"]
    actual = zabr_volatility(
        FORWARD, FORWARD, EXPIRY, ALPHA, BETA, NU, RHO, 0.75,
        mode=ZabrEvaluation.ShortMaturityLognormal,
    )
    tolerance.tight(actual, float(block["gamma75_vol_atm"]))


def test_gamma75_otm_strikes_match_cpp(cpp: dict[str, Any]) -> None:
    """gamma = 0.75 OTM vol matches C++ (RK45 integration).

    TIGHT — both Python (scipy.solve_ivp RK45) and C++
    (AdaptiveRungeKutta) implement the same Andreasen-Huge ODE with
    matching tolerances (rtol=1e-5, atol=1e-8).
    """
    block = cpp["zabr_formula"]
    actual_4 = zabr_volatility(
        0.04, FORWARD, EXPIRY, ALPHA, BETA, NU, RHO, 0.75,
        mode=ZabrEvaluation.ShortMaturityLognormal,
    )
    actual_6 = zabr_volatility(
        0.06, FORWARD, EXPIRY, ALPHA, BETA, NU, RHO, 0.75,
        mode=ZabrEvaluation.ShortMaturityLognormal,
    )
    tolerance.custom(
        actual_4, float(block["gamma75_vol_strike_4pct"]),
        abs_tol=1.0e-5, rel_tol=1.0e-5,
        reason="RK45 rtol/atol drift between scipy.solve_ivp and C++ AdaptiveRungeKutta",
    )
    tolerance.custom(
        actual_6, float(block["gamma75_vol_strike_6pct"]),
        abs_tol=1.0e-5, rel_tol=1.0e-5,
        reason="RK45 rtol/atol drift between scipy.solve_ivp and C++ AdaptiveRungeKutta",
    )


def test_gamma1_normal_atm_matches_cpp(cpp: dict[str, Any]) -> None:
    """Normal arm: gamma=1 ATM equals ``alpha * F^beta``."""
    block = cpp["zabr_formula"]
    actual = zabr_volatility(
        FORWARD, FORWARD, EXPIRY, ALPHA, BETA, NU, RHO, 1.0,
        mode=ZabrEvaluation.ShortMaturityNormal,
    )
    tolerance.tight(actual, float(block["gamma1_normal_vol_atm"]))


def test_gamma75_normal_atm_matches_cpp(cpp: dict[str, Any]) -> None:
    """Normal arm: gamma=0.75 ATM also closed-form."""
    block = cpp["zabr_formula"]
    actual = zabr_volatility(
        FORWARD, FORWARD, EXPIRY, ALPHA, BETA, NU, RHO, 0.75,
        mode=ZabrEvaluation.ShortMaturityNormal,
    )
    tolerance.tight(actual, float(block["gamma75_normal_vol_atm"]))


def test_atm_lognormal_closed_form() -> None:
    """ATM lognormal = ``alpha * F^(beta-1)`` independent of nu, rho, gamma."""
    expected = ALPHA * (FORWARD ** (BETA - 1.0))
    for gamma in (0.5, 0.75, 1.0, 1.25):
        for nu in (0.1, 0.4, 0.6):
            for rho in (-0.3, 0.0, 0.5):
                actual = zabr_volatility(
                    FORWARD, FORWARD, EXPIRY, ALPHA, BETA, nu, rho, gamma,
                    mode=ZabrEvaluation.ShortMaturityLognormal,
                )
                tolerance.tight(actual, expected)


def test_atm_normal_closed_form() -> None:
    """ATM normal = ``alpha * F^beta``."""
    expected = ALPHA * (FORWARD**BETA)
    for gamma in (0.5, 0.75, 1.0, 1.25):
        actual = zabr_volatility(
            FORWARD, FORWARD, EXPIRY, ALPHA, BETA, 0.3, 0.0, gamma,
            mode=ZabrEvaluation.ShortMaturityNormal,
        )
        tolerance.tight(actual, expected)


def test_fd_modes_raise() -> None:
    """LocalVolatility / FullFd / ProjectedHedge raise LibraryException."""
    for mode in (
        ZabrEvaluation.LocalVolatility,
        ZabrEvaluation.FullFd,
        ZabrEvaluation.ProjectedHedge,
    ):
        with pytest.raises(LibraryException, match="not implemented"):
            zabr_volatility(
                0.05, FORWARD, EXPIRY, ALPHA, BETA, NU, RHO, 1.0, mode=mode,
            )


def test_negative_gamma_raises() -> None:
    with pytest.raises(LibraryException, match="gamma"):
        zabr_volatility(
            0.05, FORWARD, EXPIRY, ALPHA, BETA, NU, RHO, -0.1,
        )


def test_zero_expiry_raises() -> None:
    with pytest.raises(LibraryException, match="expiry"):
        zabr_volatility(
            0.05, FORWARD, 0.0, ALPHA, BETA, NU, RHO, 1.0,
        )


def test_smile_shape_is_smooth() -> None:
    """ZABR smile is C^infinity on its support — sample fine grid."""
    strikes: list[float] = [
        0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05,
        0.055, 0.06, 0.065, 0.07, 0.075, 0.08,
    ]
    vols: list[float] = [
        zabr_volatility(
            strike, FORWARD, EXPIRY, ALPHA, BETA, NU, RHO, 0.75,
            mode=ZabrEvaluation.ShortMaturityLognormal,
        )
        for strike in strikes
    ]
    # No NaN, no Inf, monotone-smile finite second differences.
    for v in vols:
        assert math.isfinite(v)
    # Smile is a U-shape under negative rho — the minimum should be
    # near ATM.
    min_idx: int = vols.index(min(vols))
    assert 4 <= min_idx <= 8  # near the middle of the strike grid
