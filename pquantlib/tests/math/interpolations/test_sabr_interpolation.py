"""Tests for SabrInterpolation (fit + recover SABR params)."""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.sabr_formula import sabr_volatility
from pquantlib.math.interpolations.sabr_interpolation import SabrInterpolation
from pquantlib.testing import reference_reader, tolerance

_REF = reference_reader.load("cluster/l9c")
_SLICE = _REF["cube_sabr_slice"]


def _synthetic_slice(
    strikes: list[float],
    forward: float,
    expiry: float,
    alpha: float,
    beta: float,
    nu: float,
    rho: float,
) -> list[float]:
    """Generate a strike-vol slice from a known SABR parameter set."""
    return [
        sabr_volatility(k, forward, expiry, alpha, beta, nu, rho)
        for k in strikes
    ]


# --- round-trip recovery -----------------------------------------------


def test_sabr_interpolation_recovers_known_params_default_init() -> None:
    """Round-trip: known params -> synthetic vols -> fit -> recover."""
    strikes = [0.03, 0.035, 0.04, 0.045, 0.05]
    forward = 0.04
    expiry = 3.0
    alpha_true = 0.03
    beta_true = 0.6
    nu_true = 0.3
    rho_true = 0.0
    vols = _synthetic_slice(strikes, forward, expiry, alpha_true, beta_true, nu_true, rho_true)

    fit = SabrInterpolation(
        strikes, vols, expiry, forward,
        alpha=0.04, beta=beta_true, nu=0.2, rho=0.1,
        beta_is_fixed=True,
    )
    # Beta is pinned to the true value (a standard practice — beta is
    # under-determined on a short slice). The other three should recover
    # to LOOSE tier (1e-8) and the fit RMS error to ~ 0.
    tolerance.loose(fit.alpha(), alpha_true)
    tolerance.loose(fit.nu(), nu_true)
    tolerance.loose(fit.rho(), rho_true)
    tolerance.loose(fit.beta(), beta_true)
    assert fit.rms_error() < 1e-6


def test_sabr_interpolation_reproduces_input_vols_on_pillars() -> None:
    """Even without recovering exact params, the fit should reproduce
    the input vols at each strike to LOOSE tier."""
    strikes = _SLICE["strikes"]
    vols = _SLICE["sabr_vols"]
    forward = _SLICE["forward"]
    expiry = _SLICE["expiry"]
    beta_pinned = _SLICE["beta"]

    fit = SabrInterpolation(
        strikes, vols, expiry, forward,
        alpha=0.02, beta=beta_pinned, nu=0.4, rho=0.2,
        beta_is_fixed=True,
    )
    for k, v in zip(strikes, vols, strict=True):
        tolerance.loose(fit.value(k), v)


def test_sabr_interpolation_alpha_only_fit_is_well_conditioned() -> None:
    """With beta, nu, rho pinned to truth, alpha must converge to truth."""
    strikes = [0.03, 0.04, 0.05, 0.06, 0.07]
    forward = 0.05
    expiry = 1.0
    alpha_true = 0.025
    beta_true = 0.5
    nu_true = 0.4
    rho_true = -0.2
    vols = _synthetic_slice(strikes, forward, expiry, alpha_true, beta_true, nu_true, rho_true)

    fit = SabrInterpolation(
        strikes, vols, expiry, forward,
        alpha=0.05, beta=beta_true, nu=nu_true, rho=rho_true,
        beta_is_fixed=True, nu_is_fixed=True, rho_is_fixed=True,
    )
    tolerance.loose(fit.alpha(), alpha_true)
    assert fit.beta() == beta_true
    assert fit.nu() == nu_true
    assert fit.rho() == rho_true


def test_sabr_interpolation_all_fixed_returns_initial_values() -> None:
    """If every parameter is fixed, no optimisation runs."""
    strikes = [0.03, 0.04, 0.05]
    vols = [0.20, 0.18, 0.19]
    fit = SabrInterpolation(
        strikes, vols, 1.0, 0.05,
        alpha=0.04, beta=0.5, nu=0.4, rho=-0.1,
        alpha_is_fixed=True, beta_is_fixed=True,
        nu_is_fixed=True, rho_is_fixed=True,
    )
    assert fit.alpha() == 0.04
    assert fit.beta() == 0.5
    assert fit.nu() == 0.4
    assert fit.rho() == -0.1
    assert fit.converged() is True


# --- vega weighting ----------------------------------------------------


def test_sabr_interpolation_vega_weighting_off_matches_uniform() -> None:
    """With vega_weighted=False the residual vector is the raw model-mkt diff."""
    strikes = [0.03, 0.04, 0.05, 0.06, 0.07]
    forward = 0.05
    expiry = 1.0
    vols = _synthetic_slice(strikes, forward, expiry, 0.025, 0.5, 0.4, -0.2)
    fit = SabrInterpolation(
        strikes, vols, expiry, forward,
        alpha=0.05, beta=0.5, nu=0.5, rho=-0.1,
        beta_is_fixed=True,
        vega_weighted=False,
    )
    assert fit.rms_error() < 1e-5


def test_sabr_interpolation_vega_weighted_arm_runs() -> None:
    """Smoke test for vega-weighted path."""
    strikes = [0.03, 0.04, 0.05, 0.06, 0.07]
    forward = 0.05
    expiry = 1.0
    vols = _synthetic_slice(strikes, forward, expiry, 0.025, 0.5, 0.4, -0.2)
    fit = SabrInterpolation(
        strikes, vols, expiry, forward,
        alpha=0.05, beta=0.5, nu=0.5, rho=-0.1,
        beta_is_fixed=True,
        vega_weighted=True,
    )
    # Vega-weighted residual is much smaller (weights sum to 1).
    assert fit.rms_error() < 1e-5


# --- evaluation --------------------------------------------------------


def test_sabr_interpolation_value_matches_direct_sabr_call() -> None:
    """interp(k) must equal sabr_volatility(k, ...) with fitted params."""
    strikes = [0.03, 0.04, 0.05, 0.06, 0.07]
    forward = 0.05
    expiry = 1.0
    vols = _synthetic_slice(strikes, forward, expiry, 0.025, 0.5, 0.4, -0.2)
    fit = SabrInterpolation(
        strikes, vols, expiry, forward,
        beta_is_fixed=True, beta=0.5,
    )
    test_strike = 0.045
    direct = sabr_volatility(
        test_strike, forward, expiry,
        fit.alpha(), fit.beta(), fit.nu(), fit.rho(),
    )
    tolerance.exact(fit(test_strike), direct)


# --- input bounds ------------------------------------------------------


def test_sabr_interpolation_rejects_too_few_strikes() -> None:
    with pytest.raises(LibraryException, match="at least 2"):
        SabrInterpolation([0.05], [0.20], 1.0, 0.05)


def test_sabr_interpolation_rejects_mismatched_lengths() -> None:
    with pytest.raises(LibraryException, match="same length"):
        SabrInterpolation([0.04, 0.05], [0.20], 1.0, 0.05)


def test_sabr_interpolation_rejects_negative_expiry() -> None:
    with pytest.raises(LibraryException, match="expiry_time"):
        SabrInterpolation([0.04, 0.05], [0.20, 0.18], -1.0, 0.05)


# --- defaults ----------------------------------------------------------


def test_sabr_interpolation_default_params_apply_when_none() -> None:
    """Default values mirror C++ ``SABRSpecs::defaultValues``."""
    strikes = [0.03, 0.04, 0.05, 0.06, 0.07]
    vols = [0.30, 0.22, 0.20, 0.22, 0.30]
    fit = SabrInterpolation(strikes, vols, 1.0, 0.05)
    # Just verify a fit was produced; defaults should give a reasonable start.
    assert 0.0 < fit.alpha() < 5.0
    assert 0.0 <= fit.beta() <= 1.0
    assert 0.0 <= fit.nu() < 10.0
    assert -1.0 <= fit.rho() <= 1.0
