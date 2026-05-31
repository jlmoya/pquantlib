"""NoArbSabrInterpolation fit recovery (W6-A).

The no-arb SABR fit evaluates the Doust terminal-density model per
strike, so it is materially more expensive than the SABR/SVI fits. We
exercise the well-determined sub-problem (fix beta + rho, free alpha +
nu — the standard market convention for no-arb SABR fitting) and
confirm synthetic-parameter recovery. This is a self-consistency probe
(no separate C++ reference values — the C++ fit uses a different
optimiser, so only the recovered model is comparable).
"""

from __future__ import annotations

from pquantlib.experimental.volatility.no_arb_sabr import no_arb_sabr_volatility
from pquantlib.experimental.volatility.no_arb_sabr_interpolation import (
    NoArbSabrInterpolation,
)
from pquantlib.testing import tolerance

# Doust figure-3 generating parameters.
_TAU = 1.0
_BETA = 0.5
_ALPHA = 0.026
_RHO = -0.1
_NU = 0.4
_FWD = 0.0488


def _synthetic_slice() -> tuple[list[float], list[float]]:
    strikes = [0.03, 0.0488, 0.07, 0.09]
    vols = [
        no_arb_sabr_volatility(k, _FWD, _TAU, _ALPHA, _BETA, _NU, _RHO)
        for k in strikes
    ]
    return strikes, vols


def test_fit_recovers_alpha_nu_beta_rho_fixed() -> None:
    """Fix beta + rho, recover alpha + nu from a synthetic slice (TIGHT).

    The alpha/nu sub-problem is well-determined, so the trust-region
    solver recovers the generating params to ~1e-6.
    """
    strikes, vols = _synthetic_slice()
    fit = NoArbSabrInterpolation(
        strikes, vols, _TAU, _FWD,
        alpha=0.03, beta=_BETA, nu=0.5, rho=_RHO,
        beta_is_fixed=True, rho_is_fixed=True,
        max_nfev=80,
    )
    assert fit.converged()
    assert fit.rms_error() < 1e-6
    tolerance.loose(fit.alpha(), _ALPHA)
    tolerance.loose(fit.nu(), _NU)
    # beta / rho pinned at their fixed values.
    tolerance.exact(fit.beta(), _BETA)
    tolerance.exact(fit.rho(), _RHO)


def test_fit_recovers_vols() -> None:
    """Fitted slice reproduces the input market vols (LOOSE)."""
    strikes, vols = _synthetic_slice()
    fit = NoArbSabrInterpolation(
        strikes, vols, _TAU, _FWD,
        alpha=0.03, beta=_BETA, nu=0.5, rho=_RHO,
        beta_is_fixed=True, rho_is_fixed=True,
        max_nfev=80,
    )
    for k, v in zip(strikes, vols, strict=True):
        tolerance.loose(fit.value(k), v)


def test_fit_all_fixed_is_evaluation_only() -> None:
    """All params fixed → no optimisation, value uses the fixed params."""
    strikes, vols = _synthetic_slice()
    fit = NoArbSabrInterpolation(
        strikes, vols, _TAU, _FWD,
        alpha=_ALPHA, beta=_BETA, nu=_NU, rho=_RHO,
        alpha_is_fixed=True, beta_is_fixed=True,
        nu_is_fixed=True, rho_is_fixed=True,
    )
    assert fit.converged()
    tolerance.exact(fit.alpha(), _ALPHA)
    tolerance.exact(fit.beta(), _BETA)
    tolerance.exact(fit.nu(), _NU)
    tolerance.exact(fit.rho(), _RHO)
    # Residuals are ~0 because the fixed params generated the slice.
    assert fit.rms_error() < 1e-12


def test_callable_evaluation_matches_free_function() -> None:
    strikes, vols = _synthetic_slice()
    fit = NoArbSabrInterpolation(
        strikes, vols, _TAU, _FWD,
        alpha=_ALPHA, beta=_BETA, nu=_NU, rho=_RHO,
        alpha_is_fixed=True, beta_is_fixed=True,
        nu_is_fixed=True, rho_is_fixed=True,
    )
    expected = no_arb_sabr_volatility(0.06, _FWD, _TAU, _ALPHA, _BETA, _NU, _RHO)
    tolerance.tight(fit(0.06), expected)
