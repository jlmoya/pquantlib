"""SABR closed-form volatility — Hagan 2002 + Bachelier-vol variant.

# C++ parity: ql/termstructures/volatility/sabr.{hpp,cpp} (v1.42.1).

Implements:

* :func:`sabr_volatility` — shifted-lognormal Hagan 2002 closed form.
* :func:`sabr_normal_volatility` — Bachelier (normal) vol per Deloitte
  whitepaper (referenced in the C++ comment block).
* :func:`shifted_sabr_volatility` — `sabr_volatility(K+shift, F+shift, ...)`.
* :func:`validate_sabr_parameters` — input-bound checks shared by the
  fitter (``SabrInterpolation``) and the smile section (``SabrSmileSection``).

Hagan 2002 formula structure (lognormal arm):

.. code-block:: text

   z      = (nu / alpha) * sqrt(F * K)^(1-beta) * log(F/K)
   B      = 1 - 2 rho z + z^2
   x(z)   = log((sqrt(B) + z - rho) / (1 - rho))
   D      = sqrt(F * K)^(1-beta) * (1 + C/24 + C^2/1920)   where C = (1-beta)^2 log(F/K)^2
   sigma  = (alpha / D) * (z / x(z)) * d

with ``d`` the standard correction factor; near-ATM the ``z / x(z)``
factor is replaced by a Taylor expansion to avoid catastrophic
cancellation. The branch threshold matches the C++ source
(``z*z > QL_EPSILON * 10``).

Bachelier (normal) variant is provided for swap-rate models that prefer
normal vols; both variants share the same ``z``, ``B``, ``x(z)`` path
plus different scale factors.

The free functions use ``# noqa: N803,N806`` per the Phase 9 C task
brief — math-symbol variable names (``alpha``, ``beta``, ``nu``, ``rho``,
``K``, ``F``, ``T``, ``sigma``) are intentional carryovers from the
Hagan paper / C++ source.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.termstructures.volatility.volatility_type import VolatilityType

# Threshold for switching between the (z / x(z)) closed form and its
# Taylor expansion near ATM. Matches the C++ `m = 10` constant in
# `sabr.cpp:69` (`if (std::fabs(z*z) > QL_EPSILON * m)`).
_TAYLOR_BRANCH_THRESHOLD: float = 1.0e-15  # ~ QL_EPSILON (DBL_EPSILON) * 10


# --- bounds ------------------------------------------------------------


def validate_sabr_parameters(
    alpha: float,
    beta: float,
    nu: float,
    rho: float,
) -> None:
    """Validate (alpha, beta, nu, rho) per the C++ ``validateSabrParameters``.

    # C++ parity: ``validateSabrParameters`` (sabr.cpp:149-161).

    - ``alpha > 0``
    - ``0 <= beta <= 1``
    - ``nu >= 0``
    - ``rho * rho < 1``
    """
    qassert.require(alpha > 0.0, f"alpha must be positive: {alpha} not allowed")
    qassert.require(
        0.0 <= beta <= 1.0,
        f"beta must be in [0, 1]: {beta} not allowed",
    )
    qassert.require(nu >= 0.0, f"nu must be non-negative: {nu} not allowed")
    qassert.require(rho * rho < 1.0, f"rho square must be less than one: {rho} not allowed")


# --- lognormal (shifted) -----------------------------------------------


def _unsafe_sabr_lognormal_volatility(
    strike: float,
    forward: float,
    expiry_time: float,
    alpha: float,
    beta: float,
    nu: float,
    rho: float,
) -> float:
    """Closed-form Hagan 2002 lognormal SABR volatility — no input checks.

    # C++ parity: ``unsafeSabrLogNormalVolatility`` (sabr.cpp:37-76).
    """
    one_minus_beta = 1.0 - beta
    A = (forward * strike) ** one_minus_beta  # noqa: N806
    sqrt_a = math.sqrt(A)
    # Use exact log when forward != strike; degenerate-friendly when equal.
    if not math.isclose(forward, strike, abs_tol=1e-15, rel_tol=1e-12):
        log_m = math.log(forward / strike)
    else:
        epsilon = (forward - strike) / strike
        log_m = epsilon - 0.5 * epsilon * epsilon
    z = (nu / alpha) * sqrt_a * log_m
    B = 1.0 - 2.0 * rho * z + z * z  # noqa: N806
    C = one_minus_beta * one_minus_beta * log_m * log_m  # noqa: N806
    tmp = (math.sqrt(B) + z - rho) / (1.0 - rho)
    xx = math.log(tmp)
    D = sqrt_a * (1.0 + C / 24.0 + C * C / 1920.0)  # noqa: N806
    d = 1.0 + expiry_time * (
        one_minus_beta * one_minus_beta * alpha * alpha / (24.0 * A)
        + 0.25 * rho * beta * nu * alpha / sqrt_a
        + (2.0 - 3.0 * rho * rho) * (nu * nu / 24.0)
    )
    # Taylor-expansion branch near ATM to avoid catastrophic cancellation
    # of (z / x(z)) when z is sub-epsilon.
    if abs(z * z) > _TAYLOR_BRANCH_THRESHOLD:
        multiplier = z / xx
    else:
        multiplier = 1.0 - 0.5 * rho * z - (3.0 * rho * rho - 2.0) * z * z / 12.0
    return (alpha / D) * multiplier * d


def _unsafe_sabr_normal_volatility(
    strike: float,
    forward: float,
    expiry_time: float,
    alpha: float,
    beta: float,
    nu: float,
    rho: float,
) -> float:
    """Closed-form Bachelier SABR normal volatility — no input checks.

    # C++ parity: ``unsafeSabrNormalVolatility`` (sabr.cpp:94-132).
    """
    one_minus_beta = 1.0 - beta
    minus_beta = -beta
    A = (forward * strike) ** one_minus_beta  # noqa: N806
    sqrt_a = math.sqrt(A)
    if not math.isclose(forward, strike, abs_tol=1e-15, rel_tol=1e-12):
        log_m = math.log(forward / strike)
    else:
        epsilon = (forward - strike) / strike
        log_m = epsilon - 0.5 * epsilon * epsilon
    z = (nu / alpha) * sqrt_a * log_m
    B = 1.0 - 2.0 * rho * z + z * z  # noqa: N806
    C = one_minus_beta * one_minus_beta * log_m * log_m  # noqa: N806
    D = log_m * log_m  # noqa: N806
    tmp = (math.sqrt(B) + z - rho) / (1.0 - rho)
    xx = math.log(tmp)
    E_1 = 1.0 + D / 24.0 + D * D / 1920.0  # noqa: N806
    E_2 = 1.0 + C / 24.0 + C * C / 1920.0  # noqa: N806
    E = E_1 / E_2  # noqa: N806
    d = 1.0 + expiry_time * (
        minus_beta * (2 - beta) * alpha * alpha / (24.0 * A)
        + 0.25 * rho * beta * nu * alpha / sqrt_a
        + (2.0 - 3.0 * rho * rho) * (nu * nu / 24.0)
    )
    if abs(z * z) > _TAYLOR_BRANCH_THRESHOLD:
        multiplier = z / xx
    else:
        multiplier = 1.0 - 0.5 * rho * z - (3.0 * rho * rho - 2.0) * z * z / 12.0
    F_scale = alpha * (forward * strike) ** (beta / 2.0)  # noqa: N806
    return F_scale * E * multiplier * d


def _unsafe_sabr_volatility(
    strike: float,
    forward: float,
    expiry_time: float,
    alpha: float,
    beta: float,
    nu: float,
    rho: float,
    volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
) -> float:
    """Internal dispatch between lognormal / normal arms — no checks.

    # C++ parity: ``unsafeSabrVolatility`` (sabr.cpp:134-147).
    """
    if volatility_type == VolatilityType.Normal:
        return _unsafe_sabr_normal_volatility(
            strike, forward, expiry_time, alpha, beta, nu, rho
        )
    return _unsafe_sabr_lognormal_volatility(
        strike, forward, expiry_time, alpha, beta, nu, rho
    )


def _unsafe_shifted_sabr_volatility(
    strike: float,
    forward: float,
    expiry_time: float,
    alpha: float,
    beta: float,
    nu: float,
    rho: float,
    shift: float,
    volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
) -> float:
    """Internal shifted dispatch — no checks.

    # C++ parity: ``unsafeShiftedSabrVolatility`` (sabr.cpp:78-92).

    Reduces to the unshifted formula with both strike and forward shifted
    by ``+shift``.
    """
    return _unsafe_sabr_volatility(
        strike + shift,
        forward + shift,
        expiry_time,
        alpha,
        beta,
        nu,
        rho,
        volatility_type,
    )


# --- public API --------------------------------------------------------


def sabr_volatility(
    strike: float,
    forward: float,
    expiry_time: float,
    alpha: float,
    beta: float,
    nu: float,
    rho: float,
    volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
) -> float:
    """SABR closed-form volatility (lognormal or normal arm).

    # C++ parity: ``sabrVolatility`` (sabr.cpp:163-180).

    Args:
        strike: option strike (must be > 0).
        forward: ATM forward rate (must be > 0).
        expiry_time: option expiry in year fractions (must be >= 0).
        alpha: SABR ``alpha`` (vol of vol scale, > 0).
        beta: SABR ``beta`` (CEV exponent, in [0, 1]).
        nu: SABR ``nu`` (vol of vol, >= 0).
        rho: SABR ``rho`` (correlation, |rho| < 1).
        volatility_type: ``ShiftedLognormal`` (default) selects the
            standard Hagan 2002 formula; ``Normal`` selects the
            Bachelier variant.

    Returns:
        SABR volatility (lognormal or absolute, depending on
        ``volatility_type``).
    """
    qassert.require(strike > 0.0, f"strike must be positive: {strike} not allowed")
    qassert.require(
        forward > 0.0,
        f"at-the-money forward rate must be positive: {forward} not allowed",
    )
    qassert.require(
        expiry_time >= 0.0,
        f"expiry time must be non-negative: {expiry_time} not allowed",
    )
    validate_sabr_parameters(alpha, beta, nu, rho)
    return _unsafe_sabr_volatility(
        strike, forward, expiry_time, alpha, beta, nu, rho, volatility_type
    )


def sabr_normal_volatility(
    strike: float,
    forward: float,
    expiry_time: float,
    alpha: float,
    beta: float,
    nu: float,
    rho: float,
) -> float:
    """SABR Bachelier (normal) volatility — convenience wrapper.

    # C++ parity: equivalent to ``sabrVolatility(K, F, T, alpha, beta,
    # nu, rho, VolatilityType::Normal)``.

    Same bound-checks as :func:`sabr_volatility`.
    """
    return sabr_volatility(
        strike,
        forward,
        expiry_time,
        alpha,
        beta,
        nu,
        rho,
        VolatilityType.Normal,
    )


def shifted_sabr_volatility(
    strike: float,
    forward: float,
    expiry_time: float,
    alpha: float,
    beta: float,
    nu: float,
    rho: float,
    shift: float,
    volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
) -> float:
    """Shifted SABR closed-form volatility.

    # C++ parity: ``shiftedSabrVolatility`` (sabr.cpp:182-200).

    With ``shift = 0`` this is identical to :func:`sabr_volatility`.
    """
    qassert.require(
        strike + shift > 0.0,
        f"strike+shift must be positive: {strike}+{shift} not allowed",
    )
    qassert.require(
        forward + shift > 0.0,
        f"at-the-money forward+shift must be positive: {forward}+{shift} not allowed",
    )
    qassert.require(
        expiry_time >= 0.0,
        f"expiry time must be non-negative: {expiry_time} not allowed",
    )
    validate_sabr_parameters(alpha, beta, nu, rho)
    return _unsafe_shifted_sabr_volatility(
        strike, forward, expiry_time, alpha, beta, nu, rho, shift, volatility_type
    )
