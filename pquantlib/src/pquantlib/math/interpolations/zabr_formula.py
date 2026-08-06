"""ZABR closed-form short-maturity volatility.

# C++ parity: ql/termstructures/volatility/zabr.{hpp,cpp} (v1.42.1) —
# ``ZabrModel::lognormalVolatility`` + ``ZabrModel::normalVolatility``.

The ZABR model (Andreasen & Huge, *ZABR — Expansions for the Masses*,
SSRN 1980726, December 2011) generalises the SABR model with one
extra parameter :math:`\\gamma` controlling the elasticity of the
variance dynamics:

.. math::

   dF_t &= \\alpha_t \\, F_t^{\\beta} \\, dW_t \\\\
   d\\alpha_t &= \\nu \\, \\alpha_t^{\\gamma} \\, dZ_t \\\\
   d\\langle W, Z \\rangle_t &= \\rho \\, dt

When :math:`\\gamma = 1` the ZABR model collapses to the standard SABR
model (where the volatility :math:`\\alpha` is a geometric Brownian
motion with vol-of-vol :math:`\\nu`). For :math:`\\gamma \\neq 1` the
volatility evolves as a constant-elasticity-of-variance (CEV) process
with elasticity :math:`\\gamma`.

This module ports the closed-form short-maturity expansions used in
``ZabrModel::lognormalVolatility`` and ``ZabrModel::normalVolatility``:

**Lognormal vol (short-maturity, gamma = 1 — i.e. SABR):**
matches :func:`pquantlib.math.interpolations.sabr_formula.sabr_volatility`.

**Lognormal vol (short-maturity, gamma != 1):** uses the
Andreasen-Huge ``x(K)`` transform — for ``gamma == 1`` the transform
has a closed form (cf. ``zabr.cpp:316-339`` and the standard SABR
``x(z)`` formula); for ``gamma != 1`` C++ uses an Adaptive
Runge-Kutta to integrate the ``F(y, u)`` ODE (``zabr.cpp:340-358``).

**gamma != 1 ODE integration.** The C++ implementation uses an
``AdaptiveRungeKutta<Real>(1e-8, 1e-5, 0.0)`` integrator stepping in ``y``
from 0 to ``y(strike)`` (with ``y(strike) = (forward^(1-beta) -
strike^(1-beta)) / (1-beta) * alpha^(gamma - 2)``), and so does the port,
using :class:`~pquantlib.math.ode.adaptive_runge_kutta.AdaptiveRungeKutta`.

This used to delegate to ``scipy.integrate.solve_ivp(method="RK45",
rtol=1e-5, atol=1e-8)``, described as "the same RK45 method and default
tolerances ... matching the C++ AdaptiveRungeKutta semantics: the second
argument is relative, the first is absolute". Both halves of that were
wrong. The C++ constructor is ``(eps, h1, hmin)`` — the 1e-5 is the
*initial step size*, not a relative tolerance — and scipy's "RK45" is
Dormand-Prince while QuantLib's is Cash-Karp, with a different embedded
error estimate and a different step controller. Measured against the
L10-C probe the delegation was 8.1e-8 relative out at the 4% strike, not
the TIGHT the docstring claimed.

**Carve-out — Local / FullFd / ProjectedHedge.** The C++
``ZabrModel`` also provides three FD-based evaluation modes
(``localVolatility``, ``fdPrice``, ``fullFdPrice``) that require a
1-D or 2-D PDE solver. These are deferred — the FD framework is L5-D
material and the ZABR PDE specialisation (Dupire / 2-D backward
solver) is a non-trivial additional port. The Python ``ZabrEvaluation``
IntEnum lists them so users get a clear ``LibraryException`` if they
ask for those modes.

Math-symbol variable names — ``alpha``, ``beta``, ``nu``, ``rho``,
``gamma``, ``K``, ``F``, ``T`` — are intentional carryovers from the
Andreasen-Huge paper / C++ source.
"""

from __future__ import annotations

import math
from enum import IntEnum

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.sabr_formula import (
    sabr_volatility,
    validate_sabr_parameters,
)
from pquantlib.math.ode.adaptive_runge_kutta import AdaptiveRungeKutta
from pquantlib.termstructures.volatility.volatility_type import VolatilityType


class ZabrEvaluation(IntEnum):
    """ZABR evaluation modes.

    # C++ parity: the four tag structs ``ZabrShortMaturityLognormal``,
    # ``ZabrShortMaturityNormal``, ``ZabrLocalVolatility``, ``ZabrFullFd``
    # in ``zabrsmilesection.hpp:42-46``.
    """

    ShortMaturityLognormal = 0
    ShortMaturityNormal = 1
    LocalVolatility = 2
    FullFd = 3
    ProjectedHedge = 4  # Phase 9 follow-up — not in C++ v1.42.1 either


def _validate_zabr_parameters(
    alpha: float,
    beta: float,
    nu: float,
    rho: float,
    gamma: float,
    forward: float,
    expiry_time: float,
) -> None:
    """Validate ZABR inputs per C++ ``ZabrModel`` constructor (zabr.cpp:43-56)."""
    validate_sabr_parameters(alpha, beta, nu, rho)
    qassert.require(gamma >= 0.0, f"gamma must be non-negative: {gamma} not allowed")
    qassert.require(forward >= 0.0, f"forward must be non-negative: {forward} not allowed")
    qassert.require(
        expiry_time > 0.0,
        f"expiry time must be positive: {expiry_time} not allowed",
    )


def _zabr_y(
    strike: float,
    forward: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    """Internal ``y(strike)`` transform.

    # C++ parity: ``ZabrModel::y`` (zabr.cpp:363-375).
    """
    if math.isclose(beta, 1.0, abs_tol=1e-15, rel_tol=1e-13):
        return math.log(forward / strike) * (alpha ** (gamma - 2.0))
    # beta != 1
    if strike < 0.0:
        sign_term = (forward ** (1.0 - beta)) + ((-strike) ** (1.0 - beta))
    else:
        sign_term = (forward ** (1.0 - beta)) - (strike ** (1.0 - beta))
    return sign_term * (alpha ** (gamma - 2.0)) / (1.0 - beta)


def _zabr_f_ode(y: float, u: float, nu: float, rho: float, gamma: float) -> float:
    """The ODE right-hand side ``du/dy = F(y, u)`` for gamma != 1.

    # C++ parity: ``ZabrModel::F`` (zabr.cpp:377-385).

    .. code-block:: text

       A = 1 + (gamma - 2)^2 * nu^2 * y^2 + 2 * rho * (gamma - 2) * nu * y
       B = 2 * rho * (1 - gamma) * nu + 2 * (1 - gamma) * (gamma - 2) * nu^2 * y
       C = (1 - gamma)^2 * nu^2
       F = (-B*u + sqrt(B^2 u^2 - 4 A (C u^2 - 1))) / (2 A)
    """
    g2 = gamma - 2.0
    A = 1.0 + g2 * g2 * nu * nu * y * y + 2.0 * rho * g2 * nu * y  # noqa: N806
    B = 2.0 * rho * (1.0 - gamma) * nu + 2.0 * (1.0 - gamma) * g2 * nu * nu * y  # noqa: N806
    C = (1.0 - gamma) * (1.0 - gamma) * nu * nu  # noqa: N806
    radicand = B * B * u * u - 4.0 * A * (C * u * u - 1.0)
    return (-B * u + math.sqrt(max(radicand, 0.0))) / (2.0 * A)


def _zabr_x_general(
    strike: float,
    forward: float,
    alpha: float,
    beta: float,
    nu: float,
    rho: float,
    gamma: float,
) -> float:
    """The Andreasen-Huge ``x(K)`` transform.

    # C++ parity: ``ZabrModel::x`` (zabr.cpp:312-361).

    For ``gamma == 1`` there is a closed form. For ``gamma != 1`` the
    transform is the solution at ``y(strike)`` of the ODE
    ``du/dy = F(y, u)`` with ``u(0) = 0``, scaled by
    ``alpha^(1-gamma)``.
    """
    y_strike = _zabr_y(strike, forward, alpha, beta, gamma)
    # C++ uses transformed nu: nu_internal = nu_input * alpha^(1 - gamma)
    # (zabr.cpp:47). The y / F formulas below already presume this
    # transformed nu. To match the C++ semantics from the ZabrModel ctor
    # we must apply that transform here too.
    nu_transformed = nu * (alpha ** (1.0 - gamma))
    if math.isclose(gamma, 1.0, abs_tol=1e-15, rel_tol=1e-13):
        # Closed form (zabr.cpp:332-338):
        #   J = sqrt(1 + nu^2 y^2 - 2 rho nu y)
        #   x = log((J + nu y - rho) / (1 - rho)) / nu
        # Note the C++ code uses the *transformed* nu (alpha^(1-gamma) = 1
        # for gamma=1, so they coincide).
        nu_use = nu_transformed
        # If forward == strike, y_strike = 0 and x evaluates to 0 by
        # construction; guard log()/division.
        if math.isclose(y_strike, 0.0, abs_tol=1e-15, rel_tol=1e-13):
            return 0.0
        j = math.sqrt(1.0 + nu_use * nu_use * y_strike * y_strike - 2.0 * rho * nu_use * y_strike)
        return math.log((j + nu_use * y_strike - rho) / (1.0 - rho)) / nu_use
    # gamma != 1 — RK45 from u(0)=0 to u(y_strike).
    if math.isclose(y_strike, 0.0, abs_tol=1e-15, rel_tol=1e-13):
        return 0.0
    nu_use = nu_transformed

    def rhs(y: float, u: float) -> float:
        return _zabr_f_ode(y, u, nu_use, rho, gamma)

    # C++ parity: zabr.cpp:329 — AdaptiveRungeKutta<Real> rk(1.0E-8, 1.0E-5, 0.0),
    # i.e. eps = 1e-8, initial step h1 = 1e-5, hmin = 0. Integrated from
    # (y0, u0) = (0, 0) to y = y(strike); zabr.cpp:356-357.
    u_final = AdaptiveRungeKutta(1.0e-8, 1.0e-5, 0.0).solve_1d(rhs, 0.0, 0.0, y_strike)
    # C++ scales by alpha^(1 - gamma) (zabr.cpp:358).
    return u_final * (alpha ** (1.0 - gamma))


def _zabr_lognormal_helper(strike: float, forward: float, alpha: float, beta: float, x: float) -> float:
    """ZABR lognormal vol helper, given precomputed ``x = x(strike)``.

    # C++ parity: ``ZabrModel::lognormalVolatilityHelper``
    #             (zabr.cpp:58-64).
    """
    if math.isclose(strike, forward, abs_tol=1e-15, rel_tol=1e-13):
        return (forward ** (beta - 1.0)) * alpha
    return math.log(forward / strike) / x


def _zabr_normal_helper(strike: float, forward: float, alpha: float, beta: float, x: float) -> float:
    """ZABR normal (Bachelier) vol helper.

    # C++ parity: ``ZabrModel::normalVolatilityHelper`` (zabr.cpp:78-83).
    """
    if math.isclose(strike, forward, abs_tol=1e-15, rel_tol=1e-13):
        return (forward**beta) * alpha
    return (forward - strike) / x


def zabr_volatility(
    strike: float,
    forward: float,
    expiry_time: float,
    alpha: float,
    beta: float,
    nu: float,
    rho: float,
    gamma: float,
    mode: ZabrEvaluation = ZabrEvaluation.ShortMaturityLognormal,
) -> float:
    """ZABR closed-form volatility.

    # C++ parity: ``ZabrModel::lognormalVolatility(K)`` and
    #             ``ZabrModel::normalVolatility(K)`` (zabr.cpp:66-95).

    Args:
        strike: option strike.
        forward: ATM forward.
        expiry_time: option expiry in years (T - t). Not used by the
           short-maturity expansions themselves (they are leading-order
           in T), but validated for non-negativity.
        alpha, beta, nu, rho, gamma: ZABR parameters. ``gamma = 1``
           reduces the x(K) transform to its closed form (no ODE
           integration), giving a leading-order SABR-like vol.
        mode: one of ``ShortMaturityLognormal`` (default),
           ``ShortMaturityNormal``, or one of the FD modes
           (``LocalVolatility`` / ``FullFd`` / ``ProjectedHedge``),
           the latter three raise ``LibraryException``.

    Returns:
        ZABR volatility under the requested mode.

    Notes:
        For ``gamma == 1`` the result is the *leading-order* SABR
        formula (lognormal: ``sigma = log(F/K) / x(K)`` with closed-form
        ``x(K)``; ATM: ``sigma = alpha * F^(beta-1)``). It is NOT the
        full Hagan-2002 closed-form (which includes the correction
        ``d`` factor); for the full Hagan formula use
        :func:`sabr_volatility` directly.
    """
    _validate_zabr_parameters(alpha, beta, nu, rho, gamma, forward, expiry_time)
    if mode in (ZabrEvaluation.LocalVolatility, ZabrEvaluation.FullFd):
        # C++ declares all four evaluation tags in zabrsmilesection.hpp:42-45,
        # not in the interpolation: LocalVolatility and FullFd price a strike
        # grid off a PDE and interpolate, so they are properties of a whole
        # smile *section*, not of a pointwise volatility formula. They are
        # implemented — on ZabrSmileSection, which is where C++ puts them.
        raise LibraryException(
            f"ZABR mode {mode.name} is a smile-section mode, not a pointwise "
            "formula — use ZabrSmileSection. This function supports "
            "ShortMaturityLognormal and ShortMaturityNormal."
        )
    if mode is ZabrEvaluation.ProjectedHedge:
        # Verified against v1.43: `ProjectedHedge` appears nowhere in ql/ or
        # test-suite/. C++ has exactly four tags and this is not one of them.
        raise LibraryException(
            "ZABR mode ProjectedHedge has no counterpart in C++ v1.43 — the "
            "four evaluation tags are ShortMaturityLognormal, "
            "ShortMaturityNormal, LocalVolatility and FullFd."
        )
    # ATM short-circuit: at strike == forward both helpers return a
    # closed-form value independent of x(K). This avoids a degenerate
    # x(K) = 0 numerator/denominator that would otherwise need special
    # handling.
    if math.isclose(strike, forward, abs_tol=1e-15, rel_tol=1e-13):
        if mode == ZabrEvaluation.ShortMaturityNormal:
            return (forward**beta) * alpha
        return (forward ** (beta - 1.0)) * alpha
    # Off-ATM: compute x(K), then lognormal/normal helper. For
    # ``gamma == 1`` this still uses the closed-form branch of
    # ``_zabr_x_general``; for ``gamma != 1`` it integrates the
    # Andreasen-Huge ODE via RK45.
    _ = sabr_volatility  # keep import — used by tests for documented divergence
    _ = VolatilityType
    x = _zabr_x_general(strike, forward, alpha, beta, nu, rho, gamma)
    if mode == ZabrEvaluation.ShortMaturityNormal:
        return _zabr_normal_helper(strike, forward, alpha, beta, x)
    return _zabr_lognormal_helper(strike, forward, alpha, beta, x)
