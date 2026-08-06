"""ZABR closed-form short-maturity volatility.

# C++ parity: ql/termstructures/volatility/zabr.{hpp,cpp} (v1.43) —
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

This module is a thin functional facade over
:class:`~pquantlib.termstructures.volatility.zabr.ZabrModel`, which is
where the ``x(K)`` transform, the Andreasen-Huge ODE and the two
volatility helpers actually live. Keeping one implementation matters:
the ``gamma != 1`` branch integrates ``du/dy = F(y, u)`` with
QuantLib's own ``AdaptiveRungeKutta(eps=1e-8, h1=1e-5, hmin=0)`` — a
Cash-Karp pair with Numerical-Recipes step control, whose second
argument is the *initial step size* and not a relative tolerance.
Cross-validation against the C++ v1.43 probe is at TIGHT tier for every
gamma.

**Evaluation modes.** ``ZabrEvaluation`` also names the two
finite-difference modes ``LocalVolatility`` and ``FullFd``. Those are
properties of a whole smile *section* (they price a strike grid off a
PDE and interpolate), not of a pointwise volatility formula, so they
are not reachable from this function — use
:class:`~pquantlib.termstructures.volatility.zabr_smile_section.ZabrSmileSection`
with the corresponding ``evaluation``.

Math-symbol variable names — ``alpha``, ``beta``, ``nu``, ``rho``,
``gamma``, ``K``, ``F``, ``T`` — are intentional carryovers from the
Andreasen-Huge paper / C++ source.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.volatility.zabr import ZabrModel


class ZabrEvaluation(IntEnum):
    """ZABR evaluation modes.

    # C++ parity: the four tag structs ``ZabrShortMaturityLognormal``,
    # ``ZabrShortMaturityNormal``, ``ZabrLocalVolatility``, ``ZabrFullFd``
    # in ``zabrsmilesection.hpp`` — modelled as an enum because Python has
    # no template tag dispatch.

    ``ProjectedHedge`` has NO C++ counterpart in v1.43 and is rejected
    everywhere; it is retained only so previously-serialised values keep
    their meaning.
    """

    ShortMaturityLognormal = 0
    ShortMaturityNormal = 1
    LocalVolatility = 2
    FullFd = 3
    ProjectedHedge = 4  # no C++ counterpart


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
    """ZABR closed-form volatility at one strike.

    # C++ parity: ``ZabrModel::lognormalVolatility(K)`` and
    #             ``ZabrModel::normalVolatility(K)``.

    Args:
        strike: option strike.
        forward: ATM forward.
        expiry_time: option expiry in years (T - t). The short-maturity
           expansions are leading-order in T and do not use it, but
           ``ZabrModel`` validates that it is positive.
        alpha, beta, nu, rho, gamma: ZABR parameters. ``gamma = 1``
           reduces the x(K) transform to its closed form (no ODE
           integration).
        mode: ``ShortMaturityLognormal`` (default) or
           ``ShortMaturityNormal``. The section-level FD modes raise
           ``LibraryException``.

    Returns:
        ZABR volatility under the requested mode.

    Notes:
        For ``gamma == 1`` the result is the *leading-order* SABR
        formula (lognormal: ``sigma = log(F/K) / x(K)`` with closed-form
        ``x(K)``; ATM: ``sigma = alpha * F^(beta-1)``). It is NOT the
        full Hagan-2002 closed-form (which includes the correction
        ``d`` factor); for the full Hagan formula use
        :func:`~pquantlib.math.interpolations.sabr_formula.sabr_volatility`
        directly.
    """
    if mode in (
        ZabrEvaluation.LocalVolatility,
        ZabrEvaluation.FullFd,
    ):
        raise LibraryException(
            f"ZABR mode {mode.name} is a smile-section evaluation, not a pointwise "
            "formula — construct a ZabrSmileSection with evaluation="
            f"ZabrEvaluation.{mode.name} instead."
        )
    if mode == ZabrEvaluation.ProjectedHedge:
        raise LibraryException(
            "ZABR mode ProjectedHedge has no counterpart in C++ QuantLib v1.43."
        )
    model = ZabrModel(expiry_time, forward, alpha, beta, nu, rho, gamma)
    if mode == ZabrEvaluation.ShortMaturityNormal:
        return model.normal_volatility(strike)
    return model.lognormal_volatility(strike)


__all__ = ["ZabrEvaluation", "zabr_volatility"]
