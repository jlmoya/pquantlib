"""AnalyticH1HWEngine — Grzelak-Oosterlee H1-HW Heston/Hull-White approximation.

# C++ parity: ql/pricingengines/vanilla/analytich1hwengine.{hpp,cpp} (v1.43) —
# ``class AnalyticH1HWEngine : public AnalyticHestonHullWhiteEngine``.

Extends :class:`~pquantlib.pricingengines.vanilla.analytic_heston_hull_white_engine.AnalyticHestonHullWhiteEngine`
to a **nonzero** equity/short-rate correlation ``rho_sr``::

    dS = (r - d) S dt + sqrt(v) S dW_1
    dv = kappa (theta - v) dt + gamma sqrt(v) dW_2
    dr = (theta(t) - lambda r) dt + eta dW_3
    dW_1 dW_2 = rho_sv dt,  dW_1 dW_3 = rho_sr dt,  dW_2 dW_3 = 0

The parent's characteristic function stays exactly as it is; H1-HW adds one more
term, obtained by replacing the (non-affine) ``sqrt(v_t)`` in the ``rho_sr``
cross-term with its deterministic approximation ``E[sqrt(v_t)] ~ a + b e^{-c t}``.
The two branches of that fit are selected by the Feller-like ratio
``8 kappa theta / gamma^2``:

* ``> 1``: ``a = sqrt(theta - gamma^2/(8 kappa))``, ``b = sqrt(v0) - a`` and ``c``
  from a single evaluation of the closed-form approximation ``LambdaApprox(1)``;
* ``<= 1``: ``a`` from a ratio of Gamma functions and ``c`` from the *exact*
  ``Lambda(1/kappa)``, a truncated hypergeometric series that C++ caps at 1000
  terms with ``QL_REQUIRE(i < maxIter, "can not calculate Lambda")``.

Both branches are exercised by the reference cases ``h1hw_sv030_*``
(ratio 1.333) and ``h1hw_sv060_*`` (ratio 0.333).

References: Grzelak & Oosterlee, *On the Heston model with stochastic interest
rates*; Grzelak, *Equity and Foreign Exchange Hybrid Models for Pricing
Long-Maturity Financial Derivatives* (PhD thesis).

The constructor asymmetry in C++
--------------------------------
C++ has two constructors and they do **not** validate the same way::

    AnalyticH1HWEngine(model, hw, rhoSr, Size integrationOrder = 144)
        QL_REQUIRE(rhoSr_ >= 0.0, "Fourier integration is not stable if "
                                  "the equity interest rate correlation is negative");

    AnalyticH1HWEngine(model, hw, rhoSr, Real relTolerance, Size maxEvaluations)
        // no such requirement

Both are reproduced — :meth:`AnalyticH1HWEngine.__init__` and
:meth:`AnalyticH1HWEngine.with_lobatto` — including the missing check on the
second, because the asymmetry is observable behaviour and not obviously
intentional. The warning is not decorative: at ``rho_sr = -0.5`` on the reference
market C++ returns an "NPV" of about ``-3e16`` on a spot of 100.
``h1hw_negative_rhosr_ctor_asymmetry`` pins the throw, the non-throw, and the
fact that the non-throwing branch returns something that is not an option price —
never its digits, which are not reproducible across integrators.
"""

from __future__ import annotations

import cmath
import math
from typing import Final

from pquantlib import qassert
from pquantlib.math.distributions.gamma_function import GammaFunction
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.pricingengines.vanilla.analytic_heston_engine import Integration
from pquantlib.pricingengines.vanilla.analytic_heston_hull_white_engine import (
    AnalyticHestonHullWhiteEngine,
)

# C++ ``Null<Real>()`` == ``std::numeric_limits<float>::max()``, passed as
# GaussLobattoIntegral's ABSOLUTE tolerance so that the relative criterion always
# wins. See the same constant in analytic_heston_hull_white_engine.py.
_NULL_REAL: Final[float] = 3.4028234663852886e38
# C++ parity: analytich1hwengine.cpp:99 uses std::numeric_limits<float>::epsilon()
# — the FLOAT one, i.e. 2**-23, not double epsilon.
_FLOAT_EPSILON: Final[float] = 1.1920928955078125e-07
# C++ parity: analytich1hwengine.cpp:90.
_MAX_LAMBDA_ITERATIONS: Final[int] = 1000


class _FjHelper:
    """The H1-HW cross-term of the log characteristic function.

    # C++ parity: ``class AnalyticH1HWEngine::Fj_Helper`` — forward-declared
    # ``private:`` in analytich1hwengine.hpp:82 and defined in
    # analytich1hwengine.cpp:29-137. A genuinely private nested helper, so it is
    # module-private here.
    """

    __slots__ = ("_d", "_eta", "_gamma", "_j", "_kappa", "_lambda", "_rho_sr", "_term", "_theta", "_v0")

    def __init__(
        self,
        heston_model: HestonModel,
        hull_white_model: HullWhite,
        rho_sr: float,
        term: float,
        j: int,
    ) -> None:
        """# C++ parity: analytich1hwengine.cpp:52-67.

        The C++ signature carries an unused ``Real strike`` between ``term`` and
        ``j`` (the parameter is unnamed in the definition, and every call site
        passes ``0.0``); it is dropped here rather than reproduced as dead API.
        """
        self._j: int = j
        self._lambda: float = hull_white_model.a()
        self._eta: float = hull_white_model.sigma()
        self._v0: float = heston_model.v0()
        self._kappa: float = heston_model.kappa()
        self._theta: float = heston_model.theta()
        self._gamma: float = heston_model.sigma()
        self._d: float = 4.0 * self._kappa * self._theta / (self._gamma * self._gamma)
        self._rho_sr: float = rho_sr
        self._term: float = term

    # --- the deterministic sqrt(v) fit ----------------------------------

    def _c(self, t: float) -> float:
        """# C++ parity: ``Fj_Helper::c`` at analytich1hwengine.cpp:69-71."""
        return self._gamma * self._gamma / (4.0 * self._kappa) * (
            1.0 - math.exp(-self._kappa * t)
        )

    def _lambda_of_t(self, t: float) -> float:
        """# C++ parity: ``Fj_Helper::lambda`` at analytich1hwengine.cpp:73-76."""
        return (
            4.0 * self._kappa * self._v0 * math.exp(-self._kappa * t)
            / (self._gamma * self._gamma * (1.0 - math.exp(-self._kappa * t)))
        )

    def _lambda_approx(self, t: float) -> float:
        """# C++ parity: ``Fj_Helper::LambdaApprox`` at analytich1hwengine.cpp:78-81."""
        c = self._c(t)
        lam = self._lambda_of_t(t)
        return math.sqrt(c * (lam - 1.0) + c * self._d * (1.0 + 1.0 / (2.0 * (self._d + lam))))

    def _lambda_exact(self, t: float) -> float:
        """``E[sqrt(v_t)]`` as a truncated hypergeometric series.

        # C++ parity: ``Fj_Helper::Lambda`` at analytich1hwengine.cpp:83-102.

        The C++ loop is ``do { ... } while (s > eps && ++i < maxIter);`` — the
        increment is inside the short-circuit, so ``i`` is NOT advanced on the
        iteration that converges. That matters: the following
        ``QL_REQUIRE(i < maxIter)`` then passes. Reproduced exactly.
        """
        d = self._d
        lambda_t = self._lambda_of_t(t)
        i = 0
        ret_val = 0.0
        while True:
            k = float(i)
            s = math.exp(
                k * math.log(0.5 * lambda_t)
                + GammaFunction.log_value(0.5 * (1.0 + d) + k)
                - GammaFunction.log_value(k + 1.0)
                - GammaFunction.log_value(0.5 * d + k)
            )
            ret_val += s
            if not s > _FLOAT_EPSILON:
                break
            i += 1
            if not i < _MAX_LAMBDA_ITERATIONS:
                break

        qassert.require(i < _MAX_LAMBDA_ITERATIONS, "can not calculate Lambda")

        ret_val *= math.sqrt(2.0 * self._c(t)) * math.exp(-0.5 * lambda_t)
        return ret_val

    # --- the term itself ------------------------------------------------

    def __call__(self, u: float, /) -> complex:
        """# C++ parity: ``Fj_Helper::operator()`` at analytich1hwengine.cpp:104-137."""
        gamma2 = self._gamma * self._gamma
        kappa = self._kappa
        theta = self._theta
        v0 = self._v0
        d = self._d
        lam = self._lambda
        term = self._term

        if 8.0 * kappa * theta / gamma2 > 1.0:
            a = math.sqrt(theta - gamma2 / (8.0 * kappa))
            b = math.sqrt(v0) - a
            c = -math.log((self._lambda_approx(1.0) - a) / b)
        else:
            a = math.sqrt(gamma2 / (2.0 * kappa)) * math.exp(
                GammaFunction.log_value(0.5 * (d + 1.0))
                - GammaFunction.log_value(0.5 * d)
            )
            t1 = 0.0
            t2 = 1.0 / kappa
            lambda_t1 = math.sqrt(v0)
            lambda_t2 = self._lambda_exact(t2)
            c = math.log((lambda_t2 - a) / (lambda_t1 - a)) / (t1 - t2)
            b = math.exp(c * t1) * (lambda_t1 - a)

        i4 = (
            -1.0
            / lam
            * complex(u * u, -u if self._j == 1 else u)
            * (
                b / c * (1.0 - cmath.exp(-c * term))
                + a * term
                + a / lam * (math.exp(-lam * term) - 1.0)
                + b / (c - lam) * cmath.exp(-c * term) * (1.0 - cmath.exp(-term * (lam - c)))
            )
        )
        return self._eta * self._rho_sr * i4


class AnalyticH1HWEngine(AnalyticHestonHullWhiteEngine):
    """Heston / Hull-White engine with equity-rate correlation (H1-HW).

    # C++ parity: ``class AnalyticH1HWEngine`` in analytich1hwengine.hpp:70-86.
    """

    def __init__(
        self,
        model: HestonModel,
        hull_white_model: HullWhite,
        rho_sr: float,
        integration_order: int = 144,
    ) -> None:
        """Gauss-Laguerre form — the one that validates ``rho_sr``.

        # C++ parity: analytich1hwengine.cpp:140-148.
        """
        super().__init__(model, hull_white_model, integration_order)
        self._rho_sr: float = float(rho_sr)
        # C++ parity: cpp:146-147 — present on THIS ctor only.
        qassert.require(
            self._rho_sr >= 0.0,
            "Fourier integration is not stable if the equity interest rate "
            "correlation is negative",
        )

    @classmethod
    def with_lobatto(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        model: HestonModel,
        hull_white_model: HullWhite,
        rho_sr: float,
        rel_tolerance: float,
        max_evaluations: int,
    ) -> AnalyticH1HWEngine:
        """Adaptive Gauss-Lobatto form — the one that does NOT validate ``rho_sr``.

        # C++ parity: analytich1hwengine.cpp:150-156. The absence of the
        # QL_REQUIRE is reproduced deliberately; see the module docstring.
        """
        # Built at rho_sr = 0 so the ctor's guard is trivially satisfied, then
        # overwritten — the same idiom the base engine's own factories use.
        engine = cls(model, hull_white_model, 0.0)
        engine._rho_sr = float(rho_sr)
        engine._integration = Integration.gauss_lobatto(
            rel_tolerance, _NULL_REAL, max_evaluations
        )
        return engine

    def rho_sr(self) -> float:
        """Equity / short-rate correlation supplied at construction."""
        return self._rho_sr

    def add_on_term(self, phi: float, t: float, j: int) -> complex:
        """Parent's Hull-White term plus the H1-HW cross-term.

        # C++ parity: ``AnalyticH1HWEngine::addOnTerm`` at
        # analytich1hwengine.cpp:158-163 — a fresh Fj_Helper is built on every
        # call, with ``term_`` set to the ``t`` passed in.
        """
        return super().add_on_term(phi, t, j) + _FjHelper(
            self.model(), self.hull_white_model(), self._rho_sr, t, j
        )(phi)


__all__ = ["AnalyticH1HWEngine"]
