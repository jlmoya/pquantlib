"""AnalyticHestonHullWhiteEngine — Heston equity with a Hull-White short rate.

# C++ parity: ql/pricingengines/vanilla/analytichestonhullwhiteengine.{hpp,cpp}
# (v1.43) — ``class AnalyticHestonHullWhiteEngine : public AnalyticHestonEngine``.

Prices a European vanilla under

    dS = (r - d) S dt + sqrt(v) S dW_1
    dv = kappa (theta - v) dt + sigma sqrt(v) dW_2
    dr = (theta(t) - a r) dt + eta dW_3
    dW_1 dW_2 = rho dt,   dW_1 dW_3 = 0,   dW_2 dW_3 = 0

i.e. the equity and the short rate are **uncorrelated** — that assumption is what
makes the extra term additive in the log characteristic function. (For a nonzero
equity/rate correlation use
:class:`~pquantlib.pricingengines.vanilla.analytic_h1_hw_engine.AnalyticH1HWEngine`,
which derives from this class and adds the Grzelak-Oosterlee term.)

The whole content of the class is one scalar ``m`` and one add-on term::

    m           = eta^2/(2 a^2) * (t + 2/a e^{-a t} - 1/(2a) e^{-2 a t} - 3/(2a))
    addOnTerm(u, t, j) = complex(-m u^2, u (m - 2 m (j-1)))

so ``j = 1`` gets ``+i m u`` and ``j = 2`` gets ``-i m u``. Below
``a*t <= QL_EPSILON**0.25`` (exactly ``2**-13``) the algebraic small-``a`` limit
``m = 0.5 eta^2 t^3 (1/3 - a t/4 + 7 a^2 t^2/60)`` is used instead; the test is a
strict ``>``, so ``a*t`` exactly on the threshold takes the *low* branch.

Both C++ constructors are reproduced. Each does exactly one thing beyond storing
the Hull-White model: it pins the *base engine's* complex-log formula to
``Gatheral`` — not the base class's own ``OptimalCV`` default — and chooses the
quadrature:

* :meth:`__init__` -> ``Integration.gauss_laguerre(integration_order)``, default
  order 144, and the base's ``intOrder <= 192`` requirement applies;
* :meth:`with_lobatto` -> ``Integration.gauss_lobatto(rel_tolerance, None,
  max_evaluations)``.

Pinning ``Gatheral`` matters: the control-variate formulas the base defaults to
push the add-on term through a different integrand, and the C++ hybrid engines
never use them.

References: in 't Hout, Bierkens, von der Ploeg, in 't Panhuis, *A semi
closed-form analytic pricing formula for call options in a hybrid
Heston-Hull-White model*; A. Sepp, *Pricing European-Style Options under Jump
Diffusion Processes with Stochastic Volatility*.
"""

from __future__ import annotations

import math
from typing import Final

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.pricingengines.vanilla.analytic_heston_engine import (
    AnalyticHestonEngine,
    ComplexLogFormula,
    Integration,
)

# C++ ``Null<Real>()`` is ``QL_NULL_REAL`` == ``std::numeric_limits<float>::max()``.
# The C++ ctor passes it as GaussLobattoIntegral's ABSOLUTE accuracy, where it is
# large enough that ``min(absAccuracy, acc*relTol)`` always selects the relative
# criterion — that is how "no absolute tolerance" is spelled in QuantLib. It is
# NOT spelled as a null pointer: ``Integrator``'s ctor QL_REQUIREs
# ``absoluteAccuracy > QL_EPSILON`` and the sentinel satisfies it. Passing
# ``None`` into ``Integration.gauss_lobatto``'s ``abs_tolerance`` — which is what
# its signature invites — reaches ``GaussLobattoIntegral`` and raises TypeError on
# that comparison, so the sentinel is passed explicitly here.
_NULL_REAL: Final[float] = 3.4028234663852886e38


class AnalyticHestonHullWhiteEngine(AnalyticHestonEngine):
    """Analytic Heston engine including a stochastic Hull-White short rate.

    # C++ parity: ``class AnalyticHestonHullWhiteEngine`` in
    # analytichestonhullwhiteengine.hpp:67-92.
    """

    def __init__(
        self,
        heston_model: HestonModel,
        hull_white_model: HullWhite,
        integration_order: int = 144,
    ) -> None:
        """Gauss-Laguerre form.

        # C++ parity: analytichestonhullwhiteengine.cpp:26-38 — the base is
        # constructed as ``AnalyticHestonEngine(hestonModel, Gatheral,
        # Integration::gaussLaguerre(integrationOrder))``.
        """
        super().__init__(heston_model, integration_order)
        # C++ parity: the base's 4-argument ctor, whose cpxLog/integration are
        # given explicitly and whose andersenPiterbargEpsilon/alpha keep their
        # defaults (unused under Gatheral).
        self._cpx_log = ComplexLogFormula.Gatheral
        self._integration = Integration.gauss_laguerre(integration_order)
        self._andersen_piterbarg_epsilon = 1e-25
        self._alpha = -0.5

        self._hull_white_model: HullWhite = hull_white_model
        self._a: float = 0.0
        self._sigma: float = 0.0
        self._m: float = 0.0
        # C++ parity: cpp:36-37 — setParameters() then registerWith().
        self._set_parameters()
        hull_white_model.register_with(self)

    @classmethod
    def with_lobatto(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        heston_model: HestonModel,
        hull_white_model: HullWhite,
        rel_tolerance: float,
        max_evaluations: int,
    ) -> AnalyticHestonHullWhiteEngine:
        """Adaptive Gauss-Lobatto form.

        # C++ parity: analytichestonhullwhiteengine.cpp:40-50 — the base is
        # constructed as ``AnalyticHestonEngine(hestonModel, Gatheral,
        # Integration::gaussLobatto(relTolerance, Null<Real>(), maxEvaluations))``.

        This shadows ``AnalyticHestonEngine.with_lobatto`` with a wider signature.
        That is not an LSP violation being waved through: these factories stand in
        for C++ *constructors*, which are never inherited and never in an override
        relationship. ``AnalyticHestonEngine.with_lobatto`` is no more callable on
        this class than ``AnalyticHestonEngine``'s constructor is in C++.
        """
        engine = cls(heston_model, hull_white_model)
        engine._integration = Integration.gauss_lobatto(
            rel_tolerance, _NULL_REAL, max_evaluations
        )
        return engine

    # --- inspectors -----------------------------------------------------

    def hull_white_model(self) -> HullWhite:
        """The Hull-White model (``hullWhiteModel_``, protected in C++)."""
        return self._hull_white_model

    # --- parameter plumbing ---------------------------------------------

    def _set_parameters(self) -> None:
        """Cache ``a`` and ``sigma`` off the Hull-White model.

        # C++ parity: ``AnalyticHestonHullWhiteEngine::setParameters`` at
        # analytichestonhullwhiteengine.cpp:70-73, which reads
        # ``hullWhiteModel_->params()[0]`` and ``[1]``. HullWhite nulls out its
        # inherited Vasicek ``b`` and ``lambda`` slots, so those two positions
        # are exactly ``a`` and ``sigma``.
        """
        self._a = self._hull_white_model.a()
        self._sigma = self._hull_white_model.sigma()

    def update(self) -> None:
        """# C++ parity: ``AnalyticHestonHullWhiteEngine::update`` at
        # analytichestonhullwhiteengine.cpp:52-55.
        """
        self._set_parameters()
        super().update()

    # --- characteristic-function hook -----------------------------------

    def add_on_term(self, phi: float, t: float, j: int) -> complex:
        """``complex(-m u^2, u (m - 2 m (j-1)))``.

        # C++ parity: inline ``AnalyticHestonHullWhiteEngine::addOnTerm`` at
        # analytichestonhullwhiteengine.hpp:95-100. ``t`` is unused there too —
        # ``m_`` was already computed for the option's own maturity in
        # ``calculate()``.
        """
        del t  # C++ ignores the time argument here as well.
        m = self._m
        return complex(-m * phi * phi, phi * (m - 2.0 * m * (j - 1)))

    # --- calculate ------------------------------------------------------

    def calculate(self) -> None:
        """# C++ parity: ``AnalyticHestonHullWhiteEngine::calculate`` at
        # analytichestonhullwhiteengine.cpp:57-68.
        """
        args = self._arguments
        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None

        t = self.model().process().time(args.exercise.last_date())
        a = self._a
        sigma = self._sigma
        # C++ parity: cpp:60 — strict `>` against pow(QL_EPSILON, 0.25) == 2**-13.
        if a * t > math.pow(QL_EPSILON, 0.25):
            self._m = (
                sigma
                * sigma
                / (2.0 * a * a)
                * (
                    t
                    + 2.0 / a * math.exp(-a * t)
                    - 1.0 / (2.0 * a) * math.exp(-2.0 * a * t)
                    - 3.0 / (2.0 * a)
                )
            )
        else:
            # C++ parity: cpp:66 — low-a algebraic limit.
            self._m = (
                0.5
                * sigma
                * sigma
                * t
                * t
                * t
                * (1.0 / 3.0 - 0.25 * a * t + 7.0 / 60.0 * a * a * t * t)
            )

        super().calculate()


__all__ = ["AnalyticHestonHullWhiteEngine"]
