"""AnalyticBatesDetJumpEngine — Bates with deterministic jump intensity.

# C++ parity: ql/pricingengines/vanilla/batesengine.{hpp,cpp}
# (v1.42.1) — ``class BatesDetJumpEngine : public BatesEngine``.

Prices a plain-vanilla European option under the Bates dynamics with a
deterministic Ornstein-Uhlenbeck jump-intensity layered on top of the
base lognormal-jumps process::

    dS(t,S)     = (r - q - lambda*m) * S * dt + sqrt(V) * S * dW_1
                  + (exp(J) - 1) * S * dN
    dV(t,S)     = kappa * (theta - V) * dt + sigma * sqrt(V) * dW_2
    dlambda(t)  = kappaLambda * (thetaLambda - lambda) * dt
    dW_1 dW_2   = rho dt
    J ~ Normal(nu, delta^2)

The deterministic intensity contribution is wrapped around the base
``BatesEngine.add_on_term`` lognormal CF term ``l`` (Sepp form) as::

    add_on = (kL*t - 1 + exp(-kL*t)) * thetaL * l / (kL*t*lambda)
           + (1 - exp(-kL*t)) * l / (kL*t)

with ``kL = kappaLambda`` and ``thetaL = thetaLambda``.

* The first piece is the integrated intensity's long-term mean
  contribution.
* The second piece is the initial-value decay contribution.

Together they replace the constant-intensity ``t*lambda`` envelope of
the base ``BatesEngine`` with the integral of the OU intensity from 0
to t.

Divergences from C++:

* Numerical integration is the inherited scipy.integrate.quad
  (QUADPACK QAGI adaptive) — the C++ uses Gauss-Laguerre order 144.
  Both converge well inside the LOOSE tier on the standard parameter
  regime. See ``AnalyticHestonEngine`` for the full discussion.
* The ``(relTolerance, maxEvaluations)`` Gauss-Lobatto constructor
  collapses to the same Python signature (we ignore the integration
  controls; quad is adaptive).
"""

from __future__ import annotations

import math

from pquantlib.models.equity.bates_det_jump_model import BatesDetJumpModel
from pquantlib.pricingengines.vanilla.bates_engine import BatesEngine


class AnalyticBatesDetJumpEngine(BatesEngine):
    """Analytic Bates engine with deterministic OU jump intensity.

    # C++ parity: ``class BatesDetJumpEngine : public BatesEngine`` in
    # ql/pricingengines/vanilla/batesengine.hpp:118-127 (v1.42.1).
    """

    def __init__(
        self,
        model: BatesDetJumpModel,
        integration_order: int = 144,
    ) -> None:
        """Construct from a BatesDetJumpModel.

        # C++ parity: ``BatesDetJumpEngine(model, integrationOrder)`` in
        # batesengine.cpp:57-60.

        Parameters
        ----------
        model:
            The BatesDetJumpModel whose lognormal-jump parameters drive
            the inner ``BatesEngine.add_on_term`` and whose OU-intensity
            parameters (kappaLambda, thetaLambda) wrap that into the
            deterministic-intensity correction.
        integration_order:
            Kept for API parity. Ignored at runtime.
        """
        # super().__init__ stores the model as both ``self._model``
        # (HestonModel) and ``self._bates_model`` (BatesModel). Since
        # BatesDetJumpModel IS-A BatesModel, the base accessors pick up
        # the correct parameters for the lognormal CF.
        super().__init__(model, integration_order=integration_order)
        self._det_jump_model: BatesDetJumpModel = model

    # --- BatesDetJumpModel narrow accessor ------------------------------

    def model(self) -> BatesDetJumpModel:  # type: ignore[override]
        """The underlying ``BatesDetJumpModel`` (narrowed).

        Runtime object is the same as the base engine's ``_model``.
        """
        return self._det_jump_model

    # --- add_on_term override -------------------------------------------

    def add_on_term(self, phi: float, t: float, j: int) -> complex:
        """Deterministic-intensity wrap of the lognormal-jump CF.

        # C++ parity: ``BatesDetJumpEngine::addOnTerm`` in
        # batesengine.cpp:67-83.

        Parameters
        ----------
        phi:
            Fourier variable (real).
        t:
            Time to maturity (in year fractions).
        j:
            Heston integrand index (1 or 2).

        Returns
        -------
        complex
            The deterministic-intensity-wrapped jump add-on. ``l`` is
            the base ``BatesEngine.add_on_term`` (lognormal CF).
        """
        # C++ parity: batesengine.cpp:70-71 — l = BatesEngine::addOnTerm.
        # Python rename ``l`` → ``l_term`` (ruff E741 ambiguity).
        l_term = super().add_on_term(phi, t, j)
        # C++ parity: batesengine.cpp:76-78 — pull jump-intensity OU
        # params off the BatesDetJumpModel.
        lambda_ = self._det_jump_model.lambda_()
        kappa_lambda = self._det_jump_model.kappa_lambda()
        theta_lambda = self._det_jump_model.theta_lambda()
        # C++ parity: batesengine.cpp:80-82 — full deterministic-intensity
        # wrap formula. The ``kL*t`` and ``kL*t*lambda`` divisors are
        # numerically safe at standard parameters (kL, lambda, t all > 0).
        kl_t = kappa_lambda * t
        exp_neg_kl_t = math.exp(-kl_t)
        return (
            (kl_t - 1.0 + exp_neg_kl_t)
            * theta_lambda
            * l_term
            / (kl_t * lambda_)
            + (1.0 - exp_neg_kl_t) * l_term / kl_t
        )


__all__ = ["AnalyticBatesDetJumpEngine"]
