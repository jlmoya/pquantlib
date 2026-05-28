"""BatesEngine — analytic Bates-jump-diffusion engine via the Heston add-on hook.

# C++ parity: ql/pricingengines/vanilla/batesengine.{hpp,cpp}
# (v1.42.1) — ``class BatesEngine : public AnalyticHestonEngine``.

Prices a plain-vanilla European option under the Bates dynamics::

    dS(t,S)  = (r - q - lambda*m) * S * dt + sqrt(v) * S * dW_1
               + (exp(J) - 1) * S * dN
    dV(t,S)  = kappa * (theta - V) * dt + sigma * sqrt(V) * dW_2
    dW_1 dW_2 = rho dt
    J ~ Normal(nu, delta^2)              # log-jump-size
    N ~ Poisson(lambda * t)              # Poisson arrivals

by re-using the inherited ``AnalyticHestonEngine`` Gatheral integrator and
plugging the Bates jump compensator into the ``add_on_term`` hook.

Add-on term (Sepp form; matches batesengine.cpp:39-54)::

    delta2 = 0.5 * delta^2
    g      = (i, phi)  for j=1   i.e. complex(1, phi)
           = (0, phi)  for j=2   i.e. complex(0, phi)
    add_on = t * lambda * ( exp(nu*g + delta2*g^2)
                            - 1
                            - g * (exp(nu + delta2) - 1) )

The ``- g*(exp(nu+delta2) - 1)`` piece is the martingale compensator
that matches the ``-lambda*m`` drift correction on the spot SDE
(``m = exp(nu + delta2) - 1``) so the discounted spot stays a
martingale. Without it the engine would mis-price even at ``lambda=0``
in the limit (and would diverge for non-zero jumps).

Divergences from C++:

* Numerical integration is the inherited scipy.integrate.quad
  (QUADPACK QAGI adaptive) — the C++ uses Gauss-Laguerre order 144.
  Both converge well inside the LOOSE tier on the standard parameter
  regime. See ``AnalyticHestonEngine`` for the full discussion.
* ``ComplexLogFormula`` is fixed at Gatheral — same as L4-C.
* The ``(relTolerance, maxEvaluations)`` Gauss-Lobatto constructor
  collapses to the same Python signature (we ignore the integration
  controls; quad is adaptive).
* Sister engines ``BatesDetJumpEngine`` / ``BatesDoubleExpEngine`` /
  ``BatesDoubleExpDetJumpEngine`` are out of scope — they require the
  ``BatesDetJumpModel`` / ``BatesDoubleExpModel`` /
  ``BatesDoubleExpDetJumpModel`` parameter hierarchies which L4-C
  carved out. Listed as L6 carve-outs in ``phase6-completion.md``.
"""

from __future__ import annotations

import cmath

from pquantlib.models.equity.bates_model import BatesModel
from pquantlib.pricingengines.vanilla.analytic_heston_engine import (
    AnalyticHestonEngine,
)


class BatesEngine(AnalyticHestonEngine):
    """Analytic Bates jump-diffusion engine.

    # C++ parity: ``class BatesEngine : public AnalyticHestonEngine`` in
    # ql/pricingengines/vanilla/batesengine.hpp:106-115 (v1.42.1).
    """

    def __init__(
        self,
        model: BatesModel,
        integration_order: int = 144,
    ) -> None:
        """Construct from a BatesModel.

        # C++ parity: ``BatesEngine(model, integrationOrder)`` in
        # batesengine.cpp:26-30 — forwards to the AnalyticHestonEngine
        # base ctor with Gatheral + Gauss-Laguerre(144). The Python
        # port's ``integration_order`` kwarg is accepted for API parity
        # but ignored at runtime (scipy.quad is adaptive); same
        # treatment as the base ``AnalyticHestonEngine``.

        Parameters
        ----------
        model:
            The Bates model whose Heston parameters drive the Fourier
            integrand and whose jump parameters (lambda, nu, delta)
            drive the ``add_on_term`` compensator.
        integration_order:
            Kept as a kwarg for API parity. Ignored at runtime.
        """
        super().__init__(model, integration_order=integration_order)
        # Narrow ``self._model``'s static type for the override below.
        # At runtime ``BatesModel`` is-a ``HestonModel`` so the base
        # class's stored reference is the same object.
        self._bates_model: BatesModel = model

    # --- BatesModel narrow accessor -------------------------------------

    def model(self) -> BatesModel:  # type: ignore[override]
        """The underlying ``BatesModel``.

        Narrows the base ``AnalyticHestonEngine.model() -> HestonModel``
        return type for callers that want jump-parameter introspection
        without an explicit cast. Runtime object is the same.
        """
        return self._bates_model

    # --- add_on_term override -------------------------------------------

    def add_on_term(self, phi: float, t: float, j: int) -> complex:
        """Bates Merton-jump compensator term for the Heston CF integrand.

        # C++ parity: ``BatesEngine::addOnTerm`` in batesengine.cpp:39-54
        # (v1.42.1). The C++ uses ``dynamic_pointer_cast<BatesModel>``;
        # we know statically the model is a BatesModel because the
        # ctor enforces it.

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
            The log-CF add-on (added to the inner ``arg`` exponent
            before the ``Im(exp(arg))/phi`` Gatheral form).
        """
        nu = self._bates_model.nu()
        delta = self._bates_model.delta()
        lambda_ = self._bates_model.lambda_()
        delta2 = 0.5 * delta * delta
        # C++ parity: batesengine.cpp:48 — ``i = (j == 1) ? 1.0 : 0.0``.
        # The ``g`` complex is then ``g = (i, phi)`` = ``i + i*phi`` in
        # C++ complex<Real> ctor terms — i.e. complex(real=i, imag=phi).
        i = 1.0 if j == 1 else 0.0
        g = complex(i, phi)
        # C++ parity: batesengine.cpp:52-53 — the full Sepp form.
        return t * lambda_ * (
            cmath.exp(nu * g + delta2 * g * g)
            - 1.0
            - g * (cmath.exp(nu + delta2) - 1.0)
        )


__all__ = ["BatesEngine"]
