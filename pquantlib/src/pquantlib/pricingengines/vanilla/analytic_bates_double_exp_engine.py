"""AnalyticBatesDoubleExpEngine — Bates with double-exponential jumps.

# C++ parity: ql/pricingengines/vanilla/batesengine.{hpp,cpp}
# (v1.42.1) — ``class BatesDoubleExpEngine : public AnalyticHestonEngine``.

Prices a plain-vanilla European option under the Bates dynamics with
double-exponential (Kou 2002) jump-size distribution::

    dS(t,S)  = (r - q - lambda*m) * S * dt + sqrt(V) * S * dW_1
               + (exp(J) - 1) * S * dN
    dV(t,S)  = kappa * (theta - V) * dt + sigma * sqrt(V) * dW_2
    dW_1 dW_2 = rho dt

    omega(J) = p * (1/nuUp)   * exp(-J/nuUp)   for J > 0
             + q * (1/nuDown) * exp( J/nuDown) for J < 0
    p + q    = 1

The Sepp closed-form CF add-on (matches batesengine.cpp:101-116) is::

    g     = (i, phi)  for j=1   i.e. complex(1, phi)
          = (0, phi)  for j=2   i.e. complex(0, phi)
    chf_J = p / (1 - g*nuUp) + q / (1 + g*nuDown) - 1
    m_J   = p / (1 - nuUp)   + q / (1 + nuDown)   - 1
    add_on = t * lambda * ( chf_J - g * m_J )

The ``-g * m_J`` piece is the martingale jump compensator
(``m_J = exp(jump_size) - 1`` integrated against the double-exp law),
matching the ``-lambda*m`` drift on the spot SDE so the discounted spot
remains a martingale.

Divergences from C++:

* Numerical integration is the inherited scipy.integrate.quad
  (QUADPACK QAGI adaptive) — the C++ uses Gauss-Laguerre order 144.
  Both converge well inside the LOOSE tier on the standard parameter
  regime.
* The ``(relTolerance, maxEvaluations)`` Gauss-Lobatto constructor
  collapses to the same Python signature (we ignore the integration
  controls; quad is adaptive).
"""

from __future__ import annotations

from pquantlib.models.equity.bates_double_exp_model import BatesDoubleExpModel
from pquantlib.pricingengines.vanilla.analytic_heston_engine import (
    AnalyticHestonEngine,
)


class AnalyticBatesDoubleExpEngine(AnalyticHestonEngine):
    """Analytic Bates engine with double-exponential jump distribution.

    # C++ parity: ``class BatesDoubleExpEngine : public AnalyticHestonEngine``
    # in ql/pricingengines/vanilla/batesengine.hpp:130-141 (v1.42.1).
    """

    def __init__(
        self,
        model: BatesDoubleExpModel,
        integration_order: int = 144,
    ) -> None:
        """Construct from a BatesDoubleExpModel.

        # C++ parity: ``BatesDoubleExpEngine(model, integrationOrder)``
        # in batesengine.cpp:86-91.

        Parameters
        ----------
        model:
            The BatesDoubleExpModel whose Heston parameters drive the
            Fourier integrand and whose double-exp jump parameters
            (p, nuDown, nuUp, lambda) drive the add_on_term.
        integration_order:
            Kept for API parity. Ignored at runtime.
        """
        super().__init__(model, integration_order=integration_order)
        self._double_exp_model: BatesDoubleExpModel = model

    # --- BatesDoubleExpModel narrow accessor ----------------------------

    def model(self) -> BatesDoubleExpModel:  # type: ignore[override]
        """The underlying ``BatesDoubleExpModel`` (narrowed).

        Runtime object is the same as the base engine's ``_model``.
        """
        return self._double_exp_model

    # --- add_on_term override -------------------------------------------

    def add_on_term(self, phi: float, t: float, j: int) -> complex:
        """Double-exponential-jump compensator for the Heston CF integrand.

        # C++ parity: ``BatesDoubleExpEngine::addOnTerm`` in
        # batesengine.cpp:101-116.

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
            The double-exp jump add-on, including the martingale
            compensator.
        """
        p = self._double_exp_model.p()
        q = 1.0 - p
        nu_down = self._double_exp_model.nu_down()
        nu_up = self._double_exp_model.nu_up()
        lambda_ = self._double_exp_model.lambda_()
        # C++ parity: batesengine.cpp:111 — ``i = (j == 1) ? 1.0 : 0.0``.
        i = 1.0 if j == 1 else 0.0
        g = complex(i, phi)
        # C++ parity: batesengine.cpp:114-115 — the Sepp double-exp form.
        return t * lambda_ * (
            p / (1.0 - g * nu_up)
            + q / (1.0 + g * nu_down)
            - 1.0
            - g * (p / (1.0 - nu_up) + q / (1.0 + nu_down) - 1.0)
        )


__all__ = ["AnalyticBatesDoubleExpEngine"]
