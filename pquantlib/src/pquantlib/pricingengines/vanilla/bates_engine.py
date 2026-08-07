"""Bates model engines based on Fourier transform.

# C++ parity: ql/pricingengines/vanilla/batesengine.{hpp,cpp} (v1.43) —
# ``class BatesEngine : public AnalyticHestonEngine``,
# ``class BatesDetJumpEngine : public BatesEngine``,
# ``class BatesDoubleExpEngine : public AnalyticHestonEngine``,
# ``class BatesDoubleExpDetJumpEngine : public BatesDoubleExpEngine``.

All four C++ classes live in the single header ``batesengine.hpp``, so all four
live in this single module. They price European vanillas under

1. Jump-diffusion with stochastic volatility::

       dS(t,S)  = (r - q - lambda*m) S dt + sqrt(v) S dW_1 + (e^J - 1) S dN
       dv(t,S)  = kappa (theta - v) dt + sigma sqrt(v) dW_2
       dW_1 dW_2 = rho dt

   with ``N`` a Poisson process of intensity ``lambda`` and jump magnitude
   ``J`` drawn from ``omega(J)``.

   1.1 log-normal jump size — :class:`BatesEngine`::

           omega(J) = exp(-(J - nu)^2 / (2 delta^2)) / sqrt(2 pi delta^2)

   1.2 asymmetric double-exponential (Kou) jump size —
       :class:`BatesDoubleExpEngine`::

           omega(J) = p/nuUp   * exp(-J/nuUp)   for J > 0
                    + q/nuDown * exp( J/nuDown) for J < 0,   p + q = 1

2. The same, with a *deterministic* jump intensity::

       dlambda(t) = kappaLambda (thetaLambda - lambda) dt

   2.1 log-normal jumps — :class:`BatesDetJumpEngine`
   2.2 double-exponential jumps — :class:`BatesDoubleExpDetJumpEngine`

Each class differs from :class:`AnalyticHestonEngine` only through the virtual
``add_on_term(phi, t, j)`` hook, which contributes the jump part of the log
characteristic function.

References:

- D. Bates, *Jumps and stochastic volatility: exchange rate processes implicit
  in Deutsche mark options*, Review of Financial Studies 9, 69-107.
- A. Sepp, *Pricing European-Style Options under Jump Diffusion Processes with
  Stochastic Volatility: Applications of Fourier Transform*.

Why ``Gatheral`` and not the inherited default
----------------------------------------------
C++ does **not** delegate to ``AnalyticHestonEngine(model, integrationOrder)``.
Every one of the eight constructors below names the three-argument
``AnalyticHestonEngine(model, Gatheral, Integration::gaussLaguerre(order))`` (or
its Gauss-Lobatto twin) explicitly. That is load-bearing: the two-argument
constructor selects ``OptimalCV``, whose control variate is derived from the
*plain Heston* characteristic function and therefore ignores everything
``add_on_term`` contributes. Under ``Gatheral`` the add-on enters the integrand
directly, which is the only form that prices a jump diffusion correctly.
``AP_Helper.__call__`` even asserts ``add_on_term(...) == 0`` ("only Heston
model is supported"), so routing a Bates engine through a control-variate form
does not merely lose accuracy — it raises.
"""

from __future__ import annotations

import cmath
import math

from pquantlib.models.equity.bates_det_jump_model import BatesDetJumpModel
from pquantlib.models.equity.bates_double_exp_det_jump_model import (
    BatesDoubleExpDetJumpModel,
)
from pquantlib.models.equity.bates_double_exp_model import BatesDoubleExpModel
from pquantlib.models.equity.bates_model import BatesModel
from pquantlib.pricingengines.vanilla.analytic_heston_engine import (
    AnalyticHestonEngine,
    ComplexLogFormula,
    Integration,
)

#: Default of the three-argument ``AnalyticHestonEngine`` constructor
#: (analytichestonengine.hpp:159). Unused under ``Gatheral`` — Gauss-Laguerre
#: never asks the Andersen-Piterbarg limit for an upper bound — but stored so
#: the engine state matches C++ field for field.
_ANDERSEN_PITERBARG_EPSILON: float = 1e-25


class BatesEngine(AnalyticHestonEngine):
    """Analytic Bates jump-diffusion engine (log-normal jump size).

    # C++ parity: ``class BatesEngine : public AnalyticHestonEngine`` in
    # ql/pricingengines/vanilla/batesengine.hpp:106-115 (v1.43).
    """

    def __init__(self, model: BatesModel, integration_order: int = 144) -> None:
        """Gauss-Laguerre quadrature of ``integration_order``, Gatheral log.

        # C++ parity: ``BatesEngine(model, integrationOrder = 144)``
        # (batesengine.cpp:26-30).
        """
        super().__init__(model, integration_order=integration_order)
        self._cpx_log = ComplexLogFormula.Gatheral
        self._andersen_piterbarg_epsilon = _ANDERSEN_PITERBARG_EPSILON
        # Narrows ``self._model``'s static type for the override below. At
        # runtime a ``BatesModel`` is-a ``HestonModel``, so this is the same
        # object the base class stored.
        self._bates_model: BatesModel = model

    @classmethod
    def with_lobatto(  # type: ignore[override]
        cls, model: BatesModel, rel_tolerance: float, max_evaluations: int
    ) -> BatesEngine:
        """Adaptive Gauss-Lobatto quadrature, Gatheral log.

        The model parameter narrows ``AnalyticHestonEngine.with_lobatto``'s
        ``HestonModel``, which is why the override is silenced: in C++ these are
        constructors, not virtual functions, and each of the four classes
        redeclares the pair for its own model type. ``add_on_term`` reads
        ``nu`` / ``delta`` / ``lambda`` off that model, so accepting a bare
        ``HestonModel`` here would be unsound.

        # C++ parity: ``BatesEngine(model, relTolerance, maxEvaluations)``
        # (batesengine.cpp:32-37) — ``Integration::gaussLobatto(relTolerance,
        # Null<Real>(), maxEvaluations)``. ``None`` carries C++'s
        # ``Null<Real>()`` absolute tolerance, as everywhere else in this
        # package.
        """
        engine = cls(model)
        engine._integration = Integration.gauss_lobatto(rel_tolerance, None, max_evaluations)
        return engine

    def model(self) -> BatesModel:  # type: ignore[override]
        """The underlying ``BatesModel`` (narrowed)."""
        return self._bates_model

    def add_on_term(self, phi: float, t: float, j: int) -> complex:
        """Merton log-normal jump term of the log characteristic function.

        # C++ parity: ``BatesEngine::addOnTerm`` (batesengine.cpp:39-54). The
        # C++ recovers the model with ``dynamic_pointer_cast<BatesModel>``; the
        # constructor here types it statically instead.
        """
        nu = self._bates_model.nu()
        delta = self._bates_model.delta()
        lambda_ = self._bates_model.lambda_()
        delta2 = 0.5 * delta * delta
        # C++ parity: batesengine.cpp:48-49 — ``i = (j == 1) ? 1.0 : 0.0`` and
        # ``complex<Real> g(i, phi)``, i.e. complex(real=i, imag=phi).
        i = 1.0 if j == 1 else 0.0
        g = complex(i, phi)
        # C++ parity: batesengine.cpp:52-53.
        return t * lambda_ * (
            cmath.exp(nu * g + delta2 * g * g) - 1.0 - g * (cmath.exp(nu + delta2) - 1.0)
        )


class BatesDetJumpEngine(BatesEngine):
    """Bates engine with a deterministic Ornstein-Uhlenbeck jump intensity.

    # C++ parity: ``class BatesDetJumpEngine : public BatesEngine`` in
    # batesengine.hpp:118-127 (v1.43).
    """

    def __init__(self, model: BatesDetJumpModel, integration_order: int = 144) -> None:
        """# C++ parity: ``BatesDetJumpEngine(model, integrationOrder)`` (.cpp:57-60)."""
        super().__init__(model, integration_order=integration_order)
        self._det_jump_model: BatesDetJumpModel = model

    @classmethod
    def with_lobatto(  # type: ignore[override]
        cls, model: BatesDetJumpModel, rel_tolerance: float, max_evaluations: int
    ) -> BatesDetJumpEngine:
        """# C++ parity: ``BatesDetJumpEngine(model, relTolerance, maxEvaluations)`` (.cpp:62-65)."""
        engine = cls(model)
        engine._integration = Integration.gauss_lobatto(rel_tolerance, None, max_evaluations)
        return engine

    def model(self) -> BatesDetJumpModel:  # type: ignore[override]
        """The underlying ``BatesDetJumpModel`` (narrowed)."""
        return self._det_jump_model

    def add_on_term(self, phi: float, t: float, j: int) -> complex:
        """Deterministic-intensity wrap of the log-normal jump term.

        # C++ parity: ``BatesDetJumpEngine::addOnTerm`` (batesengine.cpp:67-83).
        """
        # C++ parity: .cpp:70-71 — ``l`` renamed (ruff E741 forbids ``l``).
        l_term = super().add_on_term(phi, t, j)
        lambda_ = self._det_jump_model.lambda_()
        kappa_lambda = self._det_jump_model.kappa_lambda()
        theta_lambda = self._det_jump_model.theta_lambda()
        # C++ parity: .cpp:80-82.
        kl_t = kappa_lambda * t
        exp_neg_kl_t = math.exp(-kl_t)
        return (kl_t - 1.0 + exp_neg_kl_t) * theta_lambda * l_term / (kl_t * lambda_) + (
            1.0 - exp_neg_kl_t
        ) * l_term / kl_t


class BatesDoubleExpEngine(AnalyticHestonEngine):
    """Bates engine with Kou's asymmetric double-exponential jump size.

    # C++ parity: ``class BatesDoubleExpEngine : public AnalyticHestonEngine``
    # in batesengine.hpp:130-141 (v1.43).
    """

    def __init__(self, model: BatesDoubleExpModel, integration_order: int = 144) -> None:
        """# C++ parity: ``BatesDoubleExpEngine(model, integrationOrder)`` (.cpp:86-91)."""
        super().__init__(model, integration_order=integration_order)
        self._cpx_log = ComplexLogFormula.Gatheral
        self._andersen_piterbarg_epsilon = _ANDERSEN_PITERBARG_EPSILON
        self._double_exp_model: BatesDoubleExpModel = model

    @classmethod
    def with_lobatto(  # type: ignore[override]
        cls, model: BatesDoubleExpModel, rel_tolerance: float, max_evaluations: int
    ) -> BatesDoubleExpEngine:
        """# C++ parity: ``BatesDoubleExpEngine(model, relTolerance, maxEvaluations)`` (.cpp:93-99)."""
        engine = cls(model)
        engine._integration = Integration.gauss_lobatto(rel_tolerance, None, max_evaluations)
        return engine

    def model(self) -> BatesDoubleExpModel:  # type: ignore[override]
        """The underlying ``BatesDoubleExpModel`` (narrowed)."""
        return self._double_exp_model

    def add_on_term(self, phi: float, t: float, j: int) -> complex:
        """Double-exponential jump term of the log characteristic function.

        # C++ parity: ``BatesDoubleExpEngine::addOnTerm`` (batesengine.cpp:101-116).
        """
        p = self._double_exp_model.p()
        q = 1.0 - p
        nu_down = self._double_exp_model.nu_down()
        nu_up = self._double_exp_model.nu_up()
        lambda_ = self._double_exp_model.lambda_()
        # C++ parity: .cpp:111-112.
        i = 1.0 if j == 1 else 0.0
        g = complex(i, phi)
        # C++ parity: .cpp:114-115.
        return t * lambda_ * (
            p / (1.0 - g * nu_up)
            + q / (1.0 + g * nu_down)
            - 1.0
            - g * (p / (1.0 - nu_up) + q / (1.0 + nu_down) - 1.0)
        )


class BatesDoubleExpDetJumpEngine(BatesDoubleExpEngine):
    """Double-exponential jumps with a deterministic OU jump intensity.

    # C++ parity: ``class BatesDoubleExpDetJumpEngine : public
    # BatesDoubleExpEngine`` in batesengine.hpp:144-155 (v1.43).
    """

    def __init__(
        self, model: BatesDoubleExpDetJumpModel, integration_order: int = 144
    ) -> None:
        """# C++ parity: ``BatesDoubleExpDetJumpEngine(model, integrationOrder)`` (.cpp:118-121)."""
        super().__init__(model, integration_order=integration_order)
        self._double_exp_det_jump_model: BatesDoubleExpDetJumpModel = model

    @classmethod
    def with_lobatto(  # type: ignore[override]
        cls, model: BatesDoubleExpDetJumpModel, rel_tolerance: float, max_evaluations: int
    ) -> BatesDoubleExpDetJumpEngine:
        """# C++ parity: the ``(relTolerance, maxEvaluations)`` ctor (.cpp:123-126)."""
        engine = cls(model)
        engine._integration = Integration.gauss_lobatto(rel_tolerance, None, max_evaluations)
        return engine

    def model(self) -> BatesDoubleExpDetJumpModel:  # type: ignore[override]
        """The underlying ``BatesDoubleExpDetJumpModel`` (narrowed)."""
        return self._double_exp_det_jump_model

    def add_on_term(self, phi: float, t: float, j: int) -> complex:
        """Deterministic-intensity wrap of the double-exponential jump term.

        # C++ parity: ``BatesDoubleExpDetJumpEngine::addOnTerm``
        # (batesengine.cpp:128-143). Identical algebra to
        # ``BatesDetJumpEngine::addOnTerm``, wrapped around the double-exp
        # ``l`` instead of the log-normal one — and C++ really does duplicate
        # it rather than share it.
        """
        l_term = super().add_on_term(phi, t, j)
        lambda_ = self._double_exp_det_jump_model.lambda_()
        kappa_lambda = self._double_exp_det_jump_model.kappa_lambda()
        theta_lambda = self._double_exp_det_jump_model.theta_lambda()
        # C++ parity: .cpp:140-142.
        kl_t = kappa_lambda * t
        exp_neg_kl_t = math.exp(-kl_t)
        return (kl_t - 1.0 + exp_neg_kl_t) * theta_lambda * l_term / (kl_t * lambda_) + (
            1.0 - exp_neg_kl_t
        ) * l_term / kl_t


__all__ = [
    "BatesDetJumpEngine",
    "BatesDoubleExpDetJumpEngine",
    "BatesDoubleExpEngine",
    "BatesEngine",
]
