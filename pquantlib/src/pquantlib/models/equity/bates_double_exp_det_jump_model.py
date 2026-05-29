"""BatesDoubleExpDetJumpModel — double-exp Bates + deterministic intensity.

# C++ parity: ql/models/equity/batesmodel.{hpp,cpp} (v1.42.1).

Combines the double-exponential jump law of ``BatesDoubleExpModel``
with the deterministic Ornstein-Uhlenbeck intensity dynamics of
``BatesDetJumpModel``::

    dS(t,S)     = (r - q - lambda*m) * S * dt + sqrt(V) * S * dW_1
                  + (exp(J) - 1) * S * dN
    dV(t,S)     = kappa * (theta - V) * dt + sigma * sqrt(V) * dW_2
    dlambda(t)  = kappaLambda * (thetaLambda - lambda) * dt
    omega(J)    = p * (1/nuUp) * exp(-J/nuUp)   for J > 0
                + q * (1/nuDown) * exp(J/nuDown) for J < 0
    p + q = 1

The parameter vector grows from 9 to 11 slots::

    arguments_ = [
        theta, kappa, sigma, rho, v0,    # inherited from Heston (0..4)
        p, nuDown, nuUp, lambda,          # inherited from DoubleExp (5..8)
        kappaLambda,                      # index 9 — OU mean-reversion (Positive)
        thetaLambda,                      # index 10 — OU long-term mean (Positive)
    ]
"""

from __future__ import annotations

from pquantlib.math.optimization.constraint import PositiveConstraint
from pquantlib.models.equity.bates_double_exp_model import BatesDoubleExpModel
from pquantlib.models.parameter import ConstantParameter
from pquantlib.processes.heston_process import HestonProcess


class BatesDoubleExpDetJumpModel(BatesDoubleExpModel):
    """Bates double-exp + deterministic-intensity model.

    # C++ parity: ``class BatesDoubleExpDetJumpModel : public
    # BatesDoubleExpModel`` in ql/models/equity/batesmodel.hpp:80-89
    # (v1.42.1).
    """

    # 9 inherited DoubleExp slots + 2 new OU-intensity slots.
    _N_ARGUMENTS: int = 11

    def __init__(
        self,
        process: HestonProcess,
        lambda_: float = 0.1,
        nu_up: float = 0.1,
        nu_down: float = 0.1,
        p: float = 0.5,
        kappa_lambda: float = 1.0,
        theta_lambda: float = 0.1,
    ) -> None:
        """Construct from a HestonProcess + double-exp + det-intensity params.

        # C++ parity: ``BatesDoubleExpDetJumpModel(process, lambda, nuUp,
        # nuDown, p, kappaLambda, thetaLambda)`` in batesmodel.cpp:74-85.

        Parameters
        ----------
        process:
            The underlying HestonProcess.
        lambda_, nu_up, nu_down, p:
            Double-exponential jump parameters (see BatesDoubleExpModel).
        kappa_lambda:
            Mean-reversion speed of the deterministic intensity OU.
            Positive.
        theta_lambda:
            Long-term mean of the deterministic intensity OU. Positive.
        """
        super().__init__(process, lambda_, nu_up, nu_down, p)
        # C++ parity: batesmodel.cpp:79 — arguments_.resize(11) + assign
        # slots 9, 10. Python pre-allocated via _N_ARGUMENTS override.
        # C++ parity: batesmodel.cpp:81-82 — kappaLambda is Positive.
        self._arguments[9] = ConstantParameter(kappa_lambda, PositiveConstraint())
        # C++ parity: batesmodel.cpp:83-84 — thetaLambda is Positive.
        self._arguments[10] = ConstantParameter(theta_lambda, PositiveConstraint())

    # --- deterministic-intensity accessors ------------------------------

    def kappa_lambda(self) -> float:
        """Mean-reversion speed of the deterministic intensity OU.

        # C++ parity: ``BatesDoubleExpDetJumpModel::kappaLambda`` in
        # batesmodel.hpp:87.
        """
        return self._arguments[9](0.0)

    def theta_lambda(self) -> float:
        """Long-term mean of the deterministic intensity OU.

        # C++ parity: ``BatesDoubleExpDetJumpModel::thetaLambda`` in
        # batesmodel.hpp:88.
        """
        return self._arguments[10](0.0)


__all__ = ["BatesDoubleExpDetJumpModel"]
