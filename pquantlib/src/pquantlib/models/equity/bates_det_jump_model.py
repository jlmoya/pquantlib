"""BatesDetJumpModel — Bates with deterministic jump intensity.

# C++ parity: ql/models/equity/batesmodel.{hpp,cpp} (v1.42.1).

Extends ``BatesModel`` by laying a deterministic Ornstein-Uhlenbeck
jump-intensity process on top of the base log-normal jumps::

    dS(t,S)     = (r - q - lambda*m) * S * dt + sqrt(V) * S * dW_1
                  + (exp(J) - 1) * S * dN
    dV(t,S)     = kappa * (theta - V) * dt + sigma * sqrt(V) * dW_2
    dlambda(t)  = kappaLambda * (thetaLambda - lambda) * dt
    dW_1 dW_2   = rho dt
    J ~ Normal(nu, delta^2)              # log-jump-size

The intensity decays from its initial value (the base ``BatesProcess``'s
lambda) toward ``thetaLambda`` at rate ``kappaLambda``. The parameter
vector grows from 8 to 10 slots::

    arguments_ = [
        theta, kappa, sigma, rho, v0,    # inherited from Heston (0..4)
        nu, delta, lambda,                # inherited from Bates (5..7)
        kappaLambda,                      # index 8 — OU mean-reversion (Positive)
        thetaLambda,                      # index 9 — OU long-term mean (Positive)
    ]

``BatesDetJumpEngine`` will eventually consume these slots via the CF
add-on hook (W1-C scope below). For now ``BatesEngine`` would only
see slots 0..7 and miss the deterministic-intensity correction.

Divergences from C++:

* None — the C++ class is a thin subclass that resizes ``arguments_``
  to 10 and assigns 2 constants. Same Python.
"""

from __future__ import annotations

from pquantlib.math.optimization.constraint import PositiveConstraint
from pquantlib.models.equity.bates_model import BatesModel
from pquantlib.models.parameter import ConstantParameter
from pquantlib.processes.bates_process import BatesProcess


class BatesDetJumpModel(BatesModel):
    """Bates model with deterministic jump intensity.

    # C++ parity: ``class BatesDetJumpModel : public BatesModel`` in
    # ql/models/equity/batesmodel.hpp:56-64 (v1.42.1).
    """

    # 8 inherited Bates slots + 2 new OU-intensity slots.
    _N_ARGUMENTS: int = 10

    def __init__(
        self,
        process: BatesProcess,
        kappa_lambda: float = 1.0,
        theta_lambda: float = 0.1,
    ) -> None:
        """Construct from a BatesProcess + deterministic-intensity OU params.

        # C++ parity: ``BatesDetJumpModel(process, kappaLambda, thetaLambda)``
        # in batesmodel.cpp:47-57.

        Parameters
        ----------
        process:
            The underlying BatesProcess. Its lambda becomes the OU
            initial intensity.
        kappa_lambda:
            Mean-reversion speed of the deterministic intensity OU.
            Positive.
        theta_lambda:
            Long-term mean of the deterministic intensity OU. Positive.
        """
        # super().__init__ runs BatesModel.__init__ which sets slots
        # 0..7 from the BatesProcess. With the overridden
        # ``_N_ARGUMENTS = 10``, CalibratedModel.__init__ pre-allocates
        # 10 slots so we can index into 8 and 9 without resizing.
        super().__init__(process)
        # C++ parity: batesmodel.cpp:51 — arguments_.resize(10) + assign
        # slots 8, 9. Python pre-allocated via _N_ARGUMENTS override.
        # C++ parity: batesmodel.cpp:53-54 — kappaLambda is Positive.
        self._arguments[8] = ConstantParameter(kappa_lambda, PositiveConstraint())
        # C++ parity: batesmodel.cpp:55-56 — thetaLambda is Positive.
        self._arguments[9] = ConstantParameter(theta_lambda, PositiveConstraint())

    # --- deterministic-intensity accessors ------------------------------

    def kappa_lambda(self) -> float:
        """Mean-reversion speed of the deterministic intensity OU.

        # C++ parity: ``BatesDetJumpModel::kappaLambda`` in
        # batesmodel.hpp:62.
        """
        return self._arguments[8](0.0)

    def theta_lambda(self) -> float:
        """Long-term mean of the deterministic intensity OU.

        # C++ parity: ``BatesDetJumpModel::thetaLambda`` in
        # batesmodel.hpp:63.
        """
        return self._arguments[9](0.0)


__all__ = ["BatesDetJumpModel"]
