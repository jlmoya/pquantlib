"""BatesDoubleExpModel — Bates with double-exponential jump distribution.

# C++ parity: ql/models/equity/batesmodel.{hpp,cpp} (v1.42.1).

Replaces the lognormal jump-size law of the base Bates model with the
Kou (2002) double-exponential law::

    dS(t,S)  = (r - q - lambda*m) * S * dt + sqrt(V) * S * dW_1
               + (exp(J) - 1) * S * dN
    dV(t,S)  = kappa * (theta - V) * dt + sigma * sqrt(V) * dW_2
    dW_1 dW_2 = rho dt

    omega(J) = p * (1/nuUp)   * exp(-J / nuUp)     for J > 0
             + q * (1/nuDown) * exp( J / nuDown)   for J < 0
    p + q = 1

where ``p`` is the probability of an up-jump and ``nuUp``, ``nuDown``
are the mean magnitudes of up/down jumps respectively. Unlike
``BatesModel`` (which holds a ``BatesProcess``), this model takes a
plain ``HestonProcess`` — the jump parameters are model-level,
not process-level.

The parameter vector grows from 5 to 9 slots::

    arguments_ = [
        theta, kappa, sigma, rho, v0,    # inherited from Heston (0..4)
        p,                                # index 5 — up-jump prob (Boundary 0..1)
        nuDown,                           # index 6 — down-jump mean (Positive)
        nuUp,                             # index 7 — up-jump mean   (Positive)
        lambda,                           # index 8 — jump intensity (Positive)
    ]

Divergences from C++:

* The C++ class extends ``HestonModel`` directly (not ``BatesModel``),
  because the jump law differs — same Python.
"""

from __future__ import annotations

from pquantlib.math.optimization.constraint import (
    BoundaryConstraint,
    PositiveConstraint,
)
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.models.parameter import ConstantParameter
from pquantlib.processes.heston_process import HestonProcess


class BatesDoubleExpModel(HestonModel):
    """Bates model with double-exponential jump distribution.

    # C++ parity: ``class BatesDoubleExpModel : public HestonModel`` in
    # ql/models/equity/batesmodel.hpp:67-77 (v1.42.1).
    """

    # 5 inherited Heston slots + 4 new jump slots.
    _N_ARGUMENTS: int = 9

    def __init__(
        self,
        process: HestonProcess,
        lambda_: float = 0.1,
        nu_up: float = 0.1,
        nu_down: float = 0.1,
        p: float = 0.5,
    ) -> None:
        """Construct from a HestonProcess + double-exp jump params.

        # C++ parity: ``BatesDoubleExpModel(process, lambda, nuUp,
        # nuDown, p)`` in batesmodel.cpp:60-71.

        Parameters
        ----------
        process:
            The underlying HestonProcess (jump params are model-level).
        lambda_:
            Poisson jump intensity. Positive.
        nu_up:
            Mean magnitude of up-jumps (J > 0). Positive.
        nu_down:
            Mean magnitude of down-jumps (J < 0). Positive.
        p:
            Probability of an up-jump (q = 1 - p). In [0, 1].
        """
        super().__init__(process)
        # C++ parity: batesmodel.cpp:66-67 — p in [0, 1].
        self._arguments[5] = ConstantParameter(p, BoundaryConstraint(0.0, 1.0))
        # C++ parity: batesmodel.cpp:68 — nuDown is Positive.
        self._arguments[6] = ConstantParameter(nu_down, PositiveConstraint())
        # C++ parity: batesmodel.cpp:69 — nuUp is Positive.
        self._arguments[7] = ConstantParameter(nu_up, PositiveConstraint())
        # C++ parity: batesmodel.cpp:70 — lambda is Positive.
        self._arguments[8] = ConstantParameter(lambda_, PositiveConstraint())

    # --- double-exp jump accessors --------------------------------------

    def p(self) -> float:
        """Up-jump probability (params[5] evaluated at t=0).

        # C++ parity: ``BatesDoubleExpModel::p`` in batesmodel.hpp:73.
        """
        return self._arguments[5](0.0)

    def nu_down(self) -> float:
        """Mean down-jump magnitude (params[6] evaluated at t=0).

        # C++ parity: ``BatesDoubleExpModel::nuDown`` in batesmodel.hpp:74.
        """
        return self._arguments[6](0.0)

    def nu_up(self) -> float:
        """Mean up-jump magnitude (params[7] evaluated at t=0).

        # C++ parity: ``BatesDoubleExpModel::nuUp`` in batesmodel.hpp:75.
        """
        return self._arguments[7](0.0)

    def lambda_(self) -> float:
        """Poisson jump intensity (params[8] evaluated at t=0).

        # C++ parity: ``BatesDoubleExpModel::lambda`` in batesmodel.hpp:76.

        Trailing underscore because ``lambda`` is a Python keyword.
        """
        return self._arguments[8](0.0)


__all__ = ["BatesDoubleExpModel"]
