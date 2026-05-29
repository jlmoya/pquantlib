"""ExtOUWithJumpsProcess — extended OU plus exponential jumps (Kluge).

# C++ parity: ql/experimental/processes/extouwithjumpsprocess.{hpp,cpp}
# (v1.42.1).

Two-factor model: log-spot ``S = exp(X_t + Y_t)`` with

* ``X`` — ExtendedOU process (slow mean reversion).
* ``Y`` — jump process: ``dY_t = -beta * Y_t dt + J_t * dN_t`` with
  ``omega(J) = eta * exp(-eta * J)`` and Poisson intensity
  ``jumpIntensity``.

This W5-A port exposes only the accessors needed by
``FdmExtOUJumpOp`` and ``FdmKlugeExtOUOp``:

* ``eta`` (rate of the exponential jump-size distribution).
* ``beta`` (mean-reversion of ``Y``).
* ``jumpIntensity`` (Poisson rate ``lambda``).
* ``getExtendedOrnsteinUhlenbeckProcess()`` — the embedded ExtOU.

The full multi-D ``initialValues`` / ``drift`` / ``diffusion`` /
``evolve`` surface — used by Monte-Carlo sampling — is deferred to a
future port.
"""

from __future__ import annotations

from typing import final

from pquantlib import qassert
from pquantlib.experimental.processes.extended_ornstein_uhlenbeck_process import (
    ExtendedOrnsteinUhlenbeckProcess,
)


@final
class ExtOUWithJumpsProcess:
    """Extended-OU + exponential-jump process.

    # C++ parity: ``class ExtOUWithJumpsProcess : public StochasticProcess``
    # in extouwithjumpsprocess.hpp:59.

    Parameters
    ----------
    process
        ExtendedOU process for the ``X`` factor.
    y0
        Initial value of the jump factor ``Y``.
    beta
        Mean-reversion speed of ``Y``.
    jump_intensity
        Poisson intensity ``lambda``.
    eta
        Rate of the exponential jump-size distribution.
    """

    __slots__ = ("_beta", "_eta", "_jump_intensity", "_ou_process", "_y0")

    def __init__(
        self,
        process: ExtendedOrnsteinUhlenbeckProcess,
        y0: float,
        beta: float,
        jump_intensity: float,
        eta: float,
    ) -> None:
        qassert.require(beta >= 0.0, f"non-positive beta given: {beta}")
        qassert.require(
            jump_intensity >= 0.0, f"non-positive jump intensity: {jump_intensity}"
        )
        qassert.require(eta > 0.0, f"non-positive eta: {eta}")
        self._ou_process: ExtendedOrnsteinUhlenbeckProcess = process
        self._y0: float = float(y0)
        self._beta: float = float(beta)
        self._jump_intensity: float = float(jump_intensity)
        self._eta: float = float(eta)

    def get_extended_ornstein_uhlenbeck_process(self) -> ExtendedOrnsteinUhlenbeckProcess:
        # C++ parity: getExtendedOrnsteinUhlenbeckProcess().
        return self._ou_process

    def beta(self) -> float:
        # C++ parity: beta().
        return self._beta

    def eta(self) -> float:
        # C++ parity: eta().
        return self._eta

    def jump_intensity(self) -> float:
        # C++ parity: jumpIntensity().
        return self._jump_intensity

    def y0(self) -> float:
        """Initial value of ``Y``."""
        return self._y0


__all__ = ["ExtOUWithJumpsProcess"]
