"""ExtendedOrnsteinUhlenbeckProcess — time-dependent-mean OU.

# C++ parity: ql/experimental/processes/extendedornsteinuhlenbeckprocess.{hpp,cpp}
# (v1.42.1).

Describes the SDE

    dx_t = a (b(t) - x_t) dt + sigma dW_t

where ``b(t)`` is a user-supplied callable. Specialises the standard
``OrnsteinUhlenbeckProcess`` (constant mean ``b``) by allowing a
time-dependent target.

This W5-A port exposes the minimal surface required by the
``FdmExtendedOrnsteinUhlenbeckOp`` / ``FdmExtOUJumpOp`` /
``FdmKlugeExtOUOp`` FD operators:

* ``x0()`` / ``speed()`` / ``volatility()`` accessors.
* ``drift(t, x)`` — ``speed * (b(t) - x)``.
* ``diffusion(t, x)`` — constant ``volatility``.

The full ``expectation`` / ``variance`` / ``stdDeviation`` integration
support from the C++ class (MidPoint / Trapezoidal / GaussLobatto
discretisation modes) is deferred to a future Monte-Carlo cluster —
the FD ops here only use the drift + volatility scalars.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import final

from pquantlib import qassert
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D


@final
class ExtendedOrnsteinUhlenbeckProcess(StochasticProcess1D):
    """Time-dependent-mean Ornstein-Uhlenbeck process.

    # C++ parity: ``class ExtendedOrnsteinUhlenbeckProcess : public
    # StochasticProcess1D`` in extendedornsteinuhlenbeckprocess.hpp:42.

    Parameters
    ----------
    speed
        Mean-reversion speed ``a``.
    sigma
        Volatility (constant).
    x0
        Initial state.
    b
        Time-dependent mean callable ``b(t) -> float``.
    """

    __slots__ = ("_b", "_speed", "_volatility", "_x0")

    def __init__(
        self,
        speed: float,
        sigma: float,
        x0: float,
        b: Callable[[float], float],
    ) -> None:
        # No 1-D discretisation: closed-form drift/diffusion only.
        super().__init__(discretization=None)
        qassert.require(sigma >= 0.0, f"negative volatility given: {sigma}")
        self._speed: float = float(speed)
        self._volatility: float = float(sigma)
        self._x0: float = float(x0)
        self._b: Callable[[float], float] = b

    def x0(self) -> float:
        # C++ parity: extendedornsteinuhlenbeckprocess.cpp x0().
        return self._x0

    def speed(self) -> float:
        # C++ parity: speed() accessor.
        return self._speed

    def volatility(self) -> float:
        # C++ parity: volatility() accessor.
        return self._volatility

    def drift_1d(self, t: float, x: float) -> float:
        # C++ parity: drift(t, x) = speed_ * (b_(t) - x).
        return self._speed * (self._b(t) - x)

    def diffusion_1d(self, t: float, x: float) -> float:
        # C++ parity: diffusion(t, x) = vol_.
        return self._volatility

    def b(self, t: float) -> float:
        """Evaluate the time-dependent mean ``b(t)``."""
        return self._b(t)


__all__ = ["ExtendedOrnsteinUhlenbeckProcess"]
