"""ExtendedOrnsteinUhlenbeckProcess — time-dependent-mean OU.

# C++ parity: ql/experimental/processes/extendedornsteinuhlenbeckprocess.{hpp,cpp}
# (v1.43).

Describes the SDE

    dx_t = a (b(t) - x_t) dt + sigma dW_t

where ``b(t)`` is a user-supplied callable. Specialises the standard
``OrnsteinUhlenbeckProcess`` (constant mean ``b``) by allowing a
time-dependent target.

``variance`` and ``stdDeviation`` are the plain OU ones — the
time-dependent mean shifts the drift, not the noise. ``expectation``
differs, because the mean reversion pulls toward a moving target, and C++
offers three ways of approximating the integral of that pull over one
step:

* ``MidPoint`` (the default) — evaluate ``b`` once, at the middle of the
  step;
* ``Trapezodial`` — C++ spells it with that typo; evaluate ``b`` at both
  ends and use the exact linear-``b`` solution;
* ``GaussLobatto`` — integrate ``b(u) exp(a u)`` adaptively over the step.

The choice is not cosmetic: ``FdmSimpleProcess1dMesher`` calls
``evolve`` (hence ``expectation``) to size the mesh, so two engines that
differ only in this enum place their grid nodes differently.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import IntEnum
from typing import final

from pquantlib import qassert
from pquantlib.math.integrals.lobatto import GaussLobattoIntegral
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D


class Discretization(IntEnum):
    """Which approximation ``expectation`` uses for the moving mean.

    # C++ parity: ``enum Discretization { MidPoint, Trapezodial, GaussLobatto }``
    # (extendedornsteinuhlenbeckprocess.hpp:44). The middle spelling is C++'s
    # own typo for "Trapezoidal" and is reproduced so call sites transcribe.
    """

    MidPoint = 0
    Trapezodial = 1
    GaussLobatto = 2


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
    discretization
        Which ``expectation`` approximation to use. C++ default: ``MidPoint``.
    int_eps
        Absolute accuracy of the ``GaussLobatto`` integral. C++ default 1e-4.
    """

    __slots__ = (
        "_b",
        "_expectation_mode",
        "_int_eps",
        "_ou_process",
        "_speed",
        "_volatility",
        "_x0",
    )

    #: Exposed as a class attribute too, so call sites can write
    #: ``ExtendedOrnsteinUhlenbeckProcess.Discretization.Trapezodial`` exactly
    #: as C++ writes the nested enum.
    Discretization = Discretization

    def __init__(
        self,
        speed: float,
        sigma: float,
        x0: float,
        b: Callable[[float], float],
        discretization: Discretization = Discretization.MidPoint,
        int_eps: float = 1e-4,
    ) -> None:
        # No generic 1-D discretisation: every moment is overridden below,
        # exactly as C++ overrides expectation/stdDeviation/variance.
        super().__init__(discretization=None)
        qassert.require(speed >= 0.0, f"negative a given: {speed}")
        qassert.require(sigma >= 0.0, f"negative volatility given: {sigma}")
        self._speed: float = float(speed)
        self._volatility: float = float(sigma)
        self._x0: float = float(x0)
        self._b: Callable[[float], float] = b
        # Named _expectation_mode, not _discretization: StochasticProcess already
        # owns a _discretization slot (the generic path-evolution scheme), and
        # this enum selects something else entirely.
        self._expectation_mode: Discretization = discretization
        self._int_eps: float = float(int_eps)
        # C++ holds an OrnsteinUhlenbeckProcess(speed, vol, x0) and forwards
        # every moment to it; level defaults to 0 there, and the moving mean
        # enters only through the correction terms below.
        self._ou_process: OrnsteinUhlenbeckProcess = OrnsteinUhlenbeckProcess(
            float(speed), float(sigma), float(x0)
        )

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

    def std_deviation_1d(self, t0: float, x0: float, dt: float) -> float:
        """# C++ parity: forwards to the plain OU process (cpp:66-69)."""
        return self._ou_process.std_deviation_1d(t0, x0, dt)

    def variance_1d(self, t0: float, x0: float, dt: float) -> float:
        """# C++ parity: forwards to the plain OU process (cpp:71-74)."""
        return self._ou_process.variance_1d(t0, x0, dt)

    def expectation_1d(self, t0: float, x0: float, dt: float) -> float:
        """Plain-OU expectation plus the moving-mean correction.

        # C++ parity: ``ExtendedOrnsteinUhlenbeckProcess::expectation``
        # (extendedornsteinuhlenbeckprocess.cpp:84-111).
        """
        base = self._ou_process.expectation_1d(t0, x0, dt)
        a = self._speed
        if self._expectation_mode == Discretization.MidPoint:
            return base + self._b(t0 + 0.5 * dt) * (1.0 - math.exp(-a * dt))
        if self._expectation_mode == Discretization.Trapezodial:
            t = t0 + dt
            u = t0
            bt = self._b(t)
            bu = self._b(u)
            ex = math.exp(-a * dt)
            return base + bt - ex * bu - (bt - bu) / (a * dt) * (1.0 - ex)
        if self._expectation_mode == Discretization.GaussLobatto:
            # C++ integrand: b(x) * exp(speed * x), integrated over [t0, t0+dt].
            integral = GaussLobattoIntegral(100000, self._int_eps)
            return base + a * math.exp(-a * (t0 + dt)) * integral(
                lambda x: self._b(x) * math.exp(a * x), t0, t0 + dt
            )
        qassert.fail("unknown discretization scheme")
        raise AssertionError  # unreachable


__all__ = ["ExtendedOrnsteinUhlenbeckProcess"]
