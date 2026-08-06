"""GeometricBrownianMotionProcess — plain geometric Brownian motion.

# C++ parity: ql/processes/geometricbrownianprocess.{hpp,cpp} (v1.43).

Describes the SDE

    dS(t, S) = mue * S dt + sigma * S dW_t

with three constants: the initial value, the (proportional) drift ``mue``
and the (proportional) volatility ``sigma``.

Note what this class deliberately is NOT: unlike the BSM family it does
**not** work in log-space, so ``apply`` stays at the base ``x0 + dx`` and
``expectation``/``stdDeviation``/``variance`` come straight from the Euler
discretization rather than from a closed form. That means
``expectation(t0, x0, dt) == x0 * (1 + mue*dt)``, not ``x0 * exp(mue*dt)`` —
an approximation the C++ makes on purpose by supplying an
``EulerDiscretization`` and overriding nothing else.
"""

from __future__ import annotations

from pquantlib.processes.euler_discretization import EulerDiscretization
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D


class GeometricBrownianMotionProcess(StochasticProcess1D):
    """Geometric Brownian-motion process.

    # C++ parity: ``class GeometricBrownianMotionProcess : public
    # StochasticProcess1D`` (geometricbrownianprocess.hpp:41-54).
    """

    __slots__ = ("_initial_value", "_mue", "_sigma")

    def __init__(self, initial_value: float, mue: float, sigma: float) -> None:
        """Construct from the initial value, drift and volatility.

        # C++ parity: geometricbrownianprocess.cpp:27-33 — the base is
        # constructed with a fresh ``EulerDiscretization``; there are no
        # sign or range checks on any argument.
        """
        super().__init__(EulerDiscretization())
        self._initial_value: float = float(initial_value)
        self._mue: float = float(mue)
        self._sigma: float = float(sigma)

    # --- inspectors ------------------------------------------------------

    def x0(self) -> float:
        # C++ parity: geometricbrownianprocess.cpp:35-37.
        return self._initial_value

    def mue(self) -> float:
        """Proportional drift.

        # C++ parity: ``mue_`` is a protected member with no accessor in
        # C++; the Python port exposes a read-only one for testability.
        """
        return self._mue

    def sigma(self) -> float:
        """Proportional volatility.

        # C++ parity: ``sigma_`` is a protected member with no accessor in
        # C++; the Python port exposes a read-only one for testability.
        """
        return self._sigma

    # --- StochasticProcess1D scalar interface ---------------------------

    def drift_1d(self, t: float, x: float) -> float:
        """``mue * x`` — independent of ``t``.

        # C++ parity: geometricbrownianprocess.cpp:39-41.
        """
        del t  # C++ leaves the Time parameter unnamed
        return self._mue * x

    def diffusion_1d(self, t: float, x: float) -> float:
        """``sigma * x`` — independent of ``t``.

        # C++ parity: geometricbrownianprocess.cpp:43-45.
        """
        del t  # C++ leaves the Time parameter unnamed
        return self._sigma * x


__all__ = ["GeometricBrownianMotionProcess"]
