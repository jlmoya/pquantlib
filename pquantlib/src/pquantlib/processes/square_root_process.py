"""SquareRootProcess — CIR-type square-root diffusion.

# C++ parity: ql/processes/squarerootprocess.{hpp,cpp} (v1.42.1).

Describes the square-root (Cox-Ingersoll-Ross) process

    dx = a (b - x_t) dt + sigma sqrt(x_t) dW_t

with mean-reversion speed ``a``, long-run mean ``b``, volatility
``sigma`` and initial state ``x0``. The default time-stepping is an
``EulerDiscretization`` (matching C++). This is the kernel process the
``SquareRootCLVModel`` collocates onto.
"""

from __future__ import annotations

import math
from typing import final

from pquantlib.processes.euler_discretization import EulerDiscretization
from pquantlib.processes.stochastic_process_1d import (
    StochasticProcess1D,
    StochasticProcess1DDiscretization,
)


@final
class SquareRootProcess(StochasticProcess1D):
    """Square-root (CIR) 1-D process.

    # C++ parity: ``class SquareRootProcess : public StochasticProcess1D``
    # in squarerootprocess.hpp:42-61.

    Parameters
    ----------
    b
        Long-run mean.
    a
        Mean-reversion speed.
    sigma
        Volatility.
    x0
        Initial state.
    discretization
        Time-stepping scheme (defaults to ``EulerDiscretization``).
    """

    __slots__ = ("_mean", "_speed", "_volatility", "_x0")

    def __init__(
        self,
        b: float,
        a: float,
        sigma: float,
        x0: float = 0.0,
        discretization: StochasticProcess1DDiscretization | None = None,
    ) -> None:
        # C++ parity: squarerootprocess.cpp:26-30.
        super().__init__(
            discretization=discretization
            if discretization is not None
            else EulerDiscretization()
        )
        self._x0: float = float(x0)
        self._mean: float = float(b)
        self._speed: float = float(a)
        self._volatility: float = float(sigma)

    def x0(self) -> float:
        # C++ parity: squarerootprocess.cpp:32-34.
        return self._x0

    def drift_1d(self, t: float, x: float) -> float:
        # C++ parity: squarerootprocess.cpp:36-38 — speed_ * (mean_ - x).
        return self._speed * (self._mean - x)

    def diffusion_1d(self, t: float, x: float) -> float:
        # C++ parity: squarerootprocess.cpp:40-42 — volatility_ * sqrt(x).
        return self._volatility * math.sqrt(x)

    def a(self) -> float:
        # C++ parity: squarerootprocess.hpp:55 — speed_.
        return self._speed

    def b(self) -> float:
        # C++ parity: squarerootprocess.hpp:56 — mean_.
        return self._mean

    def sigma(self) -> float:
        # C++ parity: squarerootprocess.hpp:57 — volatility_.
        return self._volatility


__all__ = ["SquareRootProcess"]
