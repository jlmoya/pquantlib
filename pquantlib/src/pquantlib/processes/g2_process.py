"""G2Process — two-factor Gaussian short-rate driver process.

# C++ parity: ql/processes/g2process.{hpp,cpp} (v1.42.1).

Two correlated OU processes:
``dx_t = -a x_t dt + sigma dW^1_t``
``dy_t = -b y_t dt + eta   dW^2_t``
with ``dW^1 dW^2 = rho dt``. ``x0=y0=0`` (matches C++ which only
exposes mutators not ctor args for these).

The diffusion matrix is the Cholesky factor of the correlation matrix
scaled by ``(sigma, eta)``:

::

      sigma      0
      rho*sigma  sqrt(1-rho^2)*eta

For ``std_deviation`` the C++ implementation rescales rho by the
ratio ``H/den`` where:

::

      H = (rho * sigma * eta) / (a+b) * (1 - exp(-a*dt) * exp(-b*dt))
      den = 0.5 * sigma * eta * sqrt((1 - exp(-2*a*dt)) * (1 - exp(-2*b*dt)) / (a*b))

This corresponds to the *integrated* instantaneous correlation between
the two OU increments over the interval ``[t0, t0+dt]``. We mirror
this formula exactly (no algebraic simplification).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.processes.stochastic_process import StochasticProcess


class G2Process(StochasticProcess):
    """G2 two-factor Gaussian process.

    # C++ parity: ``class G2Process`` in
    # ql/processes/g2process.hpp:34-58 (v1.42.1).
    """

    __slots__ = (
        "_a",
        "_b",
        "_eta",
        "_rho",
        "_sigma",
        "_x0",
        "_x_process",
        "_y0",
        "_y_process",
    )

    def __init__(self, a: float, sigma: float, b: float, eta: float, rho: float) -> None:
        # C++ parity: g2process.cpp:25-28 — constructs two OU processes
        # at zero level with the given mean-reversion + vol; x0=y0=0
        # are hard-coded.
        super().__init__(discretization=None)
        self._a: float = float(a)
        self._sigma: float = float(sigma)
        self._b: float = float(b)
        self._eta: float = float(eta)
        self._rho: float = float(rho)
        self._x0: float = 0.0
        self._y0: float = 0.0
        self._x_process: OrnsteinUhlenbeckProcess = OrnsteinUhlenbeckProcess(a, sigma, 0.0)
        self._y_process: OrnsteinUhlenbeckProcess = OrnsteinUhlenbeckProcess(b, eta, 0.0)

    # --- inspectors ----------------------------------------------------

    def x0(self) -> float:
        return self._x0

    def y0(self) -> float:
        return self._y0

    def a(self) -> float:
        return self._a

    def sigma(self) -> float:
        return self._sigma

    def b(self) -> float:
        return self._b

    def eta(self) -> float:
        return self._eta

    def rho(self) -> float:
        return self._rho

    # --- StochasticProcess overrides -----------------------------------

    def size(self) -> int:
        # C++ parity: g2process.cpp:30-32.
        return 2

    def initial_values(self) -> npt.NDArray[np.float64]:
        # C++ parity: g2process.cpp:34-36.
        return np.array([self._x0, self._y0], dtype=np.float64)

    def drift(self, t: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # C++ parity: g2process.cpp:38-43.
        return np.array(
            [
                self._x_process.drift_1d(t, float(x[0])),
                self._y_process.drift_1d(t, float(x[1])),
            ],
            dtype=np.float64,
        )

    def diffusion(self, t: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # C++ parity: g2process.cpp:45-59 — Cholesky factor of the
        # 2x2 correlation matrix scaled by (sigma, eta).
        out = np.zeros((2, 2), dtype=np.float64)
        out[0, 0] = self._sigma
        out[0, 1] = 0.0
        out[1, 0] = self._rho * self._sigma
        out[1, 1] = math.sqrt(1.0 - self._rho * self._rho) * self._eta
        return out

    def expectation(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        # C++ parity: g2process.cpp:61-67.
        return np.array(
            [
                self._x_process.expectation_1d(t0, float(x0[0]), dt),
                self._y_process.expectation_1d(t0, float(x0[1]), dt),
            ],
            dtype=np.float64,
        )

    def std_deviation(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        # C++ parity: g2process.cpp:69-90 — rescales rho by the
        # integrated correlation H/den.
        sigma1 = self._x_process.std_deviation_1d(t0, float(x0[0]), dt)
        sigma2 = self._y_process.std_deviation_1d(t0, float(x0[1]), dt)
        expa = math.exp(-self._a * dt)
        expb = math.exp(-self._b * dt)
        h = (
            (self._rho * self._sigma * self._eta) / (self._a + self._b) * (1.0 - expa * expb)
        )
        den = (0.5 * self._sigma * self._eta) * math.sqrt(
            (1.0 - expa * expa) * (1.0 - expb * expb) / (self._a * self._b)
        )
        new_rho = h / den
        out = np.zeros((2, 2), dtype=np.float64)
        out[0, 0] = sigma1
        out[0, 1] = 0.0
        out[1, 0] = new_rho * sigma2
        out[1, 1] = math.sqrt(1.0 - new_rho * new_rho) * sigma2
        return out

    def covariance(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        # C++ parity: g2process.cpp:92-96 — ``S * S^T``.
        sigma = self.std_deviation(t0, x0, dt)
        return sigma @ sigma.T


__all__ = ["G2Process"]
