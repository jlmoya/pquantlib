"""G2ForwardProcess — G2 process expressed under the T-forward measure.

# C++ parity: ql/processes/g2process.{hpp,cpp} (v1.42.1) — the
# ``G2ForwardProcess`` declared alongside ``G2Process``.

Same dynamics as ``G2Process`` but with the forward-measure drift
correction. The diffusion is unchanged (same Cholesky factor).

Two helper methods carry the forward-measure correction:

* ``xForwardDrift(t, T)`` — time-dependent extra drift for ``x_t``.
* ``yForwardDrift(t, T)`` — same for ``y_t``.
* ``Mx_T(s, t, T)`` / ``My_T(s, t, T)`` — integrated drift corrections
  used in ``expectation``.

The forward-measure horizon ``T_`` is mutable via
``set_forward_measure_time`` (inherited from ``ForwardMeasureProcess``).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from pquantlib.processes.forward_measure_process import ForwardMeasureProcess
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess


class G2ForwardProcess(ForwardMeasureProcess):
    """G2 process under the T-forward measure.

    # C++ parity: ``class G2ForwardProcess`` in
    # ql/processes/g2process.hpp:62-83 (v1.42.1).
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
        # C++ parity: g2process.cpp:127-130. Default ``T_`` is left
        # uninitialised in C++; callers MUST call
        # ``setForwardMeasureTime`` before pricing. Python defaults
        # to inf via the base ctor.
        super().__init__()
        self._a: float = float(a)
        self._sigma: float = float(sigma)
        self._b: float = float(b)
        self._eta: float = float(eta)
        self._rho: float = float(rho)
        self._x0: float = 0.0
        self._y0: float = 0.0
        self._x_process: OrnsteinUhlenbeckProcess = OrnsteinUhlenbeckProcess(a, sigma, 0.0)
        self._y_process: OrnsteinUhlenbeckProcess = OrnsteinUhlenbeckProcess(b, eta, 0.0)

    # --- forward-measure helpers ---------------------------------------

    def _x_forward_drift(self, t: float, T: float) -> float:  # noqa: N803 — math symbol
        # C++ parity: g2process.cpp:186-192.
        expat_t = math.exp(-self._a * (T - t))
        expbt_t = math.exp(-self._b * (T - t))
        return -(self._sigma * self._sigma / self._a) * (1.0 - expat_t) - (
            self._rho * self._sigma * self._eta / self._b
        ) * (1.0 - expbt_t)

    def _y_forward_drift(self, t: float, T: float) -> float:  # noqa: N803 — math symbol
        # C++ parity: g2process.cpp:194-200.
        expat_t = math.exp(-self._a * (T - t))
        expbt_t = math.exp(-self._b * (T - t))
        return -(self._eta * self._eta / self._b) * (1.0 - expbt_t) - (
            self._rho * self._sigma * self._eta / self._a
        ) * (1.0 - expat_t)

    def _Mx_T(self, s: float, t: float, T: float) -> float:  # noqa: N802, N803 — math symbol
        # C++ parity: g2process.cpp:202-211.
        M = (  # noqa: N806 — math symbol
            (self._sigma * self._sigma) / (self._a * self._a)
            + (self._rho * self._sigma * self._eta) / (self._a * self._b)
        ) * (1.0 - math.exp(-self._a * (t - s)))
        M += -(self._sigma * self._sigma) / (2.0 * self._a * self._a) * (  # noqa: N806  # pyright: ignore[reportConstantRedefinition]
            math.exp(-self._a * (T - t)) - math.exp(-self._a * (T + t - 2.0 * s))
        )
        M += -(self._rho * self._sigma * self._eta) / (self._b * (self._a + self._b)) * (  # noqa: N806  # pyright: ignore[reportConstantRedefinition]
            math.exp(-self._b * (T - t))
            - math.exp(-self._b * T - self._a * t + (self._a + self._b) * s)
        )
        return M

    def _My_T(self, s: float, t: float, T: float) -> float:  # noqa: N802, N803 — math symbol
        # C++ parity: g2process.cpp:213-222.
        M = (  # noqa: N806 — math symbol
            (self._eta * self._eta) / (self._b * self._b)
            + (self._rho * self._sigma * self._eta) / (self._a * self._b)
        ) * (1.0 - math.exp(-self._b * (t - s)))
        M += -(self._eta * self._eta) / (2.0 * self._b * self._b) * (  # noqa: N806  # pyright: ignore[reportConstantRedefinition]
            math.exp(-self._b * (T - t)) - math.exp(-self._b * (T + t - 2.0 * s))
        )
        M += -(self._rho * self._sigma * self._eta) / (self._a * (self._a + self._b)) * (  # noqa: N806  # pyright: ignore[reportConstantRedefinition]
            math.exp(-self._a * (T - t))
            - math.exp(-self._a * T - self._b * t + (self._a + self._b) * s)
        )
        return M

    # --- StochasticProcess overrides -----------------------------------

    def size(self) -> int:
        # C++ parity: g2process.cpp:132-134.
        return 2

    def initial_values(self) -> npt.NDArray[np.float64]:
        # C++ parity: g2process.cpp:136-138.
        return np.array([self._x0, self._y0], dtype=np.float64)

    def drift(self, t: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # C++ parity: g2process.cpp:140-145.
        return np.array(
            [
                self._x_process.drift_1d(t, float(x[0])) + self._x_forward_drift(t, self._T),
                self._y_process.drift_1d(t, float(x[1])) + self._y_forward_drift(t, self._T),
            ],
            dtype=np.float64,
        )

    def diffusion(self, t: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # C++ parity: g2process.cpp:147-154 — same Cholesky factor as G2.
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
        # C++ parity: g2process.cpp:156-162.
        return np.array(
            [
                self._x_process.expectation_1d(t0, float(x0[0]), dt)
                - self._Mx_T(t0, t0 + dt, self._T),
                self._y_process.expectation_1d(t0, float(x0[1]), dt)
                - self._My_T(t0, t0 + dt, self._T),
            ],
            dtype=np.float64,
        )

    def std_deviation(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        # C++ parity: g2process.cpp:164-178 — same rho-rescaling as G2.
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
        # C++ parity: g2process.cpp:180-184.
        sigma = self.std_deviation(t0, x0, dt)
        return sigma @ sigma.T


__all__ = ["G2ForwardProcess"]
