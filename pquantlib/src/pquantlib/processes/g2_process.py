"""G2Process — two-factor Gaussian short-rate driver process.

# C++ parity: ql/processes/g2process.{hpp,cpp} (v1.43).

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

v1.43 made the process term-structure aware. The simulated state is
shifted to ``(z1, z2) = (x + phi(t), y)``, so a path generator built on
this process produces ``r(t_i) = state[0]_i + state[1]_i`` with
curve-consistent expectation ``phi(t_i)``. Passing no term structure
degenerates the process to the pre-v1.43 pair of zero-mean OU processes
(``phi == 0``), which is what the default keeps doing.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.processes.stochastic_process import StochasticProcess
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency

# C++ parity: g2process.cpp — the phi'(t) forward difference step used by
# ``drift`` in shifted coordinates.
_PHI_DERIVATIVE_STEP: float = 1.0e-4


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
        "_term_structure",
        "_x0",
        "_x_process",
        "_y0",
        "_y_process",
    )

    def __init__(
        self,
        a: float,
        sigma: float,
        b: float,
        eta: float,
        rho: float,
        term_structure: YieldTermStructure | None = None,
    ) -> None:
        # C++ parity: g2process.cpp:25-33 (v1.43) — two OU processes at zero
        # level with the given mean-reversion + vol (x0=y0=0 are hard-coded),
        # plus the optional term structure the process registers with.
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
        self._term_structure: YieldTermStructure | None = term_structure
        if term_structure is not None:
            term_structure.register_with(self)

    # --- inspectors ----------------------------------------------------

    def x0(self) -> float:
        # C++ parity: g2process.cpp:98-100 (v1.43) — with a curve the state's
        # first component starts at phi(0), not at the OU factor's own x0.
        return self._x0 if self._term_structure is None else self.phi(0.0)

    def term_structure(self) -> YieldTermStructure | None:
        """The curve the process is fitted to, or ``None`` for the empty case.

        # C++ parity: ``G2Process::termStructure`` (g2process.cpp:102-104).
        """
        return self._term_structure

    def phi(self, t: float) -> float:
        """Deterministic shift that fits the initial term structure.

        # C++ parity: ``G2Process::phi`` (g2process.cpp:106-113).
        """
        qassert.require(
            self._term_structure is not None, "no term structure given to G2Process"
        )
        assert self._term_structure is not None
        forward = self._term_structure.forward_rate(
            t, t, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        temp1 = self._sigma * (1.0 - math.exp(-self._a * t)) / self._a
        temp2 = self._eta * (1.0 - math.exp(-self._b * t)) / self._b
        return (
            0.5 * temp1 * temp1
            + 0.5 * temp2 * temp2
            + self._rho * temp1 * temp2
            + forward
        )

    def short_rate(self, t: float, z1: float, z2: float) -> float:
        """Short rate implied by a simulated state.

        # C++ parity: ``G2Process::shortRate`` (g2process.cpp:115-118). The
        # simulated state already carries phi(t) in z1, so r is just the sum.
        """
        del t
        return z1 + z2

    def _shift_drift(self, t: float) -> float:
        """``a*phi(t) + phi'(t)`` — the extra drift in shifted coordinates.

        Zero when no term structure is attached.
        """
        if self._term_structure is None:
            return 0.0
        phi_t = self.phi(t)
        phi_th = self.phi(t + _PHI_DERIVATIVE_STEP)
        return self._a * phi_t + (phi_th - phi_t) / _PHI_DERIVATIVE_STEP

    def _shift_expectation(self, t0: float, dt: float) -> float:
        """``phi(t0+dt) - phi(t0)*exp(-a*dt)`` — the shift's contribution."""
        if self._term_structure is None:
            return 0.0
        return self.phi(t0 + dt) - self.phi(t0) * math.exp(-self._a * dt)

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
        # C++ parity: g2process.cpp:35-38 (v1.43) — the first component starts
        # at phi(0) once a curve is attached.
        z1_0 = self._x0 if self._term_structure is None else self.phi(0.0)
        return np.array([z1_0, self._y0], dtype=np.float64)

    def drift(self, t: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # C++ parity: g2process.cpp:40-58 (v1.43). In shifted coordinates
        # z1 = x + phi(t), z2 = y:
        #   dz1 = (-a*z1 + a*phi(t) + phi'(t)) dt + sigma dW1
        #   dz2 = -b*z2 dt + eta dW2
        shift_drift = self._shift_drift(t)
        return np.array(
            [
                self._x_process.drift_1d(t, float(x[0])) + shift_drift,
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
        # C++ parity: g2process.cpp:76-92 (v1.43):
        #   E[z1(t0+dt)] = z1(t0)*exp(-a*dt) + phi(t0+dt) - phi(t0)*exp(-a*dt)
        #   E[z2(t0+dt)] = z2(t0)*exp(-b*dt)
        shift_exp = self._shift_expectation(t0, dt)
        return np.array(
            [
                self._x_process.expectation_1d(t0, float(x0[0]), dt) + shift_exp,
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
