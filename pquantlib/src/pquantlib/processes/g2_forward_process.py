"""G2ForwardProcess — G2 process expressed under the T-forward measure.

# C++ parity: ql/processes/g2process.{hpp,cpp} (v1.43) — the
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

v1.43 made the process term-structure aware, exactly as for
:class:`~pquantlib.processes.g2_process.G2Process`: the simulated state is
shifted so that ``state[0] + state[1] == r(t)``, on top of the usual
T-forward convexity adjustments. Passing no term structure degenerates it to
the pre-v1.43 behaviour.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.processes.forward_measure_process import ForwardMeasureProcess
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency

# C++ parity: g2process.cpp — the phi'(t) forward difference step used by
# ``drift`` in shifted coordinates.
_PHI_DERIVATIVE_STEP: float = 1.0e-4


class G2ForwardProcess(ForwardMeasureProcess):
    """G2 process under the T-forward measure.

    # C++ parity: ``class G2ForwardProcess`` in
    # ql/processes/g2process.hpp:88-113 (v1.43).
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
        # C++ parity: g2process.cpp:169-177 (v1.43). Default ``T_`` is left
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
        self._term_structure: YieldTermStructure | None = term_structure
        if term_structure is not None:
            term_structure.register_with(self)

    # --- term-structure awareness (v1.43) -------------------------------

    def term_structure(self) -> YieldTermStructure | None:
        """The curve the process is fitted to, or ``None`` for the empty case.

        # C++ parity: ``G2ForwardProcess::termStructure``.
        """
        return self._term_structure

    def phi(self, t: float) -> float:
        """Deterministic shift that fits the initial term structure.

        # C++ parity: ``G2ForwardProcess::phi`` — same closed form as
        # ``G2Process::phi``.
        """
        qassert.require(
            self._term_structure is not None,
            "no term structure given to G2ForwardProcess",
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

        # C++ parity: ``G2ForwardProcess::shortRate`` — the state already
        # carries phi(t) in z1, so r is just the sum.
        """
        del t
        return z1 + z2

    def _shift_drift(self, t: float) -> float:
        """``a*phi(t) + phi'(t)`` — the extra drift in shifted coordinates."""
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
        # C++ parity: g2process.cpp (v1.43) — the first component starts at
        # phi(0) once a curve is attached.
        z1_0 = self._x0 if self._term_structure is None else self.phi(0.0)
        return np.array([z1_0, self._y0], dtype=np.float64)

    def drift(self, t: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # C++ parity: g2process.cpp (v1.43) — the shift drift sits on top of
        # the T-forward correction.
        shift_drift = self._shift_drift(t)
        return np.array(
            [
                self._x_process.drift_1d(t, float(x[0]))
                + self._x_forward_drift(t, self._T)
                + shift_drift,
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
        # C++ parity: g2process.cpp (v1.43) — the shift's contribution is
        # added after the T-forward correction.
        shift_exp = self._shift_expectation(t0, dt)
        return np.array(
            [
                self._x_process.expectation_1d(t0, float(x0[0]), dt)
                - self._Mx_T(t0, t0 + dt, self._T)
                + shift_exp,
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
