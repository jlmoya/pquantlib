"""EndEulerDiscretization — Euler *end-point* time-stepping discretization.

# C++ parity: ql/processes/endeulerdiscretization.{hpp,cpp} (v1.43).

Identical in shape to :class:`~pquantlib.processes.euler_discretization.EulerDiscretization`
except that every coefficient is evaluated at the END of the step,
``t0 + dt``, instead of at its start:

* drift(P, t0, x0, dt)      = mu(t0 + dt, x0) * dt
* diffusion(P, t0, x0, dt)  = sigma(t0 + dt, x0) * sqrt(dt)
* covariance(P, t0, x0, dt) = sigma(t0 + dt, x0) sigma(t0 + dt, x0)^T * dt
* variance(P, t0, x0, dt)   = sigma(t0 + dt, x0)^2 * dt   [1-D only]

Note that only ``t`` moves to the end of the step — ``x0`` stays put. This
is not a Runge-Kutta or implicit scheme; it is the same first-order Euler
step with the time argument shifted, which matters only for processes whose
drift or diffusion actually depend on ``t``.

C++ multi-inherits both ``StochasticProcess::discretization`` and
``StochasticProcess1D::discretization``; the Python port mirrors
``euler_discretization.py`` exactly, using ``typing.overload`` for the two
typed signatures and branching at runtime on ``isinstance(x0, np.ndarray)``.
"""

from __future__ import annotations

import math
from typing import overload

import numpy as np
import numpy.typing as npt

from pquantlib.processes.stochastic_process import (
    StochasticProcess,
    StochasticProcessDiscretization,
)
from pquantlib.processes.stochastic_process_1d import (
    StochasticProcess1D,
    StochasticProcess1DDiscretization,
)


class EndEulerDiscretization(StochasticProcessDiscretization, StochasticProcess1DDiscretization):
    """First-order Euler discretization evaluated at the end of the step.

    # C++ parity: ``class EndEulerDiscretization : public
    # StochasticProcess::discretization, public
    # StochasticProcess1D::discretization`` (endeulerdiscretization.hpp:33-64).
    """

    # --- drift (overloaded) ---------------------------------------------

    @overload
    def drift(
        self,
        process: StochasticProcess,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]: ...

    @overload
    def drift(
        self,
        process: StochasticProcess1D,
        t0: float,
        x0: float,
        dt: float,
    ) -> float: ...

    def drift(
        self,
        process: StochasticProcess | StochasticProcess1D,
        t0: float,
        x0: npt.NDArray[np.float64] | float,
        dt: float,
    ) -> npt.NDArray[np.float64] | float:
        """Drift over ``[t0, t0+dt]`` sampled at the end point.

        # C++ parity: ``EndEulerDiscretization::drift`` —
        # ``process.drift(t0 + dt, x0) * dt`` (endeulerdiscretization.cpp:24-33).
        """
        if isinstance(x0, np.ndarray):
            return process.drift(t0 + dt, x0) * dt
        assert isinstance(process, StochasticProcess1D)
        return process.drift_1d(t0 + dt, x0) * dt

    # --- diffusion (overloaded) -----------------------------------------

    @overload
    def diffusion(
        self,
        process: StochasticProcess,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]: ...

    @overload
    def diffusion(
        self,
        process: StochasticProcess1D,
        t0: float,
        x0: float,
        dt: float,
    ) -> float: ...

    def diffusion(
        self,
        process: StochasticProcess | StochasticProcess1D,
        t0: float,
        x0: npt.NDArray[np.float64] | float,
        dt: float,
    ) -> npt.NDArray[np.float64] | float:
        """Diffusion over ``[t0, t0+dt]`` sampled at the end point.

        # C++ parity: ``EndEulerDiscretization::diffusion`` —
        # ``process.diffusion(t0 + dt, x0) * sqrt(dt)``
        # (endeulerdiscretization.cpp:35-45).
        """
        sqrt_dt = math.sqrt(dt)
        if isinstance(x0, np.ndarray):
            return process.diffusion(t0 + dt, x0) * sqrt_dt
        assert isinstance(process, StochasticProcess1D)
        return process.diffusion_1d(t0 + dt, x0) * sqrt_dt

    # --- covariance (multi-D only) --------------------------------------

    def covariance(
        self,
        process: StochasticProcess,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        """Covariance matrix ``sigma sigma^T * dt`` sampled at the end point.

        # C++ parity: ``EndEulerDiscretization::covariance``
        # (endeulerdiscretization.cpp:47-54).
        """
        sigma = process.diffusion(t0 + dt, x0)
        return sigma @ sigma.T * dt

    # --- variance (1-D only) --------------------------------------------

    def variance(
        self,
        process: StochasticProcess1D,
        t0: float,
        x0: float,
        dt: float,
    ) -> float:
        """Scalar variance ``sigma^2 * dt`` sampled at the end point.

        # C++ parity: ``EndEulerDiscretization::variance``
        # (endeulerdiscretization.cpp:56-60).
        """
        sigma = process.diffusion_1d(t0 + dt, x0)
        return sigma * sigma * dt


__all__ = ["EndEulerDiscretization"]
