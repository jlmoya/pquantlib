"""EulerDiscretization — first-order time-stepping discretization.

# C++ parity: ql/processes/eulerdiscretization.{hpp,cpp} (v1.42.1).

C++ multi-inherits both ``StochasticProcess::discretization`` and
``StochasticProcess1D::discretization``. The single C++ class
implements:

* drift(P, t0, x0, dt) = mu(t0, x0) * dt
* diffusion(P, t0, x0, dt) = sigma(t0, x0) * sqrt(dt)
* covariance(P, t0, x0, dt) = sigma sigma^T * dt
* variance(P, t0, x0, dt) = sigma(t0, x0)^2 * dt   [1-D only]

The Python port preserves the two-discretization-base design from
StochasticProcess but uses ``typing.overload`` decorators to give
``drift`` and ``diffusion`` a typed overloaded signature. The runtime
implementation branches on ``isinstance(x0, np.ndarray)``.
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


class EulerDiscretization(StochasticProcessDiscretization, StochasticProcess1DDiscretization):
    """First-order (Euler) discretization for stochastic processes.

    Used as the default discretization in BSM-family processes.

    # C++ parity: ``class EulerDiscretization : public
    # StochasticProcess::discretization, public
    # StochasticProcess1D::discretization``.
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
        """Drift over ``[t0, t0+dt]``.

        # C++ parity: ``EulerDiscretization::drift`` —
        # ``process.drift(t0, x0) * dt``.
        """
        if isinstance(x0, np.ndarray):
            return process.drift(t0, x0) * dt
        assert isinstance(process, StochasticProcess1D)
        return process.drift_1d(t0, x0) * dt

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
        """Diffusion over ``[t0, t0+dt]``.

        # C++ parity: ``EulerDiscretization::diffusion`` —
        # ``process.diffusion(t0, x0) * sqrt(dt)``.
        """
        sqrt_dt = math.sqrt(dt)
        if isinstance(x0, np.ndarray):
            return process.diffusion(t0, x0) * sqrt_dt
        assert isinstance(process, StochasticProcess1D)
        return process.diffusion_1d(t0, x0) * sqrt_dt

    # --- covariance (multi-D only) --------------------------------------

    def covariance(
        self,
        process: StochasticProcess,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        """Covariance matrix sigma sigma^T * dt.

        # C++ parity: ``EulerDiscretization::covariance``.
        """
        sigma = process.diffusion(t0, x0)
        return sigma @ sigma.T * dt

    # --- variance (1-D only) --------------------------------------------

    def variance(
        self,
        process: StochasticProcess1D,
        t0: float,
        x0: float,
        dt: float,
    ) -> float:
        """Scalar variance sigma^2 * dt (1-D specialization).

        # C++ parity: 1-D overload of ``EulerDiscretization::variance``.
        """
        sigma = process.diffusion_1d(t0, x0)
        return sigma * sigma * dt


__all__ = ["EulerDiscretization"]
