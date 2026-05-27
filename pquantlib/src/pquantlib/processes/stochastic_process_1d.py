"""StochasticProcess1D — 1-D specialization of StochasticProcess.

# C++ parity: ql/stochasticprocess.{hpp,cpp} (v1.42.1) ``class
# StochasticProcess1D : public StochasticProcess``.

The C++ class adds scalar-valued methods (``Real`` instead of ``Array``)
that bypass the Array / Matrix machinery for 1-D processes:

* ``x0`` — scalar initial value.
* ``drift(t, x)``, ``diffusion(t, x)``, ``expectation(t0, x0, dt)``,
  ``std_deviation``, ``variance``, ``evolve``, ``apply`` — all scalar
  in / scalar out.

Python implements both sets of methods. The vector base methods
(``initial_values``, ``drift(t, Array)``, etc.) dispatch through the
scalar versions, so 1-D processes only override the scalar overloads.

Discretization is split: ``StochasticProcessDiscretization`` (multi-D)
lives in ``stochastic_process.py``; the 1-D variant lives here as
``StochasticProcess1DDiscretization``. Both are pure abstract classes;
``EulerDiscretization`` implements both interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.processes.stochastic_process import StochasticProcess


class StochasticProcess1DDiscretization(ABC):
    """Abstract discretization of a 1-D stochastic process.

    # C++ parity: nested ``StochasticProcess1D::discretization`` abstract
    # class in ``ql/stochasticprocess.hpp``.
    """

    @abstractmethod
    def drift(
        self,
        process: StochasticProcess1D,
        t0: float,
        x0: float,
        dt: float,
    ) -> float:
        """Drift over ``[t0, t0+dt]`` starting at ``x0`` (scalar)."""

    @abstractmethod
    def diffusion(
        self,
        process: StochasticProcess1D,
        t0: float,
        x0: float,
        dt: float,
    ) -> float:
        """Diffusion over ``[t0, t0+dt]`` starting at ``x0`` (scalar)."""

    @abstractmethod
    def variance(
        self,
        process: StochasticProcess1D,
        t0: float,
        x0: float,
        dt: float,
    ) -> float:
        """Variance over ``[t0, t0+dt]`` starting at ``x0`` (scalar)."""


class StochasticProcess1D(StochasticProcess, ABC):
    """Abstract 1-D stochastic process.

    # C++ parity: ``class StochasticProcess1D : public StochasticProcess``.

    Subclasses MUST implement scalar overloads ``x0`` / ``drift(t, x)`` /
    ``diffusion(t, x)``. They MAY override the scalar ``expectation`` /
    ``std_deviation`` / ``variance`` / ``evolve`` / ``apply``.

    The vector base interface is satisfied by trivial dispatch through
    the scalar methods (mirrors the C++ inline definitions).
    """

    def __init__(self, discretization: StochasticProcess1DDiscretization | None = None) -> None:
        # The C++ base ctor takes the 1-D discretization variant only.
        # We don't forward to StochasticProcess.__init__'s multi-D
        # discretization slot because that has incompatible signatures.
        # Instead the 1-D variant is stored separately, and the vector
        # base methods are reimplemented to dispatch through scalar.
        StochasticProcess.__init__(self, discretization=None)
        self._discretization_1d: StochasticProcess1DDiscretization | None = discretization

    # --- abstract scalar interface --------------------------------------

    @abstractmethod
    def x0(self) -> float:
        """Initial value of the (scalar) state variable.

        # C++ parity: ``StochasticProcess1D::x0``.
        """

    @abstractmethod
    def drift_1d(self, t: float, x: float) -> float:
        """Scalar drift at ``(t, x)``.

        # C++ parity: ``StochasticProcess1D::drift(Time, Real)``.

        Renamed ``drift_1d`` in the Python port because the multi-D
        ``drift(t, Array) -> Array`` is also defined on this class
        (overload resolution by type doesn't exist in Python).
        """

    @abstractmethod
    def diffusion_1d(self, t: float, x: float) -> float:
        """Scalar diffusion at ``(t, x)``.

        # C++ parity: ``StochasticProcess1D::diffusion(Time, Real)``.
        See note on naming under ``drift_1d``.
        """

    # --- defaulted scalar interface -------------------------------------

    def expectation_1d(self, t0: float, x0: float, dt: float) -> float:
        """Scalar expectation at ``t0 + dt``.

        # C++ parity: ``StochasticProcess1D::expectation`` —
        # ``apply(x0, discretization_->drift(*this, t0, x0, dt))``.
        """
        qassert.require(self._discretization_1d is not None, "no 1-D discretization provided")
        assert self._discretization_1d is not None
        return self.apply_1d(x0, self._discretization_1d.drift(self, t0, x0, dt))

    def std_deviation_1d(self, t0: float, x0: float, dt: float) -> float:
        """Scalar standard deviation at ``t0 + dt``.

        # C++ parity: ``StochasticProcess1D::stdDeviation`` —
        # ``discretization_->diffusion(*this, t0, x0, dt)``.
        """
        qassert.require(self._discretization_1d is not None, "no 1-D discretization provided")
        assert self._discretization_1d is not None
        return self._discretization_1d.diffusion(self, t0, x0, dt)

    def variance_1d(self, t0: float, x0: float, dt: float) -> float:
        """Scalar variance at ``t0 + dt``.

        # C++ parity: ``StochasticProcess1D::variance`` —
        # ``discretization_->variance(*this, t0, x0, dt)``.
        """
        qassert.require(self._discretization_1d is not None, "no 1-D discretization provided")
        assert self._discretization_1d is not None
        return self._discretization_1d.variance(self, t0, x0, dt)

    def evolve_1d(self, t0: float, x0: float, dt: float, dw: float) -> float:
        """Scalar evolution at ``t0 + dt`` with Brownian increment ``dw``.

        # C++ parity: ``StochasticProcess1D::evolve`` —
        # ``apply(expectation(t0, x0, dt), stdDeviation(t0, x0, dt) * dw)``.
        """
        return self.apply_1d(
            self.expectation_1d(t0, x0, dt),
            self.std_deviation_1d(t0, x0, dt) * dw,
        )

    def apply_1d(self, x0: float, dx: float) -> float:
        """Scalar apply: default ``x0 + dx``.

        # C++ parity: ``StochasticProcess1D::apply`` — default
        # ``x0 + dx``. Overridable (BSM-family overrides to
        # ``x0 * exp(dx)`` because it operates in log-space).
        """
        return x0 + dx

    # --- multi-D base dispatch (inline forwarders) -----------------------

    def size(self) -> int:
        """Always 1 for a 1-D process.

        # C++ parity: ``StochasticProcess1D::size`` (inline, returns 1).
        """
        return 1

    def initial_values(self) -> npt.NDArray[np.float64]:
        """Single-entry vector with ``x0`` inside.

        # C++ parity: ``StochasticProcess1D::initialValues``.
        """
        return np.array([self.x0()], dtype=np.float64)

    def drift(self, t: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Vector drift dispatching through ``drift_1d``.

        # C++ parity: inline definition in ``stochasticprocess.hpp``.
        """
        return np.array([self.drift_1d(t, float(x[0]))], dtype=np.float64)

    def diffusion(self, t: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Vector diffusion dispatching through ``diffusion_1d``.

        # C++ parity: inline definition in ``stochasticprocess.hpp``.
        Returns a 1x1 matrix.
        """
        return np.array([[self.diffusion_1d(t, float(x[0]))]], dtype=np.float64)

    def expectation(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        return np.array([self.expectation_1d(t0, float(x0[0]), dt)], dtype=np.float64)

    def std_deviation(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        """Returns a 1x1 matrix.

        # C++ parity: inline ``stdDeviation(Time, Array, Time)``.
        """
        return np.array([[self.std_deviation_1d(t0, float(x0[0]), dt)]], dtype=np.float64)

    def covariance(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        """Returns a 1x1 matrix containing ``variance_1d``.

        # C++ parity: inline ``covariance(Time, Array, Time)``.
        """
        return np.array([[self.variance_1d(t0, float(x0[0]), dt)]], dtype=np.float64)

    def evolve(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
        dw: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return np.array(
            [self.evolve_1d(t0, float(x0[0]), dt, float(dw[0]))],
            dtype=np.float64,
        )

    def apply(
        self,
        x0: npt.NDArray[np.float64],
        dx: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return np.array([self.apply_1d(float(x0[0]), float(dx[0]))], dtype=np.float64)


__all__ = [
    "StochasticProcess1D",
    "StochasticProcess1DDiscretization",
]
