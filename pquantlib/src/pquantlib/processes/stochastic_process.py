"""StochasticProcess — multi-dimensional stochastic process abstract base.

# C++ parity: ql/stochasticprocess.{hpp,cpp} (v1.42.1) ``class
# StochasticProcess : public Observer, public Observable``.

The C++ class describes
``dx_t = mu(t, x_t) dt + sigma(t, x_t) dW_t`` for a vector ``x_t``,
with a ``discretization`` nested class that supplies drift / diffusion /
covariance over a finite interval.

Python divergences vs C++:

* C++ uses ``Array`` (1-D real) and ``Matrix`` (2-D real); the Python
  port uses ``numpy.ndarray[float64]`` directly (no separate Array /
  Matrix classes — those are deferred per L1 carve-out). For 1-D
  callers the array helper is constructed via ``np.array([x])``.
* The C++ nested ``StochasticProcess::discretization`` is a pure
  virtual abstract class with three methods (drift / diffusion /
  covariance). Python ports it as ``StochasticProcessDiscretization``
  in this module — a public Protocol-like abstract class. The 1-D
  specialization (``StochasticProcess1D::discretization``) sits in
  ``stochastic_process_1d.py``.
* C++ uses ``ext::shared_ptr<discretization>`` for ownership; Python
  uses a direct reference (no GC concerns — Python's GC handles it).
* ``time(date)`` is ``QL_FAIL("not implemented")`` in the C++ base
  (and most subclasses); the Python port raises ``LibraryException``
  with the same message. Subclasses that have a daycounter (like
  ``GeneralizedBlackScholesProcess``) override it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.patterns.observer import Observable
from pquantlib.time.date import Date


class StochasticProcessDiscretization(ABC):
    """Abstract discretization of a (multi-D) stochastic process.

    # C++ parity: nested ``StochasticProcess::discretization`` abstract
    # class in ``ql/stochasticprocess.hpp``.
    """

    @abstractmethod
    def drift(
        self,
        process: StochasticProcess,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        """Drift over ``[t0, t0+dt]`` starting at ``x0``."""

    @abstractmethod
    def diffusion(
        self,
        process: StochasticProcess,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        """Diffusion matrix over ``[t0, t0+dt]`` starting at ``x0``."""

    @abstractmethod
    def covariance(
        self,
        process: StochasticProcess,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        """Covariance matrix over ``[t0, t0+dt]`` starting at ``x0``."""


class StochasticProcess(Observable, ABC):
    """Abstract multi-dimensional stochastic process.

    # C++ parity: ``class StochasticProcess : public Observer, public
    # Observable``. The Python ``Observer`` is a structural Protocol
    # (anything with ``update()`` qualifies), so only ``Observable``
    # needs to be inherited. Concrete subclasses provide ``update()``
    # which makes them duck-type-compatible with the ``Observer``
    # Protocol — they can be passed to ``Observable.register_with``.

    Subclasses must implement ``size`` / ``initial_values`` / ``drift`` /
    ``diffusion``. They MAY override ``factors`` (defaults to ``size``),
    ``expectation`` / ``std_deviation`` / ``covariance`` / ``evolve``,
    and ``apply``.
    """

    def __init__(self, discretization: StochasticProcessDiscretization | None = None) -> None:
        Observable.__init__(self)
        self._discretization: StochasticProcessDiscretization | None = discretization

    # --- abstract --------------------------------------------------------

    @abstractmethod
    def size(self) -> int:
        """Number of dimensions of the state vector.

        # C++ parity: ``StochasticProcess::size``.
        """

    @abstractmethod
    def initial_values(self) -> npt.NDArray[np.float64]:
        """Initial value of the state vector.

        # C++ parity: ``StochasticProcess::initialValues``.
        """

    @abstractmethod
    def drift(self, t: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Drift vector at ``(t, x)``.

        # C++ parity: ``StochasticProcess::drift(Time, const Array&)``.
        """

    @abstractmethod
    def diffusion(self, t: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Diffusion matrix at ``(t, x)``.

        # C++ parity: ``StochasticProcess::diffusion(Time, const Array&)``.
        """

    # --- defaulted -------------------------------------------------------

    def factors(self) -> int:
        """Number of independent Brownian factors.

        # C++ parity: ``StochasticProcess::factors`` — defaults to
        # ``size()``. Override if a process has fewer independent factors
        # than dimensions (e.g. a correlation-driven multi-asset model).
        """
        return self.size()

    def expectation(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        """Expectation of the state at ``t0 + dt``.

        # C++ parity: ``StochasticProcess::expectation`` —
        # ``apply(x0, discretization_->drift(*this, t0, x0, dt))``.
        """
        qassert.require(self._discretization is not None, "no discretization provided")
        assert self._discretization is not None
        return self.apply(x0, self._discretization.drift(self, t0, x0, dt))

    def std_deviation(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        """Standard-deviation matrix of the state at ``t0 + dt``.

        # C++ parity: ``StochasticProcess::stdDeviation`` —
        # ``discretization_->diffusion(*this, t0, x0, dt)``.
        """
        qassert.require(self._discretization is not None, "no discretization provided")
        assert self._discretization is not None
        return self._discretization.diffusion(self, t0, x0, dt)

    def covariance(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        """Covariance matrix of the state at ``t0 + dt``.

        # C++ parity: ``StochasticProcess::covariance``.
        """
        qassert.require(self._discretization is not None, "no discretization provided")
        assert self._discretization is not None
        return self._discretization.covariance(self, t0, x0, dt)

    def evolve(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
        dw: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Asset value at ``t0 + dt`` given a Brownian increment ``dw``.

        # C++ parity: ``StochasticProcess::evolve`` —
        # ``apply(expectation(t0, x0, dt), stdDeviation(t0, x0, dt) * dw)``.
        """
        std = self.std_deviation(t0, x0, dt)
        return self.apply(self.expectation(t0, x0, dt), std @ dw)

    def apply(
        self,
        x0: npt.NDArray[np.float64],
        dx: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Apply a change ``dx`` to the state ``x0``.

        # C++ parity: ``StochasticProcess::apply`` — default ``x0 + dx``.
        """
        return x0 + dx

    # --- utilities -------------------------------------------------------

    def time(self, date: Date) -> float:
        """Year fraction corresponding to ``date`` in the process's clock.

        # C++ parity: base ``StochasticProcess::time`` —
        # ``QL_FAIL("date/time conversion not supported")``. Override
        # in subclasses that hold a daycounter / reference date.
        """
        raise LibraryException("date/time conversion not supported")

    # --- Observer interface ---------------------------------------------

    def update(self) -> None:
        """Observer.update — propagate the notification.

        # C++ parity: ``StochasticProcess::update`` — calls
        # ``notifyObservers()``.
        """
        self.notify_observers()


__all__ = [
    "StochasticProcess",
    "StochasticProcessDiscretization",
]
