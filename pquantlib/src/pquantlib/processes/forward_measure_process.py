"""ForwardMeasureProcess — abstract base for processes under the forward measure.

# C++ parity: ql/processes/forwardmeasureprocess.{hpp,cpp} (v1.42.1).

Two abstract bases:

* ``ForwardMeasureProcess`` — multi-D StochasticProcess subclass that
  stores a forward-measure horizon ``T_``. Concrete subclasses
  (``G2ForwardProcess``, etc.) override ``drift`` / ``diffusion`` /
  ``expectation`` to inject the forward-measure drift correction.
* ``ForwardMeasureProcess1D`` — 1-D specialization with the same
  ``setForwardMeasureTime``/``getForwardMeasureTime`` mutator pair.

Both observe ``set_forward_measure_time`` as a notification trigger
(matches C++ ``setForwardMeasureTime`` which calls ``notifyObservers``).

Default ``T_`` is ``inf`` (matches C++ which leaves it uninitialised
unless an explicit ctor argument is given). Concrete clients SHOULD
call ``set_forward_measure_time`` before pricing.

# C++ parity: ``T_`` is a protected member (Real) in C++; in Python
# this is exposed via property accessors only — internal storage is
# ``_T``.
"""

from __future__ import annotations

import math

from pquantlib.processes.stochastic_process import (
    StochasticProcess,
    StochasticProcessDiscretization,
)
from pquantlib.processes.stochastic_process_1d import (
    StochasticProcess1D,
    StochasticProcess1DDiscretization,
)


class ForwardMeasureProcess(StochasticProcess):
    """Forward-measure multi-D stochastic process base.

    # C++ parity: ``class ForwardMeasureProcess`` in
    # ql/processes/forwardmeasureprocess.hpp:37-47 (v1.42.1).
    """

    def __init__(
        self,
        T: float = math.inf,  # noqa: N803 — math symbol
        discretization: StochasticProcessDiscretization | None = None,
    ) -> None:
        # C++ parity: forwardmeasureprocess.hpp:42-45 has two ctors:
        #   (a) ForwardMeasureProcess(Time T) -> stores T_ directly;
        #   (b) ForwardMeasureProcess(discretization) -> delegates to base.
        # Python merges both into a single signature.
        super().__init__(discretization=discretization)
        self._T: float = float(T)

    def set_forward_measure_time(self, T: float) -> None:  # noqa: N803 — math symbol
        """Update the forward-measure horizon and notify observers.

        # C++ parity: ``ForwardMeasureProcess::setForwardMeasureTime`` in
        # forwardmeasureprocess.cpp:30-33.
        """
        self._T = float(T)  # pyright: ignore[reportConstantRedefinition]
        self.notify_observers()

    def get_forward_measure_time(self) -> float:
        """Return the current forward-measure horizon.

        # C++ parity: ``ForwardMeasureProcess::getForwardMeasureTime`` in
        # forwardmeasureprocess.cpp:35-37.
        """
        return self._T


class ForwardMeasureProcess1D(StochasticProcess1D):
    """Forward-measure 1-D stochastic process base.

    # C++ parity: ``class ForwardMeasureProcess1D`` in
    # ql/processes/forwardmeasureprocess.hpp:55-65 (v1.42.1).
    """

    def __init__(
        self,
        T: float = math.inf,  # noqa: N803 — math symbol
        discretization: StochasticProcess1DDiscretization | None = None,
    ) -> None:
        super().__init__(discretization=discretization)
        self._T: float = float(T)

    def set_forward_measure_time(self, T: float) -> None:  # noqa: N803 — math symbol
        """Update the forward-measure horizon and notify observers.

        # C++ parity: ``ForwardMeasureProcess1D::setForwardMeasureTime`` in
        # forwardmeasureprocess.cpp:45-48.
        """
        self._T = float(T)  # pyright: ignore[reportConstantRedefinition]
        self.notify_observers()

    def get_forward_measure_time(self) -> float:
        """Return the current forward-measure horizon.

        # C++ parity: ``ForwardMeasureProcess1D::getForwardMeasureTime`` in
        # forwardmeasureprocess.cpp:50-52.
        """
        return self._T


__all__ = ["ForwardMeasureProcess", "ForwardMeasureProcess1D"]
