"""Gaussian1dGsrProcess — Gaussian1d-flavoured adapter wrapping ``GsrProcess``.

# C++ parity: this is a thin wrapper over ``GsrProcess`` to keep the
# ``Gaussian1d``-naming convention used by Phase 11 callers (especially
# ``MarkovFunctional``). The C++ code base reuses ``GsrProcess`` directly
# wherever a ``Gaussian1d``-compatible state process is needed, so this
# adapter is **PQuantLib-only**: it stands in as the
# ``Gaussian1dModel.state_process()`` factory entry point with a
# ``(term_structure, volstepdates, volatilities, reversion, T)`` ctor
# signature mirroring the C++ ``MarkovFunctional``/``Gsr`` shared
# constructor convention.

The adapter:

- Converts the date-list ``volstepdates`` into a float time-array via
  ``term_structure.time_from_reference``.
- Builds a ``GsrProcess`` with the (times, vols, reversions) tuple.
- Forwards every ``StochasticProcess1D`` method to the wrapped
  ``GsrProcess`` (delegation, not inheritance — keeps the SDE
  semantics identical without exposing the cache surface).
- Supports both constant-reversion (``reversion: float``) and
  piecewise-reversion (``reversion: list[float]`` aligned to
  ``volstepdates``) inputs.

Construction example::

    p = Gaussian1dGsrProcess(
        term_structure=yts,
        volstepdates=[d1y, d2y, d5y],
        volatilities=[0.01, 0.01, 0.01, 0.01],
        reversion=0.05,
    )

The wrapped ``GsrProcess`` is exposed via ``inner()`` for callers that
need access to the GSR-specific inspectors (``y(t)``, ``G(t, w)``,
``set_times``, ``flush_cache``, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pquantlib.processes.gsr_process import GsrProcess
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D

if TYPE_CHECKING:
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure
    from pquantlib.time.date import Date


class Gaussian1dGsrProcess(StochasticProcess1D):
    """Gaussian1d-style adapter over ``GsrProcess``.

    Constructor parameters mirror the ``Gsr`` / ``MarkovFunctional`` ctor
    convention so that the same builder code can produce either model's
    state process. Internally the adapter holds a single ``GsrProcess``
    instance and delegates every ``StochasticProcess1D`` method to it.
    """

    __slots__ = ("_inner", "_term_structure")

    def __init__(
        self,
        term_structure: YieldTermStructure,
        volstepdates: list[Date],
        volatilities: list[float],
        reversion: float | list[float],
        T: float = 60.0,  # noqa: N803 — math symbol
    ) -> None:
        super().__init__()
        self._term_structure: YieldTermStructure = term_structure
        # Convert volstepdates -> times.
        times = np.asarray(
            [term_structure.time_from_reference(d) for d in volstepdates],
            dtype=np.float64,
        )
        vols = np.asarray(volatilities, dtype=np.float64)
        revs = (
            np.asarray([reversion], dtype=np.float64)
            if isinstance(reversion, (int, float))
            else np.asarray(reversion, dtype=np.float64)
        )
        self._inner: GsrProcess = GsrProcess(
            times=times,
            vols=vols,
            reversions=revs,
            T=T,
            reference_date=term_structure.reference_date(),
            day_counter=term_structure.day_counter(),
        )

    # --- inner accessor -------------------------------------------------

    def inner(self) -> GsrProcess:
        """Wrapped ``GsrProcess`` instance.

        Exposes the GSR-specific inspectors (``y(t)``, ``G(t, w)``, etc.)
        and the cache mutators that are not part of the abstract
        ``StochasticProcess1D`` surface.
        """
        return self._inner

    # --- StochasticProcess1D delegating surface -------------------------

    def x0(self) -> float:
        return self._inner.x0()

    def drift_1d(self, t: float, x: float) -> float:
        return self._inner.drift_1d(t, x)

    def diffusion_1d(self, t: float, x: float) -> float:
        return self._inner.diffusion_1d(t, x)

    def expectation_1d(self, t0: float, x0: float, dt: float) -> float:
        return self._inner.expectation_1d(t0, x0, dt)

    def std_deviation_1d(self, t0: float, x0: float, dt: float) -> float:
        return self._inner.std_deviation_1d(t0, x0, dt)

    def variance_1d(self, t0: float, x0: float, dt: float) -> float:
        return self._inner.variance_1d(t0, x0, dt)

    def evolve_1d(self, t0: float, x0: float, dt: float, dw: float) -> float:
        return self._inner.evolve_1d(t0, x0, dt, dw)

    def apply_1d(self, x0: float, dx: float) -> float:
        return self._inner.apply_1d(x0, dx)

    # --- GSR-specific forwarders ----------------------------------------

    def sigma(self, t: float) -> float:
        """Piecewise vol at time ``t``. # C++ parity: GsrProcess::sigma."""
        return self._inner.sigma(t)

    def reversion(self, t: float) -> float:
        """Piecewise reversion at time ``t``. # C++ parity: GsrProcess::reversion."""
        return self._inner.reversion(t)

    def y(self, t: float) -> float:
        """``y(t)`` accumulated forward-state covariance. # C++ parity: GsrProcess::y."""
        return self._inner.y(t)

    def G(self, t: float, w: float, x: float = 0.0) -> float:  # noqa: N802 — math symbol
        """``G(t, w)`` Hull-White integral. # C++ parity: GsrProcess::G."""
        return self._inner.G(t, w, x)

    def get_forward_measure_time(self) -> float:
        """Forward-measure horizon ``T``. # C++ parity: GsrProcess::getForwardMeasureTime."""
        return self._inner.get_forward_measure_time()

    def set_forward_measure_time(self, T: float) -> None:  # noqa: N803 — math symbol
        """Update forward-measure horizon (flushes cache). # C++ parity: GsrProcess::setForwardMeasureTime."""
        self._inner.set_forward_measure_time(T)

    def set_times(self, times: npt.NDArray[np.float64]) -> None:
        """Update step times. # C++ parity: GsrProcess::setTimes (private)."""
        self._inner.set_times(times)

    def set_vols(self, vols: npt.NDArray[np.float64]) -> None:
        """Update piecewise vols. # C++ parity: GsrProcess::setVols (private)."""
        self._inner.set_vols(vols)

    def set_reversions(self, reversions: npt.NDArray[np.float64]) -> None:
        """Update piecewise reversions. # C++ parity: GsrProcess::setReversions (private)."""
        self._inner.set_reversions(reversions)

    def flush_cache(self) -> None:
        """Flush all memoization caches. # C++ parity: GsrProcess::flushCache."""
        self._inner.flush_cache()


__all__ = ["Gaussian1dGsrProcess"]
