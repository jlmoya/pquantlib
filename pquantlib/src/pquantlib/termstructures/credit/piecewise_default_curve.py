"""PiecewiseDefaultCurve — bootstrap default-probability curve.

# C++ parity: ql/termstructures/credit/piecewisedefaultcurve.hpp (v1.42.1).

C++ ``PiecewiseDefaultCurve<Traits, Interpolator, Bootstrap>`` is the
core piecewise bootstrap default-probability curve. It chains
``InterpolatedXxxCurve<Interpolator>`` (Xxx in {SurvivalProbability,
HazardRate, DefaultDensity}) with a per-trait bootstrap iterator that
solves for one pillar at a time given a list of ``BootstrapHelper``
instances.

L9-B wires the scaffold landed in L8-B with a real
``IterativeBootstrap[DefaultProbabilityTermStructure, Traits]`` from
L8-A. The traits class chooses which interpolated underlying curve
type is built (SurvivalProbability → InterpolatedSurvivalProbability;
HazardRate → InterpolatedHazardRate; DefaultDensity →
InterpolatedDefaultDensity), and the bootstrap solves for the
state variable directly (S(t) / h(t) / p(t)) at each pillar.

# C++ parity divergence: the C++ class supports arbitrary
   ``Bootstrap`` template parameters (``IterativeBootstrap`` vs
   ``LocalBootstrap``). The Python port hard-codes
   ``IterativeBootstrap``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.array import Array
from pquantlib.math.interpolations.backward_flat import BackwardFlatInterpolation
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.math.interpolations.log_linear import LogLinearInterpolation
from pquantlib.termstructures.bootstrap.iterative_bootstrap import IterativeBootstrap
from pquantlib.termstructures.credit.default_probability_helpers import CdsHelper
from pquantlib.termstructures.credit.default_probability_term_structure import (
    DefaultProbabilityTermStructure,
)
from pquantlib.termstructures.credit.interpolated_default_density_curve import (
    InterpolatedDefaultDensityCurve,
)
from pquantlib.termstructures.credit.interpolated_hazard_rate_curve import (
    InterpolatedHazardRateCurve,
)
from pquantlib.termstructures.credit.interpolated_survival_probability_curve import (
    InterpolatedSurvivalProbabilityCurve,
)
from pquantlib.termstructures.credit.probability_traits import (
    DefaultDensityTrait,
    HazardRateTrait,
    SurvivalProbabilityTrait,
)
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date

InterpolationFactory = Callable[[Array, Array], Interpolation]


def _instantiate(traits: Any) -> Any:
    """Accept either a class or an instance for ``traits=``."""
    if isinstance(traits, type):
        return traits()
    return traits


class PiecewiseDefaultCurve(DefaultProbabilityTermStructure):
    """Piecewise-bootstrapped default-probability term structure.

    Inputs:
    - ``traits``: ``SurvivalProbabilityTrait`` / ``HazardRateTrait`` /
      ``DefaultDensityTrait`` (class or instance).
    - ``reference_date``: curve reference date.
    - ``instruments``: list of :class:`CdsHelper` instances (spread or
      upfront CDS helpers).
    - ``day_counter``: day counter for date → time conversion.
    - ``calendar``: optional, defaults to None.
    - ``interpolator``: optional override; default depends on the
      traits (LogLinear for SurvivalProbability, BackwardFlat for
      HazardRate, Linear for DefaultDensity).
    - ``accuracy``: Brent inner tolerance. Default ``1e-12``.

    The bootstrap is lazy: it triggers on the first
    ``survival_probability`` / ``hazard_rate`` / ``default_density`` /
    ``default_probability`` evaluation.
    """

    def __init__(
        self,
        traits: Any,
        reference_date: Date,
        instruments: Sequence[CdsHelper],
        day_counter: DayCounter,
        calendar: Calendar | None = None,
        interpolator: InterpolationFactory | None = None,
        accuracy: float = 1.0e-12,
    ) -> None:
        super().__init__(
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
        )
        qassert.require(
            len(instruments) >= 1,
            "PiecewiseDefaultCurve: at least one instrument is required",
        )
        self._traits: Any = _instantiate(traits)
        self._traits_class: type = traits if isinstance(traits, type) else type(traits)
        self._instruments: list[CdsHelper] = list(instruments)
        # TermStructure already stores day_counter; keep a typed copy
        # here for ``bootstrap_install_grid``.
        self._dc_stored: DayCounter = day_counter
        self._calendar_in: Calendar | None = calendar
        self._accuracy: float = accuracy
        if interpolator is None:
            interpolator = self._default_interpolator()
        self._interpolator: InterpolationFactory = interpolator
        self._underlying: (
            InterpolatedSurvivalProbabilityCurve
            | InterpolatedHazardRateCurve
            | InterpolatedDefaultDensityCurve
            | None
        ) = None
        self._bootstrap_done: bool = False

    # ---- inspectors -------------------------------------------------------

    def traits(self) -> type:
        return self._traits_class

    def instruments(self) -> list[CdsHelper]:
        return list(self._instruments)

    def max_date(self) -> Date:
        u: Any = self._underlying
        if u is not None and u._dates:
            return u._dates[-1]
        return max(h.latest_date() for h in self._instruments)

    def dates(self) -> list[Date]:
        self._ensure_bootstrap()
        u: Any = self._underlying
        assert u is not None
        return list(u._dates)

    def data(self) -> list[float]:
        self._ensure_bootstrap()
        u: Any = self._underlying
        assert u is not None
        return list(u._data)

    def nodes(self) -> list[tuple[Date, float]]:
        self._ensure_bootstrap()
        return list(zip(self.dates(), self.data(), strict=True))

    # ---- traits → underlying class plumbing -------------------------------

    def _underlying_class(self) -> type:
        if self._traits_class is SurvivalProbabilityTrait:
            return InterpolatedSurvivalProbabilityCurve
        if self._traits_class is HazardRateTrait:
            return InterpolatedHazardRateCurve
        if self._traits_class is DefaultDensityTrait:
            return InterpolatedDefaultDensityCurve
        qassert.fail(
            f"PiecewiseDefaultCurve: unsupported traits {self._traits_class}",
        )
        return InterpolatedSurvivalProbabilityCurve  # unreachable

    def _underlying_data_kwarg(self) -> str:
        if self._traits_class is SurvivalProbabilityTrait:
            return "probabilities"
        if self._traits_class is HazardRateTrait:
            return "hazard_rates"
        if self._traits_class is DefaultDensityTrait:
            return "densities"
        qassert.fail(
            f"PiecewiseDefaultCurve: unsupported traits {self._traits_class}",
        )
        return "data"

    def _seed_data(self, n: int) -> list[float]:
        """Seed values that pass the underlying curve's constructor checks."""
        if self._traits_class is SurvivalProbabilityTrait:
            # Probabilities must be monotonically non-increasing in (0, 1].
            return [1.0] + [1.0 / (1.0 + 0.01 * i) for i in range(1, n)]
        # Hazard / density: non-negative.
        return [0.01] * n

    def _default_interpolator(self) -> InterpolationFactory:
        if self._traits_class is SurvivalProbabilityTrait:
            return LogLinearInterpolation
        if self._traits_class is HazardRateTrait:
            return BackwardFlatInterpolation
        return LinearInterpolation  # DefaultDensity

    # ---- BootstrapCurveProtocol surface ----------------------------------

    def base_date(self) -> Date:
        return self.reference_date()

    def times(self) -> list[float]:
        if self._underlying is None:
            return []
        return list(self._underlying.times())

    def data_live(self) -> list[float]:
        u: Any = self._underlying
        assert u is not None
        return u._data

    def set_data_at(self, i: int, level: float) -> None:
        u: Any = self._underlying
        assert u is not None
        u._data[i] = level

    def refresh_interpolation_through(self, up_to: int) -> None:
        u: Any = self._underlying
        assert u is not None
        partial_t = u._times[: up_to + 1]
        partial_d = u._data[: up_to + 1]
        u._interpolation = self._interpolator(
            np.asarray(partial_t, dtype=np.float64),
            np.asarray(partial_d, dtype=np.float64),
        )
        # Notify observers so downstream lazy instruments
        # (CDS, swaps wired into helpers) invalidate their NPV cache.
        # # C++ parity: InterpolatedCurve::setupInterpolation triggers
        # the curve's Observable::notifyObservers via its parent
        # TermStructure.
        self.notify_observers()

    def bootstrap_install_grid(
        self, dates: list[Date], times: list[float], data: list[float],
    ) -> None:
        underlying_cls = self._underlying_class()
        seed_data = self._seed_data(len(dates))
        kwargs: dict[str, Any] = {
            "dates": list(dates),
            self._underlying_data_kwarg(): seed_data,
            "day_counter": self._dc_stored,
            "calendar": self._calendar_in,
            "interpolator": self._interpolator,
        }
        self._underlying = underlying_cls(**kwargs)
        u: Any = self._underlying
        u._dates = list(dates)
        u._times = list(times)
        u._data = list(data)
        u._interpolation = self._interpolator(
            np.asarray(times, dtype=np.float64),
            np.asarray(data, dtype=np.float64),
        )

    # ---- bootstrap trigger -----------------------------------------------

    def _ensure_bootstrap(self) -> None:
        # If the underlying is already allocated (we may be re-entrant
        # mid-bootstrap from a helper engine call), bail out — the
        # IterativeBootstrap is using the live interpolation as it
        # solves, so a partial-progress query is meaningful.
        if self._bootstrap_done or self._underlying is not None:
            return
        bootstrapper = IterativeBootstrap(
            curve=self, instruments=self._instruments,
            traits=self._traits, accuracy=self._accuracy,
        )
        bootstrapper.calculate()
        self._bootstrap_done = True

    # ---- DefaultProbabilityTermStructure hooks ---------------------------

    def _survival_probability_impl(self, t: float) -> float:
        self._ensure_bootstrap()
        u: Any = self._underlying
        assert u is not None
        return u._survival_probability_impl(t)

    def _default_density_impl(self, t: float) -> float:
        self._ensure_bootstrap()
        u: Any = self._underlying
        assert u is not None
        return u._default_density_impl(t)

    def _hazard_rate_impl(self, t: float) -> float:
        self._ensure_bootstrap()
        u: Any = self._underlying
        assert u is not None
        return u._hazard_rate_impl(t)


__all__ = ["PiecewiseDefaultCurve"]
