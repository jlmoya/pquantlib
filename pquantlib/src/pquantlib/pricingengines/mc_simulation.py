"""McSimulation — base class for Monte Carlo pricing engines.

# C++ parity: ql/pricingengines/mcsimulation.hpp (v1.42.1) —
# ``template <template <class> class MC, class RNG, class S>
#  class McSimulation``.

This base mixes into concrete MC engines (e.g. ``MCEuropeanEngine``)
and supplies the iteration / sample-size / tolerance machinery:

* ``calculate(required_tolerance, required_samples, max_samples)`` —
  invoked by the engine's ``calculate()`` override.  Constructs an
  internal :class:`pquantlib.methods.montecarlo.monte_carlo_model.MonteCarloModel`,
  then drives sampling either to a target absolute-tolerance
  (``value(...)``) or a target sample count (``value_with_samples``).

* ``value(tolerance, max_samples, min_samples)`` — adaptive sample
  growth: keep doubling until the error estimate (1-sigma on the mean)
  is below ``tolerance``.

* ``value_with_samples(n)`` — straight to ``n`` samples; no tolerance
  loop.

The engine subclass MUST implement four hooks:

  - ``path_generator()`` — fresh ``PathGenerator`` (or compatible)
    per ``calculate()``.
  - ``path_pricer()`` — fresh ``PathPricer`` per ``calculate()``.
  - ``time_grid()`` — engine's ``TimeGrid`` (used for steps logic).
  - (optionally) ``control_path_pricer()`` /
    ``control_variate_value()`` for CV.

Python deviates from C++ by using protocols rather than template
parameters; the ``MC`` trait (``SingleVariate`` vs ``MultiVariate``)
collapses into a runtime check on whether the path-pricer accepts
``Path`` or ``MultiPath``.  No compile-time switch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.math.statistics.general_statistics import GeneralStatistics
from pquantlib.methods.montecarlo.monte_carlo_model import (
    MonteCarloModel,
    PathGeneratorTypeProtocol,
)
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.time.time_grid import TimeGrid

_MIN_DEFAULT_SAMPLES = 1023
"""C++ default ``minSamples`` for tolerance-driven sampling
(``McSimulation::value`` first argument default in mcsimulation.hpp:55)."""


class McSimulation[PathT](ABC):
    """Abstract base for MC pricing engines.

    # C++ parity: ``McSimulation<MC, RNG, S>`` (mcsimulation.hpp).
    """

    __slots__ = ("_antithetic_variate", "_control_variate", "_mc_model")

    def __init__(self, antithetic_variate: bool, control_variate: bool) -> None:
        self._antithetic_variate: bool = antithetic_variate
        self._control_variate: bool = control_variate
        self._mc_model: MonteCarloModel[PathT] | None = None

    # --- subclass hooks ---------------------------------------------------

    @abstractmethod
    def path_pricer(self) -> PathPricer[PathT]:
        """Build a fresh path pricer (called once per ``calculate``)."""

    @abstractmethod
    def path_generator(self) -> PathGeneratorTypeProtocol[PathT]:
        """Build a fresh path generator (called once per ``calculate``)."""

    @abstractmethod
    def time_grid(self) -> TimeGrid:
        """Return the engine's TimeGrid (called once per ``calculate``)."""

    def control_path_pricer(self) -> PathPricer[PathT] | None:
        """Optional CV path pricer — None if no CV used."""
        return None

    def control_path_generator(self) -> PathGeneratorTypeProtocol[PathT] | None:
        """Optional CV path generator — None if the same path drives CV."""
        return None

    def control_variate_value(self) -> float | None:
        """Optional CV reference value — None disables CV.

        # C++ parity: ``McSimulation::controlVariateValue`` —
        # ``Null<result_type>()`` by default.
        """
        return None

    # --- driver -----------------------------------------------------------

    def run_mc(
        self,
        required_tolerance: float | None,
        required_samples: int | None,
        max_samples: int | None,
    ) -> None:
        """Run the MC simulation per the requested termination criterion.

        # C++ parity: ``McSimulation::calculate`` (mcsimulation.hpp:158-206).
        # Renamed in the Python port to avoid clashing with
        # ``PricingEngine.calculate`` when an engine multi-inherits
        # ``GenericEngine`` and ``McSimulation``.
        """
        qassert.require(
            required_tolerance is not None or required_samples is not None,
            "neither tolerance nor number of samples set",
        )

        if self._control_variate:
            cv_value = self.control_variate_value()
            qassert.require(cv_value is not None, "engine does not provide control-variation price")
            assert cv_value is not None
            cv_pp = self.control_path_pricer()
            qassert.require(cv_pp is not None, "engine does not provide control-variation path pricer")
            assert cv_pp is not None
            cv_pg = self.control_path_generator()
            self._mc_model = MonteCarloModel[PathT](
                path_generator=self.path_generator(),
                path_pricer=self.path_pricer(),
                sample_accumulator=GeneralStatistics(),
                antithetic_variate=self._antithetic_variate,
                cv_path_pricer=cv_pp,
                cv_option_value=cv_value,
                cv_path_generator=cv_pg,
            )
        else:
            self._mc_model = MonteCarloModel[PathT](
                path_generator=self.path_generator(),
                path_pricer=self.path_pricer(),
                sample_accumulator=GeneralStatistics(),
                antithetic_variate=self._antithetic_variate,
            )

        if required_tolerance is not None:
            if max_samples is not None:
                self.value(required_tolerance, max_samples=max_samples)
            else:
                self.value(required_tolerance)
        else:
            assert required_samples is not None
            self.value_with_samples(required_samples)

    # --- sampling ---------------------------------------------------------

    def value(
        self,
        tolerance: float,
        max_samples: int | None = None,
        min_samples: int = _MIN_DEFAULT_SAMPLES,
    ) -> float:
        """Add samples until error estimate falls under ``tolerance``.

        # C++ parity: ``McSimulation::value`` (mcsimulation.hpp:104-138).

        Algorithm: start with ``min_samples`` (1023 by C++ default);
        compute the 1-sigma error; if above tolerance, grow batches
        until either the estimate drops below ``tolerance`` or
        ``max_samples`` is hit.
        """
        qassert.require(self._mc_model is not None, "MC model not initialized")
        assert self._mc_model is not None
        sample_number = self._mc_model.sample_accumulator().samples()
        if sample_number < min_samples:
            self._mc_model.add_samples(min_samples - sample_number)
            sample_number = self._mc_model.sample_accumulator().samples()

        error = self._mc_model.sample_accumulator().error_estimate()
        while error > tolerance:
            if max_samples is not None and sample_number >= max_samples:
                raise LibraryException(
                    f"max number of samples ({max_samples}) reached, "
                    f"while error ({error}) is still above tolerance ({tolerance})"
                )
            # Conservative estimate of how many samples are needed (C++ formula).
            order = (error * error) / (tolerance * tolerance)
            next_batch = max(
                int(sample_number * order * 0.8 - sample_number),
                min_samples,
            )
            if max_samples is not None:
                next_batch = min(next_batch, max_samples - sample_number)
            sample_number += next_batch
            self._mc_model.add_samples(next_batch)
            error = self._mc_model.sample_accumulator().error_estimate()
        return self._mc_model.sample_accumulator().mean()

    def value_with_samples(self, samples: int) -> float:
        """Run exactly ``samples`` MC iterations.

        # C++ parity: ``McSimulation::valueWithSamples`` (mcsimulation.hpp:142-154).
        """
        qassert.require(self._mc_model is not None, "MC model not initialized")
        assert self._mc_model is not None
        sample_number = self._mc_model.sample_accumulator().samples()
        qassert.require(
            samples >= sample_number,
            f"number of already simulated samples ({sample_number}) "
            f"greater than requested samples ({samples})",
        )
        self._mc_model.add_samples(samples - sample_number)
        return self._mc_model.sample_accumulator().mean()

    def error_estimate(self) -> float:
        """Standard error (sigma / sqrt(N)) of the current mean estimate.

        # C++ parity: ``McSimulation::errorEstimate``.
        """
        qassert.require(self._mc_model is not None, "MC model not initialized")
        assert self._mc_model is not None
        return self._mc_model.sample_accumulator().error_estimate()

    def sample_accumulator(self) -> GeneralStatistics:
        """The underlying ``GeneralStatistics`` instance.

        # C++ parity: ``McSimulation::sampleAccumulator``.
        """
        qassert.require(self._mc_model is not None, "MC model not initialized")
        assert self._mc_model is not None
        return self._mc_model.sample_accumulator()


__all__ = ["McSimulation"]
