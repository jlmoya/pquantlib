"""MCDiscreteAveragingAsianEngineBase — MC engine base for discrete-average Asians.

# C++ parity: ql/pricingengines/asian/mcdiscreteasianenginebase.hpp (v1.43) —
# ``template <template <class> class MC, class RNG = PseudoRandom,
#            class S = Statistics>
#  class MCDiscreteAveragingAsianEngineBase : public
#      DiscreteAveragingAsianOption::engine,
#      public McSimulation<MC, RNG, S>``, plus ``detail::PastFixingsOnly``.

Equivalent role to :class:`~pquantlib.pricingengines.vanilla.mc_vanilla_engine.MCVanillaEngine`
but for Asians: the time grid comes from the option's *fixing dates* rather
than from a step count, and the fixing-date list is what makes the grid
mandatory times meaningful to the path pricers.

Template parameters, and where they went
----------------------------------------

``MC`` (``SingleVariate`` / ``MultiVariate``, ql/methods/montecarlo/mctraits.hpp)
    becomes the class type parameter ``PathT`` plus the ``multi_variate``
    constructor flag, exactly as in ``MCVanillaEngine[PathT]``. The
    Black-Scholes engines are ``SingleVariate`` and drive a
    :class:`~pquantlib.methods.montecarlo.path_generator.PathGenerator` over
    :class:`~pquantlib.methods.montecarlo.path.Path`; the two Heston engines
    are ``MultiVariate`` and drive a
    :class:`~pquantlib.methods.montecarlo.multi_path_generator.MultiPathGenerator`
    over :class:`~pquantlib.methods.montecarlo.multi_path.MultiPath`.

``RNG`` (``PseudoRandom`` / ``LowDiscrepancy``, ql/math/randomnumbers/rngtraits.hpp)
    becomes the ``rng_traits`` constructor argument. It is a real parameter,
    not decoration: ``allows_error_estimate`` gates whether
    ``results.error_estimate`` is filled at all (C++
    ``if constexpr (RNG::allowsErrorEstimate)``), and
    ``make_sequence_generator`` decides whether the path is driven by a
    Mersenne Twister or by a Sobol sequence.

``S`` (statistics accumulator)
    is always ``GeneralStatistics``.

Seeds
-----
``seed`` is passed straight to the traits' ``make_sequence_generator``. Seed 0
therefore reaches ``SeedGenerator`` through the Mersenne Twister and is
clock-derived, exactly as in C++ — this port does *not* silently substitute a
different seed.
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.instruments.asian_option import DiscreteAveragingAsianOptionArguments
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.methods.montecarlo.monte_carlo_model import (
    PathGeneratorTypeProtocol,
)
from pquantlib.methods.montecarlo.multi_path_generator import MultiPathGenerator
from pquantlib.methods.montecarlo.path_generator import PathGenerator
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.mc_simulation import McSimulation
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.pricingengines.vanilla.mc_vanilla_engine import RngTraits
from pquantlib.processes.stochastic_process import StochasticProcess
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.time.time_grid import TimeGrid


class PastFixingsOnly(LibraryException):
    """Raised when no fixing of a discrete-average Asian is still in the future.

    # C++ parity: ``detail::PastFixingsOnly``
    # (mcdiscreteasianenginebase.hpp:39-44) —
    # ``Error("n/a", 0, "n/a", "all fixings are in the past")``.

    This is a *distinct type*, not just a message: C++
    ``MCDiscreteAveragingAsianEngineBase::calculate`` catches it explicitly
    (mcdiscreteasianenginebase.hpp:83-89) so a future revision can compute the
    fully-determined payoff instead of failing, and re-throws it for now.
    """

    def __init__(self) -> None:
        super().__init__("all fixings are in the past")


class MCDiscreteAveragingAsianEngineBase[PathT](
    GenericEngine[DiscreteAveragingAsianOptionArguments, OneAssetOptionResults],
    McSimulation[PathT],
):
    """Abstract MC engine base for discrete-average Asian options.

    # C++ parity: ``MCDiscreteAveragingAsianEngineBase<MC,RNG,S>``
    # (mcdiscreteasianenginebase.hpp:53-128, 133-206).
    """

    def __init__(
        self,
        process: StochasticProcess,
        *,
        brownian_bridge: bool = False,
        antithetic_variate: bool = False,
        control_variate: bool = False,
        required_samples: int | None = None,
        required_tolerance: float | None = None,
        max_samples: int | None = None,
        seed: int = 0,
        time_steps: int | None = None,
        time_steps_per_year: int | None = None,
        include_exercise_date: bool = False,
        rng_traits: RngTraits = PseudoRandom,
        multi_variate: bool = False,
    ) -> None:
        # NOTE: pyright cannot track explicit base-class __init__ forwarding
        # through PEP-695 generic bases; the ignores keep the checker quiet
        # without changing runtime behaviour.
        GenericEngine.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            DiscreteAveragingAsianOptionArguments(),
            OneAssetOptionResults(),
        )
        McSimulation.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            antithetic_variate=antithetic_variate,
            control_variate=control_variate,
        )
        # C++ parity: mcdiscreteasianenginebase.hpp:133-151.
        self._process: StochasticProcess = process
        self._brownian_bridge: bool = brownian_bridge
        self._required_samples: int | None = required_samples
        self._required_tolerance: float | None = required_tolerance
        self._max_samples: int | None = max_samples
        self._seed: int = seed
        self._time_steps: int | None = time_steps
        self._time_steps_per_year: int | None = time_steps_per_year
        self._include_exercise_date: bool = include_exercise_date
        self._rng_traits: RngTraits = rng_traits
        self._multi_variate: bool = multi_variate
        process.register_with(self)

    # --- engine entry-point ----------------------------------------------

    def calculate(self) -> None:
        """Run the MC and fill ``self._results``.

        # C++ parity: ``MCDiscreteAveragingAsianEngineBase::calculate``
        # (mcdiscreteasianenginebase.hpp:78-105).
        """
        try:
            self.run_mc(
                required_tolerance=self._required_tolerance,
                required_samples=self._required_samples,
                max_samples=self._max_samples,
            )
        except PastFixingsOnly:
            # C++ parity: the catch block exists so a future revision can write
            # the (fully determined) payoff into the results; for now it
            # re-throws unchanged.
            raise
        assert self._mc_model is not None
        accumulator = self._mc_model.sample_accumulator()
        self._results.value = accumulator.mean()

        if self._control_variate:
            # C++ parity: mcdiscreteasianenginebase.hpp:93-97 — the control
            # variate can push deep-OTM estimates slightly negative.
            assert self._results.value is not None
            self._results.value = max(0.0, self._results.value)

        # C++ parity: ``if constexpr (RNG::allowsErrorEstimate)``. A
        # low-discrepancy point set is not i.i.d., so C++ leaves
        # ``results_.errorEstimate`` at ``Null<Real>()`` and
        # ``Instrument::errorEstimate()`` throws. Reproduced, not smoothed over.
        if self._rng_traits.allows_error_estimate:
            self._results.error_estimate = accumulator.error_estimate()

        # C++ parity: mcdiscreteasianenginebase.hpp:104 — "Allow inspection of
        # the timeGrid via additional results".
        self._results.additional_results["TimeGrid"] = self.time_grid()

    # --- McSimulation hooks ----------------------------------------------

    def time_grid(self) -> TimeGrid:
        """Build a time grid from the option's fixing dates.

        # C++ parity: ``MCDiscreteAveragingAsianEngineBase::timeGrid``
        # (mcdiscreteasianenginebase.hpp:153-185).

        Fixing times come from ``process.time(date)`` — i.e. the *risk-free*
        curve's day counter and reference date — and only non-negative ones
        survive.
        """
        args = self._arguments
        process = self._process
        fixing_times: list[float] = []
        for fd in args.fixing_dates:
            t = process.time(fd)
            if t >= 0:
                fixing_times.append(t)

        if not fixing_times or (len(fixing_times) == 1 and fixing_times[0] == 0.0):
            raise PastFixingsOnly

        # Some models (eg. Heston) might request additional points in the time
        # grid to improve the accuracy of the discretization.
        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        t = process.time(args.exercise.last_date())

        if self._include_exercise_date and t > fixing_times[-1]:
            fixing_times.append(t)

        if self._time_steps is not None:
            return TimeGrid.with_mandatory_and_steps(fixing_times, self._time_steps)
        if self._time_steps_per_year is not None:
            # C++ ``static_cast<Size>(this->timeStepsPerYear_*t)`` truncates,
            # and uses the *exercise* time even when it was not appended.
            return TimeGrid.with_mandatory_and_steps(
                fixing_times, int(self._time_steps_per_year * t)
            )

        return TimeGrid.with_mandatory(fixing_times)

    def path_generator(self) -> PathGeneratorTypeProtocol[PathT]:
        """Build a fresh path generator per ``calculate()``.

        # C++ parity: ``MCDiscreteAveragingAsianEngineBase::pathGenerator``
        # (mcdiscreteasianenginebase.hpp:110-119) — the dimension is
        # ``process->factors() * (grid.size()-1)`` and the generator comes from
        # ``RNG::make_sequence_generator(dimensions, seed_)``.
        """
        grid = self.time_grid()
        dimensions = self._process.factors() * (len(grid) - 1)
        generator = self._rng_traits.make_sequence_generator(dimensions, self._seed)
        if self._multi_variate:
            # C++ ``MultiVariate<RNG>::path_generator_type``.
            return MultiPathGenerator(  # type: ignore[return-value]
                self._process, grid, generator, self._brownian_bridge
            )
        # C++ ``SingleVariate<RNG>::path_generator_type``.
        qassert.require(
            isinstance(self._process, StochasticProcess1D),
            "1-D process required for a single-variate MC engine",
        )
        assert isinstance(self._process, StochasticProcess1D)
        return PathGenerator.with_time_grid(  # type: ignore[return-value]
            self._process, grid, generator, brownian_bridge=self._brownian_bridge
        )

    # --- abstract subclass hooks -----------------------------------------

    @abstractmethod
    def path_pricer(self) -> PathPricer[PathT]:
        """Subclass: build the arithmetic / geometric average pricer."""

    def control_pricing_engine(self) -> PricingEngine | None:
        """Engine supplying the control-variate reference value.

        # C++ parity: ``McSimulation::controlPricingEngine`` — returns a null
        # ``shared_ptr`` by default (mcsimulation.hpp).
        """
        return None

    def control_variate_value(self) -> float | None:
        """Run the control pricing engine on our arguments and return its NPV.

        # C++ parity: ``MCDiscreteAveragingAsianEngineBase::controlVariateValue``
        # (mcdiscreteasianenginebase.hpp:187-206) — ``*controlArguments =
        # arguments_; controlPE->calculate();`` then
        # ``dynamic_cast<const DiscreteAveragingAsianOption::results*>(...)->value``.
        """
        control_pe = self.control_pricing_engine()
        qassert.require(
            control_pe is not None,
            "engine does not provide control variation pricing engine",
        )
        assert control_pe is not None
        control_arguments = control_pe.get_arguments()
        qassert.require(
            isinstance(control_arguments, DiscreteAveragingAsianOptionArguments),
            "engine is using inconsistent arguments",
        )
        assert isinstance(control_arguments, DiscreteAveragingAsianOptionArguments)
        # ``*controlArguments = arguments_`` — a whole-struct copy, so every
        # field crosses, seasoning included.
        control_arguments.payoff = self._arguments.payoff
        control_arguments.exercise = self._arguments.exercise
        control_arguments.average_type = self._arguments.average_type
        control_arguments.running_accumulator = self._arguments.running_accumulator
        control_arguments.past_fixings = self._arguments.past_fixings
        control_arguments.fixing_dates = list(self._arguments.fixing_dates)
        control_pe.calculate()
        control_results = control_pe.get_results()
        qassert.require(
            isinstance(control_results, OneAssetOptionResults),
            "engine returns an inconsistent result type",
        )
        assert isinstance(control_results, OneAssetOptionResults)
        value = control_results.value
        if value is None:
            raise LibraryException("control engine did not produce a value")
        return float(value)


__all__ = ["MCDiscreteAveragingAsianEngineBase", "PastFixingsOnly"]
