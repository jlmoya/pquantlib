"""MCVanillaEngine — abstract MC pricing engine for vanilla options.

# C++ parity: ql/pricingengines/vanilla/mcvanillaengine.hpp (v1.43) —
# ``template <template <class> class MC, class RNG, class S, class Inst>
#  class MCVanillaEngine``.

C++ folds the inheritance chain ``MCVanillaEngine : Inst::engine,
McSimulation<MC, RNG, S>`` so a single class supplies both the
pricing-engine interface (``calculate()`` filling ``results_.value``
+ optional ``errorEstimate``) and the MC orchestrator hooks. The Python
port keeps the same role-split but uses multiple inheritance:
:class:`~pquantlib.pricingengines.generic_engine.GenericEngine` for the
arguments/results pair and
:class:`~pquantlib.pricingengines.mc_simulation.McSimulation` for the MC
machinery.

Template parameters, and where they went
----------------------------------------

``MC`` (``SingleVariate`` / ``MultiVariate``, ql/methods/montecarlo/mctraits.hpp)
    becomes the class type parameter ``PathT`` plus the ``multi_variate``
    constructor flag. ``SingleVariate`` drives a
    :class:`~pquantlib.methods.montecarlo.path_generator.PathGenerator` over
    :class:`~pquantlib.methods.montecarlo.path.Path`; ``MultiVariate`` drives a
    :class:`~pquantlib.methods.montecarlo.multi_path_generator.MultiPathGenerator`
    over :class:`~pquantlib.methods.montecarlo.multi_path.MultiPath`. Concrete
    engines pin it: ``MCEuropeanEngine(MCVanillaEngine[Path])``,
    ``MCEuropeanHestonEngine(MCVanillaEngine[MultiPath])``.

``RNG`` (``PseudoRandom`` / ``LowDiscrepancy``, ql/math/randomnumbers/rngtraits.hpp)
    becomes the ``rng_traits`` constructor argument, defaulting to
    :class:`~pquantlib.math.randomnumbers.rng_traits.PseudoRandom` exactly as
    the C++ template default does. It is a real parameter, not decoration:
    ``allows_error_estimate`` gates whether ``results.error_estimate`` is
    filled at all (C++ ``if constexpr (RNG::allowsErrorEstimate)``), and
    ``make_sequence_generator`` decides whether the path is driven by a
    Mersenne Twister or by a Sobol sequence.

``S`` (statistics accumulator)
    is always ``GeneralStatistics``; no engine in v1.43 instantiates
    ``MCVanillaEngine`` with anything else that changes ``mean()`` /
    ``errorEstimate()``.

``Inst``
    is always ``VanillaOption`` for the engines in this package, so the
    arguments/results pair is fixed at ``OptionArguments`` /
    ``OneAssetOptionResults``.

Seeds
-----
``seed`` is passed straight to the traits' ``make_sequence_generator``.
Seed 0 therefore reaches ``SeedGenerator`` through the Mersenne Twister and
is clock-derived, exactly as in C++ — this port does *not* silently
substitute a different seed. Deterministic results require an explicit
nonzero seed (a Sobol generator is deterministic for seed 0 too, since
``SobolRsg`` treats 0 as "no scrambling").
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.randomnumbers.rng_traits import LowDiscrepancy, PseudoRandom
from pquantlib.methods.montecarlo.monte_carlo_model import (
    PathGeneratorTypeProtocol,
)
from pquantlib.methods.montecarlo.multi_path_generator import MultiPathGenerator
from pquantlib.methods.montecarlo.path_generator import PathGenerator
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.option import OptionArguments
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.mc_simulation import McSimulation
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.processes.stochastic_process import StochasticProcess
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.time.time_grid import TimeGrid

#: The two RNG policies C++ instantiates these engines with.
#:
#: # C++ parity: ``typedef GenericPseudoRandom<MersenneTwisterUniformRng,
#: # InverseCumulativeNormal> PseudoRandom`` and
#: # ``typedef GenericLowDiscrepancy<SobolRsg, InverseCumulativeNormal>
#: # LowDiscrepancy`` (rngtraits.hpp:70-71, 103-104).
type RngTraits = type[PseudoRandom] | type[LowDiscrepancy]


class MCVanillaEngine[PathT](
    GenericEngine[OptionArguments, OneAssetOptionResults],
    McSimulation[PathT],
):
    """Abstract MC engine for vanilla options.

    # C++ parity: ``MCVanillaEngine<MC, RNG, S, Inst>``
    # (mcvanillaengine.hpp:37-90).

    The constructor reproduces the four C++ guards verbatim:
    ``timeSteps`` or ``timeStepsPerYear`` must be given, not both, and
    neither may be zero.
    """

    def __init__(
        self,
        process: StochasticProcess,
        *,
        time_steps: int | None = None,
        time_steps_per_year: int | None = None,
        brownian_bridge: bool = False,
        antithetic_variate: bool = False,
        control_variate: bool = False,
        required_samples: int | None = None,
        required_tolerance: float | None = None,
        max_samples: int | None = None,
        seed: int = 0,
        rng_traits: RngTraits = PseudoRandom,
        multi_variate: bool = False,
    ) -> None:
        # NOTE: pyright cannot track explicit base-class __init__ forwarding
        # through PEP-695 generic bases; the ignores keep the checker quiet
        # without changing runtime behaviour.
        GenericEngine.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, OptionArguments(), OneAssetOptionResults()
        )
        McSimulation.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            antithetic_variate=antithetic_variate,
            control_variate=control_variate,
        )

        # C++ parity: mcvanillaengine.hpp:111-122.
        qassert.require(
            (time_steps is not None) or (time_steps_per_year is not None),
            "no time steps provided",
        )
        qassert.require(
            (time_steps is None) or (time_steps_per_year is None),
            "both time steps and time steps per year were provided",
        )
        if time_steps is not None:
            qassert.require(time_steps != 0, f"timeSteps must be positive, {time_steps} not allowed")
        if time_steps_per_year is not None:
            qassert.require(
                time_steps_per_year != 0,
                f"timeStepsPerYear must be positive, {time_steps_per_year} not allowed",
            )

        self._process: StochasticProcess = process
        self._time_steps: int | None = time_steps
        self._time_steps_per_year: int | None = time_steps_per_year
        self._required_samples: int | None = required_samples
        self._max_samples: int | None = max_samples
        self._required_tolerance: float | None = required_tolerance
        self._brownian_bridge: bool = brownian_bridge
        self._seed: int = seed
        self._rng_traits: RngTraits = rng_traits
        self._multi_variate: bool = multi_variate
        process.register_with(self)

    # --- engine entry-point ----------------------------------------------

    def calculate(self) -> None:
        """Run the MC and fill ``self._results``.

        # C++ parity: ``MCVanillaEngine::calculate`` (mcvanillaengine.hpp:40-48).
        """
        self.run_mc(
            required_tolerance=self._required_tolerance,
            required_samples=self._required_samples,
            max_samples=self._max_samples,
        )
        assert self._mc_model is not None
        accumulator = self._mc_model.sample_accumulator()
        self._results.value = accumulator.mean()
        # C++ parity: ``if constexpr (RNG::allowsErrorEstimate)``. A
        # low-discrepancy point set is not i.i.d., so C++ leaves
        # ``results_.errorEstimate`` at ``Null<Real>()`` and
        # ``Instrument::errorEstimate()`` throws. Reproduced, not smoothed over.
        if self._rng_traits.allows_error_estimate:
            self._results.error_estimate = accumulator.error_estimate()

    # --- McSimulation hooks ----------------------------------------------

    def time_grid(self) -> TimeGrid:
        """Build the engine's TimeGrid from the option's last exercise date.

        # C++ parity: ``MCVanillaEngine::timeGrid`` (mcvanillaengine.hpp:153-164).
        """
        qassert.require(self._arguments.exercise is not None, "no exercise given")
        assert self._arguments.exercise is not None
        last_exercise_date = self._arguments.exercise.last_date()
        t = self._process.time(last_exercise_date)
        if self._time_steps is not None:
            return TimeGrid.regular(t, self._time_steps)
        assert self._time_steps_per_year is not None
        # C++ ``Size(timeStepsPerYear*t)`` truncates, and ``max(steps, 1)``
        # rescues the zero case.
        steps = int(self._time_steps_per_year * t)
        return TimeGrid.regular(t, max(steps, 1))

    def path_generator(self) -> PathGeneratorTypeProtocol[PathT]:
        """Build a fresh path generator per ``calculate()``.

        # C++ parity: ``MCVanillaEngine::pathGenerator`` (mcvanillaengine.hpp:72-81).
        """
        return self._build_path_generator(self._seed, self.time_grid())

    @abstractmethod
    def path_pricer(self) -> PathPricer[PathT]:
        """Build the path pricer (concrete engine supplies this)."""

    # --- control variate --------------------------------------------------

    def control_pricing_engine(self) -> PricingEngine | None:
        """Engine supplying the control-variate reference value.

        # C++ parity: ``McSimulation::controlPricingEngine`` — returns a null
        # ``shared_ptr`` by default (mcsimulation.hpp:83-85).
        """
        return None

    def control_variate_value(self) -> float | None:
        """Price the option with :meth:`control_pricing_engine`.

        # C++ parity: ``MCVanillaEngine::controlVariateValue``
        # (mcvanillaengine.hpp:127-149) — copies ``arguments_`` into the
        # control engine, calls ``calculate()``, returns ``results->value``.
        """
        control_engine = self.control_pricing_engine()
        qassert.require(
            control_engine is not None,
            "engine does not provide control variation pricing engine",
        )
        assert control_engine is not None
        return self._value_with(control_engine, self._arguments)

    # --- helpers ----------------------------------------------------------

    def _value_with(
        self, control_engine: PricingEngine, arguments: OptionArguments
    ) -> float:
        """Run ``control_engine`` on ``arguments`` and return its NPV.

        # C++ parity: the body of ``MCVanillaEngine::controlVariateValue`` —
        # ``*controlArguments = this->arguments_; controlPE->calculate();``
        # then ``dynamic_cast<const Inst::results*>(...)->value``. The two
        # ``dynamic_cast`` guards become ``isinstance`` checks with the same
        # messages.
        """
        control_arguments = control_engine.get_arguments()
        qassert.require(
            isinstance(control_arguments, OptionArguments),
            "engine is using inconsistent arguments",
        )
        assert isinstance(control_arguments, OptionArguments)
        control_arguments.payoff = arguments.payoff
        control_arguments.exercise = arguments.exercise
        control_engine.reset()
        control_arguments.validate()
        control_engine.calculate()
        control_results = control_engine.get_results()
        qassert.require(
            isinstance(control_results, OneAssetOptionResults),
            "engine returns an inconsistent result type",
        )
        assert isinstance(control_results, OneAssetOptionResults)
        value = control_results.value
        if value is None:
            raise LibraryException("control engine did not produce a value")
        return value

    def _build_path_generator(
        self, seed: int, grid: TimeGrid
    ) -> PathGeneratorTypeProtocol[PathT]:
        """Wire ``factors * (grid.size() - 1)`` Gaussians into a path generator.

        # C++ parity: mcvanillaengine.hpp:74-80 — the dimension is
        # ``process->factors() * (grid.size()-1)`` and the generator comes from
        # ``RNG::make_sequence_generator(dimensions, seed_)``.
        """
        dimensions = self._process.factors() * (len(grid) - 1)
        generator = self._rng_traits.make_sequence_generator(dimensions, seed)
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


__all__ = ["MCVanillaEngine", "RngTraits"]
