"""MCForwardVanillaEngine — abstract MC engine for forward-starting vanillas.

# C++ parity: ql/pricingengines/forward/mcforwardvanillaengine.hpp (v1.43) —
# ``template <template <class> class MC, class RNG = PseudoRandom,
#  class S = Statistics> class MCForwardVanillaEngine
#  : public GenericEngine<ForwardOptionArguments<VanillaOption::arguments>,
#                         VanillaOption::results>,
#    public McSimulation<MC, RNG, S>``.

Same shape as
:class:`~pquantlib.pricingengines.vanilla.mc_vanilla_engine.MCVanillaEngine`,
over :class:`~pquantlib.instruments.forward_vanilla_option.ForwardOptionArguments`
instead of plain option arguments — but the two differ in the one place that
matters, and it is the place a port gets wrong:

**The time grid carries the reset date as a mandatory point.**
``MCVanillaEngine`` builds ``TimeGrid(T, steps)``, a uniform grid over
``[0, T]``. ``MCForwardVanillaEngine`` builds::

    TimeGrid(fixingTimes.begin(), fixingTimes.end(), totalSteps)
    with fixingTimes = {t(resetDate), t(exerciseDate)}

i.e. the *mandatory points* constructor, which subdivides ``[0, t1]`` and
``[t1, t2]`` separately so that the reset time lands exactly on a node. Without
that, ``resetIndex = timeGrid.closestIndex(resetTime)`` picks a different node,
the strike is read off a different point of the path, and every price is wrong
at a fixed seed while still looking statistically plausible.

Note the asymmetry in ``totalSteps``: when the engine is configured by
``timeStepsPerYear`` the count is ``Size(timeStepsPerYear * t2)`` — driven by
the *exercise* time, not the reset time.

Template parameters, and where they went
----------------------------------------
``MC`` (``SingleVariate`` / ``MultiVariate``) becomes the class type parameter
``PathT`` plus the ``multi_variate`` constructor flag, exactly as in
``MCVanillaEngine``: ``MCForwardEuropeanBSEngine`` pins
``MCForwardVanillaEngine[Path]``, ``MCForwardEuropeanHestonEngine`` pins
``MCForwardVanillaEngine[MultiPath]``.

``RNG`` becomes the ``rng_traits`` argument; ``allows_error_estimate`` gates
whether ``results.error_estimate`` is filled at all (C++
``if constexpr (RNG::allowsErrorEstimate)``).

Seeds
-----
``seed`` reaches the traits' ``make_sequence_generator`` unchanged. Seed 0
therefore routes through ``SeedGenerator`` and is clock-derived, exactly as in
C++; deterministic results need an explicit nonzero seed.
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.instruments.forward_vanilla_option import ForwardOptionArguments
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.methods.montecarlo.monte_carlo_model import PathGeneratorTypeProtocol
from pquantlib.methods.montecarlo.multi_path_generator import MultiPathGenerator
from pquantlib.methods.montecarlo.path_generator import PathGenerator
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.option import OptionArguments
from pquantlib.payoffs import PlainVanillaPayoff, StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.mc_simulation import McSimulation
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.pricingengines.vanilla.mc_vanilla_engine import RngTraits
from pquantlib.processes.stochastic_process import StochasticProcess
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.time.time_grid import TimeGrid


class MCForwardVanillaEngine[PathT](
    GenericEngine[ForwardOptionArguments, OneAssetOptionResults],
    McSimulation[PathT],
):
    """Abstract MC engine for forward-starting (strike-resetting) vanillas.

    # C++ parity: ``MCForwardVanillaEngine<MC, RNG, S>``
    # (mcforwardvanillaengine.hpp:35-87).
    """

    def __init__(
        self,
        process: StochasticProcess,
        *,
        time_steps: int | None = None,
        time_steps_per_year: int | None = None,
        brownian_bridge: bool = False,
        antithetic_variate: bool = False,
        required_samples: int | None = None,
        required_tolerance: float | None = None,
        max_samples: int | None = None,
        seed: int = 0,
        control_variate: bool = False,
        rng_traits: RngTraits = PseudoRandom,
        multi_variate: bool = False,
    ) -> None:
        # NOTE: pyright cannot track explicit base-class __init__ forwarding
        # through PEP-695 generic bases; the ignores keep the checker quiet
        # without changing runtime behaviour.
        GenericEngine.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, ForwardOptionArguments(), OneAssetOptionResults()
        )
        McSimulation.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            antithetic_variate=antithetic_variate,
            control_variate=control_variate,
        )

        # C++ parity: mcforwardvanillaengine.hpp:105-116 — the four guards,
        # verbatim.
        qassert.require(
            (time_steps is not None) or (time_steps_per_year is not None),
            "no time steps provided",
        )
        qassert.require(
            (time_steps is None) or (time_steps_per_year is None),
            "both time steps and time steps per year were provided",
        )
        if time_steps is not None:
            qassert.require(
                time_steps != 0, f"timeSteps must be positive, {time_steps} not allowed"
            )
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

    # --- engine entry-point -------------------------------------------------

    def calculate(self) -> None:
        """Run the MC and fill ``self._results``.

        # C++ parity: ``MCForwardVanillaEngine::calculate``
        # (mcforwardvanillaengine.hpp:57-65).
        """
        self.run_mc(
            required_tolerance=self._required_tolerance,
            required_samples=self._required_samples,
            max_samples=self._max_samples,
        )
        assert self._mc_model is not None
        accumulator = self._mc_model.sample_accumulator()
        self._results.value = accumulator.mean()
        # C++ parity: ``if constexpr (RNG::allowsErrorEstimate)``. Sobol point
        # sets are not i.i.d., so C++ leaves ``results_.errorEstimate`` at
        # ``Null<Real>()`` and ``Instrument::errorEstimate()`` throws.
        if self._rng_traits.allows_error_estimate:
            self._results.error_estimate = accumulator.error_estimate()

    # --- McSimulation hooks -------------------------------------------------

    def time_grid(self) -> TimeGrid:
        """Mandatory-point grid over ``{t(reset), t(exercise)}``.

        # C++ parity: ``MCForwardVanillaEngine::timeGrid``
        # (mcforwardvanillaengine.hpp:120-141).
        """
        args = self._arguments
        qassert.require(args.reset_date is not None, "null reset date given")
        assert args.reset_date is not None
        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None

        t1 = self._process.time(args.reset_date)
        t2 = self._process.time(args.exercise.last_date())

        if self._time_steps is not None:
            total_steps = self._time_steps
        else:
            assert self._time_steps_per_year is not None
            # C++ ``static_cast<Size>(timeStepsPerYear_ * t2)`` truncates, and
            # it is driven by the EXERCISE time, not the reset time.
            total_steps = int(self._time_steps_per_year * t2)

        return TimeGrid.with_mandatory_and_steps([t1, t2], total_steps)

    def path_generator(self) -> PathGeneratorTypeProtocol[PathT]:
        """Build a fresh path generator per ``calculate()``.

        # C++ parity: ``MCForwardVanillaEngine::pathGenerator``
        # (mcforwardvanillaengine.hpp:71-80).
        """
        return self._build_path_generator(self._seed, self.time_grid())

    @abstractmethod
    def path_pricer(self) -> PathPricer[PathT]:
        """Build the path pricer (concrete engine supplies this)."""

    # --- control variate ----------------------------------------------------

    def control_pricing_engine(self) -> PricingEngine | None:
        """Engine supplying the control-variate reference value.

        # C++ parity: ``McSimulation::controlPricingEngine`` returns a null
        # ``shared_ptr`` by default (mcsimulation.hpp:83-85); only
        # ``MCForwardEuropeanHestonEngine`` overrides it.
        """
        return None

    def control_variate_value(self) -> float | None:
        """Price a *plain* vanilla struck at ``moneyness * spot``.

        # C++ parity: ``MCForwardVanillaEngine::controlVariateValue``
        # (mcforwardvanillaengine.hpp:143-175).

        This is NOT the same as ``MCVanillaEngine::controlVariateValue``, which
        copies the whole arguments bundle across. Here the control instrument is
        a different option: the forward-start strike is replaced by
        ``moneyness * process->initialValues()[0]``, i.e. the strike the option
        *would* have if it reset today, and the exercise is carried over
        unchanged.
        """
        control_engine = self.control_pricing_engine()
        qassert.require(
            control_engine is not None,
            "engine does not provide control variation pricing engine",
        )
        assert control_engine is not None

        args = self._arguments
        payoff = args.payoff
        qassert.require(isinstance(payoff, StrikedTypePayoff), "non-plain payoff given")
        assert isinstance(payoff, StrikedTypePayoff)
        assert args.moneyness is not None

        # C++ reads ``process_->initialValues()[0]`` — the asset leg, which for
        # a Heston process is element 0 of a 2-vector, not ``x0()``.
        spot = float(self._process.initial_values()[0])
        strike = args.moneyness * spot
        new_payoff = PlainVanillaPayoff(payoff.option_type(), strike)

        control_arguments = control_engine.get_arguments()
        qassert.require(
            isinstance(control_arguments, OptionArguments),
            "engine is using inconsistent arguments",
        )
        assert isinstance(control_arguments, OptionArguments)
        control_arguments.payoff = new_payoff
        control_arguments.exercise = args.exercise
        control_engine.reset()
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

    # --- helpers ------------------------------------------------------------

    def _build_path_generator(
        self, seed: int, grid: TimeGrid
    ) -> PathGeneratorTypeProtocol[PathT]:
        """Wire ``factors * (grid.size() - 1)`` Gaussians into a path generator.

        # C++ parity: mcforwardvanillaengine.hpp:73-79 — the dimension is
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


__all__ = ["MCForwardVanillaEngine"]
