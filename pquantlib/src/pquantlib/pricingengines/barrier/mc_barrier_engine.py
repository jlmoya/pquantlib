"""MCBarrierEngine — Monte Carlo pricing for single-barrier options.

# C++ parity: ql/pricingengines/barrier/mcbarrierengine.{hpp,cpp} (v1.43) —
# ``class BarrierPathPricer``, ``class BiasedBarrierPathPricer``,
# ``template <class RNG, class S> class MCBarrierEngine`` and
# ``template <class RNG, class S> class MakeMCBarrierEngine``.

Two path pricers, and the difference between them is the whole point
--------------------------------------------------------------------

:class:`BiasedBarrierPathPricer`
    Looks only at the discrete path nodes.  A path that dips below a down
    barrier *between* two sampling dates is never seen, so a knock-out is
    over-priced and a knock-in under-priced — the discretisation bias the
    class is named for.  Its scan starts at ``path[1]``, so the path's own
    starting value is never tested (the engine's ``triggered(spot)`` guard is
    what makes that safe), and it does **not** stop at the first crossing.

:class:`BarrierPathPricer`
    Adds the Brownian-bridge continuity correction of Beaglehole-Dybvig-Zhou
    and El Babsiri-Noel: conditional on the two endpoints of a step, the
    running extremum of the bridge is sampled from a uniform draw ``u``::

        x   = log(path[i+1] / path[i])
        vol = diffusion(timeGrid[i], path[i])            # time i, value at i
        y   = 0.5 * (x -/+ sqrt(x*x - 2*vol*vol*dt*log(...)))
        y   = path[i] * exp(y)

    with the ``Down`` branches taking ``-`` and ``log(u[i])`` (the minimum)
    and the ``Up`` branches taking ``+`` and ``log(1 - u[i])`` (the maximum).

    The uniforms come from a **separate, hard-coded** generator built by
    :meth:`MCBarrierEngine.path_pricer`::

        PseudoRandom::ursg_type(grid.size()-1, PseudoRandom::urng_type(5))

    — a Mersenne Twister seeded 5, producing *uniforms* rather than Gaussians,
    independent of the engine seed and of the RNG policy (a LowDiscrepancy
    engine still uses this MT19937 stream).  It is a member of the path pricer
    and therefore **stateful across paths**: the n-th path consumes the n-th
    uniform block.  See :data:`BRIDGE_UNIFORM_SEED`.

Both pricers discount a knock-**out** rebate at ``discounts[knock_node]`` —
the *first* crossing — and a never-knocked-in rebate at ``discounts[-1]``.

No control variate
------------------
``MCBarrierEngine`` passes ``controlVariate = false`` to ``McSimulation`` and
overrides neither ``controlPathPricer`` nor ``controlPricingEngine``
(mcbarrierengine.hpp:197).  There is no control variate in v1.43 and
``AnalyticBarrierEngine`` is not involved.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.instruments.barrier_option import BarrierOptionArguments, BarrierType
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng
from pquantlib.math.randomnumbers.random_sequence_generator import (
    RandomSequenceGenerator,
)
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.methods.montecarlo.monte_carlo_model import PathGeneratorTypeProtocol
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_generator import PathGenerator
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.mc_simulation import McSimulation
from pquantlib.pricingengines.vanilla.mc_vanilla_engine import RngTraits
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.time.time_grid import TimeGrid

#: Seed of the uniform generator driving the Brownian-bridge crossing
#: probability.
#:
#: # C++ parity: ``PseudoRandom::urng_type(5)`` (mcbarrierengine.hpp:256).
#: Not configurable in C++, so not configurable here.
BRIDGE_UNIFORM_SEED: Final[int] = 5


class BarrierPathPricer(PathPricer[Path]):
    """Continuity-corrected single-path pricer for a barrier option.

    # C++ parity: ``class BarrierPathPricer``
    # (mcbarrierengine.hpp:140-160 + mcbarrierengine.cpp:27-156).
    """

    __slots__ = (
        "_barrier",
        "_barrier_type",
        "_diff_process",
        "_discounts",
        "_payoff",
        "_rebate",
        "_sequence_gen",
    )

    def __init__(
        self,
        barrier_type: BarrierType,
        barrier: float,
        rebate: float,
        option_type: OptionType,
        strike: float,
        discounts: Sequence[float],
        diff_process: StochasticProcess1D,
        sequence_gen: RandomSequenceGenerator[MersenneTwisterUniformRng],
    ) -> None:
        """# C++ parity: ``BarrierPathPricer::BarrierPathPricer``
        # (mcbarrierengine.cpp:27-42)."""
        qassert.require(strike >= 0.0, "strike less than zero not allowed")
        qassert.require(barrier > 0.0, "barrier less/equal zero not allowed")
        self._barrier_type: BarrierType = barrier_type
        self._barrier: float = barrier
        self._rebate: float = rebate
        self._diff_process: StochasticProcess1D = diff_process
        self._sequence_gen: RandomSequenceGenerator[MersenneTwisterUniformRng] = (
            sequence_gen
        )
        self._payoff: PlainVanillaPayoff = PlainVanillaPayoff(option_type, strike)
        self._discounts: tuple[float, ...] = tuple(discounts)

    def __call__(self, path: Path) -> float:  # noqa: PLR0915 - one-to-one with the C++ body
        """# C++ parity: ``BarrierPathPricer::operator()``
        # (mcbarrierengine.cpp:45-156)."""
        n = path.length()
        qassert.require(n > 1, "the path cannot be empty")

        knock_node: int | None = None
        asset_price = path.front()
        time_grid = path.time_grid
        # One fresh uniform sequence per path, from the seed-5 generator.
        u = self._sequence_gen.next_sequence().value

        # The four branches differ in the sign of the sqrt term, in whether
        # `u` or `1 - u` is used, and in the comparison direction. They are
        # written out rather than folded so the asymmetry stays visible.
        if self._barrier_type == BarrierType.DownIn:
            is_option_active = False
            for i in range(n - 1):
                new_asset_price = path[i + 1]
                # terminal or initial vol?
                vol = self._diff_process.diffusion_1d(time_grid[i], asset_price)
                dt = time_grid.dt(i)
                x = math.log(new_asset_price / asset_price)
                y = 0.5 * (
                    x - math.sqrt(x * x - 2 * vol * vol * dt * math.log(float(u[i])))
                )
                y = asset_price * math.exp(y)
                if y <= self._barrier:
                    is_option_active = True
                    if knock_node is None:
                        knock_node = i + 1
                asset_price = new_asset_price
        elif self._barrier_type == BarrierType.UpIn:
            is_option_active = False
            for i in range(n - 1):
                new_asset_price = path[i + 1]
                vol = self._diff_process.diffusion_1d(time_grid[i], asset_price)
                dt = time_grid.dt(i)
                x = math.log(new_asset_price / asset_price)
                y = 0.5 * (
                    x
                    + math.sqrt(
                        x * x - 2 * vol * vol * dt * math.log(1 - float(u[i]))
                    )
                )
                y = asset_price * math.exp(y)
                if y >= self._barrier:
                    is_option_active = True
                    if knock_node is None:
                        knock_node = i + 1
                asset_price = new_asset_price
        elif self._barrier_type == BarrierType.DownOut:
            is_option_active = True
            for i in range(n - 1):
                new_asset_price = path[i + 1]
                vol = self._diff_process.diffusion_1d(time_grid[i], asset_price)
                dt = time_grid.dt(i)
                x = math.log(new_asset_price / asset_price)
                y = 0.5 * (
                    x - math.sqrt(x * x - 2 * vol * vol * dt * math.log(float(u[i])))
                )
                y = asset_price * math.exp(y)
                if y <= self._barrier:
                    is_option_active = False
                    if knock_node is None:
                        knock_node = i + 1
                asset_price = new_asset_price
        elif self._barrier_type == BarrierType.UpOut:
            is_option_active = True
            for i in range(n - 1):
                new_asset_price = path[i + 1]
                vol = self._diff_process.diffusion_1d(time_grid[i], asset_price)
                dt = time_grid.dt(i)
                x = math.log(new_asset_price / asset_price)
                y = 0.5 * (
                    x
                    + math.sqrt(
                        x * x - 2 * vol * vol * dt * math.log(1 - float(u[i]))
                    )
                )
                y = asset_price * math.exp(y)
                if y >= self._barrier:
                    is_option_active = False
                    if knock_node is None:
                        knock_node = i + 1
                asset_price = new_asset_price
        else:  # pragma: no cover - BarrierType has no fifth member
            raise LibraryException("unknown barrier type")

        return _settle(
            self._barrier_type,
            is_option_active=is_option_active,
            payoff_value=self._payoff(asset_price),
            rebate=self._rebate,
            discounts=self._discounts,
            knock_node=knock_node,
        )


class BiasedBarrierPathPricer(PathPricer[Path]):
    """Discretely-monitored single-path pricer (no continuity correction).

    # C++ parity: ``class BiasedBarrierPathPricer``
    # (mcbarrierengine.hpp:163-179 + mcbarrierengine.cpp:159-247).
    """

    __slots__ = ("_barrier", "_barrier_type", "_discounts", "_payoff", "_rebate")

    def __init__(
        self,
        barrier_type: BarrierType,
        barrier: float,
        rebate: float,
        option_type: OptionType,
        strike: float,
        discounts: Sequence[float],
    ) -> None:
        """# C++ parity: ``BiasedBarrierPathPricer::BiasedBarrierPathPricer``
        # (mcbarrierengine.cpp:159-171)."""
        qassert.require(strike >= 0.0, "strike less than zero not allowed")
        qassert.require(barrier > 0.0, "barrier less/equal zero not allowed")
        self._barrier_type: BarrierType = barrier_type
        self._barrier: float = barrier
        self._rebate: float = rebate
        self._payoff: PlainVanillaPayoff = PlainVanillaPayoff(option_type, strike)
        self._discounts: tuple[float, ...] = tuple(discounts)

    def __call__(self, path: Path) -> float:
        """# C++ parity: ``BiasedBarrierPathPricer::operator()``
        # (mcbarrierengine.cpp:174-247).

        The scan starts at ``path[1]`` — the path's own starting value is
        never tested — and it does not break on the first crossing, so
        ``asset_price`` ends up holding ``path.back()``.
        """
        n = path.length()
        qassert.require(n > 1, "the path cannot be empty")

        knock_node: int | None = None
        asset_price = path.front()

        if self._barrier_type == BarrierType.DownIn:
            is_option_active = False
            for i in range(1, n):
                asset_price = path[i]
                if asset_price <= self._barrier:
                    is_option_active = True
                    if knock_node is None:
                        knock_node = i
        elif self._barrier_type == BarrierType.UpIn:
            is_option_active = False
            for i in range(1, n):
                asset_price = path[i]
                if asset_price >= self._barrier:
                    is_option_active = True
                    if knock_node is None:
                        knock_node = i
        elif self._barrier_type == BarrierType.DownOut:
            is_option_active = True
            for i in range(1, n):
                asset_price = path[i]
                if asset_price <= self._barrier:
                    is_option_active = False
                    if knock_node is None:
                        knock_node = i
        elif self._barrier_type == BarrierType.UpOut:
            is_option_active = True
            for i in range(1, n):
                asset_price = path[i]
                if asset_price >= self._barrier:
                    is_option_active = False
                    if knock_node is None:
                        knock_node = i
        else:  # pragma: no cover - BarrierType has no fifth member
            raise LibraryException("unknown barrier type")

        return _settle(
            self._barrier_type,
            is_option_active=is_option_active,
            payoff_value=self._payoff(asset_price),
            rebate=self._rebate,
            discounts=self._discounts,
            knock_node=knock_node,
        )


def _settle(
    barrier_type: BarrierType,
    *,
    is_option_active: bool,
    payoff_value: float,
    rebate: float,
    discounts: tuple[float, ...],
    knock_node: int | None,
) -> float:
    """Common tail of both ``operator()`` bodies.

    # C++ parity: mcbarrierengine.cpp:142-155 and the byte-identical
    # mcbarrierengine.cpp:233-246. A live path pays its payoff discounted from
    # the last grid point; a knocked-OUT path pays the rebate discounted from
    # the FIRST crossing; a never-knocked-IN path pays the rebate discounted
    # from the last grid point.
    """
    if is_option_active:
        return payoff_value * discounts[-1]
    if barrier_type in (BarrierType.UpIn, BarrierType.DownIn):
        return rebate * discounts[-1]
    # UpOut / DownOut: `knockNode` is the C++ `Null<Size>()` sentinel when the
    # path never knocked, but that branch is unreachable — `isOptionActive`
    # only goes false for a KO path after `knockNode` has been set.
    assert knock_node is not None
    return rebate * discounts[knock_node]


class MCBarrierEngine(
    GenericEngine[BarrierOptionArguments, OneAssetOptionResults],
    McSimulation[Path],
):
    """Monte Carlo pricing engine for single-barrier options.

    # C++ parity: ``template <class RNG, class S> class MCBarrierEngine``
    # (mcbarrierengine.hpp:56-111, 185-270).

    ``S`` is fixed at ``GeneralStatistics`` (the only accumulator C++
    instantiates this with); ``RNG`` survives as the ``rng_traits`` argument,
    defaulting to :class:`~pquantlib.math.randomnumbers.rng_traits.PseudoRandom`
    exactly as the C++ template default does.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        *,
        time_steps: int | None = None,
        time_steps_per_year: int | None = None,
        brownian_bridge: bool = False,
        antithetic_variate: bool = False,
        required_samples: int | None = None,
        required_tolerance: float | None = None,
        max_samples: int | None = None,
        is_biased: bool = False,
        seed: int = 0,
        rng_traits: RngTraits = PseudoRandom,
    ) -> None:
        """# C++ parity: ``MCBarrierEngine::MCBarrierEngine``
        # (mcbarrierengine.hpp:186-214). ``None`` plays the role of
        # ``Null<Size>()`` / ``Null<Real>()``.

        Seed 0 reaches ``SeedGenerator`` through the Mersenne Twister and is
        therefore clock-derived, exactly as in C++ — this port does not
        silently substitute a different seed.
        """
        GenericEngine.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, BarrierOptionArguments(), OneAssetOptionResults()
        )
        McSimulation.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            antithetic_variate=antithetic_variate,
            # C++ parity: ``McSimulation<SingleVariate, RNG, S>(antitheticVariate,
            # false)`` — the control variate is hard-wired off.
            control_variate=False,
        )
        qassert.require(
            time_steps is not None or time_steps_per_year is not None,
            "no time steps provided",
        )
        qassert.require(
            time_steps is None or time_steps_per_year is None,
            "both time steps and time steps per year were provided",
        )
        qassert.require(
            time_steps != 0, f"timeSteps must be positive, {time_steps} not allowed"
        )
        qassert.require(
            time_steps_per_year != 0,
            f"timeStepsPerYear must be positive, {time_steps_per_year} not allowed",
        )

        self._process: GeneralizedBlackScholesProcess = process
        self._time_steps: int | None = time_steps
        self._time_steps_per_year: int | None = time_steps_per_year
        self._required_samples: int | None = required_samples
        self._max_samples: int | None = max_samples
        self._required_tolerance: float | None = required_tolerance
        self._is_biased: bool = is_biased
        self._brownian_bridge: bool = brownian_bridge
        self._seed: int = seed
        self._rng_traits: RngTraits = rng_traits
        process.register_with(self)

    # --- engine entry point ----------------------------------------------

    def _triggered(self, underlying: float) -> bool:
        """# C++ parity: ``BarrierOption::engine::triggered``
        # (ql/instruments/barrieroption.cpp:126-137). Strict ``<`` / ``>``."""
        barrier = self._arguments.barrier
        assert barrier is not None
        if self._arguments.barrier_type in (BarrierType.DownIn, BarrierType.DownOut):
            return underlying < barrier
        return underlying > barrier

    def calculate(self) -> None:
        """# C++ parity: ``MCBarrierEngine::calculate``
        # (mcbarrierengine.hpp:78-89)."""
        spot = self._process.x0()
        qassert.require(spot > 0.0, "negative or null underlying given")
        qassert.require(not self._triggered(spot), "barrier touched")
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
        # ``results_.errorEstimate`` at Null and ``Instrument::errorEstimate()``
        # throws. Reproduced, not smoothed over.
        if self._rng_traits.allows_error_estimate:
            self._results.error_estimate = accumulator.error_estimate()

    # --- McSimulation hooks ----------------------------------------------

    def time_grid(self) -> TimeGrid:
        """# C++ parity: ``MCBarrierEngine::timeGrid``
        # (mcbarrierengine.hpp:216-228)."""
        exercise = self._arguments.exercise
        qassert.require(exercise is not None, "no exercise given")
        assert exercise is not None
        residual_time = self._process.time(exercise.last_date())
        if self._time_steps is not None:
            return TimeGrid.regular(residual_time, self._time_steps)
        if self._time_steps_per_year is not None:
            # C++ ``Size(timeStepsPerYear*t)`` truncates, and ``max(steps, 1)``
            # rescues the zero case.
            steps = int(self._time_steps_per_year * residual_time)
            return TimeGrid.regular(residual_time, max(steps, 1))
        raise LibraryException("time steps not specified")

    def path_generator(self) -> PathGeneratorTypeProtocol[Path]:
        """# C++ parity: ``MCBarrierEngine::pathGenerator``
        # (mcbarrierengine.hpp:94-101)."""
        grid = self.time_grid()
        generator = self._rng_traits.make_sequence_generator(len(grid) - 1, self._seed)
        return PathGenerator.with_time_grid(
            self._process, grid, generator, brownian_bridge=self._brownian_bridge
        )

    def path_pricer(self) -> PathPricer[Path]:
        """# C++ parity: ``MCBarrierEngine::pathPricer``
        # (mcbarrierengine.hpp:231-270)."""
        args = self._arguments
        payoff = args.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)

        grid = self.time_grid()
        discounts = [
            self._process.risk_free_rate().discount(grid[i]) for i in range(len(grid))
        ]

        barrier_type = args.barrier_type
        barrier = args.barrier
        rebate = args.rebate
        assert barrier_type is not None
        assert barrier is not None
        assert rebate is not None

        if self._is_biased:
            return BiasedBarrierPathPricer(
                barrier_type,
                barrier,
                rebate,
                payoff.option_type(),
                payoff.strike(),
                discounts,
            )
        sequence_gen = RandomSequenceGenerator(
            len(grid) - 1, MersenneTwisterUniformRng(BRIDGE_UNIFORM_SEED)
        )
        return BarrierPathPricer(
            barrier_type,
            barrier,
            rebate,
            payoff.option_type(),
            payoff.strike(),
            discounts,
            self._process,
            sequence_gen,
        )


class MakeMCBarrierEngine:
    """Fluent builder for :class:`MCBarrierEngine`.

    # C++ parity: ``template <class RNG, class S> class MakeMCBarrierEngine``
    # (mcbarrierengine.hpp:115-137, 273-367).

    Python has no implicit conversion operator, so the chain ends with the
    explicit :meth:`engine` call — the port of
    ``operator ext::shared_ptr<PricingEngine>() const``, including its two
    guards, which fire at *conversion* time and not at ``with_*`` time.

    Defaults reproduce the C++ member initialisers exactly:
    ``brownianBridge_ = antithetic_ = biased_ = false``, ``seed_ = 0``, and
    ``steps_ / stepsPerYear_ / samples_ / maxSamples_ / tolerance_`` all
    ``Null`` (``None`` here).
    """

    __slots__ = (
        "_antithetic",
        "_biased",
        "_brownian_bridge",
        "_max_samples",
        "_process",
        "_rng_traits",
        "_samples",
        "_seed",
        "_steps",
        "_steps_per_year",
        "_tolerance",
    )

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        rng_traits: RngTraits = PseudoRandom,
    ) -> None:
        """# C++ parity: ``MakeMCBarrierEngine::MakeMCBarrierEngine``
        # (mcbarrierengine.hpp:274-277)."""
        self._process: GeneralizedBlackScholesProcess = process
        self._rng_traits: RngTraits = rng_traits
        self._brownian_bridge: bool = False
        self._antithetic: bool = False
        self._biased: bool = False
        self._steps: int | None = None
        self._steps_per_year: int | None = None
        self._samples: int | None = None
        self._max_samples: int | None = None
        self._tolerance: float | None = None
        self._seed: int = 0

    def with_steps(self, steps: int) -> MakeMCBarrierEngine:
        """# C++ parity: ``withSteps`` (mcbarrierengine.hpp:280-285).

        Deliberately unguarded: the mutual exclusion with
        ``with_steps_per_year`` is checked at conversion time, not here.
        """
        self._steps = steps
        return self

    def with_steps_per_year(self, steps: int) -> MakeMCBarrierEngine:
        """# C++ parity: ``withStepsPerYear`` (mcbarrierengine.hpp:287-292)."""
        self._steps_per_year = steps
        return self

    def with_brownian_bridge(self, brownian_bridge: bool = True) -> MakeMCBarrierEngine:
        """# C++ parity: ``withBrownianBridge`` (mcbarrierengine.hpp:294-299)."""
        self._brownian_bridge = brownian_bridge
        return self

    def with_antithetic_variate(self, b: bool = True) -> MakeMCBarrierEngine:
        """# C++ parity: ``withAntitheticVariate`` (mcbarrierengine.hpp:301-306)."""
        self._antithetic = b
        return self

    def with_samples(self, samples: int) -> MakeMCBarrierEngine:
        """# C++ parity: ``withSamples`` (mcbarrierengine.hpp:308-314)."""
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(self, tolerance: float) -> MakeMCBarrierEngine:
        """# C++ parity: ``withAbsoluteTolerance`` (mcbarrierengine.hpp:316-326).

        The ``QL_REQUIRE(RNG::allowsErrorEstimate)`` is a compile-time constant
        in C++; here it is a runtime check on the traits class.
        """
        qassert.require(self._samples is None, "number of samples already set")
        qassert.require(
            bool(self._rng_traits.allows_error_estimate),
            "chosen random generator policy does not allow an error estimate",
        )
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCBarrierEngine:
        """# C++ parity: ``withMaxSamples`` (mcbarrierengine.hpp:328-333)."""
        self._max_samples = samples
        return self

    def with_bias(self, biased: bool = True) -> MakeMCBarrierEngine:
        """# C++ parity: ``withBias`` (mcbarrierengine.hpp:335-340).

        The default argument is ``true``: ``with_bias()`` selects the
        *biased* (discretely monitored) path pricer.
        """
        self._biased = biased
        return self

    def with_seed(self, seed: int) -> MakeMCBarrierEngine:
        """# C++ parity: ``withSeed`` (mcbarrierengine.hpp:342-347)."""
        self._seed = seed
        return self

    def engine(self) -> MCBarrierEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>() const``
        # (mcbarrierengine.hpp:349-367)."""
        qassert.require(
            self._steps is not None or self._steps_per_year is not None,
            "number of steps not given",
        )
        qassert.require(
            self._steps is None or self._steps_per_year is None,
            "number of steps overspecified",
        )
        return MCBarrierEngine(
            self._process,
            time_steps=self._steps,
            time_steps_per_year=self._steps_per_year,
            brownian_bridge=self._brownian_bridge,
            antithetic_variate=self._antithetic,
            required_samples=self._samples,
            required_tolerance=self._tolerance,
            max_samples=self._max_samples,
            is_biased=self._biased,
            seed=self._seed,
            rng_traits=self._rng_traits,
        )


__all__ = [
    "BRIDGE_UNIFORM_SEED",
    "BarrierPathPricer",
    "BiasedBarrierPathPricer",
    "MCBarrierEngine",
    "MakeMCBarrierEngine",
]
