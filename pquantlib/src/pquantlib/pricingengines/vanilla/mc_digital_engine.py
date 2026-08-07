"""MCDigitalEngine — Monte Carlo pricing for American cash-or-nothing digitals.

# C++ parity: ql/pricingengines/vanilla/mcdigitalengine.{hpp,cpp} (v1.43) —
# ``template <class RNG, class S> class MCDigitalEngine``,
# ``class DigitalPathPricer``, and
# ``template <class RNG, class S> class MakeMCDigitalEngine``.

Prices an *American* (at-hit) cash-or-nothing digital by simulating the asset
path and detecting the first barrier crossing. Because a discrete path can step
over the strike and back without any sampled point ever being past it, the
pricer applies the Brownian-bridge hit correction of

    Beaglehole, Dybvig & Zhou, "Going to Extremes: Correcting Simulation Bias in
    Exotic Option Valuation", Financial Analysts Journal 53(1), 1997, 62-68

    El Babsiri & Noel, "Simulating path-dependent options: A new approach",
    Journal of Derivatives 6(2), 1998, 65-83

which is what makes the engine converge at a usable rate on a coarse grid.

The bridge draws its uniforms from a generator that is *hard-coded* inside
``MCDigitalEngine::pathPricer()`` (mcdigitalengine.hpp:176-177)::

    PseudoRandom::ursg_type sequenceGen(grid.size()-1,
                                        PseudoRandom::urng_type(76));

i.e. a ``RandomSequenceGenerator<MersenneTwisterUniformRng>`` on seed **76**,
producing raw *uniforms* (not Gaussians), one fresh sequence per path, and
completely independent of the engine's own ``seed``. Neither the seed nor the
uniform-vs-Gaussian choice is configurable, and both change every price.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng
from pquantlib.math.randomnumbers.random_sequence_generator import (
    RandomSequenceGenerator,
)
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.payoffs import CashOrNothingPayoff, OptionType
from pquantlib.pricingengines.vanilla.mc_vanilla_engine import MCVanillaEngine, RngTraits
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.termstructures.yield_term_structure import YieldTermStructure

#: Seed of the uniform generator driving the Brownian-bridge hit probability.
#:
#: # C++ parity: ``PseudoRandom::urng_type(76)`` (mcdigitalengine.hpp:177).
#: Not configurable in C++, so not configurable here.
BRIDGE_UNIFORM_SEED = 76


class DigitalPathPricer(PathPricer[Path]):
    """At-hit cash-or-nothing pricer with the Brownian-bridge correction.

    # C++ parity: ``DigitalPathPricer`` (mcdigitalengine.hpp:112-127 +
    # mcdigitalengine.cpp:27-109).
    """

    __slots__ = (
        "_diff_process",
        "_discount_ts",
        "_exercise",
        "_payoff",
        "_sequence_gen",
    )

    def __init__(
        self,
        payoff: CashOrNothingPayoff,
        exercise: AmericanExercise,
        discount_ts: YieldTermStructure,
        diff_process: StochasticProcess1D,
        sequence_gen: RandomSequenceGenerator[MersenneTwisterUniformRng],
    ) -> None:
        self._payoff: CashOrNothingPayoff = payoff
        self._exercise: AmericanExercise = exercise
        self._discount_ts: YieldTermStructure = discount_ts
        self._diff_process: StochasticProcess1D = diff_process
        self._sequence_gen: RandomSequenceGenerator[MersenneTwisterUniformRng] = sequence_gen

    def __call__(self, path: Path) -> float:
        """# C++ parity: ``DigitalPathPricer::operator()``
        # (mcdigitalengine.cpp:36-109)."""
        n = path.length()
        qassert.require(n > 1, "the path cannot be empty")

        log_asset_price = math.log(path.front())
        time_grid = path.time_grid
        # One fresh uniform sequence per path, from the seed-76 generator.
        u = self._sequence_gen.next_sequence().value
        log_strike = math.log(self._payoff.strike())
        option_type = self._payoff.option_type()

        if option_type == OptionType.Call:
            for i in range(n - 1):
                x = math.log(path[i + 1] / path[i])
                # "terminal or initial vol?" -- C++ evaluates the diffusion at
                # timeGrid[i+1] with the asset value at the *start* of the step
                # (the alternative is present but commented out upstream).
                vol = self._diff_process.diffusion_1d(
                    time_grid[i + 1], math.exp(log_asset_price)
                )
                dt = time_grid.dt(i)
                # NOTE the asymmetry with the Put branch: log(1 - u[i]) here.
                y = log_asset_price + 0.5 * (
                    x + math.sqrt(x * x - 2 * vol * vol * dt * math.log(1 - float(u[i])))
                )
                if y >= log_strike:
                    return self._payoff.cash_payoff() * self._hit_discount(path, i)
                log_asset_price += x
        elif option_type == OptionType.Put:
            for i in range(n - 1):
                x = math.log(path[i + 1] / path[i])
                vol = self._diff_process.diffusion_1d(
                    time_grid[i + 1], math.exp(log_asset_price)
                )
                dt = time_grid.dt(i)
                y = log_asset_price + 0.5 * (
                    x - math.sqrt(x * x - 2 * vol * vol * dt * math.log(float(u[i])))
                )
                if y <= log_strike:
                    return self._payoff.cash_payoff() * self._hit_discount(path, i)
                log_asset_price += x
        else:
            # C++ parity: ``default: QL_FAIL("unknown option type")``.
            raise LibraryException("unknown option type")

        return 0.0

    def _hit_discount(self, path: Path, i: int) -> float:
        """Discount factor applied on a hit in step ``i``.

        # C++ parity: mcdigitalengine.cpp:62-72 (and the identical Put block at
        # 89-99). On ``payoffAtExpiry()`` the cash is paid at maturity and is
        # discounted from ``timeGrid.back()``; otherwise it is paid on the hit
        # and is discounted from ``timeGrid[i+1]`` -- the C++ comment notes the
        # exercise time is really somewhere inside ``[timeGrid[i+1],
        # timeGrid[i+2]]``, but the code uses the left endpoint.
        """
        grid = path.time_grid
        if self._exercise.payoff_at_expiry():
            return self._discount_ts.discount(grid.back())
        return self._discount_ts.discount(grid[i + 1])


class MCDigitalEngine(MCVanillaEngine[Path]):
    """Monte Carlo engine for American cash-or-nothing digital options.

    # C++ parity: ``MCDigitalEngine<RNG, S>`` (mcdigitalengine.hpp:60-85,
    # 133-186).

    ``controlVariate`` is hard-wired ``false`` in the C++ constructor
    (mcdigitalengine.hpp:149).
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
        seed: int = 0,
        rng_traits: RngTraits = PseudoRandom,
    ) -> None:
        super().__init__(
            process,
            time_steps=time_steps,
            time_steps_per_year=time_steps_per_year,
            brownian_bridge=brownian_bridge,
            antithetic_variate=antithetic_variate,
            control_variate=False,
            required_samples=required_samples,
            required_tolerance=required_tolerance,
            max_samples=max_samples,
            seed=seed,
            rng_traits=rng_traits,
            multi_variate=False,
        )

    def path_pricer(self) -> PathPricer[Path]:
        """# C++ parity: ``MCDigitalEngine::pathPricer``
        # (mcdigitalengine.hpp:157-186)."""
        payoff = self._arguments.payoff
        qassert.require(isinstance(payoff, CashOrNothingPayoff), "wrong payoff given")
        assert isinstance(payoff, CashOrNothingPayoff)

        exercise = self._arguments.exercise
        qassert.require(isinstance(exercise, AmericanExercise), "wrong exercise given")
        assert isinstance(exercise, AmericanExercise)

        process = self._process
        qassert.require(
            isinstance(process, GeneralizedBlackScholesProcess),
            "Black-Scholes process required",
        )
        assert isinstance(process, GeneralizedBlackScholesProcess)

        grid = self.time_grid()
        sequence_gen = RandomSequenceGenerator(
            len(grid) - 1, MersenneTwisterUniformRng(BRIDGE_UNIFORM_SEED)
        )
        return DigitalPathPricer(
            payoff, exercise, process.risk_free_rate(), process, sequence_gen
        )


class MakeMCDigitalEngine:
    """Fluent builder for :class:`MCDigitalEngine`.

    # C++ parity: ``MakeMCDigitalEngine<RNG, S>``
    # (mcdigitalengine.hpp:88-110, 190-275).

    Same shape as :class:`~pquantlib.pricingengines.vanilla.mc_european_engine.MakeMCEuropeanEngine`:
    every ``with_*`` returns ``self``, and :meth:`engine` is the port of
    ``operator ext::shared_ptr<PricingEngine>() const``, carrying its two
    steps guards.
    """

    __slots__ = (
        "_antithetic",
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
        self._process: GeneralizedBlackScholesProcess = process
        self._rng_traits: RngTraits = rng_traits
        self._antithetic: bool = False
        self._steps: int | None = None
        self._steps_per_year: int | None = None
        self._samples: int | None = None
        self._max_samples: int | None = None
        self._tolerance: float | None = None
        self._brownian_bridge: bool = False
        self._seed: int = 0

    def with_steps(self, steps: int) -> MakeMCDigitalEngine:
        """# C++ parity: ``withSteps`` (mcdigitalengine.hpp:196-200)."""
        self._steps = steps
        return self

    def with_steps_per_year(self, steps: int) -> MakeMCDigitalEngine:
        """# C++ parity: ``withStepsPerYear`` (mcdigitalengine.hpp:203-207)."""
        self._steps_per_year = steps
        return self

    def with_brownian_bridge(self, brownian_bridge: bool = True) -> MakeMCDigitalEngine:
        """# C++ parity: ``withBrownianBridge`` (mcdigitalengine.hpp:245-249)."""
        self._brownian_bridge = brownian_bridge
        return self

    def with_samples(self, samples: int) -> MakeMCDigitalEngine:
        """# C++ parity: ``withSamples`` (mcdigitalengine.hpp:210-216)."""
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(self, tolerance: float) -> MakeMCDigitalEngine:
        """# C++ parity: ``withAbsoluteTolerance`` (mcdigitalengine.hpp:219-228)."""
        qassert.require(self._samples is None, "number of samples already set")
        qassert.require(
            bool(self._rng_traits.allows_error_estimate),
            "chosen random generator policy does not allow an error estimate",
        )
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCDigitalEngine:
        """# C++ parity: ``withMaxSamples`` (mcdigitalengine.hpp:231-235)."""
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> MakeMCDigitalEngine:
        """# C++ parity: ``withSeed`` (mcdigitalengine.hpp:238-242)."""
        self._seed = seed
        return self

    def with_antithetic_variate(self, b: bool = True) -> MakeMCDigitalEngine:
        """# C++ parity: ``withAntitheticVariate`` (mcdigitalengine.hpp:252-256)."""
        self._antithetic = b
        return self

    def engine(self) -> MCDigitalEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>() const``
        # (mcdigitalengine.hpp:259-275)."""
        qassert.require(
            (self._steps is not None) or (self._steps_per_year is not None),
            "number of steps not given",
        )
        qassert.require(
            (self._steps is None) or (self._steps_per_year is None),
            "number of steps overspecified",
        )
        return MCDigitalEngine(
            self._process,
            time_steps=self._steps,
            time_steps_per_year=self._steps_per_year,
            brownian_bridge=self._brownian_bridge,
            antithetic_variate=self._antithetic,
            required_samples=self._samples,
            required_tolerance=self._tolerance,
            max_samples=self._max_samples,
            seed=self._seed,
            rng_traits=self._rng_traits,
        )


__all__ = [
    "BRIDGE_UNIFORM_SEED",
    "DigitalPathPricer",
    "MCDigitalEngine",
    "MakeMCDigitalEngine",
]
