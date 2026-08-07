"""Monte Carlo double-barrier option engine.

# C++ parity: ql/experimental/barrieroption/mcdoublebarrierengine.{hpp,cpp}
# (v1.43).

Three classes:

* :class:`DoubleBarrierPathPricer` — prices a single path. Only
  ``KnockIn`` and ``KnockOut`` are implemented; ``KIKO`` and ``KOKI`` reach
  the C++ ``QL_FAIL("unknown barrier type")`` and raise here too.
* :class:`MCDoubleBarrierEngine` — ``DoubleBarrierOption::engine`` plus
  ``McSimulation<SingleVariate, RNG, S>``.
* :class:`MakeMCDoubleBarrierEngine` — the fluent builder. Python kwargs
  make builders redundant, but this one is part of the C++ public surface
  (the C++ test suite constructs the engine exclusively through it), so it
  is ported rather than folded away.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.instruments.double_barrier_option import (
    DoubleBarrierOptionArguments,
    DoubleBarrierType,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.methods.montecarlo.gaussian_sequence_generator import (
    make_pseudo_random_rsg,
)
from pquantlib.methods.montecarlo.monte_carlo_model import (
    PathGeneratorTypeProtocol,
)
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_generator import PathGenerator
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.mc_simulation import McSimulation
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.time_grid import TimeGrid


class DoubleBarrierPathPricer(PathPricer[Path]):
    """Single-path pricer for a continuously monitored double barrier.

    # C++ parity: ``class DoubleBarrierPathPricer``
    # (mcdoublebarrierengine.hpp:112-130 + .cpp:25-96).
    """

    __slots__ = (
        "_barrier_high",
        "_barrier_low",
        "_barrier_type",
        "_discounts",
        "_payoff",
        "_rebate",
    )

    def __init__(
        self,
        barrier_type: DoubleBarrierType,
        barrier_low: float,
        barrier_high: float,
        rebate: float,
        option_type: OptionType,
        strike: float,
        discounts: Sequence[float],
    ) -> None:
        qassert.require(strike >= 0.0, "strike less than zero not allowed")
        qassert.require(barrier_low > 0.0, "low barrier less/equal zero not allowed")
        qassert.require(barrier_high > 0.0, "high barrier less/equal zero not allowed")
        self._barrier_type: DoubleBarrierType = barrier_type
        self._barrier_low: float = barrier_low
        self._barrier_high: float = barrier_high
        self._rebate: float = rebate
        self._payoff: PlainVanillaPayoff = PlainVanillaPayoff(option_type, strike)
        self._discounts: tuple[float, ...] = tuple(discounts)

    def __call__(self, path: Path) -> float:
        """# C++ parity: ``DoubleBarrierPathPricer::operator()`` (.cpp:42-96)."""
        n = path.length()
        qassert.require(n > 1, "the path cannot be empty")

        terminal_price = path.back()
        knock_node: int | None = None

        # # C++ parity note: the scan starts at path[1], so the path's own
        # # starting value (path[0]) is never tested against either barrier —
        # # a spot that is already outside the corridor at t=0 does NOT knock.
        # # The engine's `QL_REQUIRE(!triggered(spot))` is what keeps that from
        # # mattering in practice. Reproduced verbatim (.cpp:56, 69).
        if self._barrier_type == DoubleBarrierType.KnockOut:
            is_option_active = True
            for i in range(n - 1):
                new_asset_price = path[i + 1]
                if (
                    new_asset_price >= self._barrier_high
                    or new_asset_price <= self._barrier_low
                ):
                    is_option_active = False
                    if knock_node is None:
                        knock_node = i + 1
                    break
        elif self._barrier_type == DoubleBarrierType.KnockIn:
            is_option_active = False
            for i in range(n - 1):
                new_asset_price = path[i + 1]
                if (
                    new_asset_price >= self._barrier_high
                    or new_asset_price <= self._barrier_low
                ):
                    is_option_active = True
                    if knock_node is None:
                        knock_node = i + 1
                    break
        else:
            raise LibraryException("unknown barrier type")

        if is_option_active:
            return self._payoff(terminal_price) * self._discounts[-1]
        if self._barrier_type == DoubleBarrierType.KnockOut:
            # # C++ parity note: `knockNode` is the C++ `Null<Size>()`
            # # sentinel when the KO path never knocked — but that branch is
            # # unreachable, since `isOptionActive` is only false for a KO
            # # path after `knockNode` has been set (.cpp:89).
            assert knock_node is not None
            return self._rebate * self._discounts[knock_node]
        # KnockIn that never knocked in
        return self._rebate * self._discounts[-1]


class MCDoubleBarrierEngine(
    GenericEngine[DoubleBarrierOptionArguments, OneAssetOptionResults],
    McSimulation[Path],
):
    """Monte Carlo pricing engine for double-barrier options.

    # C++ parity: ``template <class RNG, class S> class MCDoubleBarrierEngine``
    # (mcdoublebarrierengine.hpp:35-86, 134-201).

    The Python port fixes ``RNG = PseudoRandom`` and ``S = Statistics``,
    matching the defaults and the only instantiation the C++ test suite uses.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        time_steps: int | None = None,
        time_steps_per_year: int | None = None,
        brownian_bridge: bool = False,
        antithetic_variate: bool = False,
        required_samples: int | None = None,
        required_tolerance: float | None = None,
        max_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        GenericEngine.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, DoubleBarrierOptionArguments(), OneAssetOptionResults()
        )
        McSimulation.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            antithetic_variate=antithetic_variate,
            control_variate=False,
        )
        # # C++ parity: the four QL_REQUIREs of the ctor
        # # (mcdoublebarrierengine.hpp:149-160). ``None`` plays the role of
        # # ``Null<Size>()``.
        qassert.require(
            time_steps is not None or time_steps_per_year is not None,
            "no time steps provided",
        )
        qassert.require(
            time_steps is None or time_steps_per_year is None,
            "both time steps and time steps per year were provided",
        )
        qassert.require(
            time_steps != 0,
            f"timeSteps must be positive, {time_steps} not allowed",
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
        self._brownian_bridge: bool = brownian_bridge
        self._seed: int = seed
        process.register_with(self)

    # --- engine entry point ----------------------------------------------

    def _triggered(self, underlying: float) -> bool:
        """# C++ parity: ``DoubleBarrierOption::engine::triggered``
        # (doublebarrieroption.cpp:109-111)."""
        barrier_lo = self._arguments.barrier_lo
        barrier_hi = self._arguments.barrier_hi
        assert barrier_lo is not None
        assert barrier_hi is not None
        return underlying <= barrier_lo or underlying >= barrier_hi

    def calculate(self) -> None:
        """# C++ parity: ``MCDoubleBarrierEngine::calculate``
        # (mcdoublebarrierengine.hpp:53-64)."""
        spot = self._process.x0()
        qassert.require(spot > 0.0, "negative or null underlying given")
        qassert.require(not self._triggered(spot), "barrier touched")
        self.run_mc(
            required_tolerance=self._required_tolerance,
            required_samples=self._required_samples,
            max_samples=self._max_samples,
        )
        assert self._mc_model is not None
        self._results.value = self._mc_model.sample_accumulator().mean()
        # PseudoRandom::allowsErrorEstimate is 1.
        self._results.error_estimate = (
            self._mc_model.sample_accumulator().error_estimate()
        )

    # --- McSimulation hooks ----------------------------------------------

    def time_grid(self) -> TimeGrid:
        """# C++ parity: ``MCDoubleBarrierEngine::timeGrid``
        # (mcdoublebarrierengine.hpp:164-176)."""
        exercise = self._arguments.exercise
        qassert.require(exercise is not None, "no exercise given")
        assert exercise is not None
        residual_time = self._process.time(exercise.last_date())
        if self._time_steps is not None:
            return TimeGrid.regular(residual_time, self._time_steps)
        if self._time_steps_per_year is not None:
            steps = int(self._time_steps_per_year * residual_time)
            return TimeGrid.regular(residual_time, max(steps, 1))
        raise LibraryException("time steps not specified")

    def path_generator(self) -> PathGeneratorTypeProtocol[Path]:
        """# C++ parity: ``MCDoubleBarrierEngine::pathGenerator``
        # (mcdoublebarrierengine.hpp:69-76)."""
        grid = self.time_grid()
        # Python divergence, shared with MCVanillaEngine: C++ routes seed 0
        # through the clock-based SeedGenerator; pquantlib substitutes a
        # deterministic 1 so results stay reproducible.
        seed = self._seed if self._seed != 0 else 1
        gsg = make_pseudo_random_rsg(len(grid) - 1, seed)
        return PathGenerator.with_time_grid(
            self._process, grid, gsg, brownian_bridge=self._brownian_bridge
        )

    def path_pricer(self) -> PathPricer[Path]:
        """# C++ parity: ``MCDoubleBarrierEngine::pathPricer``
        # (mcdoublebarrierengine.hpp:178-201)."""
        payoff = self._arguments.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)

        grid = self.time_grid()
        discounts = [self._process.risk_free_rate().discount(grid[i]) for i in range(len(grid))]

        barrier_type = self._arguments.barrier_type
        barrier_lo = self._arguments.barrier_lo
        barrier_hi = self._arguments.barrier_hi
        rebate = self._arguments.rebate
        assert barrier_type is not None
        assert barrier_lo is not None
        assert barrier_hi is not None
        assert rebate is not None

        return DoubleBarrierPathPricer(
            barrier_type,
            barrier_lo,
            barrier_hi,
            rebate,
            payoff.option_type(),
            payoff.strike(),
            discounts,
        )


class MakeMCDoubleBarrierEngine:
    """Fluent builder for :class:`MCDoubleBarrierEngine`.

    # C++ parity: ``template <class RNG, class S> class MakeMCDoubleBarrierEngine``
    # (mcdoublebarrierengine.hpp:89-110, 203-290).

    ``operator ext::shared_ptr<PricingEngine>()`` becomes :meth:`build`
    (Python has no implicit conversion operator); every ``withXxx`` setter
    keeps its C++ ordering constraints.
    """

    __slots__ = (
        "_antithetic",
        "_brownian_bridge",
        "_max_samples",
        "_process",
        "_samples",
        "_seed",
        "_steps",
        "_steps_per_year",
        "_tolerance",
    )

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        self._process: GeneralizedBlackScholesProcess = process
        self._brownian_bridge: bool = False
        self._antithetic: bool = False
        self._steps: int | None = None
        self._steps_per_year: int | None = None
        self._samples: int | None = None
        self._max_samples: int | None = None
        self._tolerance: float | None = None
        self._seed: int = 0

    def with_steps(self, steps: int) -> MakeMCDoubleBarrierEngine:
        """# C++ parity: ``withSteps`` (mcdoublebarrierengine.hpp:209-214)."""
        self._steps = steps
        return self

    def with_steps_per_year(self, steps: int) -> MakeMCDoubleBarrierEngine:
        """# C++ parity: ``withStepsPerYear`` (mcdoublebarrierengine.hpp:216-221)."""
        self._steps_per_year = steps
        return self

    def with_brownian_bridge(
        self, brownian_bridge: bool = True
    ) -> MakeMCDoubleBarrierEngine:
        """# C++ parity: ``withBrownianBridge`` (mcdoublebarrierengine.hpp:223-228)."""
        self._brownian_bridge = brownian_bridge
        return self

    def with_antithetic_variate(self, b: bool = True) -> MakeMCDoubleBarrierEngine:
        """# C++ parity: ``withAntitheticVariate`` (mcdoublebarrierengine.hpp:230-235)."""
        self._antithetic = b
        return self

    def with_samples(self, samples: int) -> MakeMCDoubleBarrierEngine:
        """# C++ parity: ``withSamples`` (mcdoublebarrierengine.hpp:237-244)."""
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(self, tolerance: float) -> MakeMCDoubleBarrierEngine:
        """# C++ parity: ``withAbsoluteTolerance`` (mcdoublebarrierengine.hpp:246-256).

        The ``RNG::allowsErrorEstimate`` QL_REQUIRE always passes here: the
        Python port fixes ``RNG = PseudoRandom``, whose flag is 1.
        """
        qassert.require(self._samples is None, "number of samples already set")
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCDoubleBarrierEngine:
        """# C++ parity: ``withMaxSamples`` (mcdoublebarrierengine.hpp:258-263)."""
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> MakeMCDoubleBarrierEngine:
        """# C++ parity: ``withSeed`` (mcdoublebarrierengine.hpp:265-270)."""
        self._seed = seed
        return self

    def build(self) -> PricingEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>()``
        # (mcdoublebarrierengine.hpp:272-290)."""
        qassert.require(
            self._steps is not None or self._steps_per_year is not None,
            "number of steps not given",
        )
        qassert.require(
            self._steps is None or self._steps_per_year is None,
            "number of steps overspecified",
        )
        return MCDoubleBarrierEngine(
            self._process,
            self._steps,
            self._steps_per_year,
            self._brownian_bridge,
            self._antithetic,
            self._samples,
            self._tolerance,
            self._max_samples,
            self._seed,
        )


__all__ = [
    "DoubleBarrierPathPricer",
    "MCDoubleBarrierEngine",
    "MakeMCDoubleBarrierEngine",
]
