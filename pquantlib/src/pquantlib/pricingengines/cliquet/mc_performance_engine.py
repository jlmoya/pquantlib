"""MCPerformanceEngine — Monte Carlo engine for performance (ratchet) options.

# C++ parity: ql/pricingengines/cliquet/mcperformanceengine.{hpp,cpp} (v1.43) —
# ``template <class RNG = PseudoRandom, class S = Statistics>
#  class MCPerformanceEngine : public CliquetOption::engine,
#                              public McSimulation<SingleVariate, RNG, S>``,
# ``class PerformanceOptionPathPricer : public PathPricer<Path>``, and
# ``template <class RNG, class S> class MakeMCPerformanceEngine``.

A performance option pays, per reset period, ``max(S(t_i)/S(t_{i-1}) - k, 0)``
(call) or ``max(k - S(t_i)/S(t_{i-1}), 0)`` (put), each discounted to today. The
MC counterpart of
:class:`~pquantlib.pricingengines.cliquet.analytic_performance_engine.AnalyticPerformanceEngine`.

Four things a port gets wrong
-----------------------------

1. **The per-period sum starts at ``i = 2``, not ``i = 1``.** The C++ loop is::

       for (Size i = 2; i < n; i++)
           sum += discounts_[i-1] * payoff(path[i]/path[i-1]);

   With a grid ``{0, t_1, ..., t_k, T}`` that pays the periods
   ``(t_1 -> t_2), ..., (t_k -> T)`` and **does not pay** the first period
   ``(0 -> t_1)``. A port that starts the loop at 1 prices one extra period.

2. **The discount for a period is taken at the period's END date**, and by
   *date* — ``riskFreeRate()->discount(resetDates[k])`` — not by grid time. Every
   other engine in this wave discounts at a grid time.

3. **There is no step count.** ``timeGrid()`` is
   ``TimeGrid(fixingTimes.begin(), fixingTimes.end())`` over the reset schedule
   plus the exercise date: the fixing schedule *is* the grid. There is no
   ``withSteps`` / ``withStepsPerYear`` on the builder, and
   ``MakeMCPerformanceEngine``'s conversion operator validates **nothing** — the
   only builder in this wave that does not.

4. **A first reset on the evaluation date breaks the engine.** ``TimeGrid``'s
   iterator constructor prepends 0 only when the first mandatory time is
   ``> 0``. If ``resetDates[0]`` is today, ``t = 0`` is already a fixing, no
   extra node is inserted, and the grid ends up one point shorter than the
   discount vector — so ``PerformanceOptionPathPricer`` throws
   ``"discounts/options mismatch"`` instead of pricing. That is v1.43
   behaviour, pinned by ``perf_first_reset_today_throws``.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.cliquet_option import CliquetOptionArguments
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.methods.montecarlo.monte_carlo_model import PathGeneratorTypeProtocol
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_generator import PathGenerator
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.payoffs import OptionType, PercentageStrikePayoff, PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.mc_simulation import McSimulation
from pquantlib.pricingengines.vanilla.mc_vanilla_engine import RngTraits
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.time_grid import TimeGrid


class PerformanceOptionPathPricer(PathPricer[Path]):
    """Sum of discounted per-period performance payoffs along one path.

    # C++ parity: ``PerformanceOptionPathPricer``
    # (mcperformanceengine.hpp:107-118, .cpp:25-42).
    """

    __slots__ = ("_discounts", "_option_type", "_strike")

    def __init__(
        self,
        option_type: OptionType,
        strike: float,
        discounts: Sequence[float],
    ) -> None:
        self._strike: float = strike
        self._option_type: OptionType = option_type
        self._discounts: tuple[float, ...] = tuple(discounts)

    def __call__(self, path: Path) -> float:
        """# C++ parity: ``PerformanceOptionPathPricer::operator()``
        # (mcperformanceengine.cpp:30-42)."""
        n = path.length()
        qassert.require(n == len(self._discounts) + 1, "discounts/options mismatch")
        payoff = PlainVanillaPayoff(self._option_type, self._strike)

        total = 0.0
        # C++ starts at 2: the first period is deliberately not paid.
        for i in range(2, n):
            total += self._discounts[i - 1] * payoff(path[i] / path[i - 1])
        return total


class MCPerformanceEngine(
    GenericEngine[CliquetOptionArguments, OneAssetOptionResults],
    McSimulation[Path],
):
    """Monte Carlo engine for performance (ratchet) options.

    # C++ parity: ``MCPerformanceEngine<RNG, S>``
    # (mcperformanceengine.hpp:33-80, 123-182).

    There is no ``time_steps`` / ``time_steps_per_year``: the reset schedule is
    the time grid.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        *,
        brownian_bridge: bool = False,
        antithetic_variate: bool = False,
        required_samples: int | None = None,
        required_tolerance: float | None = None,
        max_samples: int | None = None,
        seed: int = 0,
        rng_traits: RngTraits = PseudoRandom,
    ) -> None:
        GenericEngine.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, CliquetOptionArguments(), OneAssetOptionResults()
        )
        McSimulation.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            antithetic_variate=antithetic_variate,
            # C++ passes a literal ``false`` (mcperformanceengine.hpp:132).
            control_variate=False,
        )
        self._process: GeneralizedBlackScholesProcess = process
        self._required_samples: int | None = required_samples
        self._max_samples: int | None = max_samples
        self._required_tolerance: float | None = required_tolerance
        self._brownian_bridge: bool = brownian_bridge
        self._seed: int = seed
        self._rng_traits: RngTraits = rng_traits
        process.register_with(self)

    # --- engine entry-point -------------------------------------------------

    def calculate(self) -> None:
        """# C++ parity: ``MCPerformanceEngine::calculate``
        # (mcperformanceengine.hpp:51-59)."""
        self.run_mc(
            required_tolerance=self._required_tolerance,
            required_samples=self._required_samples,
            max_samples=self._max_samples,
        )
        assert self._mc_model is not None
        accumulator = self._mc_model.sample_accumulator()
        self._results.value = accumulator.mean()
        if self._rng_traits.allows_error_estimate:
            self._results.error_estimate = accumulator.error_estimate()

    # --- McSimulation hooks -------------------------------------------------

    def time_grid(self) -> TimeGrid:
        """The fixing schedule itself: reset times plus the exercise time.

        # C++ parity: ``MCPerformanceEngine::timeGrid``
        # (mcperformanceengine.hpp:139-149).
        """
        args = self._arguments
        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        fixing_times = [self._process.time(d) for d in args.reset_dates]
        fixing_times.append(self._process.time(args.exercise.last_date()))
        # C++ TimeGrid(begin, end) prepends 0 only when the first time is > 0.
        return TimeGrid.with_mandatory(fixing_times)

    def path_generator(self) -> PathGeneratorTypeProtocol[Path]:
        """# C++ parity: ``MCPerformanceEngine::pathGenerator``
        # (mcperformanceengine.hpp:64-72).

        Dimension is ``grid.size() - 1``, with no ``factors()`` multiplier — the
        C++ text.
        """
        grid = self.time_grid()
        generator = self._rng_traits.make_sequence_generator(len(grid) - 1, self._seed)
        return PathGenerator.with_time_grid(
            self._process, grid, generator, brownian_bridge=self._brownian_bridge
        )

    def path_pricer(self) -> PathPricer[Path]:
        """# C++ parity: ``MCPerformanceEngine::pathPricer``
        # (mcperformanceengine.hpp:152-182)."""
        args = self._arguments
        payoff = args.payoff
        qassert.require(
            isinstance(payoff, PercentageStrikePayoff), "non-percentage payoff given"
        )
        assert isinstance(payoff, PercentageStrikePayoff)

        qassert.require(
            isinstance(args.exercise, EuropeanExercise), "wrong exercise given"
        )
        assert args.exercise is not None

        risk_free = self._process.risk_free_rate()
        # Discounts by DATE, one per reset plus one at the exercise date.
        discounts = [risk_free.discount(d) for d in args.reset_dates]
        discounts.append(risk_free.discount(args.exercise.last_date()))

        return PerformanceOptionPathPricer(
            payoff.option_type(), payoff.strike(), discounts
        )


class MakeMCPerformanceEngine:
    """Fluent builder for :class:`MCPerformanceEngine`.

    # C++ parity: ``MakeMCPerformanceEngine<RNG, S>``
    # (mcperformanceengine.hpp:85-103, 186-252).

    Deliberately missing, because C++ does not have them: ``with_steps`` and
    ``with_steps_per_year``. And :meth:`engine` performs **no** validation —
    a builder with neither samples nor tolerance converts fine and only fails
    later, inside ``McSimulation``, with "neither tolerance nor number of
    samples set".
    """

    __slots__ = (
        "_antithetic",
        "_brownian_bridge",
        "_max_samples",
        "_process",
        "_rng_traits",
        "_samples",
        "_seed",
        "_tolerance",
    )

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        rng_traits: RngTraits = PseudoRandom,
    ) -> None:
        self._process: GeneralizedBlackScholesProcess = process
        self._rng_traits: RngTraits = rng_traits
        self._brownian_bridge: bool = False
        self._antithetic: bool = False
        self._samples: int | None = None
        self._max_samples: int | None = None
        self._tolerance: float | None = None
        self._seed: int = 0

    def with_brownian_bridge(self, b: bool = True) -> MakeMCPerformanceEngine:
        """# C++ parity: ``withBrownianBridge(bool b = true)``
        # (mcperformanceengine.hpp:191-196)."""
        self._brownian_bridge = b
        return self

    def with_antithetic_variate(self, b: bool = True) -> MakeMCPerformanceEngine:
        """# C++ parity: ``withAntitheticVariate(bool b = true)`` (hpp:198-203)."""
        self._antithetic = b
        return self

    def with_samples(self, samples: int) -> MakeMCPerformanceEngine:
        """# C++ parity: ``withSamples`` (hpp:205-212)."""
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(self, tolerance: float) -> MakeMCPerformanceEngine:
        """# C++ parity: ``withAbsoluteTolerance`` (hpp:214-224)."""
        qassert.require(self._samples is None, "number of samples already set")
        qassert.require(
            bool(self._rng_traits.allows_error_estimate),
            "chosen random generator policy does not allow an error estimate",
        )
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCPerformanceEngine:
        """# C++ parity: ``withMaxSamples`` (hpp:226-231)."""
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> MakeMCPerformanceEngine:
        """# C++ parity: ``withSeed`` (hpp:233-238)."""
        self._seed = seed
        return self

    def engine(self) -> MCPerformanceEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>() const``
        # (hpp:240-252) — no guards at all, deliberately."""
        return MCPerformanceEngine(
            self._process,
            brownian_bridge=self._brownian_bridge,
            antithetic_variate=self._antithetic,
            required_samples=self._samples,
            required_tolerance=self._tolerance,
            max_samples=self._max_samples,
            seed=self._seed,
            rng_traits=self._rng_traits,
        )


__all__ = [
    "MCPerformanceEngine",
    "MakeMCPerformanceEngine",
    "PerformanceOptionPathPricer",
]
